# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for `VmemArena` and `ArenaSlice`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jax import export
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.dma import ArenaSlice, VmemArena
from vllm_torchtpu.megakernel.dma.arena import ALIGNMENT_BYTES
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32

if TYPE_CHECKING:
    from typing import Any

    RefLike = Any


def test_vmem_arena_interval_reuse_and_4kib_alignment() -> None:
    """Verify 4 KiB alignment, disjoint interval reuse, and O(max_s B_s) peak."""
    arena = VmemArena(capacity_bytes=64 * 1024)
    assert arena.capacity_bytes == 64 * 1024
    assert arena.peak_bytes == 0

    # stage_a: (32, 256) bf16 = 16 KiB -> [0, 16384)
    slc_a = arena.allocate("stage_a", (32, 256), BF16)
    assert type(slc_a) is ArenaSlice
    assert slc_a.offset_bytes == 0
    assert slc_a.size_bytes == 16 * 1024
    assert slc_a.offset_bytes % ALIGNMENT_BYTES == 0
    assert slc_a.size_bytes % ALIGNMENT_BYTES == 0
    assert arena.peak_bytes == 16 * 1024

    # stage_b replaces stage_a: (64, 256) fp8_e4m3fn = 16 KiB -> reuses [0, 16384)
    slc_b = arena.allocate("stage_b", (64, 256), FP8, replaces=("stage_a",))
    assert slc_b.offset_bytes == 0
    assert slc_b.size_bytes == 16 * 1024
    assert arena.peak_bytes == 16 * 1024

    # stage_c concurrent with stage_b: (16, 256) f32 = 16 KiB -> [16384, 32768)
    slc_c = arena.allocate("stage_c", (16, 256), F32)
    assert slc_c.offset_bytes == 16 * 1024
    assert slc_c.size_bytes == 16 * 1024
    assert arena.peak_bytes == 32 * 1024

    # Non-tile shape rounds up to 4 KiB boundary: (5, 128) i32 -> (8, 128)
    slc_pad = arena.allocate("stage_pad", (5, 128), I32)
    assert slc_pad.shape == (5, 128)
    assert slc_pad.tiled_shape == (8, 128)
    assert slc_pad.offset_bytes == 32 * 1024
    assert slc_pad.size_bytes == 4 * 1024
    assert arena.peak_bytes == 36 * 1024

    # stage_d replaces stage_b and stage_c: (32, 256) f32 = 32 KiB -> fits in [0, 32768)
    slc_d = arena.allocate("stage_d", (32, 256), F32, replaces=("stage_b", "stage_c"))
    assert slc_d.offset_bytes == 0
    assert slc_d.size_bytes == 32 * 1024
    total_allocated = (
        slc_a.size_bytes
        + slc_b.size_bytes
        + slc_c.size_bytes
        + slc_pad.size_bytes
        + slc_d.size_bytes
    )
    assert total_allocated == 84 * 1024
    assert arena.peak_bytes == 36 * 1024 < total_allocated


def test_vmem_arena_pallas_view_read_write_parity() -> None:
    """Verify `VmemArena.view` in CPU `interpret=True` and TPU `jax.export`."""
    arena = VmemArena(capacity_bytes=32 * 1024)
    slc_s1 = arena.allocate("s1", (32, 256), BF16)  # [0, 16 KiB)
    slc_s2 = arena.allocate("s2", (16, 256), F32)  # [16 KiB, 32 KiB)
    arena.release("s1")
    slc_s3 = arena.allocate("s3", (64, 256), FP8)  # reuses [0, 16 KiB)
    assert slc_s3.offset_bytes == slc_s1.offset_bytes == 0
    assert arena.peak_bytes == 32 * 1024

    def kernel_interp(
        x_ref: RefLike,
        y_ref: RefLike,
        out_s2_ref: RefLike,
        out_s3_ref: RefLike,
        pool_ref: RefLike,
    ) -> None:
        """Read and write aliased arena views in CPU interpret mode."""
        v1 = arena.view(pool_ref, "s1")
        v2 = arena.view(pool_ref, slc_s2)
        v1[...] = x_ref[...]
        v2[...] = v1[:16, :].astype(F32) * 2.0
        out_s2_ref[...] = v2[...]

        # Stage 3 reuses Stage 1's reclaimed interval without corrupting Stage 2.
        v3 = arena.view(pool_ref, slc_s3)
        v3[...] = y_ref[...]
        out_s3_ref[...] = v3[...]
        out_s2_ref[...] = v2[...] + 1.0

    call_interp = pl.pallas_call(
        kernel_interp,
        out_shape=(
            jax.ShapeDtypeStruct((16, 256), F32),
            jax.ShapeDtypeStruct((64, 256), FP8),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        scratch_shapes=[arena.scratch_spec(interpret=True)],
        interpret=True,
    )
    x = (jnp.arange(32 * 256, dtype=jnp.int32) % 31).reshape(32, 256).astype(BF16)
    y = (jnp.arange(64 * 256, dtype=jnp.int32) % 19).reshape(64, 256).astype(FP8)
    got_s2, got_s3 = jax.jit(call_interp)(x, y)
    assert jnp.array_equal(got_s2, x[:16, :].astype(F32) * 2.0 + 1.0)
    assert jnp.array_equal(got_s3, y)

    # Cross-lower for TPU via jax.export to verify Mosaic tpu.reinterpret_cast lowering.
    def kernel_tpu(
        x_ref: RefLike,
        o_ref: RefLike,
        pool_ref: RefLike,
        sem: RefLike,
    ) -> None:
        """Lower aliased VMEM pool views with async HBM-to-VMEM DMA copies on TPU."""
        v1 = arena.view(pool_ref, slc_s1)
        v2 = arena.view(pool_ref, slc_s2)
        cp_in = pltpu.make_async_copy(x_ref, v1, sem)
        cp_in.start()
        cp_in.wait()
        v2[...] = v1[:16, :].astype(F32)
        cp_out = pltpu.make_async_copy(v2, o_ref, sem)
        cp_out.start()
        cp_out.wait()

    call_tpu = pl.pallas_call(
        kernel_tpu,
        out_shape=jax.ShapeDtypeStruct((16, 256), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[arena.scratch_spec(), pltpu.SemaphoreType.DMA(())],
    )
    exported = export.export(jax.jit(call_tpu), platforms=["tpu"])(
        jax.ShapeDtypeStruct((32, 256), BF16)
    )
    assert "tpu_custom_call" in exported.mlir_module()


def test_vmem_arena_rejects_overrun_and_loop_signature_drift() -> None:
    """Verify `MegakernelConfigError` on capacity overrun and loop signature drift."""
    for bad_cap in (0, -4096, 1024, 6000):
        with pytest.raises(MegakernelConfigError, match="capacity_bytes"):
            VmemArena(capacity_bytes=bad_cap)

    arena = VmemArena(capacity_bytes=16 * 1024)
    with pytest.raises(MegakernelConfigError, match="at least 2D"):
        arena.allocate("bad_rank", (256,), BF16)

    arena.allocate("live_buf", (32, 256), BF16)  # fills 16 KiB
    with pytest.raises((MegakernelConfigError, ValueError), match="already live"):
        arena.allocate("live_buf", (16, 128), BF16)
    with pytest.raises(MegakernelConfigError, match="not currently live"):
        arena.release("missing_buf")
    with pytest.raises(MegakernelConfigError, match="not currently live"):
        arena.allocate("next_buf", (16, 128), BF16, replaces=("missing_buf",))
    with pytest.raises(MegakernelConfigError, match="exceeds capacity"):
        arena.allocate("overflow_buf", (16, 128), BF16)

    # Failed allocate(..., replaces=...) must leave the replaced slice live.
    with pytest.raises(MegakernelConfigError, match="exceeds capacity"):
        arena.allocate("too_big", (64, 256), BF16, replaces=("live_buf",))
    assert [s.name for s in arena.live_slices()] == ["live_buf"]

    # Sub-byte dtype rejection.
    with pytest.raises(ValueError, match="Sub-byte dtype"):
        arena.allocate("sub_byte", (16, 128), jnp.int4, replaces=("live_buf",))
    assert [s.name for s in arena.live_slices()] == ["live_buf"]

    # Unknown slice name in view() raises MegakernelConfigError.
    with pytest.raises(MegakernelConfigError, match="unknown arena slice"):
        arena.view({}, "nonexistent_slice")

    # Duplicate slice name after release must raise ValueError.
    arena.release("live_buf")
    with pytest.raises(ValueError, match="Duplicate slice name"):
        arena.allocate("live_buf", (16, 128), BF16)

    # Allocating or releasing after scratch_spec() freezes the arena.
    _ = arena.scratch_spec()
    with pytest.raises(RuntimeError, match="after scratch_spec"):
        arena.allocate("post_freeze", (16, 128), BF16)
    with pytest.raises(RuntimeError, match="after scratch_spec"):
        arena.release("live_buf")

    # Loop-carry invariant check: signature before and after a loop body must match.
    loop_arena = VmemArena(capacity_bytes=32 * 1024)
    loop_arena.allocate("carry", (16, 256), BF16)
    before_sig = loop_arena.signature()

    # Valid iteration: allocates and releases temporary buffer, restoring signature.
    loop_arena.allocate("tmp", (16, 256), F32)
    loop_arena.release("tmp")
    loop_arena.assert_matching_signature(before_sig)

    # Invalid iteration: leaks `tmp` across loop boundary -> raises error.
    loop_arena.allocate("tmp_leak", (16, 256), F32)
    with pytest.raises(MegakernelConfigError, match="signature drift"):
        loop_arena.assert_matching_signature(before_sig)
