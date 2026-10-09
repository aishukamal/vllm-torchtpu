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
"""Sleep-mode memory mechanism for the TPU worker.

vLLM's sleep mode on CUDA remaps the allocator's virtual ranges; TPU has no
such allocator, so the worker swaps the storage behind every live tensor
instead. The tensor objects stay where they are (module attributes, the
runner's ``kv_caches`` list, the forward-context bindings, compiled graphs)
and only their ``.data`` is replaced: by a zero-element tensor while asleep
and by a fresh allocation of the recorded geometry on wake.

KV cache tensors may alias one device allocation (the block-major bundle is
one tensor with a ``bundle[:, index]`` view per layer; unified pools are one
object held by every layer). Aliases are grouped by storage, the root
allocation is re-created once, and each view is rebuilt from the root with
the same op that produced it, so the wake leaves exactly the layout the
runner allocated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_device

logger = init_logger(__name__)

SLEEP_TAGS: tuple[str, ...] = ("weights", "kv_cache")


@dataclass
class WeightRecord:
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    host: torch.Tensor | None

    @property
    def nbytes(self) -> int:
        return _nbytes(self.shape, self.dtype)


@dataclass(frozen=True)
class SelectView:
    dim: int
    index: int

    def apply(self, root: torch.Tensor) -> torch.Tensor:
        return root.select(self.dim, self.index)


@dataclass(frozen=True)
class ReshapeView:
    shape: tuple[int, ...]

    def apply(self, root: torch.Tensor) -> torch.Tensor:
        return root.view(self.shape)


@dataclass
class KVViewRecord:
    tensor: torch.Tensor
    recipe: SelectView | ReshapeView


@dataclass
class KVStorageRecord:
    root: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    views: list[KVViewRecord]

    @property
    def nbytes(self) -> int:
        return _nbytes(self.shape, self.dtype)


@dataclass
class SleepState:
    level: int
    sleeping_tags: set[str]
    weights: list[WeightRecord]
    kv_cache: list[KVStorageRecord]


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * dtype.itemsize


def _release_storage(tensor: torch.Tensor) -> None:
    tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)


def _named_tensors(
    module: nn.Module, seen: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Non-empty parameters and buffers, each tensor object once."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if tensor.numel() == 0 or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        out.append((name, tensor))
    return out


def release_weights(modules: Iterable[nn.Module], level: int) -> list[WeightRecord]:
    """Drop the device storage of every parameter and buffer of ``modules``.

    Level 1 keeps a host copy of everything so ``restore_weights`` reinstalls
    the model as it was. Level 2 keeps host copies of buffers only, as the GPU
    worker does: a weight load after wake rewrites parameters, but nothing
    rewrites buffers such as rotary caches.
    """
    assert level in (1, 2), f"sleep level must be 1 or 2, got {level}"
    records: list[WeightRecord] = []
    seen: set[int] = set()
    for module in modules:
        buffer_ids = {id(buffer) for _, buffer in module.named_buffers()}
        for name, tensor in _named_tensors(module, seen):
            keep_host_copy = level == 1 or id(tensor) in buffer_ids
            records.append(
                WeightRecord(
                    name=name,
                    tensor=tensor,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    host=tensor.detach().to("cpu") if keep_host_copy else None,
                )
            )
    synchronize_device()
    for record in records:
        _release_storage(record.tensor)
    return records


def restore_weights(records: Sequence[WeightRecord]) -> None:
    """Give every recorded tensor fresh device storage and reinstall host copies.

    Tensors without a host copy come back uninitialized; the caller loads
    weights into them in place, as on GPU after a level-2 sleep.
    """
    for record in records:
        record.tensor.data = torch.empty(
            record.shape, dtype=record.dtype, device=record.device
        )
    with torch.no_grad():
        for record in records:
            if record.host is not None:
                record.tensor.copy_(record.host)
                record.host = None
    synchronize_device()


def _is_storage_root(tensor: torch.Tensor) -> bool:
    return (
        tensor.storage_offset() == 0
        and tensor.is_contiguous()
        and tensor.numel() * tensor.element_size() == tensor.untyped_storage().nbytes()
    )


def _view_recipe(root: torch.Tensor, view: torch.Tensor) -> SelectView | ReshapeView:
    root_shape = tuple(root.shape)
    root_stride = tuple(root.stride())
    view_shape = tuple(view.shape)
    view_stride = tuple(view.stride())
    offset = view.storage_offset()
    for dim in range(root.dim()):
        rest_shape = root_shape[:dim] + root_shape[dim + 1 :]
        rest_stride = root_stride[:dim] + root_stride[dim + 1 :]
        step = root_stride[dim]
        if (
            view_shape == rest_shape
            and view_stride == rest_stride
            and step > 0
            and offset % step == 0
            and offset // step < root_shape[dim]
        ):
            return SelectView(dim=dim, index=offset // step)
    if offset == 0 and view.is_contiguous() and view.numel() == root.numel():
        return ReshapeView(shape=view_shape)
    raise NotImplementedError(
        "sleep mode cannot rebuild a KV cache view with "
        f"shape={view_shape} stride={view_stride} offset={offset} of a root "
        f"with shape={root_shape} stride={root_stride}; only select and "
        "reshape views are supported"
    )


def record_kv_cache(tensors: Iterable[torch.Tensor]) -> list[KVStorageRecord]:
    """Group the live KV cache tensors by device allocation.

    ``tensors`` must include the root of every allocation (the runner's raw
    pools and the block-major bundle, not only the per-layer views).
    """
    by_storage: dict[int, list[torch.Tensor]] = {}
    seen: set[int] = set()
    for tensor in tensors:
        if tensor.numel() == 0 or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        by_storage.setdefault(tensor.untyped_storage().data_ptr(), []).append(tensor)

    records: list[KVStorageRecord] = []
    for aliases in by_storage.values():
        roots = [tensor for tensor in aliases if _is_storage_root(tensor)]
        if not roots:
            raise NotImplementedError(
                "sleep mode found KV cache views whose root allocation is not "
                "tracked by the runner: shapes "
                f"{[tuple(tensor.shape) for tensor in aliases]}"
            )
        root = roots[0]
        records.append(
            KVStorageRecord(
                root=root,
                shape=tuple(root.shape),
                dtype=root.dtype,
                device=root.device,
                views=[
                    KVViewRecord(tensor=tensor, recipe=_view_recipe(root, tensor))
                    for tensor in aliases
                    if tensor is not root
                ],
            )
        )
    return records


def release_kv_cache(records: Sequence[KVStorageRecord]) -> None:
    for record in records:
        for view in record.views:
            _release_storage(view.tensor)
        _release_storage(record.root)


def restore_kv_cache(records: Sequence[KVStorageRecord]) -> None:
    """Re-create every KV allocation zero-filled, as ``initialize_kv_cache`` did,
    and re-point the views at it."""
    for record in records:
        record.root.data = torch.zeros(
            record.shape, dtype=record.dtype, device=record.device
        )
        for view in record.views:
            view.tensor.data = view.recipe.apply(record.root)
    synchronize_device()
