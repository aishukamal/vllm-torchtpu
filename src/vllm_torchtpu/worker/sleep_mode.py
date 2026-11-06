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

vLLM's GPU worker hands sleep and wake to a ``SleepModeBackend`` (``cumem``
by default) that offloads or drops whole allocator pools. TPU has no such
allocator, so the worker swaps the storage behind every live tensor instead;
there is no backend to select. The tensor objects stay where they are
(module attributes, the runner's ``kv_caches`` list, the forward-context
bindings, compiled graphs) and only their ``.data`` is replaced: by a
zero-element tensor while asleep and by a fresh allocation of the recorded
geometry on wake.

Parameters are named as the GPU worker names them: the model's own names,
and ``draft.<name>`` for draft-model parameters the model does not share. The
``sleep_preserve_parameter_names`` globs of ``ModelConfig`` resolve against
those names at level 2; a matching parameter is left out of the release plan
and stays resident on device, which is the TPU form of what the GPU worker
achieves by saving such parameters to host and writing them back on wake.

KV cache tensors may alias one device allocation (the block-major bundle is
one tensor with a ``bundle[:, index]`` view per layer; unified pools are one
object held by every layer). Aliases are grouped by storage, the root
allocation is re-created once, and each view is rebuilt from the root with
the same op that produced it, so the wake leaves exactly the layout the
runner allocated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import filter as fnmatch_filter

import torch
from torch import nn

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_device

logger = init_logger(__name__)

SLEEP_TAGS: tuple[str, ...] = ("weights", "kv_cache")
DRAFT_PREFIX = "draft."


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
    preserved: list[str]
    """Parameter names left resident on device by a level-2 sleep."""


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * dtype.itemsize


def _release_storage(tensor: torch.Tensor) -> None:
    tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)


def device_memory_info(devices: Iterable[torch.device]) -> tuple[int, int]:
    """Free and total device bytes summed over ``devices``, after a sync.

    Reads ``torch.accelerator.get_memory_info`` per device, as the worker's
    HBM budget does, so the freed-bytes figure the worker logs is on the same
    scale as its startup accounting.
    """
    synchronize_device()
    free = total = 0
    for device in devices:
        device_free, device_total = torch.accelerator.get_memory_info(device)
        free += device_free
        total += device_total
    return free, total


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


def sleep_parameters(
    model: nn.Module, draft_model: nn.Module | None = None
) -> dict[str, nn.Parameter]:
    """Parameters by the names sleep-retention globs match against.

    Same naming as the GPU worker: the model's ``named_parameters``, then the
    draft model's as ``draft.<name>``, leaving out draft parameters that are
    the model's own objects (a shared embedding or lm_head).
    """
    parameters = dict(model.named_parameters())
    if draft_model is not None:
        shared = {id(parameter) for parameter in parameters.values()}
        parameters.update(
            (DRAFT_PREFIX + name, parameter)
            for name, parameter in draft_model.named_parameters()
            if id(parameter) not in shared
        )
    return parameters


def preserved_parameter_names(
    parameters: Mapping[str, torch.Tensor], patterns: Sequence[str]
) -> set[str]:
    """Resolve ``sleep_preserve_parameter_names`` globs against ``parameters``.

    A glob that matches nothing is an error, as on GPU, except ``draft.*``,
    which is allowed to match nothing when the model has no draft model.
    """
    names: set[str] = set()
    for pattern in patterns:
        matches = fnmatch_filter(parameters, pattern)
        if not matches and pattern != DRAFT_PREFIX + "*":
            raise ValueError(f"No parameter matches sleep retention: {pattern}")
        names.update(matches)
    return names


def record_weights(
    model: nn.Module,
    level: int,
    draft_model: nn.Module | None = None,
    preserve_patterns: Sequence[str] = (),
) -> tuple[list[WeightRecord], list[str]]:
    """Plan the release of the model's and the draft model's weights.

    Level 1 keeps a host copy of everything so ``restore_weights`` reinstalls
    the model as it was. Level 2 keeps host copies of buffers only, of the
    model and of the draft model, as the GPU worker does: a weight load after
    wake rewrites parameters, but nothing rewrites buffers such as rotary
    caches. At level 2, parameters matching ``preserve_patterns`` are left out
    of the plan and keep their device storage. Nothing is released here, so an
    unmatched glob fails before any storage is dropped.

    Returns the records to release and the sorted names of the parameters
    that stay resident.
    """
    assert level in (1, 2), f"sleep level must be 1 or 2, got {level}"
    preserved: set[str] = set()
    if level == 2 and preserve_patterns:
        preserved = preserved_parameter_names(
            sleep_parameters(model, draft_model), preserve_patterns
        )
    records: list[WeightRecord] = []
    seen: set[int] = set()
    for prefix, module in (("", model), (DRAFT_PREFIX, draft_model)):
        if module is None:
            continue
        buffer_ids = {id(buffer) for _, buffer in module.named_buffers()}
        for name, tensor in _named_tensors(module, seen):
            if prefix + name in preserved:
                continue
            keep_host_copy = level == 1 or id(tensor) in buffer_ids
            records.append(
                WeightRecord(
                    name=prefix + name,
                    tensor=tensor,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    host=tensor.detach().to("cpu") if keep_host_copy else None,
                )
            )
    return records, sorted(preserved)


def release_weights(records: Sequence[WeightRecord]) -> None:
    """Drop the device storage of every recorded tensor."""
    synchronize_device()
    for record in records:
        _release_storage(record.tensor)


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
