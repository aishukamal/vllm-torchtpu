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
"""Runner-facing hook types for the megakernel adapter.

Kept free of `jax` / `torch` imports so the runner can import them at module
load without pulling in the megakernel library.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from vllm_torchtpu import envs

from .config import import_plugin, missing_template, resolve_callable, select_plugin

_SUPPRESS_ATTR = "_megakernel_suppress_depth"
_LOAD_TENSOR_PARAMS_WITH_OPTIONS: int = 3


@dataclasses.dataclass(frozen=True)
class StepContext:
    """The inputs of one runner forward, exactly as the stock model gets them.

    `model_kwargs` are the extra keyword arguments the runner passes to the
    stock model's forward; a megakernel that can't consume them declines the
    step. A megakernel that handles the step returns what the stock model
    would: hidden states, or `(hidden_states, aux_hidden_states)`; the runner
    normalizes both paths the same way.
    """

    input_ids: Any
    positions: Any
    inputs_embeds: Any = None
    intermediate_tensors: Any = None
    model_kwargs: Mapping[str, Any] | None = None
    # The step's token ids when the runner fed the stock model
    # `inputs_embeds` built from them alone (a multimodal-capable model, but
    # no multimodal embeddings in this step); `None` otherwise.
    token_ids: Any = None


@contextlib.contextmanager
def suppressed(runner: Any) -> Iterator[None]:
    """Force the stock model forward for every step of `runner` in this scope.

    The runner wraps its dummy / precompile / profile forwards in this, so
    every stock graph those runs compile is compiled at startup even when the
    megakernel could have served the bucket. Nests; `runner` may be any object
    that accepts attributes.
    """
    depth = _suppress_depth(runner)
    setattr(runner, _SUPPRESS_ATTR, depth + 1)
    try:
        yield
    finally:
        setattr(runner, _SUPPRESS_ATTR, depth)


def _suppress_depth(runner: Any) -> int:
    # Read the instance dict: attribute-synthesizing objects (mocks) must not
    # count as suppressed.
    depth = getattr(runner, "__dict__", {}).get(_SUPPRESS_ATTR, 0)
    return depth if isinstance(depth, int) else 0


def is_suppressed(runner: Any) -> bool:
    """Whether `runner` is inside a `suppressed(runner)` scope."""
    return runner is not None and _suppress_depth(runner) > 0


def resolve_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Call `plugin.layout_request(vllm_config, options)` if defined."""
    model_cfg = getattr(vllm_config, "model_config", None)
    cfg, resolved = select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        getattr(model_cfg, "hf_config", None),
    )
    if not cfg.enabled or resolved is None:
        return None
    ref, options = resolved
    if missing_template(ref) is not None:
        return None
    plugin, module = import_plugin(ref)
    layout_req = getattr(plugin, "layout_request", None)
    if layout_req is None:
        return None
    fn = resolve_callable(layout_req, module, what="layout_request")
    req = fn(vllm_config, dict(options))
    if req is None:
        return None
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    return dict(req) if isinstance(req, Mapping) else None


_LAYOUT_APPLIED_ATTR = "_megakernel_layout_applied"


def _apply_cache_layout_fields(cache_cfg: object, req: Mapping[str, Any]) -> None:
    """Apply layout request fields onto cache configuration."""
    for k, v in req.items():
        if hasattr(cache_cfg, k):
            if k == "block_size" and getattr(
                cache_cfg, "user_specified_block_size", False
            ):
                continue
            setattr(cache_cfg, k, int(v) if k == "block_size" else v)
    for sk in ("mamba_slot_tokens", "mamba_block_size", "slot_stride_tokens"):
        if req.get(sk) is not None:
            vars(cache_cfg)["_megakernel_mamba_slot_tokens"] = int(req[sk])
            break


def apply_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Apply requested pre-sizing layout fields onto `vllm_config`."""
    cached = getattr(vllm_config, _LAYOUT_APPLIED_ATTR, None)
    if cached is not None:
        return cached
    req = resolve_layout_request(vllm_config)
    if req is None:
        return None
    cache_cfg = getattr(vllm_config, "cache_config", None)
    add_cfg = getattr(vllm_config, "additional_config", None)
    if add_cfg is None and hasattr(vllm_config, "additional_config"):
        add_cfg = {}
        attr_name = "additional_config"
        setattr(vllm_config, attr_name, add_cfg)
    if cache_cfg is not None:
        _apply_cache_layout_fields(cache_cfg, req)
    if isinstance(add_cfg, dict):
        for k, v in req.items():
            add_cfg[k] = v
    with contextlib.suppress(Exception):
        setattr(vllm_config, _LAYOUT_APPLIED_ATTR, dict(req))
    return req


def load_tensor_stream(
    vllm_config: object,
    weights: Iterable[tuple[str, object]],
) -> Iterator[tuple[str, Any]]:
    """Stream weights through `plugin.load_tensor` before loading.

    Yields `(name, tensor)` unchanged. If the megakernel is enabled and
    the plugin defines `load_tensor`, calls `load_tensor(name, tensor, options)`
    (or 2-arg `load_tensor(name, tensor)`) on each pair.
    """
    model_cfg = getattr(vllm_config, "model_config", None)
    cfg, resolved = select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        getattr(model_cfg, "hf_config", None),
    )
    fn = None
    options: dict[str, Any] = {}
    if cfg.enabled and resolved is not None:
        ref, opts = resolved
        options = dict(opts)
        if missing_template(ref) is None:
            plugin, module = import_plugin(ref)
            hook = getattr(plugin, "load_tensor", None)
            if hook is not None:
                fn = resolve_callable(hook, module, what="load_tensor")
    if fn is None:
        yield from weights
        return

    try:
        sig = inspect.signature(fn)
        takes_options = len(sig.parameters) >= _LOAD_TENSOR_PARAMS_WITH_OPTIONS
    except (ValueError, TypeError):
        takes_options = True
    for name, tensor in weights:
        if takes_options:
            fn(name, tensor, options)
        else:
            fn(name, tensor)
        yield name, tensor


__all__ = [
    "StepContext",
    "apply_layout_request",
    "is_suppressed",
    "load_tensor_stream",
    "resolve_layout_request",
    "suppressed",
]
