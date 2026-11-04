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
"""Tests for Qwen3.8 shapes, pure-JAX compute, weights, and FP64 oracle."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tests.models.qwen38_2_4t.oracle_fp64 import _bf16_f64
from vllm_torchtpu.models.qwen38_2_4t.megakernel.compute import rmsnorm
from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import (
    Qwen38Config,
    _validate_config,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.weights import WEIGHT_KEYS


def test_qwen38_config_and_validation() -> None:
    cfg = Qwen38Config()
    _validate_config(cfg)
    assert cfg.R == 32
    assert cfg.DCP == 8
    assert cfg.L == 92
    assert cfg.KV_H == 4
    assert cfg.VH == 64


def test_qwen38_weights_keys() -> None:
    assert len(WEIGHT_KEYS) == 28
    assert "wqkv" in WEIGHT_KEYS
    assert "in_norm" in WEIGHT_KEYS


def test_qwen38_rmsnorm_compute() -> None:
    x = jnp.ones((2, 64), dtype=jnp.float32)
    weight = jnp.ones((64,), dtype=jnp.bfloat16)
    out = rmsnorm(x, weight)
    assert out.shape == (2, 64)
    assert out.dtype == jnp.bfloat16


def test_qwen38_oracle_bf16_cast() -> None:
    val = np.array([1.0, 2.5, 3.125], dtype=np.float64)
    out = _bf16_f64(val)
    np.testing.assert_allclose(out, val, atol=1e-3)
