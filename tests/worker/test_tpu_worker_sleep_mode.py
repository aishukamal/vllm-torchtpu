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
"""Tests for TPUWorker.sleep / wake_up with a fake runner on host tensors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.worker import sleep_mode
from vllm_torchtpu.worker.tpu_worker import TPUWorker


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.lm_head = nn.Linear(4, 3, bias=False)
        self.lm_head.weight = self.linear.weight
        self.register_buffer("rotary", torch.arange(6, dtype=torch.float32))


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in list(model.named_parameters())
        + list(model.named_buffers())
    }


def _worker(monkeypatch, kv_caches=None, raw_tensors=(), bundle=None, drafter=None):
    monkeypatch.setattr(sleep_mode, "synchronize_device", Mock())
    model = _Model()
    if kv_caches is None:
        kv_caches = [torch.ones(8, 2, 4, dtype=torch.bfloat16) for _ in range(2)]
    worker = TPUWorker.__new__(TPUWorker)
    worker._sleep_state = None
    worker.model_runner = SimpleNamespace(
        get_model=lambda: model,
        drafter=drafter,
        kv_caches=list(kv_caches),
        kv_cache_raw_tensors=list(raw_tensors),
        _kv_cache_bundle=bundle,
    )
    return worker, model


def _all_tensors(worker, model):
    return list(dict.fromkeys(model.parameters())) + list(model.buffers()) + list(
        worker.model_runner.kv_caches
    )


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_records_geometry_and_drops_storage(monkeypatch, level):
    worker, model = _worker(monkeypatch)
    before = _all_tensors(worker, model)
    geometry = [(tuple(t.shape), t.dtype) for t in before]

    worker.sleep(level)

    assert worker.is_sleeping
    assert _all_tensors(worker, model) == before
    assert all(t.numel() == 0 for t in before)
    state = worker._sleep_state
    recorded = [(r.shape, r.dtype) for r in state.weights] + [
        (r.shape, r.dtype) for r in state.kv_cache
    ]
    assert sorted(map(str, recorded)) == sorted(map(str, geometry))
    assert state.level == level
    assert state.sleeping_tags == {"weights", "kv_cache"}


def test_tied_weights_are_recorded_once(monkeypatch):
    worker, model = _worker(monkeypatch)
    worker.sleep(2)
    names = [r.name for r in worker._sleep_state.weights]
    assert names == ["linear.weight", "linear.bias", "rotary"]


def test_level1_keeps_host_copies_and_wake_restores_them(monkeypatch):
    worker, model = _worker(monkeypatch)
    expected = _snapshot(model)

    worker.sleep(1)
    for record in worker._sleep_state.weights:
        assert record.host is not None
        assert record.host.device.type == "cpu"
        torch.testing.assert_close(record.host, expected[record.name])

    worker.wake_up(["weights"])
    for name, tensor in _snapshot(model).items():
        torch.testing.assert_close(tensor, expected[name])
    assert worker.is_sleeping
    assert all(t.numel() == 0 for t in worker.model_runner.kv_caches)

    worker.wake_up(["kv_cache"])
    assert not worker.is_sleeping
    for cache in worker.model_runner.kv_caches:
        assert cache.shape == (8, 2, 4) and cache.dtype == torch.bfloat16
        assert torch.count_nonzero(cache) == 0


def test_level2_keeps_buffers_only_and_leaves_parameters_to_the_loader(
    monkeypatch,
):
    worker, model = _worker(monkeypatch)
    rotary = model.rotary.clone()

    worker.sleep(2)
    hosts = {r.name: r.host for r in worker._sleep_state.weights}
    assert hosts["linear.weight"] is None and hosts["linear.bias"] is None
    torch.testing.assert_close(hosts["rotary"], rotary)

    worker.wake_up(["weights"])
    assert model.linear.weight.shape == (3, 4)
    assert model.linear.bias.shape == (3,)
    assert model.lm_head.weight is model.linear.weight
    torch.testing.assert_close(model.rotary, rotary)

    loaded = torch.full((3, 4), 7.0)
    with torch.no_grad():
        model.linear.weight.copy_(loaded)
    torch.testing.assert_close(model.lm_head.weight, loaded)


def test_wake_up_processes_weights_before_kv_cache(monkeypatch):
    worker, _ = _worker(monkeypatch)
    order = []
    monkeypatch.setattr(
        sleep_mode, "restore_weights", lambda records: order.append("weights")
    )
    monkeypatch.setattr(
        sleep_mode, "restore_kv_cache", lambda records: order.append("kv_cache")
    )

    worker.sleep(2)
    worker.wake_up(["kv_cache", "weights"])
    assert order == ["weights", "kv_cache"]
    assert not worker.is_sleeping

    worker.sleep(2)
    worker.wake_up()
    assert order == ["weights", "kv_cache"] * 2


def test_repeated_sleep_and_wake_are_no_ops(monkeypatch):
    worker, model = _worker(monkeypatch)
    expected = _snapshot(model)

    worker.sleep(1)
    state = worker._sleep_state
    worker.sleep(2)
    assert worker._sleep_state is state
    assert state.level == 1

    worker.wake_up(["weights"])
    weight = model.linear.weight
    worker.wake_up(["weights"])
    assert model.linear.weight is weight
    torch.testing.assert_close(weight, expected["linear.weight"])
    assert worker.is_sleeping

    worker.wake_up()
    assert not worker.is_sleeping
    worker.wake_up()
    assert not worker.is_sleeping


def test_invalid_level_and_tags_are_rejected(monkeypatch):
    worker, model = _worker(monkeypatch)
    with pytest.raises(AssertionError, match="sleep level"):
        worker.sleep(3)
    assert not worker.is_sleeping
    assert model.linear.weight.numel() == 12

    worker.sleep(2)
    with pytest.raises(ValueError, match="unknown wake_up tags"):
        worker.wake_up(["scheduling"])
    assert worker._sleep_state.sleeping_tags == {"weights", "kv_cache"}


def test_block_major_bundle_views_are_rebuilt_on_the_new_bundle(monkeypatch):
    bundle = torch.ones(4, 3, 2, 2, dtype=torch.bfloat16)
    views = [bundle[:, index] for index in range(3)]
    geometry = [(v.shape, v.stride(), v.storage_offset()) for v in views]
    worker, _ = _worker(monkeypatch, kv_caches=views, bundle=bundle)

    worker.sleep(2)
    assert bundle.numel() == 0
    assert all(v.numel() == 0 for v in views)
    [record] = worker._sleep_state.kv_cache
    assert record.root is bundle
    assert [v.recipe for v in record.views] == [
        sleep_mode.SelectView(dim=1, index=index) for index in range(3)
    ]

    worker.wake_up(["kv_cache"])
    assert bundle.shape == (4, 3, 2, 2) and bundle.dtype == torch.bfloat16
    assert torch.count_nonzero(bundle) == 0
    for view, (shape, stride, offset) in zip(views, geometry, strict=True):
        assert (view.shape, view.stride(), view.storage_offset()) == (
            shape,
            stride,
            offset,
        )
        assert view.untyped_storage().data_ptr() == bundle.untyped_storage().data_ptr()
    views[1].fill_(5)
    assert torch.equal(bundle[:, 1], views[1])
    assert torch.count_nonzero(bundle[:, 0]) == 0


def test_unified_pool_shared_by_layers_is_one_allocation(monkeypatch):
    pool_a = torch.zeros(6, 2, 2)
    pool_b = torch.zeros(6, 2, 2)
    worker, _ = _worker(
        monkeypatch, kv_caches=[pool_a, pool_a, pool_b], raw_tensors=[pool_a, pool_b]
    )

    worker.sleep(2)
    assert [r.root for r in worker._sleep_state.kv_cache] == [pool_a, pool_b]
    assert all(not r.views for r in worker._sleep_state.kv_cache)

    worker.wake_up(["kv_cache"])
    assert worker.model_runner.kv_caches[0] is pool_a
    assert pool_a.shape == (6, 2, 2)


def test_unsupported_view_fails_before_anything_is_released(monkeypatch):
    pool = torch.zeros(8, 4)
    worker, model = _worker(monkeypatch, kv_caches=[pool[2:5, 1:3]], raw_tensors=[pool])

    with pytest.raises(NotImplementedError, match="cannot rebuild"):
        worker.sleep(2)
    assert not worker.is_sleeping
    assert model.linear.weight.numel() == 12
    assert pool.numel() == 32


def test_draft_model_weights_are_included(monkeypatch):
    drafter = Eagle3Proposer.__new__(Eagle3Proposer)
    drafter.draft_model = nn.Linear(2, 2)
    worker, _ = _worker(monkeypatch, drafter=drafter)

    worker.sleep(2)
    assert drafter.draft_model.weight.numel() == 0
    worker.wake_up(["weights"])
    assert drafter.draft_model.weight.shape == (2, 2)
