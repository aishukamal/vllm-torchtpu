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
"""Unit tests for per-request ``thinking_token_budget`` on ``TPUModelRunner``."""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch._dynamo.utils
from vllm import SamplingParams
from vllm.config.reasoning import ReasoningConfig
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import greedy_top_tokens, min_tokens, thinking_budget
from vllm_torchtpu.runner.thinking_budget import (
    INT32_MAX,
    NO_FORCED_TOKEN,
    UNTRACKED_STATE,
    ThinkingBudgetMarkers,
    ThinkingBudgetTracker,
    _force_logits_rows_impl,
    _force_token_ids_1d_impl,
    _prepare_spec_verify_drafts,
    advance_spec_states,
    advance_state_py,
    compute_kmp_failure,
    compute_sequence_state,
    override_and_advance_plain,
    prepare_spec_verify_logits,
    prepare_spec_verify_tokens,
    query_forced_token_py,
    seed_states,
)
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _ThinkingBudgetSpecChunk,
    _ThinkingBudgetStepState,
)
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState
from vllm_torchtpu.utils import synchronize_tensors

# Representative reasoning marker token IDs.
QWEN3_START = (151667,)  # <think>
QWEN3_END = (151668,)  # </think>
DEEPSEEK_START = (128798,)  # <｜begin▁of▁thought｜> / <think>
DEEPSEEK_END = (128799,)  # <｜end▁of▁thought｜> / </think>
GEMMA4_START = (105, 2364, 107)  # <|channel>thought\n
GEMMA4_END = (106, 107)  # <channel|>\n


def _reasoning_config(
    start_ids: list[int] | None,
    end_ids: list[int] | None,
    natural_end_ids: list[int] | None = None,
    enabled: bool = True,
) -> ReasoningConfig:
    cfg = ReasoningConfig(reasoning_parser="qwen3" if enabled else "")
    cfg._reasoning_start_token_ids = start_ids
    cfg._reasoning_end_token_ids = end_ids
    cfg._natural_reasoning_end_token_ids = (
        end_ids if natural_end_ids is None else natural_end_ids
    )
    cfg._enabled = enabled
    return cfg


def _request_state(
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams | None,
    output_token_ids: list[int] | None = None,
    num_computed_tokens: int | None = None,
) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        mm_features=[],
        sampling_params=sampling_params,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=(
            len(prompt_token_ids)
            if num_computed_tokens is None
            else num_computed_tokens
        ),
        output_token_ids=[] if output_token_ids is None else list(output_token_ids),
    )


def _make_input_batch(vocab_size: int = 128, max_num_reqs: int = 8) -> InputBatch:
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=128,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=vocab_size,
        block_sizes=[16],
        max_num_blocks_per_req=[8],
        logitsprocs=LogitsProcessors(),
        logitsprocs_need_output_token_ids=False,
    )


def _run_sample_tokens(
    runner: SimpleNamespace | TPUModelRunner,
    grammar_output: object | None = None,
):
    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=[c.clone() for c in chunks],
            chunk_real_lens=list(lengths),
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
            return_value=False,
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        return TPUModelRunner.sample_tokens(runner, grammar_output)


class TestMarkersAndKMP:
    """Marker extraction from ``ReasoningConfig`` and KMP fallback tables."""

    def test_disabled_or_incomplete_reasoning_config_returns_none(self) -> None:
        assert ThinkingBudgetMarkers.from_reasoning_config(None) is None
        assert (
            ThinkingBudgetMarkers.from_reasoning_config(
                _reasoning_config([10], [11], enabled=False)
            )
            is None
        )
        assert (
            ThinkingBudgetMarkers.from_reasoning_config(_reasoning_config([], [11]))
            is None
        )
        assert (
            ThinkingBudgetMarkers.from_reasoning_config(_reasoning_config([10], []))
            is None
        )
        assert (
            ThinkingBudgetMarkers.from_reasoning_config(_reasoning_config(None, [11]))
            is None
        )

    @pytest.mark.parametrize(
        ("start_ids", "end_ids"),
        [
            (QWEN3_START, QWEN3_END),
            (DEEPSEEK_START, DEEPSEEK_END),
            (GEMMA4_START, GEMMA4_END),
        ],
    )
    def test_resolves_qwen3_deepseek_and_gemma4_markers(
        self, start_ids: tuple[int, ...], end_ids: tuple[int, ...]
    ) -> None:
        cfg = _reasoning_config(list(start_ids), list(end_ids))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None
        assert markers.start_ids == start_ids
        assert markers.end_ids == end_ids
        assert markers.natural_end_ids == end_ids
        assert len(markers.start_fail) == len(start_ids) + 1
        assert len(markers.end_fail) == len(end_ids) + 1

    def test_initialize_token_ids_from_reasoning_parser(self) -> None:
        fake_tok = MagicMock()
        vocab_map = {
            "<|channel>": [105],
            "<|channel>thought\n": list(GEMMA4_START),
            "<channel|>": list(GEMMA4_END),
            "\nThinking stopped.<channel|>": [900, 901, *GEMMA4_END],
        }
        fake_tok.get_vocab.return_value = {
            "<|channel>": 105,
            "<channel|>": 106,
            "<|tool_call>": 108,
            "<|turn>": 109,
            "<|tool_response>": 110,
        }
        fake_tok.encode.side_effect = lambda s, add_special_tokens=False: vocab_map.get(
            s, [999]
        )
        cfg = ReasoningConfig(
            reasoning_parser="gemma4",
            reasoning_end_str="\nThinking stopped.<channel|>",
        )
        with patch(
            "vllm.config.reasoning.cached_tokenizer_from_config",
            return_value=fake_tok,
        ):
            cfg.initialize_token_ids(MagicMock())

        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None
        assert markers.start_ids == (105,)
        assert markers.natural_end_ids == GEMMA4_END
        assert markers.end_ids == (900, 901, *GEMMA4_END)

        cfg_explicit = ReasoningConfig(
            reasoning_parser="gemma4",
            reasoning_start_str="<|channel>thought\n",
            reasoning_end_str="\nThinking stopped.<channel|>",
        )
        with patch(
            "vllm.config.reasoning.cached_tokenizer_from_config",
            return_value=fake_tok,
        ):
            cfg_explicit.initialize_token_ids(MagicMock())
        markers_explicit = ThinkingBudgetMarkers.from_reasoning_config(cfg_explicit)
        assert markers_explicit is not None
        assert markers_explicit.start_ids == GEMMA4_START

    def test_distinct_forced_end_vs_natural_end_and_disjoint_markers(self) -> None:
        # Case 1: forced transition extends natural end (e.g. </think>\n\n).
        cfg_ext = _reasoning_config(
            start_ids=[90],
            end_ids=[91, 92, 93],
            natural_end_ids=[91],
        )
        markers_ext = ThinkingBudgetMarkers.from_reasoning_config(cfg_ext)
        assert markers_ext is not None

        # Natural [91] before budget closes thinking immediately without forcing 92/93.
        st = compute_sequence_state([1, 90, 10, 91, 20, 21, 22], 3, markers_ext)
        assert st[1] == 0
        assert query_forced_token_py(st, markers_ext) == NO_FORCED_TOKEN

        # Reaching budget=2 forces the full 3-token sequence [91, 92, 93], not just 91.
        st_force = compute_sequence_state([1, 90, 10, 11], 2, markers_ext)
        emitted: list[int] = []
        for _ in range(3):
            forced = query_forced_token_py(st_force, markers_ext)
            assert forced != NO_FORCED_TOKEN
            emitted.append(forced)
            st_force = advance_state_py(st_force, forced, markers_ext)
        assert emitted == [91, 92, 93]
        assert st_force[1] == 0
        assert query_forced_token_py(st_force, markers_ext) == NO_FORCED_TOKEN

        # Tensor path must also wrap nat_end_match cleanly across [91, 92, 93].
        tracker_ext = ThinkingBudgetTracker(cfg_ext, max_num_reqs=2)
        p_ext = SamplingParams(thinking_token_budget=2)
        req_ext = _request_state("r_ext", [1, 90, 10, 11], p_ext)
        tracker_ext.add("r_ext", p_ext, req_ext.prompt_token_ids)
        tbl_ext = tracker_ext.prepare_step(
            torch.device("cpu"), ["r_ext"], {"r_ext": req_ext}
        )
        assert tbl_ext is not None
        slots_ext = tracker_ext.build_chunk_slot_ids(
            torch.device("cpu"), ["r_ext"], 0, 1, 2
        )
        assert slots_ext is not None
        tensor_emitted: list[int] = []
        for raw_cand in (40, 41, 42, 43):
            out_t, tbl_ext = override_and_advance_plain(
                tbl_ext,
                slots_ext,
                torch.tensor([[raw_cand], [0]], dtype=torch.int64),
                markers_ext,
            )
            tracker_ext.commit_state_table(tbl_ext)
            tensor_emitted.append(int(out_t[0, 0].item()))
        assert tensor_emitted == [91, 92, 93, 43]

        # Case 2: completely disjoint forced end [92, 93] vs natural end [91]
        # (matches upstream test_gpu_thinking_budget MockDistinctEndReasoningConfig)
        # must terminate after emitting [92, 93] without looping.
        cfg_disjoint = _reasoning_config(
            start_ids=[90],
            end_ids=[92, 93],
            natural_end_ids=[91],
        )
        markers_disjoint = ThinkingBudgetMarkers.from_reasoning_config(cfg_disjoint)
        assert markers_disjoint is not None
        st_dis = compute_sequence_state([1, 90, 10, 11, 12], 3, markers_disjoint)
        forced_dis: list[int] = []
        for _ in range(4):
            tok = query_forced_token_py(st_dis, markers_disjoint)
            if tok == NO_FORCED_TOKEN:
                break
            forced_dis.append(tok)
            st_dis = advance_state_py(st_dis, tok, markers_disjoint)
        assert forced_dis == [92, 93]
        assert st_dis[1] == 0

    def test_kmp_overlapping_self_prefix_pattern(self) -> None:
        # Pattern [90, 90, 91] fed [90, 90, 90, 91]: naive reset to 0 after the
        # third 90 would miss the match; KMP fail table (0, 0, 1, 0) catches it.
        pat = (90, 90, 91)
        assert compute_kmp_failure(pat) == (0, 0, 1, 0)
        assert compute_kmp_failure((90, 91, 90)) == (0, 0, 0, 1)

        cfg = _reasoning_config([90, 90, 91], [80, 80, 81])
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None

        # Overlapping start sequence in prompt enters thinking cleanly.
        st = compute_sequence_state([5, 90, 90, 90, 91], budget=2, markers=markers)
        assert st[1] == 1 and st[2] == 0

        # Overlapping natural end [80, 80, 80, 81] inside thinking closes it.
        st_closed = compute_sequence_state(
            [5, 90, 90, 90, 91, 80, 80, 80, 81], budget=10, markers=markers
        )
        assert st_closed[1] == 0

    def test_vectorized_sequence_state_matches_scalar_kmp_and_handles_long_prompts(
        self,
    ) -> None:
        def scalar_ref(
            seq: list[int], budget: int, markers: ThinkingBudgetMarkers
        ) -> tuple[int, int, int, int, int, int, int]:
            st = (min(int(budget), thinking_budget.INT32_MAX), 0, 0, 0, 0, 0, 0)
            for tok in seq:
                st = advance_state_py(st, int(tok), markers)
            return st

        marker_configs = [
            _reasoning_config(list(QWEN3_START), list(QWEN3_END)),
            _reasoning_config(list(GEMMA4_START), list(GEMMA4_END)),
            _reasoning_config([90, 90, 91], [80, 80, 81]),
            _reasoning_config(
                start_ids=[90, 91],
                end_ids=[95, 96, 97],
                natural_end_ids=[80, 81],
            ),
        ]
        rng = np.random.default_rng(77)
        for cfg in marker_configs:
            markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
            assert markers is not None
            vocab = [*markers.start_ids, *markers.natural_end_ids, *markers.end_ids]
            vocab.extend([1, 2, 3, 4, 5, 50])
            for length in (0, 1, 31, 32, 33, 64, 256, 1024):
                for budget in (0, 1, 3, 25):
                    arr = rng.choice(vocab, size=length)
                    seq = arr.tolist()
                    exp = scalar_ref(seq, budget, markers)
                    assert compute_sequence_state(seq, budget, markers) == exp
                    assert compute_sequence_state(arr, budget, markers) == exp

        # Long-context 128K-token prompt with embedded <think>...</think> history
        # and an open final <think> block: verifies multi-chunk KMP accuracy.
        qwen_markers = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        )
        assert qwen_markers is not None
        long_prompt = (
            [42] * 60_000
            + [QWEN3_START[0]]
            + [43] * 1_000
            + [QWEN3_END[0]]
            + [44] * 66_000
            + [QWEN3_START[0]]
            + [45] * 997
        )
        long_state = compute_sequence_state(
            long_prompt, budget=1000, markers=qwen_markers
        )
        assert long_state == (1000, 1, 997, 0, 0, 0, 0)


class TestUpstreamSemanticParity:
    """Differential parity against upstream V1 and V2 thinking-budget semantics."""

    @staticmethod
    def _upstream_v2_forced_token(
        all_tokens: list[int],
        budget: int | None,
        start_ids: list[int],
        natural_end_ids: list[int],
        end_ids: list[int],
    ) -> int | None:
        """Direct Python specification of upstream GPU Model Runner V2's
        ``_thinking_budget_kernel`` (``vllm/v1/worker/gpu/sample/thinking_budget.py``).
        """
        if budget is None or budget < 0:
            return None
        n = len(all_tokens)
        s_len = len(start_ids)
        nat_len = len(natural_end_ids)
        last_start = -1
        for i in range(0, n - s_len + 1):
            if all_tokens[i : i + s_len] == start_ids:
                last_start = i
        last_end = -1
        for i in range(0, n - nat_len + 1):
            if all_tokens[i : i + nat_len] == natural_end_ids:
                last_end = i
        if last_start < 0 or last_start <= last_end:
            return None
        num_reasoning_tokens = n - (last_start + s_len)
        if num_reasoning_tokens < budget:
            return None
        end_prefix_len = 0
        for prefix_len in range(1, min(len(end_ids) - 1, n) + 1):
            if all_tokens[n - prefix_len :] == end_ids[:prefix_len]:
                end_prefix_len = prefix_len
        return end_ids[end_prefix_len]

    @pytest.mark.parametrize(
        ("start_ids", "end_ids"),
        [
            (list(QWEN3_START), list(QWEN3_END)),
            (list(DEEPSEEK_START), list(DEEPSEEK_END)),
            (list(GEMMA4_START), list(GEMMA4_END)),
        ],
    )
    @pytest.mark.parametrize("budget", [0, 1, 3, 5, None])
    @pytest.mark.parametrize("prompt_open", [True, False])
    def test_differential_parity_with_upstream_v1_and_v2(
        self,
        start_ids: list[int],
        end_ids: list[int],
        budget: int | None,
        prompt_open: bool,
    ) -> None:
        vocab_size = max(*start_ids, *end_ids, 64) + 16
        cfg = _reasoning_config(start_ids, end_ids)
        prompt = [1, 2, *start_ids] if prompt_open else [1, 2, 3]
        params = SamplingParams(temperature=0.0, thinking_token_budget=budget)

        # Upstream V1 reference holder.
        holder = ThinkingBudgetStateHolder(
            cfg,
            max_num_seqs=4,
            num_spec_tokens=0,
            device=torch.device("cpu"),
            is_pin_memory=False,
        )
        v1_out: list[int] = []
        holder.sync_batch(
            BatchUpdate(
                batch_size=1,
                removed=(),
                added=[(0, params, list(prompt), v1_out)],
                moved=(),
            )
        )

        # TPU tracker under test.
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        tracker.add("req0", params, prompt)
        req_state = _request_state("req0", prompt, params)
        requests = {"req0": req_state}

        # Scripted base argmax stream: if thinking is not open in prompt, emit
        # 2 non-thinking tokens, then dynamically emit ``start_ids``, then 8
        # reasoning body tokens, then post-thinking content tokens.
        if prompt_open:
            planned_model_tokens = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        else:
            planned_model_tokens = [
                11,
                12,
                *start_ids,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
            ]

        v2_out: list[int] = []
        tpu_out: list[int] = []
        for base_tok in planned_model_tokens:
            # 1. Run upstream V1 holder.
            holder.update_state([v1_out], spec_token_ids=None)
            ref_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
            ref_logits[0, base_tok] = 5.0
            ref_logits = holder.apply_to_logits(
                ref_logits, predict_bonus_token=False, spec_token_ids=None
            )
            v1_out.append(int(torch.argmax(ref_logits, dim=-1).item()))

            # 2. Run upstream V2 specification.
            forced_v2 = self._upstream_v2_forced_token(
                [*prompt, *v2_out], budget, start_ids, end_ids, end_ids
            )
            v2_out.append(base_tok if forced_v2 is None else forced_v2)

            # 3. Run TPU tracker + override_and_advance_plain.
            state_table = tracker.prepare_step(torch.device("cpu"), ["req0"], requests)
            selected = torch.tensor([[base_tok]], dtype=torch.int64)
            if state_table is not None:
                assert tracker.markers is not None
                slot_ids = tracker.build_chunk_slot_ids(
                    torch.device("cpu"), ["req0"], 0, 1, padded_num_reqs=4
                )
                assert slot_ids is not None
                padded_sel = torch.zeros((4, 1), dtype=torch.int64)
                padded_sel[0, 0] = base_tok
                padded_sel, state_table = override_and_advance_plain(
                    state_table, slot_ids, padded_sel, tracker.markers
                )
                tracker.commit_state_table(state_table)
                selected = padded_sel[:1]
            tpu_tok = int(selected[0, 0].item())
            tpu_out.append(tpu_tok)
            req_state.output_token_ids.append(tpu_tok)

        # Exact parity with upstream GPU Model Runner V2 across ALL 30 cases.
        assert tpu_out == v2_out

        # Note: upstream V1's ThinkingBudgetStateHolder._init_state_entry sets
        # `start_thinking = len(prompt_tok_ids) - think_count - 1` (assuming
        # `len(start_ids) == 1`), which undercounts prompt-open thinking tokens
        # by `len(start_ids) - 1` when `len(start_ids) > 1` and `budget > 0`.
        # On all other 27 cases, V1 and V2 agree and match TPU output verbatim.
        if not (prompt_open and len(start_ids) > 1 and budget not in (0, None)):
            assert tpu_out == v1_out
        else:
            assert budget is not None
            assert tpu_out[: budget + len(end_ids)] == [
                *planned_model_tokens[:budget],
                *end_ids,
            ]

    def test_partial_end_prefix_continuation_from_prompt_and_generation(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None

        # Resumed prompt already ends with the first token of GEMMA4_END (106)
        # right as budget=3 is reached: must force 107 next without duplicating 106.
        prompt = [1, *GEMMA4_START, 20, 21, GEMMA4_END[0]]
        st = compute_sequence_state(prompt, budget=3, markers=markers)
        assert query_forced_token_py(st, markers) == GEMMA4_END[1]
        st = advance_state_py(st, GEMMA4_END[1], markers)
        assert st[1] == 0
        assert query_forced_token_py(st, markers) == NO_FORCED_TOKEN

        # Same during generation: model naturally samples GEMMA4_END[0] as the
        # budget-th token; next step forces only GEMMA4_END[1].
        st_gen = compute_sequence_state([1, *GEMMA4_START], budget=2, markers=markers)
        st_gen = advance_state_py(st_gen, 50, markers)
        assert query_forced_token_py(st_gen, markers) == NO_FORCED_TOKEN
        st_gen = advance_state_py(st_gen, GEMMA4_END[0], markers)
        assert query_forced_token_py(st_gen, markers) == GEMMA4_END[1]
        st_gen = advance_state_py(st_gen, GEMMA4_END[1], markers)
        assert st_gen[1] == 0

    def test_multi_block_thinking_and_prefill_boundaries(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None
        s, e = QWEN3_START[0], QWEN3_END[0]

        # Closed thinking block in prompt disables forcing.
        closed_prompt = [1, s, 10, 11, 12, e, 13]
        st_closed = compute_sequence_state(closed_prompt, budget=3, markers=markers)
        assert st_closed[1] == 0
        assert query_forced_token_py(st_closed, markers) == NO_FORCED_TOKEN

        # Re-opened thinking block in multi-turn prompt counts from latest start.
        reopened = [1, s, 10, 11, 12, e, 13, s, 14, 15, 16]
        st_reopened = compute_sequence_state(reopened, budget=3, markers=markers)
        assert query_forced_token_py(st_reopened, markers) == e

        # Re-entering <think> dynamically after a forced </think> gets a fresh budget.
        st = advance_state_py(st_reopened, e, markers)
        assert st[1] == 0
        st = advance_state_py(st, s, markers)
        assert st[1] == 1 and st[2] == 0
        for tok in (30, 31, 32):
            assert query_forced_token_py(st, markers) == NO_FORCED_TOKEN
            st = advance_state_py(st, tok, markers)
        assert query_forced_token_py(st, markers) == e

    def test_oversized_and_negative_budgets(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)

        # Oversized budget (> int32 max) clamps cleanly to INT32_MAX.
        tracker.add(
            "huge",
            SamplingParams(thinking_token_budget=2**40),
            [1, QWEN3_START[0], 10, 11, 12],
        )
        assert tracker.requests["huge"] == INT32_MAX
        req = _request_state(
            "huge",
            [1, QWEN3_START[0], 10, 11, 12],
            SamplingParams(thinking_token_budget=2**40),
        )
        table = tracker.prepare_step(torch.device("cpu"), ["huge"], {"huge": req})
        assert table is not None
        assert tracker.markers is not None
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["huge"], 0, 1, 4)
        assert slots is not None
        out, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[42], [0], [0], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        assert out[0, 0].item() == 42

        # Negative budget (-1) is treated as disabled/untracked.
        neg_params = SamplingParams()
        neg_params.thinking_token_budget = -1
        tracker.add("neg", neg_params, [1, QWEN3_START[0]])
        assert "neg" not in tracker.requests


class TestVectorizedKernels:
    """Fuzz and edge-case tests for static-shape tensor KMP kernels."""

    @pytest.mark.parametrize(
        ("start_ids", "end_ids", "nat_ids"),
        [
            ([10], [11], [11]),
            ([10, 20, 30], [40, 50], [40, 50]),
            ([10, 10, 11], [20, 20, 21], [20, 20, 21]),
            ([10, 20], [30, 31, 32], [29]),
        ],
    )
    def test_randomized_tensor_vs_scalar_kmp_fuzz_parity(
        self,
        start_ids: list[int],
        end_ids: list[int],
        nat_ids: list[int],
    ) -> None:
        cfg = _reasoning_config(start_ids, end_ids, natural_end_ids=nat_ids)
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None

        rng = np.random.default_rng(12345)
        capacity = 9
        padded_b = 8
        state_table = torch.zeros((capacity, 7), dtype=torch.int32)
        state_table[:, 0] = -1

        # Seed slots 1..6 with budgets 0, 1, 2, 4, 0, 3; slots 7..8 untracked.
        budgets = {1: 0, 2: 1, 3: 2, 4: 4, 5: 0, 6: 3}
        scalar_states: dict[int, thinking_budget.StateTuple] = {0: UNTRACKED_STATE}
        seed_slots = torch.zeros(6, dtype=torch.int32)
        seed_vals = torch.zeros((6, 7), dtype=torch.int32)
        seed_vals[:, 0] = -1
        for i, (slot, b) in enumerate(budgets.items()):
            init_prompt = start_ids if slot % 2 == 1 else [1, 2]
            st = compute_sequence_state(init_prompt, b, markers)
            scalar_states[slot] = st
            seed_slots[i] = slot
            seed_vals[i] = torch.tensor(st, dtype=torch.int32)
        state_table = seed_states(state_table, seed_slots, seed_vals)

        vocab_pool = list({*start_ids, *end_ids, *nat_ids, 1, 2, 3, 4, 5})
        for _step in range(40):
            # Permute active slots + zero-padding slots across batch rows.
            chosen_slots = [1, 2, 3, 0, 4, 5, 6, 0]
            rng.shuffle(chosen_slots)
            slot_ids = torch.tensor(chosen_slots, dtype=torch.int32)
            sampled = torch.tensor(
                [[int(rng.choice(vocab_pool))] for _ in range(padded_b)],
                dtype=torch.int64,
            )

            expected_out: list[int] = []
            for row_i, slot in enumerate(chosen_slots):
                raw_tok = int(sampled[row_i, 0].item())
                if slot == 0:
                    expected_out.append(raw_tok)
                    continue
                st = scalar_states[slot]
                forced = query_forced_token_py(st, markers)
                chosen = forced if forced >= 0 else raw_tok
                expected_out.append(chosen)
                scalar_states[slot] = advance_state_py(st, chosen, markers)

            prev_snapshot = state_table.clone()
            got, next_table = override_and_advance_plain(
                state_table, slot_ids, sampled, markers
            )
            assert torch.equal(state_table, prev_snapshot)
            state_table = next_table
            assert got.squeeze(-1).tolist() == expected_out
            assert tuple(state_table[0].tolist()) == UNTRACKED_STATE
            for slot, st in scalar_states.items():
                assert tuple(state_table[slot].tolist()) == st

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_force_logits_rows_dtypes_boundary_ids_and_immutability(
        self, dtype: torch.dtype
    ) -> None:
        vocab_size = 64
        logits = torch.linspace(-3.0, 3.0, steps=4 * vocab_size, dtype=dtype).reshape(
            4, vocab_size
        )
        # Row 2 simulates a stop/end token previously masked to -inf by min_tokens.
        logits = logits.clone()
        logits[2, vocab_size - 1] = float("-inf")
        orig_snapshot = logits.clone()

        forced_ids = torch.tensor(
            [0, NO_FORCED_TOKEN, vocab_size - 1, NO_FORCED_TOKEN],
            dtype=torch.int32,
        )
        out = _force_logits_rows_impl(logits, forced_ids)

        # Input tensor must never be mutated in place (preserves raw logprobs).
        assert torch.equal(logits, orig_snapshot)
        assert out.dtype == dtype

        # Unforced rows (1 and 3) are bit-for-bit identical.
        assert torch.equal(out[1], orig_snapshot[1])
        assert torch.equal(out[3], orig_snapshot[3])

        # Forced row 0: token 0 is 0.0, all others -inf.
        assert out[0, 0].item() == 0.0
        assert torch.isneginf(out[0, 1:]).all()
        assert int(torch.argmax(out[0]).item()) == 0

        # Forced row 2: boundary token (vocab_size - 1) restored from -inf to 0.0.
        assert out[2, vocab_size - 1].item() == 0.0
        assert torch.isneginf(out[2, : vocab_size - 1]).all()
        assert int(torch.argmax(out[2]).item()) == vocab_size - 1

        raw_1d = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
        assert _force_token_ids_1d_impl(raw_1d, forced_ids).tolist() == [
            0,
            22,
            vocab_size - 1,
            44,
        ]

    def test_prepare_and_advance_spec_verify_matches_scalar_reference(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None

        # 3 real requests in a 4-row verify chunk, with max_num_reqs=6 slot_ids,
        # K=3, padded_logits_length=16:
        # req 0 (slot 1): prompt-open, budget=3, drafts=[20, 21, 22] -> all 3
        #   drafts within budget; if all 3 pass, raw bonus 99 at col 3 hits
        #   think_count==3 and MUST be overridden to GEMMA4_END[0] (106) inside
        #   advance_spec_states!
        # req 1 (slot 0): untracked, drafts=[30, 31], untouched.
        # req 2 (slot 2): prompt-open, budget=1, drafts=[40, 106, 50] -> pos 0 ok,
        #   pos 1 forced to 106 (draft matches 106!), pos 2 forced to 107 (mismatch).
        state_table = torch.zeros((7, 7), dtype=torch.int32)
        state_table[:, 0] = -1
        st1 = compute_sequence_state([1, *GEMMA4_START], budget=3, markers=markers)
        st2 = compute_sequence_state([1, *GEMMA4_START], budget=1, markers=markers)
        state_table = seed_states(
            state_table,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([st1, st2], dtype=torch.int32),
        )

        # slot_ids is sized to max_num_reqs=6 while next_tokens has 4 padded rows.
        slot_ids = torch.tensor([1, 0, 2, 0, 0, 0], dtype=torch.int32)
        raw_drafts = [20, 21, 22, 30, 31, 40, GEMMA4_END[0], 50]
        draft_token_ids = torch.tensor(
            raw_drafts + [0] * (16 - len(raw_drafts)), dtype=torch.int32
        )
        segment_ids = torch.tensor(
            [0, 0, 0, 1, 1, 2, 2, 2] + [3] * 8, dtype=torch.int64
        )
        group_indices = torch.tensor(
            [0, 1, 2, 0, 1, 0, 1, 2] + [0] * 8, dtype=torch.int32
        )

        forced_target, eff_drafts = _prepare_spec_verify_drafts(
            state_table,
            slot_ids,
            draft_token_ids,
            segment_ids,
            group_indices,
            3,
            *markers.kernel_args(),
        )
        assert forced_target[:8].tolist() == [
            -1,
            -1,
            -1,  # req 0 (within budget=3 across the 3 draft slots)
            -1,
            -1,  # req 1 (untracked)
            -1,
            GEMMA4_END[0],
            GEMMA4_END[1],  # req 2: draft[1]==106 matched, so pos 2 forces 107!
        ]
        assert (forced_target[8:] == -1).all()
        assert eff_drafts[:8].tolist() == [
            20,
            21,
            22,
            30,
            31,
            40,
            GEMMA4_END[0],
            -1,
        ]

        raw_logits = torch.zeros((16, 128), dtype=torch.float32)
        raw_logits[:, 15] = 4.0
        raw_argmax = torch.full((16,), 15, dtype=torch.int64)
        flogits, eff_from_logits = prepare_spec_verify_logits(
            state_table,
            slot_ids,
            draft_token_ids,
            raw_logits,
            segment_ids,
            group_indices,
            max_draft_tokens=3,
            markers=markers,
        )
        ftokens, eff_from_tokens = prepare_spec_verify_tokens(
            state_table,
            slot_ids,
            draft_token_ids,
            raw_argmax,
            segment_ids,
            group_indices,
            max_draft_tokens=3,
            markers=markers,
        )
        assert torch.equal(eff_from_logits, eff_drafts)
        assert torch.equal(eff_from_tokens, eff_drafts)
        assert torch.argmax(flogits, dim=-1).tolist() == ftokens.tolist()
        assert ftokens[:8].tolist() == [
            15,
            15,
            15,
            15,
            15,
            15,
            GEMMA4_END[0],
            GEMMA4_END[1],
        ]

        # Commit rejection-sampler output rows:
        # - req 0 accepted all 3 drafts [20, 21, 22] and sampled raw bonus 99 ->
        #   advance_spec_states overrides 99 -> GEMMA4_END[0] (106) and advances!
        # - req 1 untracked -> [30, 31, 99, -1] unchanged.
        # - req 2 accepted [40, 106] and recovered 107 -> finishes GEMMA4_END!
        # - row 3 padded junk -> ignored because slot 0.
        committed = torch.tensor(
            [
                [20, 21, 22, 99],
                [30, 31, 99, -1],
                [40, GEMMA4_END[0], GEMMA4_END[1], -1],
                [77, 78, 79, 80],
            ],
            dtype=torch.int32,
        )
        prev_snapshot = state_table.clone()
        committed_out, state_table = advance_spec_states(
            state_table, slot_ids, committed, max_draft_tokens=3, markers=markers
        )
        assert committed_out[0].tolist() == [20, 21, 22, GEMMA4_END[0]]
        assert torch.equal(committed_out[1:], committed[1:])
        assert not torch.equal(state_table, prev_snapshot)

        exp_st1 = compute_sequence_state(
            [1, *GEMMA4_START, 20, 21, 22, GEMMA4_END[0]], budget=3, markers=markers
        )
        exp_st2 = compute_sequence_state(
            [1, *GEMMA4_START, 40, *GEMMA4_END], budget=1, markers=markers
        )
        assert tuple(state_table[1].tolist()) == exp_st1
        assert query_forced_token_py(exp_st1, markers) == GEMMA4_END[1]
        assert tuple(state_table[2].tolist()) == exp_st2
        assert exp_st2[1] == 0
        assert tuple(state_table[0].tolist()) == UNTRACKED_STATE


class TestTrackerAndRunnerLifecycle:
    """Lifecycle hooks, batch reordering, preemption, chunked prefill, and slots."""

    def test_zero_overhead_when_no_budgeted_requests(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        assert tracker.enabled
        assert not tracker
        assert tracker._state_table is None

        tracker.add("plain", SamplingParams(temperature=0.0), [1, QWEN3_START[0]])
        tracker.add("pool", None, [1, 2])
        assert not tracker
        assert tracker._state_table is None

        req = _request_state("plain", [1, QWEN3_START[0]], SamplingParams())
        assert (
            tracker.prepare_step(torch.device("cpu"), ["plain"], {"plain": req}) is None
        )
        assert tracker._state_table is None

    def test_update_states_add_discard_resubmit_and_streaming_reuse(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        runner = object.__new__(TPUModelRunner)
        runner._min_tokens = min_tokens.MinTokensTracker()
        runner._thinking_budget = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
        runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
        runner._pre_async_results = None
        runner.requests = {}
        runner._install_spec_token_room_guard = lambda: None

        runner._thinking_budget.add(
            "old", SamplingParams(thinking_token_budget=4), [1, QWEN3_START[0]]
        )
        new_req = SimpleNamespace(
            req_id="stream",
            sampling_params=SamplingParams(thinking_token_budget=2),
            prompt_token_ids=[1, QWEN3_START[0], 10],
        )
        so = SimpleNamespace(
            finished_req_ids={"old"},
            preempted_req_ids=None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            kv_cache_block_copies=None,
            scheduled_new_reqs=[new_req],
        )
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so)
        assert dict(runner._thinking_budget.requests) == {"stream": 2}

        # Same-step abort + resubmit under identical req_id updates budget cleanly.
        resub_req = SimpleNamespace(
            req_id="stream",
            sampling_params=SamplingParams(thinking_token_budget=5),
            prompt_token_ids=[1, QWEN3_START[0]],
        )
        so.finished_req_ids = {"stream"}
        so.scheduled_new_reqs = [resub_req]
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so)
        assert dict(runner._thinking_budget.requests) == {"stream": 5}

        # Streaming-input continuation without a budget clears tracking for req_id.
        no_budget_req = SimpleNamespace(
            req_id="stream",
            sampling_params=SamplingParams(temperature=0.0),
            prompt_token_ids=[1, QWEN3_START[0], 10, QWEN3_END[0], 20],
        )
        so.finished_req_ids = set()
        so.scheduled_new_reqs = [no_budget_req]
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so)
        assert not runner._thinking_budget

    @pytest.mark.parametrize("explicit_preempt_set", [True, False])
    def test_preemption_and_resume_restores_exact_state(
        self, explicit_preempt_set: bool
    ) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        runner = object.__new__(TPUModelRunner)
        runner._min_tokens = min_tokens.MinTokensTracker()
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        runner._thinking_budget = tracker
        runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
        runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
        runner._pre_async_results = None
        runner._install_spec_token_room_guard = lambda: None

        params = SamplingParams(thinking_token_budget=3)
        prompt = [1, *GEMMA4_START]
        req = _request_state("p", prompt, params, output_token_ids=[20])
        runner.requests = {"p": req}
        tracker.add("p", params, prompt)

        # Step 1 before preemption: advance on-device state to stale count=3
        # (simulating unsettled optimistic tokens prior to preemption rollback).
        table = tracker.prepare_step(torch.device("cpu"), ["p"], runner.requests)
        assert table is not None and tracker.markers is not None
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["p"], 0, 1, 4)
        assert slots is not None
        for tok in (20, 21, 22):
            _, table = override_and_advance_plain(
                table,
                slots,
                torch.tensor([[tok], [0], [0], [0]], dtype=torch.int64),
                tracker.markers,
            )
            tracker.commit_state_table(table)

        # Preempt (with explicit set or Model Runner V1's None), then resume
        # with settled output_token_ids=[20, 21] (2 of 3 thinking tokens used).
        so_preempt = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"p"} if explicit_preempt_set else None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            kv_cache_block_copies=None,
            scheduled_new_reqs=[],
        )
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so_preempt)

        req.output_token_ids = [20, 21]
        so_resume = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"p"}),
            kv_cache_block_copies=None,
            scheduled_new_reqs=[],
        )
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so_resume)

        # Next decode token (22) is the 3rd thinking token -> MUST NOT be forced
        # yet; the step after that MUST force GEMMA4_END[0], then GEMMA4_END[1].
        table = tracker.prepare_step(torch.device("cpu"), ["p"], runner.requests)
        assert table is not None
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["p"], 0, 1, 4)
        assert slots is not None
        out_3rd, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[22], [0], [0], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        assert out_3rd[0, 0].item() == 22

        out_end0, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[23], [0], [0], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        out_end1, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[24], [0], [0], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        assert [out_end0[0, 0].item(), out_end1[0, 0].item()] == list(GEMMA4_END)

    def test_batch_reordering_follows_req_id(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        batch = _make_input_batch()

        p_a = SamplingParams(thinking_token_budget=1)
        p_b = SamplingParams()
        req_a = _request_state("a", [1, QWEN3_START[0], 10], p_a)
        req_b = _request_state("b", [1, QWEN3_START[0], 10], p_b)
        batch.add_request(req_a)
        batch.add_request(req_b)
        batch.refresh_metadata()
        tracker.add("a", p_a, req_a.prompt_token_ids)
        tracker.add("b", p_b, req_b.prompt_token_ids)

        # Simulate _reorder_batch_for_rpa swapping row 0 ("a") and row 1 ("b").
        batch.swap_states(0, 1)
        assert batch.req_ids[:2] == ["b", "a"]

        table = tracker.prepare_step(
            torch.device("cpu"), batch.req_ids[:2], {"a": req_a, "b": req_b}
        )
        assert table is not None and tracker.markers is not None
        slots = tracker.build_chunk_slot_ids(
            torch.device("cpu"), batch.req_ids, 0, 2, padded_num_reqs=4
        )
        assert slots is not None
        assert slots[0].item() == 0 and slots[1].item() > 0

        out, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[55], [66], [0], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        # Row 0 ("b", unbudgeted) keeps 55; row 1 ("a", exhausted) forces QWEN3_END.
        assert out[:2, 0].tolist() == [55, QWEN3_END[0]]

    def test_partial_chunked_prefill_does_not_force_or_advance_state(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        logits = torch.zeros((1, 256), dtype=torch.float32)
        logits[0, 42] = 5.0

        prompt = [1, 2, 3, 4, 5, *GEMMA4_START]  # 8 prompt tokens, open thinking
        params = SamplingParams(temperature=0.0, thinking_token_budget=0)
        req = _request_state("r0", prompt, params, num_computed_tokens=0)

        class _OffsetGenerator:
            def __init__(self, offset: int) -> None:
                self.offset = offset

            def get_offset(self) -> int:
                return self.offset

            def set_offset(self, offset: int) -> None:
                self.offset = offset

        fake_gen = _OffsetGenerator(16)
        runner = SimpleNamespace(
            execute_model_state=SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": 4},  # 0 + 4 < 8 -> partial prefill!
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[logits],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=False,
                kv_save_nan_check=None,
            ),
            input_batch=SimpleNamespace(
                max_num_logprobs=None,
                num_reqs=1,
                req_ids=["r0"],
                req_id_to_index={"r0": 0},
                all_greedy=True,
                generators={0: fake_gen},
                num_tokens_no_spec=[4],
                token_ids_cpu=np.zeros((1, 16), dtype=np.int32),
            ),
            requests={"r0": req},
            _pre_async_results=None,
            _pp_wave=None,
            _dp_step_other_decode=False,
            _min_tokens=min_tokens.MinTokensTracker(),
            _thinking_budget=ThinkingBudgetTracker(cfg, max_num_reqs=4),
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            _is_async_drafter=False,
            _fence_spec_kv_under_token_copy=False,
            _is_dspark_pd_producer=False,
            is_pooling_model=False,
            _get_sampling_generator=lambda: None,
            _greedy_sample=lambda x: torch.argmax(x, dim=-1, keepdim=True),
            _update_mamba_slot_read_offsets=lambda *_args: None,
            _update_num_xla_graphs=lambda *_args: None,
            _modify_prev_results=lambda: None,
            _stage_timing=MagicMock(),
        )
        runner._stage_timing.take_samples.return_value = ([], [])
        runner._thinking_budget.add("r0", params, prompt)
        for name in (
            "_mask_min_tokens",
            "_min_tokens_batch_state",
            "_settle_draft_placeholders_for_min_tokens",
            "_num_output_tokens",
            "maybe_wait_for_kv_save",
            "get_finished_kv_transfers",
        ):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))

        # Chunk 1 (4/8 tokens): single pre-sampling discard pass suppresses
        # thinking-budget advancement, discards sampled_token_ids, and rewinds
        # generator offset by -4.
        out_partial = _run_sample_tokens(runner)
        assert out_partial.sampled_token_ids == [[]]
        assert fake_gen.offset == 12

        # Chunk 2 (remaining 4/8 tokens): completes prefill and forces GEMMA4_END[0],
        # then next decode forces GEMMA4_END[1] (without further generator rewinds).
        req.num_computed_tokens = 4
        emitted: list[int] = []
        for scheduled in (4, 1, 1):
            runner.execute_model_state = SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": scheduled},
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[logits],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=False,
                kv_save_nan_check=None,
            )
            step_out = _run_sample_tokens(runner)
            tok = step_out.sampled_token_ids[0][0]
            emitted.append(tok)
            req.num_computed_tokens += scheduled
        assert emitted == [GEMMA4_END[0], GEMMA4_END[1], 42]
        assert req.output_token_ids == emitted
        assert fake_gen.offset == 12

    def test_slot_pool_sized_at_max_num_reqs_plus_one_reconciles_and_asserts(
        self,
    ) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        # max_num_reqs=2 -> capacity=3 (slot 0 sentinel + 2 active slots 1..2).
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=2)
        assert tracker._capacity == 3
        sp = SamplingParams(thinking_token_budget=2)
        r0 = _request_state("r0", [1, QWEN3_START[0]], sp, output_token_ids=[10, 11])
        r1 = _request_state("r1", [1, QWEN3_START[0]], sp, output_token_ids=[10])
        r2 = _request_state("r2", [1, QWEN3_START[0]], sp, output_token_ids=[])
        for req in (r0, r1, r2):
            tracker.add(req.req_id, sp, [1, QWEN3_START[0]])

        table = tracker.prepare_step(
            torch.device("cpu"), ["r0", "r1"], {"r0": r0, "r1": r1}
        )
        assert table is not None
        assert table.shape == (3, thinking_budget.NUM_STATE_COLS)

        # Exceeding max_num_reqs live slotted requests fails fast with AssertionError.
        with pytest.raises(AssertionError, match="slot pool exhausted"):
            tracker.prepare_step(
                torch.device("cpu"),
                ["r0", "r1", "r2"],
                {"r0": r0, "r1": r1, "r2": r2},
            )

        # Once r0 leaves active requests (e.g. finished or cleaned up), its slot
        # is reclaimed automatically when r2 enters the batch.
        slot_r1_before = tracker._slot_by_req_id["r1"]
        tracker.prepare_step(torch.device("cpu"), ["r1", "r2"], {"r1": r1, "r2": r2})
        assert tracker._slot_by_req_id["r1"] == slot_r1_before
        assert "r0" not in tracker._slot_by_req_id
        assert "r2" in tracker._slot_by_req_id

        # If r0 re-enters requests after r2 leaves, prepare_step re-derives r0's
        # state from prompt + output_token_ids ([10, 11] -> budget=2 reached).
        table = tracker.prepare_step(torch.device("cpu"), ["r0"], {"r0": r0, "r1": r1})
        assert table is not None and tracker.markers is not None
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["r0"], 0, 1, 2)
        assert slots is not None
        out, table = override_and_advance_plain(
            table,
            slots,
            torch.tensor([[50], [0]], dtype=torch.int64),
            tracker.markers,
        )
        tracker.commit_state_table(table)
        assert out[0, 0].item() == QWEN3_END[0]

    def test_scatter_state_rows_and_chunk_slots_enforce_unique_positive_slots(
        self,
    ) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        sp = SamplingParams(thinking_token_budget=2)
        req_a = _request_state("a", [1, QWEN3_START[0]], sp)
        tracker.add("a", sp, [1, QWEN3_START[0]])
        table = tracker.prepare_step(torch.device("cpu"), ["a"], {"a": req_a})
        assert table is not None

        # Duplicate padded sentinel 0 lanes are valid; duplicate positive slots fail.
        valid = thinking_budget.seed_states(
            table,
            torch.tensor([1, 0, 0, 0], dtype=torch.int32),
            torch.zeros((4, thinking_budget.NUM_STATE_COLS), dtype=torch.int32),
        )
        assert valid.shape == table.shape
        with pytest.raises(AssertionError, match="Duplicate positive slot IDs"):
            thinking_budget.seed_states(
                table,
                torch.tensor([1, 1, 0, 0], dtype=torch.int32),
                torch.zeros((4, thinking_budget.NUM_STATE_COLS), dtype=torch.int32),
            )
        with pytest.raises(AssertionError, match="Duplicate slot"):
            tracker.build_chunk_slot_ids(torch.device("cpu"), ["a", "a"], 0, 2, 4)

    def test_steady_state_prepare_step_and_chunk_fast_cache_invalidation(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        dev = torch.device("cpu")
        sp = SamplingParams(thinking_token_budget=2)
        r0 = _request_state("r0", [1, QWEN3_START[0]], sp)
        r1 = _request_state("r1", [1, QWEN3_START[0]], SamplingParams())
        reqs: dict[str, CachedRequestState] = {"r0": r0, "r1": r1}
        tracker.add("r0", sp, r0.prompt_token_ids)
        tracker.add("r1", r1.sampling_params, r1.prompt_token_ids)

        t1 = tracker.prepare_step(dev, ["r0", "r1"], reqs)
        assert t1 is not None
        epoch_after_seed = tracker._state_epoch
        s1 = tracker.build_chunk_slot_ids(dev, ["r0", "r1"], 0, 2, 4)
        s2 = tracker.build_chunk_slot_ids(dev, ["r0", "r1"], 0, 2, 4)
        assert s1 is not None and s2 is s1
        assert tracker.prepare_step(dev, ["r0", "r1"], reqs) is t1
        assert tracker._state_epoch == epoch_after_seed

        # Untracked-only batch also caches None cleanly.
        assert tracker.prepare_step(dev, ["r1"], reqs) is None
        assert tracker.prepare_step(dev, ["r1"], reqs) is None
        assert tracker.build_chunk_slot_ids(dev, ["r1"], 0, 1, 4) is None
        assert tracker.build_chunk_slot_ids(dev, ["r1"], 0, 1, 4) is None

        # Preemption + resume bump _state_epoch and invalidate _chunk_fast_cache.
        tracker.on_preempted(["r0"])
        assert tracker._state_epoch > epoch_after_seed
        assert not tracker._chunk_fast_cache
        assert tracker.build_chunk_slot_ids(dev, ["r0", "r1"], 0, 2, 4) is None

        tracker.on_resumed(["r0"], reqs)
        t_resumed = tracker.prepare_step(dev, ["r0", "r1"], reqs)
        assert t_resumed is not None
        s_resumed = tracker.build_chunk_slot_ids(dev, ["r0", "r1"], 0, 2, 4)
        assert s_resumed is not None and int(s_resumed[0].item()) > 0

        # Discard bumps epoch and clears fast cache.
        tracker.discard(["r0"])
        assert not tracker._chunk_fast_cache
        assert tracker.prepare_step(dev, ["r0", "r1"], reqs) is None


class TestSampleTokensIntegration:
    """End-to-end ``TPUModelRunner.sample_tokens`` coverage across runner modes."""

    def _make_plain_runner(
        self,
        reasoning_cfg: ReasoningConfig,
        req_ids: list[str],
        requests: dict[str, CachedRequestState],
        vocab_size: int = 256,
        all_greedy: bool = True,
        async_scheduling: bool = False,
        top_tokens: bool = False,
    ) -> SimpleNamespace:
        tracker = ThinkingBudgetTracker(reasoning_cfg, max_num_reqs=4)
        min_tracker = min_tokens.MinTokensTracker()
        for rid, rstate in requests.items():
            tracker.add(rid, rstate.sampling_params, rstate.prompt_token_ids)
            min_tracker.add(rid, rstate.sampling_params)

        runner = SimpleNamespace(
            execute_model_state=None,
            input_batch=SimpleNamespace(
                max_num_logprobs=None,
                num_reqs=len(req_ids),
                req_ids=list(req_ids),
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                all_greedy=all_greedy,
                generators={},
                num_tokens_no_spec=[
                    len(requests[rid].prompt_token_ids)
                    + len(requests[rid].output_token_ids)
                    for rid in req_ids
                ],
                token_ids_cpu=np.zeros((max(len(req_ids), 1), 64), dtype=np.int32),
            ),
            requests=requests,
            _pre_async_results=None,
            _pp_wave=None,
            _dp_step_other_decode=False,
            _min_tokens=min_tracker,
            _thinking_budget=tracker,
            _greedy_head_owner=object() if top_tokens else None,
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
            _is_async_drafter=False,
            _mtp_prefill_only=False,
            _tp_rank0_mask=None,
            _fence_spec_kv_under_token_copy=False,
            _is_dspark_pd_producer=False,
            is_pooling_model=False,
            max_num_reqs=4,
            vocab_size=vocab_size,
            _get_sampling_generator=lambda: torch.Generator().manual_seed(7),
            _greedy_sample=lambda x: torch.argmax(x, dim=-1, keepdim=True),
            _update_mamba_slot_read_offsets=lambda *_args: None,
            _update_num_xla_graphs=lambda *_args: None,
            _drafts_per_chunk=TPUModelRunner._drafts_per_chunk,
            _stage_timing=MagicMock(),
        )
        runner._stage_timing.take_samples.return_value = ([], [])
        for name in (
            "_mask_min_tokens",
            "_min_tokens_batch_state",
            "_settle_draft_placeholders_for_min_tokens",
            "_num_output_tokens",
            "_greedy_top_tokens_this_step",
            "_apply_temperature",
            "_sampler_applies_topk_topp",
            "_modify_prev_results",
            "_update_placeholder",
            "_sync_replicated_drafts_across_tp",
            "_sync_one_replicated_draft",
            "_assemble_async_spec_substitution",
            "_assemble_async_prefill_bootstrap",
            "maybe_wait_for_kv_save",
            "get_finished_kv_transfers",
        ):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
        runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(
            runner
        )
        return runner

    def test_plain_step_random_sampling_raw_logprobs_and_min_tokens(self) -> None:
        from vllm.v1.outputs import LogprobsTensors

        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        # min_tokens=5 masks stop token 99, while thinking_token_budget=1 forces
        # GEMMA4_END (106, 107) after 1 thinking token under random sampling.
        params = SamplingParams(
            temperature=0.7,
            top_k=4,
            top_p=0.95,
            min_tokens=5,
            stop_token_ids=[99],
            thinking_token_budget=1,
        )
        prompt = [1, *GEMMA4_START]
        req = _request_state("r0", prompt, params)
        runner = self._make_plain_runner(
            cfg, ["r0"], {"r0": req}, vocab_size=256, all_greedy=False
        )
        runner.input_batch.no_top_k = False
        runner.input_batch.no_top_p = False
        runner.input_batch.max_num_logprobs = 0
        runner._build_padded_sampling_params = lambda *_args: (
            torch.full((4, 1), 0.7, dtype=torch.float32),
            torch.full((4, 1), 4, dtype=torch.int32),
            torch.full((4, 1), 0.95, dtype=torch.float32),
        )
        runner._sampler_topk_candidates = lambda *_args: 0

        raw_logprob_inputs: list[torch.Tensor] = []
        chosen_for_logprobs: list[int] = []

        def gather_logprobs(logprob_logits: torch.Tensor, selected: torch.Tensor):
            raw_logprob_inputs.append(logprob_logits.clone())
            chosen_for_logprobs.append(int(selected[0, 0].item()))
            n = logprob_logits.shape[0]
            return LogprobsTensors(
                selected.view(-1, 1),
                torch.zeros((n, 1), dtype=torch.float32),
                torch.zeros(n, dtype=torch.int64),
            )

        runner.gather_logprobs = gather_logprobs

        # Logits strongly prefer stop token 99 (masked by min_tokens<5), then 50;
        # GEMMA4_END tokens (106, 107) have logit -100 so random sampling with
        # top_k=4 could NEVER pick them without budget forcing!
        base_logits = torch.full((4, 256), -100.0, dtype=torch.float32)
        base_logits[0, 99] = 20.0
        base_logits[0, 50] = 10.0

        emitted: list[int] = []
        for _ in range(4):
            runner.execute_model_state = SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": 1},
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[base_logits.clone()],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=False,
                kv_save_nan_check=None,
            )
            out = _run_sample_tokens(runner)
            tok = out.sampled_token_ids[0][0]
            emitted.append(tok)
            req.num_computed_tokens += 1

        # Step 0: min_tokens masks 99 -> picks 50 (1st thinking token).
        # Step 1 & 2: budget=1 reached -> forces GEMMA4_END [106, 107].
        # Step 3: post-thinking, still below min_tokens=5 (3 < 5) -> picks 50.
        assert emitted == [50, GEMMA4_END[0], GEMMA4_END[1], 50]
        assert req.output_token_ids == emitted
        assert chosen_for_logprobs == emitted
        # Raw logprobs saw unmasked/unforced logits (token 99 == 20.0, 106 == -100.0).
        for seen in raw_logprob_inputs:
            assert seen[0, 99].item() == 20.0
            assert seen[0, GEMMA4_END[0]].item() == -100.0

    def test_greedy_top_tokens_stays_enabled_for_plain_and_spec_verify(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        params = SamplingParams(temperature=0.0, thinking_token_budget=2)
        req = _request_state("r0", [1, *GEMMA4_START], params)
        runner = self._make_plain_runner(cfg, ["r0"], {"r0": req}, top_tokens=True)
        # TPU_GREEDY_TOP_TOKENS eligibility must remain True with thinking_token_budget!
        assert greedy_top_tokens.step_eligible(runner.input_batch, runner.requests)
        assert runner._greedy_top_tokens_this_step()

        # Plain decode with top_tokens=True (logits_list carries [padded, 1] int64).
        emitted: list[int] = []
        for raw_greedy in (30, 31, 32, 33, 34):
            top_tok_tensor = torch.tensor(
                [[raw_greedy], [0], [0], [0]], dtype=torch.int64
            )
            runner.execute_model_state = SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": 1},
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[top_tok_tensor],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=True,
                kv_save_nan_check=None,
            )
            out = _run_sample_tokens(runner)
            tok = out.sampled_token_ids[0][0]
            emitted.append(tok)
            req.num_computed_tokens += 1
        assert emitted == [30, 31, GEMMA4_END[0], GEMMA4_END[1], 34]
        assert req.output_token_ids == emitted

        # Speculative verify with top_tokens=True vs top_tokens=False must match!
        spec_cfg = SimpleNamespace(
            num_speculative_tokens=3,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        runner.speculative_config = spec_cfg
        runner.rejection_sampler = RejectionSampler(spec_cfg, torch.device("cpu"))
        runner.spec_bonus_and_target_tokens = (
            TPUModelRunner.spec_bonus_and_target_tokens.__get__(runner)
        )
        runner.spec_bonus_and_target_logits = (
            TPUModelRunner.spec_bonus_and_target_logits.__get__(runner)
        )

        # 4 rows (3 drafts + 1 bonus) for 1 request padded to num_reqs=2, num_tokens=4.
        full_logits = torch.zeros((4, 256), dtype=torch.float32)
        for row_i, pref in enumerate((40, 41, 42, 43)):
            full_logits[row_i, pref] = 8.0
        top_tokens_rows = torch.argmax(full_logits, dim=-1, keepdim=True)
        md = SimpleNamespace(
            draft_token_ids=torch.tensor([40, 41, 42, 0], dtype=torch.int32),
            draft_lengths=torch.tensor([3, 0], dtype=torch.int32),
            draft_lengths_cpu=np.array([3]),
            target_logits_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([3, 0], dtype=torch.int32),
            segment_ids=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            group_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        )

        outputs = []
        for use_top_tokens, values in (
            (False, full_logits),
            (True, top_tokens_rows),
        ):
            tb = ThinkingBudgetTracker(cfg, max_num_reqs=4)
            tb.add("r0", SamplingParams(thinking_token_budget=1), [1, *GEMMA4_START])
            r_fresh = _request_state("r0", [1, *GEMMA4_START], params)
            table = tb.prepare_step(torch.device("cpu"), ["r0"], {"r0": r_fresh})
            assert table is not None and tb.markers is not None
            step_state = _ThinkingBudgetStepState(tb, table, tb.markers, frozenset())
            tb_chunk = step_state.spec_chunk(
                torch.device("cpu"), ["r0"], 0, 1, md.draft_lengths.shape[0]
            )
            assert tb_chunk is not None and isinstance(
                tb_chunk, _ThinkingBudgetSpecChunk
            )
            with torch.compiler.set_stance("force_eager"):
                res = TPUModelRunner._sample_spec_verify_chunk(
                    runner,
                    values,
                    md,
                    None,
                    SimpleNamespace(scheduled_spec_decode_tokens={}),
                    0,
                    1,
                    True,
                    None,
                    use_top_tokens,
                    None,
                    tb_chunk,
                )
            outputs.append(res[0].tolist())
        # Draft 0 (40) accepted as the 1 thinking token; draft 1 (41 != 106)
        # rejected and recovers GEMMA4_END[0] (106)!
        assert outputs[0] == [40, GEMMA4_END[0], -1, -1]
        assert outputs[1] == outputs[0]

    def test_async_scheduling_plain_decode_zero_lag_multi_token_end(self) -> None:
        """Under ``--async-scheduling``, ``req.output_token_ids`` holds ``-1``
        placeholders when ``sample_tokens`` begins (settled later in
        ``_modify_prev_results``). Dynamic multi-token start + multi-token
        forced end must have zero 1-step lag, zero duplicates, and propagate
        forced tokens into ``_pre_async_results.next_tokens_tpu``."""
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        params = SamplingParams(temperature=0.0, thinking_token_budget=2)
        prompt = [1, 2]  # Thinking NOT open in prompt!
        req = _request_state("r0", prompt, params, num_computed_tokens=0)
        runner = self._make_plain_runner(
            cfg, ["r0"], {"r0": req}, vocab_size=4096, async_scheduling=True
        )

        # Model greedily proposes: 10, <|channel>thought\n (3 tokens), 50..53, 60, 61.
        # With budget=2:
        #   - 10 (outside thinking)
        #   - 105, 2364, 107 (opens thinking)
        #   - 50, 51 (2 thinking tokens)
        #   - 52 -> forced to 106 (GEMMA4_END[0])
        #   - 53 -> forced to 107 (GEMMA4_END[1])
        #   - 60, 61 (post-thinking content)
        model_stream = [10, *GEMMA4_START, 50, 51, 52, 53, 60, 61]
        next_tpu_log: list[int] = []

        for step_idx, raw_tok in enumerate(model_stream):
            if step_idx > 0:
                # Previous step's placeholder is still unsettled (-1) on host!
                assert req.output_token_ids[-1] == -1

            logits = torch.zeros((4, 4096), dtype=torch.float32)
            logits[0, raw_tok] = 9.0
            sched_tokens = len(prompt) if step_idx == 0 else 1
            runner.execute_model_state = SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": sched_tokens},
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[logits],
                num_reqs_list=[1],
                spec_decode_metadata_list=[None],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=False,
                kv_save_nan_check=None,
            )
            async_out = _run_sample_tokens(runner)
            assert runner._pre_async_results is not None
            # Verify the on-device next_tokens_tpu tensor for placeholder substitution
            # carries the post-forcing token ID!
            next_tpu_log.append(
                int(runner._pre_async_results.next_tokens_tpu[0].item())
            )
            # Mark the newly appended placeholder as -1 before next step's settlement.
            req.output_token_ids[-1] = -1
            _ = async_out.get_output()
            req.num_computed_tokens += sched_tokens

        # Settle final in-flight async step.
        runner._modify_prev_results()

        expected = [10, *GEMMA4_START, 50, 51, *GEMMA4_END, 60, 61]
        assert req.output_token_ids == expected
        assert next_tpu_log == expected

    def test_async_scheduling_spec_decode_with_placeholder_host_drafts(self) -> None:
        """Async EAGLE3/MTP: ``scheduled_spec_decode_tokens`` and
        ``req.output_token_ids`` on host hold ``-1`` placeholders while real
        drafts live on device in ``md.draft_token_ids``. Multi-step verify must
        force ``GEMMA4_END`` across step boundaries without lag or corruption."""
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        spec_cfg = SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=2,
            draft_tensor_parallel_size=None,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        params = SamplingParams(temperature=0.0, thinking_token_budget=3)
        prompt = [1, *GEMMA4_START]
        req = _request_state("r0", prompt, params, num_computed_tokens=0)

        runner = self._make_plain_runner(
            cfg, ["r0"], {"r0": req}, vocab_size=4096, async_scheduling=True
        )
        runner.speculative_config = spec_cfg
        runner.drafter = SimpleNamespace(bucketed_drafts=False)
        runner._is_async_drafter = True
        runner._async_drafter = SimpleNamespace(
            draft_chunks=None, bucketed_drafts=False
        )
        runner.rejection_sampler = RejectionSampler(spec_cfg, torch.device("cpu"))
        runner.spec_bonus_and_target_logits = (
            TPUModelRunner.spec_bonus_and_target_logits.__get__(runner)
        )
        runner._sample_spec_verify_chunk = (
            TPUModelRunner._sample_spec_verify_chunk.__get__(runner)
        )
        runner.spec_decode_manager = SimpleNamespace(
            propose_draft_token_ids=lambda **_kwargs: torch.tensor(
                [[41, 42]], dtype=torch.int32
            ),
            stage_draft_token_ids_for_host=lambda *_args, **_kwargs: None,
        )

        # Step 0: async prefill bootstrap samples 40 (1st thinking token) and
        # parks 1+K=3 optimistic placeholders on host.
        prefill_logits = torch.zeros((4, 4096), dtype=torch.float32)
        prefill_logits[0, 40] = 7.0
        runner.execute_model_state = SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": len(prompt)},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            draft_chunks=[],
            logits_list=[prefill_logits],
            num_reqs_list=[1],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
        )
        out0 = _run_sample_tokens(runner).get_output()
        assert out0.sampled_token_ids == [[40]]
        assert len(req.output_token_ids) == 1
        req.output_token_ids[-1] = -1
        req.num_computed_tokens = len(prompt)

        # Step 1: device drafts [41, 42], target prefers [41, 42, 43].
        #   41 (2nd thinking), 42 (3rd thinking == budget!) -> bonus forced to 106!
        # Step 2: device drafts [44, 45], target prefers [44, 45, 46].
        #   Device already has end_match=1, forcing_end=1 -> draft 0 recovers 107!
        # Step 3: device drafts [70, 71], target prefers [70, 71, 72].
        #   Thinking is closed on device -> [70, 71, 72] all accepted!
        steps = [
            ([41, 42], [41, 42, 43], [41, 42, GEMMA4_END[0]]),
            ([44, 45], [44, 45, 46], [GEMMA4_END[1]]),
            ([70, 71], [70, 71, 72], [70, 71, 72]),
        ]
        for step_i, (dev_drafts, row_prefs, exp_tokens) in enumerate(steps):
            if step_i == 0:
                assert req.output_token_ids[-1:] == [-1]
                req.num_computed_tokens = req.num_tokens - 1
            else:
                assert req.output_token_ids[-3:] == [-1, -1, -1]
                req.num_computed_tokens = req.num_tokens - 3
            logits = torch.zeros((4, 4096), dtype=torch.float32)
            for r_i, tok in enumerate(row_prefs):
                logits[r_i, tok] = 7.0
            md = SimpleNamespace(
                draft_token_ids=torch.tensor([*dev_drafts, 0, 0], dtype=torch.int32),
                draft_lengths=torch.tensor([2, 0], dtype=torch.int32),
                draft_lengths_cpu=np.array([2]),
                target_logits_indices=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
                bonus_logits_indices=torch.tensor([2, 0], dtype=torch.int32),
                final_logits_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
                segment_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                group_indices=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
            )
            runner.execute_model_state = SimpleNamespace(
                scheduler_output=SimpleNamespace(
                    num_scheduled_tokens={"r0": 3},
                    scheduled_spec_decode_tokens={"r0": [-1, -1]},
                    finished_req_ids=set(),
                    has_structured_output_requests=False,
                ),
                draft_chunks=[],
                logits_list=[logits],
                num_reqs_list=[1],
                spec_decode_metadata_list=[md],
                mamba_state_indices_list=None,
                pooler_output_list=[],
                top_tokens=False,
                kv_save_nan_check=None,
            )
            out = _run_sample_tokens(runner).get_output()
            assert out.sampled_token_ids == [exp_tokens]
            req.output_token_ids[-3:] = [-1, -1, -1]

        runner._modify_prev_results()
        assert req.output_token_ids == [40, 41, 42, *GEMMA4_END, 70, 71, 72]

    @pytest.mark.parametrize("mode", ["greedy", "random", "synthetic"])
    def test_spec_verify_chunk_modes_smart_drafter_bonus_and_early_reject(
        self, mode: str
    ) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None

        synthetic = mode == "synthetic"
        greedy = mode != "random"
        spec_cfg = SimpleNamespace(
            num_speculative_tokens=3,
            rejection_sample_method="synthetic" if synthetic else "strict",
            # Conditional acceptance rate 1.0 would blindly accept mismatching
            # drafts unless prepare_spec_verify invalidates them to -1!
            synthetic_acceptance_rates=[1.0, 1.0, 1.0] if synthetic else None,
        )
        runner = SimpleNamespace(
            speculative_config=spec_cfg,
            rejection_sampler=RejectionSampler(spec_cfg, torch.device("cpu")),
            _min_tokens=min_tokens.MinTokensTracker(),
            input_batch=SimpleNamespace(no_top_k=True, no_top_p=True),
            _sampler_topk_candidates=lambda *_args: 0,
            _build_padded_sampling_params=lambda *_args: (
                torch.full((4, 1), 0.01, dtype=torch.float32),
                torch.zeros((4, 1), dtype=torch.int32),
                torch.ones((4, 1), dtype=torch.float32),
            ),
        )
        for name in (
            "spec_bonus_and_target_logits",
            "spec_gather_bonus_and_target_logits",
            "_apply_temperature",
            "_sampler_applies_topk_topp",
            "_greedy_sample",
        ):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
        runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(
            runner
        )

        # 3 requests in one verify chunk (padded to 4 reqs, 16 tokens), K=3:
        #   req 0 (slot 1, budget=1): smart drafter proposes [40, 106, 107] and
        #     bonus prefers 88 -> ALL 4 tokens [40, 106, 107, 88] accepted in one step!
        #   req 1 (slot 2, budget=3): drafts [50, 51, 52] all match target, bonus
        #     prefers 53 -> bonus is at think_count=3, so bonus forced to 106!
        #   req 2 (slot 3, budget=2): drafts [60, 61, 62], target [69, 61, 62, 63]
        #     -> in strict greedy/random, draft 0 (60 != 69) rejects BEFORE budget,
        #     so state only advances by 1 thinking token (69), NOT 3! (In synthetic
        #     rate=1.0 mode, draft 0 is synthetically accepted, draft 1 is 2nd
        #     thinking token, and draft 2 is forced-invalidated -> recovers 106!)
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        reqs = {}
        for idx, b in enumerate((1, 3, 2)):
            rid = f"r{idx}"
            sp = SamplingParams(
                temperature=0.0 if greedy else 0.01, thinking_token_budget=b
            )
            reqs[rid] = _request_state(rid, [1, *GEMMA4_START], sp)
            tracker.add(rid, sp, [1, *GEMMA4_START])

        table = tracker.prepare_step(torch.device("cpu"), ["r0", "r1", "r2"], reqs)
        assert table is not None
        step_state = _ThinkingBudgetStepState(tracker, table, markers, frozenset())
        tb_chunk = step_state.spec_chunk(
            torch.device("cpu"), ["r0", "r1", "r2"], 0, 3, 4
        )
        assert tb_chunk is not None

        row_prefs = [
            40,
            55,
            56,
            88,  # req 0 (target prefers 55/56 at pos 1/2, overridden by 106/107!)
            50,
            51,
            52,
            53,  # req 1 (bonus 53 overridden by 106!)
            69,
            61,
            62,
            63,  # req 2
        ]
        logits = torch.full((16, 256), -50.0, dtype=torch.float32)
        for r_i, pref in enumerate(row_prefs):
            logits[r_i, pref] = 10.0

        drafts = [40, GEMMA4_END[0], GEMMA4_END[1], 50, 51, 52, 60, 61, 62]
        md = SimpleNamespace(
            draft_token_ids=torch.tensor(drafts + [0] * 7, dtype=torch.int32),
            draft_lengths=torch.tensor([3, 3, 3, 0], dtype=torch.int32),
            draft_lengths_cpu=np.array([3, 3, 3]),
            target_logits_indices=torch.tensor(
                [0, 1, 2, 4, 5, 6, 8, 9, 10] + [0] * 7, dtype=torch.int32
            ),
            bonus_logits_indices=torch.tensor([3, 7, 11, 0], dtype=torch.int32),
            segment_ids=torch.tensor(
                [0, 0, 0, 1, 1, 1, 2, 2, 2] + [3] * 7, dtype=torch.int64
            ),
            group_indices=torch.tensor(
                [0, 1, 2, 0, 1, 2, 0, 1, 2] + [0] * 7, dtype=torch.int32
            ),
        )

        with torch.compiler.set_stance("force_eager"):
            out = TPUModelRunner._sample_spec_verify_chunk(
                runner,
                logits,
                md,
                None,
                SimpleNamespace(scheduled_spec_decode_tokens={}),
                0,
                3,
                greedy,
                torch.Generator().manual_seed(0),
                False,
                None,
                tb_chunk,
            )
        table = tracker.ensure_state_table(torch.device("cpu"))

        # req 0 accepted smart drafter's [40, 106, 107] AND sampled post-think bonus 88!
        assert out[0].tolist() == [40, GEMMA4_END[0], GEMMA4_END[1], 88]
        assert int(table[tracker._slot_by_req_id["r0"], 1].item()) == 0

        # req 1 accepted [50, 51, 52] and forced bonus to GEMMA4_END[0] (106)!
        assert out[1].tolist() == [50, 51, 52, GEMMA4_END[0]]
        slot_r1 = tracker._slot_by_req_id["r1"]
        assert (
            query_forced_token_py(tuple(table[slot_r1].tolist()), markers)
            == (GEMMA4_END[1])
        )

        slot_r2 = tracker._slot_by_req_id["r2"]
        if synthetic:
            # Even with 100% synthetic acceptance, mismatching draft 62 at forced
            # position 2 was invalidated and recovered GEMMA4_END[0]!
            assert out[2].tolist() == [60, 61, GEMMA4_END[0], -1]
        else:
            # Early target rejection at pos 0 recovered 69 -> think_count == 1 (< 2)!
            assert out[2].tolist() == [69, -1, -1, -1]
            assert int(table[slot_r2, 2].item()) == 1
            assert (
                query_forced_token_py(tuple(table[slot_r2].tolist()), markers)
                == NO_FORCED_TOKEN
            )

    def test_mixed_spec_step_forces_both_verify_and_draftless_chunks(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        spec_cfg = SimpleNamespace(
            method="ngram",
            num_speculative_tokens=2,
            draft_tensor_parallel_size=None,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        p0 = SamplingParams(temperature=0.0, thinking_token_budget=1)
        p1 = SamplingParams(temperature=0.0, thinking_token_budget=0)
        r0 = _request_state("r0", [1, *GEMMA4_START], p0)
        r1 = _request_state("r1", [1, *GEMMA4_START], p1)

        runner = self._make_plain_runner(
            cfg, ["r0", "r1"], {"r0": r0, "r1": r1}, vocab_size=4096
        )
        runner.speculative_config = spec_cfg
        runner.rejection_sampler = RejectionSampler(spec_cfg, torch.device("cpu"))
        runner.spec_bonus_and_target_logits = (
            TPUModelRunner.spec_bonus_and_target_logits.__get__(runner)
        )
        runner._sample_spec_verify_chunk = (
            TPUModelRunner._sample_spec_verify_chunk.__get__(runner)
        )
        runner.drafter = SimpleNamespace()
        runner.spec_decode_manager = SimpleNamespace(
            propose_draft_token_ids=lambda **_kwargs: None
        )

        verify_logits = torch.zeros((4, 4096), dtype=torch.float32)
        for i, t in enumerate((30, 31, 32)):
            verify_logits[i, t] = 6.0
        plain_logits = torch.zeros((2, 4096), dtype=torch.float32)
        plain_logits[0, 77] = 6.0

        md0 = SimpleNamespace(
            draft_token_ids=torch.tensor([30, 31, 0, 0], dtype=torch.int32),
            draft_lengths=torch.tensor([2, 0], dtype=torch.int32),
            draft_lengths_cpu=np.array([2]),
            target_logits_indices=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([2, 0], dtype=torch.int32),
            final_logits_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
            segment_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            group_indices=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
        )
        runner.execute_model_state = SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 3, "r1": 1},
                scheduled_spec_decode_tokens={"r0": [30, 31]},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            draft_chunks=[],
            logits_list=[verify_logits, plain_logits],
            num_reqs_list=[1, 1],
            spec_decode_metadata_list=[md0, None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
        )
        out = _run_sample_tokens(runner)
        assert out.sampled_token_ids == [[30, GEMMA4_END[0]], [GEMMA4_END[0]]]

    def test_spec_verify_with_grammar_invalid_drafts_does_not_overcount_budget(
        self,
    ) -> None:
        from vllm_torchtpu.structured_decoding.structured_decoding_manager import (
            StructuredDecodingManager,
        )

        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        spec_cfg = SimpleNamespace(
            method="ngram",
            num_speculative_tokens=3,
            draft_tensor_parallel_size=None,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        p0 = SamplingParams(temperature=0.0, thinking_token_budget=2)
        r0 = _request_state("r0", [1, *QWEN3_START], p0)

        runner = self._make_plain_runner(cfg, ["r0"], {"r0": r0}, vocab_size=152064)
        runner.speculative_config = spec_cfg
        runner.rejection_sampler = RejectionSampler(spec_cfg, torch.device("cpu"))
        runner.spec_gather_bonus_and_target_logits = (
            TPUModelRunner.spec_gather_bonus_and_target_logits.__get__(runner)
        )
        runner._sample_spec_verify_chunk = (
            TPUModelRunner._sample_spec_verify_chunk.__get__(runner)
        )
        runner.drafter = SimpleNamespace()
        runner.spec_decode_manager = SimpleNamespace(
            propose_draft_token_ids=lambda **_kwargs: None
        )
        sd_mgr = StructuredDecodingManager.__new__(StructuredDecodingManager)
        sd_mgr.runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r0": 0})
        )
        runner.structured_decoding_manager = SimpleNamespace(
            mask_spec_logits=lambda target, bonus, *_a, **_kw: (target, bonus),
            mask_invalid_draft_tokens=sd_mgr.mask_invalid_draft_tokens,
        )

        verify_logits = torch.zeros((4, 152064), dtype=torch.float32)
        for i, t in enumerate((30, 31, 32, 33)):
            verify_logits[i, t] = 6.0

        md0 = SimpleNamespace(
            draft_token_ids=torch.tensor([30, 31, 32, 0], dtype=torch.int32),
            draft_lengths=torch.tensor([3, 0], dtype=torch.int32),
            draft_lengths_cpu=np.array([3]),
            target_logits_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([3, 0], dtype=torch.int32),
            final_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            segment_ids=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            group_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        )
        grammar_out = SimpleNamespace(
            structured_output_request_ids=["r0"],
            num_acceptable_drafts=[1],
        )
        runner.execute_model_state = SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 4},
                scheduled_spec_decode_tokens={"r0": [30, 31, 32]},
                finished_req_ids=set(),
                has_structured_output_requests=True,
            ),
            draft_chunks=[],
            logits_list=[verify_logits],
            num_reqs_list=[1],
            spec_decode_metadata_list=[md0],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
        )
        out = _run_sample_tokens(runner, grammar_out)
        # Drafts 31 and 32 were pinned to -1 by num_acceptable_drafts=[1], so
        # draft 0 (30) is accepted and position 1 recovers 31 without premature
        # budget forcing; state_table now records think_count == 2 == budget, so
        # the next step immediately forces QWEN3_END.
        assert out.sampled_token_ids == [[30, 31]]
        r0.num_computed_tokens += len(out.sampled_token_ids[0])
        slot = runner._thinking_budget._slot_by_req_id["r0"]
        assert (
            int(
                runner._thinking_budget._state_table[
                    slot, thinking_budget.COL_THINK_COUNT
                ].item()
            )
            == 2
        )

        next_logits = torch.zeros((2, 152064), dtype=torch.float32)
        next_logits[0, 34] = 6.0
        runner.execute_model_state = SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 1},
                scheduled_spec_decode_tokens={},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            draft_chunks=[],
            logits_list=[next_logits],
            num_reqs_list=[1],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
        )
        out2 = _run_sample_tokens(runner)
        assert out2.sampled_token_ids == [[QWEN3_END[0]]]


@contextlib.contextmanager
def _dynamo_runner_limits() -> Iterator[None]:
    """Mirror ``TPUModelRunner.__init__``'s Dynamo cache limits in standalone tests."""
    torch._dynamo.reset()
    cfg = torch._dynamo.config
    patch_kwargs: dict[str, int] = {
        "cache_size_limit": 1024,
        "accumulated_recompile_limit": 1024,
    }
    if hasattr(cfg, "recompile_limit"):
        patch_kwargs["recompile_limit"] = 1024
    try:
        with cfg.patch(**patch_kwargs):
            yield
    finally:
        torch._dynamo.reset()


class TestStaticShapesAndZeroRecompilations:
    """Bucket precompile coverage and ``torch.compile(dynamic=False)`` checks."""

    @pytest.mark.parametrize("use_top_tokens", [False, True])
    def test_precompile_covers_every_runtime_shape_and_dtype(
        self, use_top_tokens: bool
    ) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        runner = object.__new__(TPUModelRunner)
        runner.max_num_reqs = 16
        runner.num_reqs_paddings = [8, 16]
        runner.num_tokens_paddings = [16, 32, 64, 128]
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
        runner._mtp_prefill_only = False
        runner._greedy_head_owner = object() if use_top_tokens else None
        runner.device = torch.device("cpu")
        runner.vocab_size = 128
        runner._hidden_states_dtype = torch.bfloat16
        runner._precompile_timed = MagicMock()
        runner.structured_decoding_manager = SimpleNamespace(
            target_grammar_bitmask_cpu=torch.zeros((64, 1), dtype=torch.int32)
        )
        runner._thinking_budget = ThinkingBudgetTracker(cfg, max_num_reqs=16)

        calls: set[tuple[object, ...]] = set()
        orig_seed = thinking_budget.seed_states
        orig_plain = thinking_budget.override_and_advance_plain
        orig_prep_logits = thinking_budget.prepare_spec_verify_logits
        orig_prep_tokens = thinking_budget.prepare_spec_verify_tokens
        orig_adv = thinking_budget.advance_spec_states

        def sig(
            *tensors: torch.Tensor,
        ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
            return tuple((tuple(t.shape), t.dtype) for t in tensors)

        with (
            patch.object(
                thinking_budget,
                "seed_states",
                side_effect=lambda st, sl, sv: (
                    calls.add(("seed", *sig(st, sl, sv))) or orig_seed(st, sl, sv)
                ),
            ),
            patch.object(
                thinking_budget,
                "override_and_advance_plain",
                side_effect=lambda st, sl, sel, m: (
                    calls.add(("plain", *sig(st, sl, sel)))
                    or orig_plain(st, sl, sel, m)
                ),
            ),
            patch.object(
                thinking_budget,
                "prepare_spec_verify_logits",
                side_effect=lambda st, sl, dt, lg, seg, grp, k, m: (
                    calls.add(("prep_logits", k, *sig(st, sl, dt, lg, seg, grp)))
                    or orig_prep_logits(st, sl, dt, lg, seg, grp, k, m)
                ),
            ),
            patch.object(
                thinking_budget,
                "prepare_spec_verify_tokens",
                side_effect=lambda st, sl, dt, tk, seg, grp, k, m: (
                    calls.add(("prep_tokens", k, *sig(st, sl, dt, tk, seg, grp)))
                    or orig_prep_tokens(st, sl, dt, tk, seg, grp, k, m)
                ),
            ),
            patch.object(
                thinking_budget,
                "advance_spec_states",
                side_effect=lambda st, sl, nt, k, m: (
                    calls.add(("adv", k, *sig(st, sl, nt)))
                    or orig_adv(st, sl, nt, k, m)
                ),
            ),
        ):
            runner._precompile_thinking_budget()
            precompiled, calls = calls, set()
            if not use_top_tokens:
                assert not any(c[0] == "prep_tokens" for c in precompiled)

            # Replay live serving calls across all padded request & verify buckets
            # (including every reachable (num_tokens, num_reqs) pair).
            tb = runner._thinking_budget
            assert tb.markers is not None
            tb.add("r0", SamplingParams(thinking_token_budget=2), [1, *GEMMA4_START])
            req = _request_state(
                "r0", [1, *GEMMA4_START], SamplingParams(thinking_token_budget=2)
            )
            st_table = tb.prepare_step(torch.device("cpu"), ["r0"], {"r0": req})
            assert st_table is not None

            for num_reqs in runner.num_reqs_paddings:
                slots = tb.build_chunk_slot_ids(
                    torch.device("cpu"), ["r0"], 0, 1, num_reqs
                )
                assert slots is not None
                _, st_table = thinking_budget.override_and_advance_plain(
                    st_table,
                    slots,
                    torch.zeros((num_reqs, 1), dtype=torch.int64),
                    tb.markers,
                )
                tb.commit_state_table(st_table)

            step_state = _ThinkingBudgetStepState(tb, st_table, tb.markers, frozenset())
            for num_tokens in runner._verify_logits_paddings():
                for num_reqs in runner.num_reqs_paddings:
                    if not runner._spec_verify_bucket_reachable(num_tokens, num_reqs):
                        continue
                    tb_chunk = step_state.spec_chunk(
                        torch.device("cpu"), ["r0"], 0, 1, num_reqs
                    )
                    assert tb_chunk is not None
                    md = SimpleNamespace(
                        segment_ids=torch.full((num_tokens,), 1, dtype=torch.int64),
                        group_indices=torch.zeros(num_tokens, dtype=torch.int32),
                    )
                    tb_chunk.prepare_logits(
                        torch.zeros(num_tokens, dtype=torch.int32),
                        torch.zeros((num_tokens, 128), dtype=torch.bfloat16),
                        md,
                        3,
                    )
                    if use_top_tokens:
                        tb_chunk.prepare_tokens(
                            torch.zeros(num_tokens, dtype=torch.int32),
                            torch.zeros(num_tokens, dtype=torch.int64),
                            md,
                            3,
                        )
                    tb_chunk.finish(torch.full((num_reqs, 4), -1, dtype=torch.int32), 3)

        assert calls
        assert calls <= precompiled

    @pytest.mark.parametrize("use_top_tokens", [False, True])
    def test_torch_compile_fullgraph_dynamic_false_zero_recompilations(
        self, monkeypatch: pytest.MonkeyPatch, use_top_tokens: bool
    ) -> None:
        """Compile all kernels with ``fullgraph=True, dynamic=False`` and a
        graph-counting backend. After ``precompile()``, 20 heterogeneous serving
        steps (varying active request counts, reorderings, budgets, preemptions,
        plain decode, and speculative verify) must trigger **0** new compiles."""
        graph_compiles: list[str] = []

        def make_counting_backend(label: str):
            def backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
                _ = example_inputs
                graph_compiles.append(label)
                return gm.forward

            return backend

        with _dynamo_runner_limits():
            compiled_seed = torch.compile(
                thinking_budget._seed_states_impl,
                backend=make_counting_backend("seed"),
                fullgraph=True,
                dynamic=False,
            )
            compiled_plain = torch.compile(
                thinking_budget._override_and_advance_plain_impl,
                backend=make_counting_backend("plain"),
                fullgraph=True,
                dynamic=False,
            )
            compiled_prep_logits = torch.compile(
                thinking_budget._prepare_spec_verify_logits_impl,
                backend=make_counting_backend("prep_logits"),
                fullgraph=True,
                dynamic=False,
            )
            compiled_prep_tokens = torch.compile(
                thinking_budget._prepare_spec_verify_tokens_impl,
                backend=make_counting_backend("prep_tokens"),
                fullgraph=True,
                dynamic=False,
            )
            compiled_adv = torch.compile(
                thinking_budget._advance_spec_states_impl,
                backend=make_counting_backend("adv"),
                fullgraph=True,
                dynamic=False,
            )

            monkeypatch.setattr(thinking_budget, "_seed_states_impl", compiled_seed)
            monkeypatch.setattr(
                thinking_budget, "_override_and_advance_plain_impl", compiled_plain
            )
            monkeypatch.setattr(
                thinking_budget,
                "_prepare_spec_verify_logits_impl",
                compiled_prep_logits,
            )
            monkeypatch.setattr(
                thinking_budget,
                "_prepare_spec_verify_tokens_impl",
                compiled_prep_tokens,
            )
            monkeypatch.setattr(
                thinking_budget, "_advance_spec_states_impl", compiled_adv
            )

            cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
            tracker = ThinkingBudgetTracker(cfg, max_num_reqs=8)
            assert tracker.markers is not None

            num_reqs_paddings = [4, 8]
            verify_logits_paddings = [8, 16, 32]

            tracker.precompile(
                device=torch.device("cpu"),
                num_reqs_paddings=num_reqs_paddings,
                verify_logits_paddings=verify_logits_paddings,
                max_draft_tokens=3,
                vocab_size=64,
                logits_dtype=torch.float32,
                sync_fn=lambda _x: None,
                precompile_top_tokens=use_top_tokens,
            )
            warmup_count = len(graph_compiles)
            expected_warmup = (
                1
                + len(num_reqs_paddings)
                + len(verify_logits_paddings) * (2 if use_top_tokens else 1)
                + len(num_reqs_paddings)
            )
            assert warmup_count == expected_warmup

            # Run 20 varied serving steps across varying active request counts (1..8),
            # budgets (0, 1, 3, None), batch permutations, preemptions, plain + spec.
            rng = np.random.default_rng(99)
            requests: dict[str, CachedRequestState] = {}
            for i in range(8):
                rid = f"req_{i}"
                b = [0, 1, 3, None][i % 4]
                sp = SamplingParams(thinking_token_budget=b)
                prompt = [1, *GEMMA4_START] if i % 2 == 0 else [1, 2]
                requests[rid] = _request_state(rid, prompt, sp)
                tracker.add(rid, sp, prompt)

            all_ids = list(requests)
            for step in range(20):
                n_active = int(rng.integers(1, 9))
                batch_ids = list(rng.choice(all_ids, size=n_active, replace=False))
                padded_reqs = 4 if n_active <= 4 else 8

                if step == 7:
                    tracker.on_preempted(["req_0"])
                elif step == 10:
                    tracker.on_resumed(["req_0"], requests)

                st_table = tracker.prepare_step(
                    torch.device("cpu"), batch_ids, requests
                )
                if st_table is None:
                    continue

                if step % 2 == 0:
                    slots = tracker.build_chunk_slot_ids(
                        torch.device("cpu"), batch_ids, 0, n_active, padded_reqs
                    )
                    if slots is None:
                        continue
                    sel = torch.randint(0, 64, (padded_reqs, 1), dtype=torch.int64)
                    _, st_table = override_and_advance_plain(
                        st_table, slots, sel, tracker.markers
                    )
                    tracker.commit_state_table(st_table)
                else:
                    spec_slots = tracker.build_chunk_slot_ids(
                        torch.device("cpu"),
                        batch_ids,
                        0,
                        n_active,
                        tracker.max_num_reqs,
                    )
                    if spec_slots is None:
                        continue
                    padded_toks = 16 if padded_reqs == 4 else 32
                    seg = torch.full((padded_toks,), n_active, dtype=torch.int64)
                    grp = torch.zeros(padded_toks, dtype=torch.int32)
                    drafts = torch.randint(0, 64, (padded_toks,), dtype=torch.int32)
                    prepare_spec_verify_logits(
                        st_table,
                        spec_slots,
                        drafts,
                        torch.randn(padded_toks, 64),
                        seg,
                        grp,
                        3,
                        tracker.markers,
                    )
                    if use_top_tokens:
                        prepare_spec_verify_tokens(
                            st_table,
                            spec_slots,
                            drafts,
                            torch.randint(0, 64, (padded_toks,), dtype=torch.int64),
                            seg,
                            grp,
                            3,
                            tracker.markers,
                        )
                    _, st_table = advance_spec_states(
                        st_table,
                        spec_slots,
                        torch.randint(-1, 64, (padded_reqs, 4), dtype=torch.int32),
                        3,
                        tracker.markers,
                    )
                    tracker.commit_state_table(st_table)

            assert len(graph_compiles) == warmup_count

    def test_h2d_uploads_are_synchronous_and_slot_cache_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orig_to = torch.Tensor.to
        non_blocking_flags: list[bool] = []
        to_calls = 0

        def tracking_to(self_tensor: torch.Tensor, *args: object, **kwargs: object):
            nonlocal to_calls
            to_calls += 1
            nb = bool(kwargs.get("non_blocking", False))
            for arg in args:
                if isinstance(arg, bool):
                    nb = arg
            non_blocking_flags.append(nb)
            return orig_to(self_tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", tracking_to)

        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        sp_a = SamplingParams(thinking_token_budget=2)
        sp_b = SamplingParams(thinking_token_budget=4)
        reqs = {
            "a": _request_state("a", [1, *GEMMA4_START], sp_a),
            "b": _request_state("b", [1, *GEMMA4_START], sp_b),
        }
        tracker.add("a", sp_a, [1, *GEMMA4_START])
        tracker.add("b", sp_b, [1, *GEMMA4_START])

        dev = torch.device("cpu")
        st_table = tracker.prepare_step(dev, ["a", "b"], reqs)
        assert st_table is not None

        calls_before_slots = to_calls
        slots_1 = tracker.build_chunk_slot_ids(dev, ["a", "b"], 0, 2, 4)
        calls_after_first = to_calls
        slots_2 = tracker.build_chunk_slot_ids(dev, ["a", "b"], 0, 2, 4)
        calls_after_second = to_calls

        assert slots_1 is not None and slots_2 is slots_1
        assert calls_after_first == calls_before_slots + 1
        assert calls_after_second == calls_after_first

        # Batch reorder changes slot layout -> cache updates cleanly.
        slots_reordered = tracker.build_chunk_slot_ids(dev, ["b", "a"], 0, 2, 4)
        assert slots_reordered is not None
        assert not torch.equal(slots_reordered, slots_1)

        # Partial prefill masks in-flight chunked prefill rows (set and callable).
        for disc in ({"b"}, lambda rid: rid == "b"):
            slots_partial = tracker.build_chunk_slot_ids(
                dev, ["b", "a"], 0, 2, 4, discard_req_ids=disc
            )
            assert slots_partial is not None
            assert int(slots_partial[0].item()) == 0
            assert int(slots_partial[1].item()) > 0

        # Bounding check on _slot_upload_cache & _chunk_fast_cache across >32 keys.
        padded_batch = ["a"] + [None] * 64
        for start in range(40):
            padded_batch[start] = "a"
            tracker.build_chunk_slot_ids(dev, padded_batch, start, start + 1, 4)
        assert len(tracker._slot_upload_cache) <= thinking_budget._MAX_SLOT_UPLOAD_CACHE
        assert len(tracker._chunk_fast_cache) <= thinking_budget._MAX_SLOT_UPLOAD_CACHE
        assert not any(non_blocking_flags)

        # Unindexed vs indexed device equivalence (e.g. tpu vs tpu:0).
        assert thinking_budget._same_target_device(
            torch.device("cuda", 0), torch.device("cuda")
        )
        assert thinking_budget._same_target_device(
            torch.device("cuda", 0), torch.device("cuda", 0)
        )
        assert not thinking_budget._same_target_device(
            torch.device("cuda", 0), torch.device("cuda", 1)
        )
        assert not thinking_budget._same_target_device(
            torch.device("cpu"), torch.device("cuda")
        )

    def test_precompile_graph_count_budget_plain_ngram_and_mtp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod

        monkeypatch.setattr(tpu_runner_mod, "synchronize_tensors", lambda _out: None)
        monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)

        def count_graphs(
            *,
            max_reqs: int,
            req_pads: list[int],
            tok_pads: list[int],
            max_verify_rows: int,
            k: int | None,
            mtp_prefill_only: bool = False,
            top_tokens: bool = False,
        ) -> int:
            cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
            runner = SimpleNamespace(
                _thinking_budget=ThinkingBudgetTracker(cfg, max_num_reqs=max_reqs),
                speculative_config=(
                    None if k is None else SimpleNamespace(num_speculative_tokens=k)
                ),
                _mtp_prefill_only=mtp_prefill_only,
                _greedy_head_owner=object() if top_tokens else None,
                num_reqs_paddings=req_pads,
                num_tokens_paddings=tok_pads,
                max_num_reqs=max_reqs,
                vocab_size=16,
                _hidden_states_dtype=torch.bfloat16,
                device=torch.device("cpu"),
                structured_decoding_manager=SimpleNamespace(
                    target_grammar_bitmask_cpu=torch.zeros(
                        (max_verify_rows, 1), dtype=torch.int32
                    )
                ),
                _precompile_timed=lambda _name: MagicMock(
                    __enter__=lambda *_a: None, __exit__=lambda *_a: False
                ),
            )
            runner._verify_logits_paddings = (
                TPUModelRunner._verify_logits_paddings.__get__(runner)
            )
            n = 0
            for fn_name in (
                "seed_states",
                "override_and_advance_plain",
                "prepare_spec_verify_logits",
                "prepare_spec_verify_tokens",
                "advance_spec_states",
            ):
                orig = getattr(thinking_budget, fn_name)

                def wrap(*args, _orig=orig):
                    nonlocal n
                    n += 1
                    return _orig(*args)

                monkeypatch.setattr(thinking_budget, fn_name, wrap)
            TPUModelRunner._precompile_thinking_budget(runner)  # type: ignore[arg-type]
            return n

        qwen3_reqs = [8, 16, 32, 64, 128]
        qwen3_toks = [16, 32, 64, 128, 256, 512, 1024, 2048]
        # Plain decode (max_num_seqs=128): 1 seed + 5 plain = 6 graphs.
        assert (
            count_graphs(
                max_reqs=128,
                req_pads=qwen3_reqs,
                tok_pads=qwen3_toks,
                max_verify_rows=512,
                k=None,
            )
            == 6
        )
        # Spec decode ngram K=3 (max_num_seqs=128): 1 + 5 + 6 + 5 = 17 graphs.
        assert (
            count_graphs(
                max_reqs=128,
                req_pads=qwen3_reqs,
                tok_pads=qwen3_toks,
                max_verify_rows=512,
                k=3,
            )
            == 17
        )
        # With TPU_GREEDY_TOP_TOKENS=1: adds 6 prepare_spec_verify_tokens graphs (23).
        assert (
            count_graphs(
                max_reqs=128,
                req_pads=qwen3_reqs,
                tok_pads=qwen3_toks,
                max_verify_rows=512,
                k=3,
                top_tokens=True,
            )
            == 23
        )
        # Spec decode MTP K=2 (max_num_seqs=32): 1 + 3 + 4 + 3 = 11 graphs.
        assert (
            count_graphs(
                max_reqs=32,
                req_pads=[8, 16, 32],
                tok_pads=qwen3_toks,
                max_verify_rows=128,
                k=2,
            )
            == 11
        )
        # MTP prefill-only mode skips all spec-verify kernels: 1 + 3 = 4 graphs.
        assert (
            count_graphs(
                max_reqs=32,
                req_pads=[8, 16, 32],
                tok_pads=qwen3_toks,
                max_verify_rows=128,
                k=2,
                mtp_prefill_only=True,
            )
            == 4
        )

    def test_precompile_sweep_rotation_and_zero_collectives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.runner.precompile_sweep_checks import ops_outside_aten
        from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod

        monkeypatch.setattr(tpu_runner_mod, "synchronize_tensors", lambda _out: None)
        paddings = [8, 16, 32, 64, 128, 256, 512, 1024]

        def run_sweep(parallel_flag: bool, rank: int) -> list[int]:
            monkeypatch.setenv("LOCAL_RANK", str(rank))
            if parallel_flag:
                monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
            else:
                monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
            seen_plain: list[int] = []
            cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
            runner = SimpleNamespace(
                _thinking_budget=ThinkingBudgetTracker(cfg, max_num_reqs=paddings[-1]),
                speculative_config=None,
                _mtp_prefill_only=False,
                _greedy_head_owner=None,
                num_reqs_paddings=paddings,
                num_tokens_paddings=paddings,
                max_num_reqs=paddings[-1],
                vocab_size=16,
                _hidden_states_dtype=torch.float32,
                device=torch.device("cpu"),
                _precompile_timed=lambda _name: MagicMock(
                    __enter__=lambda *_a: None, __exit__=lambda *_a: False
                ),
            )
            runner._verify_logits_paddings = (
                TPUModelRunner._verify_logits_paddings.__get__(runner)
            )
            orig_plain = thinking_budget.override_and_advance_plain
            with patch.object(
                thinking_budget,
                "override_and_advance_plain",
                side_effect=lambda st, sl, sel, m: (
                    seen_plain.append(int(sl.shape[0])) or orig_plain(st, sl, sel, m)
                ),
            ):
                TPUModelRunner._precompile_thinking_budget(runner)  # type: ignore[arg-type]
            return seen_plain

        assert run_sweep(False, rank=0) == paddings
        assert run_sweep(False, rank=5) == paddings
        for rank in (0, 3, 7, 11):
            start = (rank % 8) * len(paddings) // 8
            assert run_sweep(True, rank=rank) == paddings[start:] + paddings[:start]

        # Every compiled thinking_budget kernel must contain zero collectives so
        # per-rank rotated sweeps cannot deadlock.
        traced_graphs: list[torch.fx.GraphModule] = []

        def record_backend(
            gm: torch.fx.GraphModule, _example_inputs: list[torch.Tensor]
        ):
            traced_graphs.append(gm)
            return gm.forward

        with _dynamo_runner_limits():
            st = torch.zeros((9, thinking_budget.NUM_STATE_COLS), dtype=torch.int32)
            slots = torch.zeros(4, dtype=torch.int32)
            for start_ids, end_ids in (
                (QWEN3_START, QWEN3_END),
                (GEMMA4_START, GEMMA4_END),
            ):
                m = ThinkingBudgetMarkers.from_reasoning_config(
                    _reasoning_config(list(start_ids), list(end_ids))
                )
                assert m is not None
                torch.compile(
                    thinking_budget._seed_states_impl,
                    backend=record_backend,
                    fullgraph=True,
                    dynamic=False,
                )(st, slots, torch.zeros((4, 7), dtype=torch.int32))
                torch.compile(
                    thinking_budget._override_and_advance_plain_impl,
                    backend=record_backend,
                    fullgraph=True,
                    dynamic=False,
                )(
                    st,
                    slots,
                    torch.zeros((4, 1), dtype=torch.int64),
                    *m.kernel_args(),
                )
                torch.compile(
                    thinking_budget._prepare_spec_verify_logits_impl,
                    backend=record_backend,
                    fullgraph=True,
                    dynamic=False,
                )(
                    st,
                    slots,
                    torch.zeros(8, dtype=torch.int32),
                    torch.zeros((8, 32), dtype=torch.bfloat16),
                    torch.full((8,), 4, dtype=torch.int64),
                    torch.zeros(8, dtype=torch.int32),
                    3,
                    *m.kernel_args(),
                )
                torch.compile(
                    thinking_budget._prepare_spec_verify_tokens_impl,
                    backend=record_backend,
                    fullgraph=True,
                    dynamic=False,
                )(
                    st,
                    slots,
                    torch.zeros(8, dtype=torch.int32),
                    torch.zeros(8, dtype=torch.int64),
                    torch.full((8,), 4, dtype=torch.int64),
                    torch.zeros(8, dtype=torch.int32),
                    3,
                    *m.kernel_args(),
                )
                torch.compile(
                    thinking_budget._advance_spec_states_impl,
                    backend=record_backend,
                    fullgraph=True,
                    dynamic=False,
                )(
                    st,
                    slots,
                    torch.full((4, 4), -1, dtype=torch.int32),
                    3,
                    *m.kernel_args(),
                )

        assert traced_graphs
        assert [op for gm in traced_graphs for op in ops_outside_aten(gm)] == []

    @pytest.mark.tpu_test
    def test_tpu_compiled_kernels_match_cpu_reference(self) -> None:
        import torch_tpu  # noqa: F401

        with _dynamo_runner_limits():
            tpu = torch.device("tpu")
            cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
            markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
            assert markers is not None

            cpu_tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
            tpu_tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
            reqs: dict[str, CachedRequestState] = {}
            prompts = {
                "a": [1, *GEMMA4_START],
                "b": [1, *GEMMA4_START],
                "c": [1, 2],
                "d": [1, *GEMMA4_START],
            }
            for rid, b in (("a", 0), ("b", 2), ("c", 1), ("d", None)):
                sp = SamplingParams(thinking_token_budget=b)
                reqs[rid] = _request_state(rid, prompts[rid], sp)
                cpu_tracker.add(rid, sp, prompts[rid])
                tpu_tracker.add(rid, sp, prompts[rid])

            batch_ids = ["a", "b", "c", "d"]
            cpu_table = cpu_tracker.prepare_step(torch.device("cpu"), batch_ids, reqs)
            tpu_table = tpu_tracker.prepare_step(tpu, batch_ids, reqs)
            assert cpu_table is not None and tpu_table is not None
            synchronize_tensors(tpu_table)
            assert torch.equal(tpu_table.cpu(), cpu_table)

            cpu_slots = cpu_tracker.build_chunk_slot_ids(
                torch.device("cpu"), batch_ids, 0, 4, 4
            )
            tpu_slots = tpu_tracker.build_chunk_slot_ids(tpu, batch_ids, 0, 4, 4)
            assert cpu_slots is not None and tpu_slots is not None

            # 1. Plain decode kernel parity across multi-token start/end transitions.
            for step_toks in ([40, 41, 105, 50], [42, 43, 2364, 51], [44, 45, 107, 52]):
                sel_cpu = torch.tensor([[t] for t in step_toks], dtype=torch.int64)
                out_cpu, cpu_table = override_and_advance_plain(
                    cpu_table, cpu_slots, sel_cpu, markers
                )
                cpu_tracker.commit_state_table(cpu_table)
                out_tpu, tpu_table = override_and_advance_plain(
                    tpu_table, tpu_slots, sel_cpu.to(tpu), markers
                )
                tpu_tracker.commit_state_table(tpu_table)
                synchronize_tensors([out_tpu, tpu_table])
                assert torch.equal(out_tpu.cpu(), out_cpu)
                assert torch.equal(tpu_table.cpu(), cpu_table)

            # 2. Re-seed fresh budgets and verify all speculative-decoding kernels:
            #    prepare_spec_verify_logits, prepare_spec_verify_tokens, and
            #    bonus-overriding advance_spec_states.
            for rid, b in (("a", 0), ("b", 1), ("c", 2), ("d", None)):
                sp = SamplingParams(thinking_token_budget=b)
                reqs[rid] = _request_state(rid, [1, *GEMMA4_START], sp)
                cpu_tracker.add(rid, sp, [1, *GEMMA4_START])
                tpu_tracker.add(rid, sp, [1, *GEMMA4_START])

            cpu_table = cpu_tracker.prepare_step(torch.device("cpu"), batch_ids, reqs)
            tpu_table = tpu_tracker.prepare_step(tpu, batch_ids, reqs)
            assert cpu_table is not None and tpu_table is not None
            cpu_slots = cpu_tracker.build_chunk_slot_ids(
                torch.device("cpu"), batch_ids, 0, 4, 4
            )
            tpu_slots = tpu_tracker.build_chunk_slot_ids(tpu, batch_ids, 0, 4, 4)
            assert cpu_slots is not None and tpu_slots is not None

            # Padded verify bucket (num_tokens=16, num_reqs=4, max_draft_tokens=3):
            # req a (len 2, budget=0): [106, 99] -> 106 accepted, 99 forced to 107
            # req b (len 3, budget=1): [50, 51, 52] -> 1 tok allowed, force 106, 107
            # req c (len 2, budget=2): [60, 61] -> 2 drafts ok, bonus forced to 106!
            # req d (len 1, budget=None): [70] -> untracked
            draft_tokens_cpu = torch.full((16,), 0, dtype=torch.int32)
            segment_ids_cpu = torch.full((16,), 4, dtype=torch.int64)
            group_indices_cpu = torch.full((16,), 0, dtype=torch.int32)
            per_req_drafts = [[106, 99], [50, 51, 52], [60, 61], [70]]
            cursor = 0
            for req_idx, r_drafts in enumerate(per_req_drafts):
                for g_idx, d_tok in enumerate(r_drafts):
                    draft_tokens_cpu[cursor] = d_tok
                    segment_ids_cpu[cursor] = req_idx
                    group_indices_cpu[cursor] = g_idx
                    cursor += 1
                cursor += 1  # bonus slot row in verify layout

            raw_logits_cpu = torch.randn((16, 128), dtype=torch.bfloat16)
            flogits_cpu, eff_cpu = prepare_spec_verify_logits(
                cpu_table,
                cpu_slots,
                draft_tokens_cpu,
                raw_logits_cpu,
                segment_ids_cpu,
                group_indices_cpu,
                3,
                markers,
            )
            flogits_tpu, eff_tpu = prepare_spec_verify_logits(
                tpu_table,
                tpu_slots,
                draft_tokens_cpu.to(tpu),
                raw_logits_cpu.to(tpu),
                segment_ids_cpu.to(tpu),
                group_indices_cpu.to(tpu),
                3,
                markers,
            )
            synchronize_tensors([flogits_tpu, eff_tpu])
            assert torch.equal(flogits_tpu.cpu(), flogits_cpu)
            assert torch.equal(eff_tpu.cpu(), eff_cpu)

            raw_target_ids_cpu = torch.arange(16, dtype=torch.int64) + 10
            ftarget_cpu, eff_tok_cpu = prepare_spec_verify_tokens(
                cpu_table,
                cpu_slots,
                draft_tokens_cpu,
                raw_target_ids_cpu,
                segment_ids_cpu,
                group_indices_cpu,
                3,
                markers,
            )
            ftarget_tpu, eff_tok_tpu = prepare_spec_verify_tokens(
                tpu_table,
                tpu_slots,
                draft_tokens_cpu.to(tpu),
                raw_target_ids_cpu.to(tpu),
                segment_ids_cpu.to(tpu),
                group_indices_cpu.to(tpu),
                3,
                markers,
            )
            synchronize_tensors([ftarget_tpu, eff_tok_tpu])
            assert torch.equal(ftarget_tpu.cpu(), ftarget_cpu)
            assert torch.equal(eff_tok_tpu.cpu(), eff_tok_cpu)

            # Row 2 (req c) accepted [60, 61] and sampled raw bonus 82 ->
            # advance_spec_states overrides 82 to GEMMA4_END[0] (106)!
            next_tokens_cpu = torch.tensor(
                [
                    [106, 107, -1, -1],
                    [50, 106, -1, -1],
                    [60, 61, 82, -1],
                    [70, 83, -1, -1],
                ],
                dtype=torch.int32,
            )
            out_adv_cpu, cpu_table = advance_spec_states(
                cpu_table, cpu_slots, next_tokens_cpu, 3, markers
            )
            cpu_tracker.commit_state_table(cpu_table)
            out_adv_tpu, tpu_table = advance_spec_states(
                tpu_table, tpu_slots, next_tokens_cpu.to(tpu), 3, markers
            )
            tpu_tracker.commit_state_table(tpu_table)
            synchronize_tensors([out_adv_tpu, tpu_table])
            assert out_adv_cpu[2].tolist() == [60, 61, GEMMA4_END[0], -1]
            assert torch.equal(out_adv_tpu.cpu(), out_adv_cpu)
            assert torch.equal(tpu_table.cpu(), cpu_table)

    @pytest.mark.tpu_test
    def test_tpu_precompile_zero_recompilations_plain_and_spec_verify(self) -> None:
        import torch_tpu  # noqa: F401

        with _dynamo_runner_limits():
            tpu = torch.device("tpu")
            cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
            cpu_tracker = ThinkingBudgetTracker(cfg, max_num_reqs=8)
            tpu_tracker = ThinkingBudgetTracker(cfg, max_num_reqs=8)
            assert cpu_tracker.markers is not None and tpu_tracker.markers is not None

            num_reqs_paddings = [4, 8]
            verify_logits_paddings = [8, 16, 32]

            tpu_tracker.precompile(
                device=tpu,
                num_reqs_paddings=num_reqs_paddings,
                verify_logits_paddings=verify_logits_paddings,
                max_draft_tokens=3,
                vocab_size=128,
                logits_dtype=torch.bfloat16,
                sync_fn=synchronize_tensors,
                precompile_top_tokens=True,
            )

            # Pre-allocate steady-state step input shapes before sampling the XLA
            # cache count so test-harness random tensor creation isn't counted.
            warm_sel_4 = torch.zeros((4, 1), dtype=torch.int64).to(tpu)
            warm_sel_8 = torch.zeros((8, 1), dtype=torch.int64).to(tpu)
            warm_logits_16 = torch.zeros((16, 128), dtype=torch.bfloat16).to(tpu)
            warm_logits_32 = torch.zeros((32, 128), dtype=torch.bfloat16).to(tpu)
            synchronize_tensors(
                [warm_sel_4, warm_sel_8, warm_logits_16, warm_logits_32]
            )
            warm_logits_16_cpu = warm_logits_16.cpu()
            warm_logits_32_cpu = warm_logits_32.cpu()

            dynamo_graphs_after_precompile = torch._dynamo.utils.counters["stats"][
                "unique_graphs"
            ]
            graphs_after_precompile = len(torch.tpu._get_cache_stats().per_entry_stats)
            assert dynamo_graphs_after_precompile > 0
            assert graphs_after_precompile > 0

            rng = np.random.default_rng(2026)
            requests: dict[str, CachedRequestState] = {}
            for i in range(8):
                rid = f"req_{i}"
                b = [0, 1, 3, None][i % 4]
                sp = SamplingParams(thinking_token_budget=b)
                prompt = [1, *GEMMA4_START] if i % 2 == 0 else [1, 2]
                requests[rid] = _request_state(rid, prompt, sp)
                cpu_tracker.add(rid, sp, prompt)
                tpu_tracker.add(rid, sp, prompt)

            all_ids = list(requests)
            for step in range(16):
                n_active = int(rng.integers(1, 9))
                batch_ids = [
                    str(x) for x in rng.choice(all_ids, size=n_active, replace=False)
                ]
                padded_reqs = 4 if n_active <= 4 else 8

                if step == 5:
                    cpu_tracker.on_preempted(["req_0"])
                    tpu_tracker.on_preempted(["req_0"])
                elif step == 9:
                    cpu_tracker.on_resumed(["req_0"], requests)
                    tpu_tracker.on_resumed(["req_0"], requests)

                cpu_table = cpu_tracker.prepare_step(
                    torch.device("cpu"), batch_ids, requests
                )
                tpu_table = tpu_tracker.prepare_step(tpu, batch_ids, requests)
                if tpu_table is None:
                    assert cpu_table is None
                    continue
                assert cpu_table is not None
                synchronize_tensors(tpu_table)

                if step % 2 == 0:
                    cpu_slots = cpu_tracker.build_chunk_slot_ids(
                        torch.device("cpu"), batch_ids, 0, n_active, padded_reqs
                    )
                    tpu_slots = tpu_tracker.build_chunk_slot_ids(
                        tpu, batch_ids, 0, n_active, padded_reqs
                    )
                    if tpu_slots is None:
                        assert cpu_slots is None
                        continue
                    assert cpu_slots is not None
                    sel_np = rng.integers(0, 128, size=(padded_reqs, 1), dtype=np.int64)
                    sel_cpu = torch.from_numpy(sel_np)
                    out_cpu, cpu_table = override_and_advance_plain(
                        cpu_table, cpu_slots, sel_cpu, cpu_tracker.markers
                    )
                    cpu_tracker.commit_state_table(cpu_table)
                    out_tpu, tpu_table = override_and_advance_plain(
                        tpu_table, tpu_slots, sel_cpu.to(tpu), tpu_tracker.markers
                    )
                    tpu_tracker.commit_state_table(tpu_table)
                    synchronize_tensors([out_tpu, tpu_table])
                    assert torch.equal(out_tpu.cpu(), out_cpu)
                else:
                    cpu_slots = cpu_tracker.build_chunk_slot_ids(
                        torch.device("cpu"),
                        batch_ids,
                        0,
                        n_active,
                        cpu_tracker.max_num_reqs,
                    )
                    tpu_slots = tpu_tracker.build_chunk_slot_ids(
                        tpu, batch_ids, 0, n_active, tpu_tracker.max_num_reqs
                    )
                    if tpu_slots is None:
                        assert cpu_slots is None
                        continue
                    assert cpu_slots is not None
                    padded_toks = 16 if padded_reqs == 4 else 32
                    d_lens_np = rng.integers(0, 4, size=n_active, dtype=np.int32)
                    draft_np = rng.integers(0, 128, size=padded_toks, dtype=np.int32)
                    seg_np = np.full(padded_toks, n_active, dtype=np.int64)
                    grp_np = np.zeros(padded_toks, dtype=np.int32)
                    cur = 0
                    for r_i in range(n_active):
                        for g_i in range(int(d_lens_np[r_i])):
                            if cur < padded_toks:
                                seg_np[cur] = r_i
                                grp_np[cur] = g_i
                                cur += 1
                        cur += 1

                    draft_cpu = torch.from_numpy(draft_np)
                    seg_cpu = torch.from_numpy(seg_np)
                    grp_cpu = torch.from_numpy(grp_np)
                    logits_tpu = warm_logits_16 if padded_toks == 16 else warm_logits_32
                    logits_cpu = (
                        warm_logits_16_cpu if padded_toks == 16 else warm_logits_32_cpu
                    )
                    flogits_cpu, eff_cpu = prepare_spec_verify_logits(
                        cpu_table,
                        cpu_slots,
                        draft_cpu,
                        logits_cpu,
                        seg_cpu,
                        grp_cpu,
                        3,
                        cpu_tracker.markers,
                    )
                    flogits_tpu, eff_tpu = prepare_spec_verify_logits(
                        tpu_table,
                        tpu_slots,
                        draft_cpu.to(tpu),
                        logits_tpu,
                        seg_cpu.to(tpu),
                        grp_cpu.to(tpu),
                        3,
                        tpu_tracker.markers,
                    )
                    t_ids_cpu = torch.from_numpy(
                        rng.integers(0, 128, size=padded_toks, dtype=np.int64)
                    )
                    ft_ids_cpu, eff_tok_cpu = prepare_spec_verify_tokens(
                        cpu_table,
                        cpu_slots,
                        draft_cpu,
                        t_ids_cpu,
                        seg_cpu,
                        grp_cpu,
                        3,
                        cpu_tracker.markers,
                    )
                    ft_ids_tpu, eff_tok_tpu = prepare_spec_verify_tokens(
                        tpu_table,
                        tpu_slots,
                        draft_cpu.to(tpu),
                        t_ids_cpu.to(tpu),
                        seg_cpu.to(tpu),
                        grp_cpu.to(tpu),
                        3,
                        tpu_tracker.markers,
                    )
                    next_toks_cpu = torch.from_numpy(
                        rng.integers(-1, 128, size=(padded_reqs, 4), dtype=np.int32)
                    )
                    out_adv_cpu, cpu_table = advance_spec_states(
                        cpu_table, cpu_slots, next_toks_cpu, 3, cpu_tracker.markers
                    )
                    cpu_tracker.commit_state_table(cpu_table)
                    out_adv_tpu, tpu_table = advance_spec_states(
                        tpu_table,
                        tpu_slots,
                        next_toks_cpu.to(tpu),
                        3,
                        tpu_tracker.markers,
                    )
                    tpu_tracker.commit_state_table(tpu_table)
                    synchronize_tensors(
                        [
                            eff_tpu,
                            flogits_tpu,
                            ft_ids_tpu,
                            eff_tok_tpu,
                            out_adv_tpu,
                            tpu_table,
                        ]
                    )
                    assert torch.equal(eff_tpu.cpu(), eff_cpu)
                    assert torch.equal(flogits_tpu.cpu(), flogits_cpu)
                    assert torch.equal(ft_ids_tpu.cpu(), ft_ids_cpu)
                    assert torch.equal(eff_tok_tpu.cpu(), eff_tok_cpu)
                    assert torch.equal(out_adv_tpu.cpu(), out_adv_cpu)

                assert torch.equal(tpu_table.cpu(), cpu_table)

            assert (
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
                == dynamo_graphs_after_precompile
            )
            graphs_after_sweep = len(torch.tpu._get_cache_stats().per_entry_stats)
            assert graphs_after_sweep == graphs_after_precompile
