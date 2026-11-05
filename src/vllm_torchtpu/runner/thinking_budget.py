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
"""Per-request ``SamplingParams.thinking_token_budget`` tracking and forcing.

``TPUModelRunner`` runs its own compiled sampling and speculative rejection
pipeline, reorders batch rows mid-step in ``_reorder_batch_for_rpa``, and under
``--async-scheduling`` leaves the previous step's sampled tokens unsettled on
device when ``sample_tokens()`` begins.

``ThinkingBudgetTracker`` keys budgets by ``req_id`` and maintains a
fixed-shape ``[capacity, 7]`` ``int32`` Knuth-Morris-Pratt (KMP) state table on
device so plain decode, ``TPU_GREEDY_TOP_TOKENS``, async scheduling, and
speculative verification advance without host stalls or XLA recompilations.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config.reasoning import ReasoningConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)

INT32_MAX: Final[int] = 2_147_483_647
_AMAX_SENTINEL: Final[int] = -2_147_483_647
NO_FORCED_TOKEN: Final[int] = -1
_MAX_SLOT_UPLOAD_CACHE: Final[int] = 32

# Per-slot int32 columns in ``state_table`` ([capacity, NUM_STATE_COLS]).
COL_BUDGET: Final[int] = 0
COL_IN_THINK: Final[int] = 1
COL_THINK_COUNT: Final[int] = 2
COL_START_MATCH: Final[int] = 3
COL_NAT_END_MATCH: Final[int] = 4
COL_END_MATCH: Final[int] = 5
COL_FORCING_END: Final[int] = 6
NUM_STATE_COLS: Final[int] = 7

StateTuple = tuple[int, int, int, int, int, int, int]
UNTRACKED_STATE: Final[StateTuple] = (-1, 0, 0, 0, 0, 0, 0)


def compute_kmp_failure(pattern: Sequence[int]) -> tuple[int, ...]:
    """Return KMP fallback table ``fail[0 .. len(pattern)]`` for ``pattern``."""
    m = len(pattern)
    if m == 0:
        return ()
    lps = [0] * m
    length = 0
    idx = 1
    while idx < m:
        if pattern[idx] == pattern[length]:
            length += 1
            lps[idx] = length
            idx += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[idx] = 0
            idx += 1
    return (0, *lps)


@dataclass(frozen=True)
class ThinkingBudgetMarkers:
    """Tokenized start/end markers and KMP fallback tables for reasoning."""

    start_ids: tuple[int, ...]
    start_fail: tuple[int, ...]
    end_ids: tuple[int, ...]
    end_fail: tuple[int, ...]
    natural_end_ids: tuple[int, ...]
    natural_end_fail: tuple[int, ...]

    @classmethod
    def from_reasoning_config(
        cls, reasoning_config: "ReasoningConfig | None"
    ) -> "ThinkingBudgetMarkers | None":
        if reasoning_config is None or not reasoning_config.enabled:
            return None
        start_raw = reasoning_config.reasoning_start_token_ids
        end_raw = reasoning_config.reasoning_end_token_ids
        if not start_raw or not end_raw:
            return None
        nat_raw = reasoning_config.natural_reasoning_end_token_ids or end_raw
        start_ids = tuple(int(t) for t in start_raw)
        end_ids = tuple(int(t) for t in end_raw)
        natural_end_ids = tuple(int(t) for t in nat_raw)
        return cls(
            start_ids=start_ids,
            start_fail=compute_kmp_failure(start_ids),
            end_ids=end_ids,
            end_fail=compute_kmp_failure(end_ids),
            natural_end_ids=natural_end_ids,
            natural_end_fail=compute_kmp_failure(natural_end_ids),
        )

    def kernel_args(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.start_ids,
            self.start_fail,
            self.natural_end_ids,
            self.natural_end_fail,
            self.end_ids,
            self.end_fail,
        )


def _kmp_step_py(
    pattern: tuple[int, ...], fail: tuple[int, ...], match_len: int, tok: int
) -> int:
    q = match_len
    while q > 0 and tok != pattern[q]:
        q = fail[q]
    return q + 1 if tok == pattern[q] else 0


def advance_state_py(
    state: StateTuple, tok: int, markers: ThinkingBudgetMarkers
) -> StateTuple:
    """Advance scalar 7-tuple KMP thinking-budget state on one token ``tok``."""
    (
        budget,
        in_think,
        think_count,
        start_match,
        nat_end_match,
        end_match,
        forcing_end,
    ) = state
    if budget < 0:
        return UNTRACKED_STATE
    if in_think == 0:
        next_start = _kmp_step_py(
            markers.start_ids, markers.start_fail, start_match, tok
        )
        if next_start == len(markers.start_ids):
            return (budget, 1, 0, 0, 0, 0, 0)
        return (budget, 0, 0, next_start, 0, 0, 0)

    forcing_now = forcing_end == 1 or think_count >= budget
    actively_forcing_seq = forcing_now and tok == markers.end_ids[end_match]
    next_nat = _kmp_step_py(
        markers.natural_end_ids, markers.natural_end_fail, nat_end_match, tok
    )
    next_end = _kmp_step_py(markers.end_ids, markers.end_fail, end_match, tok)
    closed = (
        next_end == len(markers.end_ids)
        if actively_forcing_seq
        else (
            next_end == len(markers.end_ids) or next_nat == len(markers.natural_end_ids)
        )
    )
    if closed:
        return (budget, 0, 0, 0, 0, 0, 0)
    if next_nat == len(markers.natural_end_ids):
        next_nat = markers.natural_end_fail[len(markers.natural_end_ids)]
    if next_end == len(markers.end_ids):
        next_end = markers.end_fail[len(markers.end_ids)]
    next_count = think_count + 1 if think_count < INT32_MAX else INT32_MAX
    return (budget, 1, next_count, 0, next_nat, next_end, int(actively_forcing_seq))


def query_forced_token_py(state: StateTuple, markers: ThinkingBudgetMarkers) -> int:
    """Return the forced token ID for ``state``, or ``NO_FORCED_TOKEN`` (-1)."""
    budget, in_think, think_count, _, _, end_match, forcing_end = state
    if budget < 0 or in_think == 0:
        return NO_FORCED_TOKEN
    if forcing_end == 1 or think_count >= budget:
        return markers.end_ids[end_match]
    return NO_FORCED_TOKEN


def compute_sequence_state(
    tokens: Sequence[int], budget: int, markers: ThinkingBudgetMarkers
) -> StateTuple:
    """Compute the 7-tuple KMP state after consuming ``tokens``."""
    state: StateTuple = (min(int(budget), INT32_MAX), 0, 0, 0, 0, 0, 0)
    for tok in tokens:
        state = advance_state_py(state, int(tok), markers)
    return state


def _kmp_step_tensor(
    pattern: tuple[int, ...],
    fail: tuple[int, ...],
    match_len: torch.Tensor,
    tok: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros_like(match_len)
    out_for_q: list[torch.Tensor] = []
    for q, pat_tok in enumerate(pattern):
        fallback = zero if q == 0 else out_for_q[fail[q]]
        out_for_q.append(torch.where(tok == pat_tok, q + 1, fallback))
    res = out_for_q[0]
    for q in range(1, len(pattern)):
        res = torch.where(match_len == q, out_for_q[q], res)
    return res


def _gather_end_token_tensor(
    end_ids: tuple[int, ...], end_match: torch.Tensor
) -> torch.Tensor:
    res = torch.full_like(end_match, end_ids[0])
    for q in range(1, len(end_ids)):
        res = torch.where(end_match == q, end_ids[q], res)
    return res


def _query_forced_token_tensor(
    st: torch.Tensor, end_ids: tuple[int, ...]
) -> torch.Tensor:
    budget = st[:, COL_BUDGET]
    should_force = (
        (budget >= 0)
        & (st[:, COL_IN_THINK] != 0)
        & ((st[:, COL_FORCING_END] != 0) | (st[:, COL_THINK_COUNT] >= budget))
    )
    return torch.where(
        should_force,
        _gather_end_token_tensor(end_ids, st[:, COL_END_MATCH]),
        torch.full_like(budget, NO_FORCED_TOKEN),
    )


def _advance_state_tensor(
    st: torch.Tensor,
    tok: torch.Tensor,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> torch.Tensor:
    budget = st[:, COL_BUDGET]
    in_think = st[:, COL_IN_THINK]
    think_count = st[:, COL_THINK_COUNT]
    zero = torch.zeros_like(budget)
    one = torch.ones_like(budget)

    next_start = _kmp_step_tensor(start_ids, start_fail, st[:, COL_START_MATCH], tok)
    entered_think = next_start == len(start_ids)
    out_not_in_think = torch.stack(
        [
            budget,
            torch.where(entered_think, one, zero),
            zero,
            torch.where(entered_think, zero, next_start),
            zero,
            zero,
            zero,
        ],
        dim=-1,
    )

    forcing_now = (st[:, COL_FORCING_END] != 0) | (think_count >= budget)
    expected_end_tok = _gather_end_token_tensor(end_ids, st[:, COL_END_MATCH])
    actively_forcing_seq = forcing_now & (tok == expected_end_tok)

    next_nat = _kmp_step_tensor(nat_ids, nat_fail, st[:, COL_NAT_END_MATCH], tok)
    next_end = _kmp_step_tensor(end_ids, end_fail, st[:, COL_END_MATCH], tok)
    end_completed = next_end == len(end_ids)
    nat_completed = next_nat == len(nat_ids)
    closed = torch.where(
        actively_forcing_seq, end_completed, end_completed | nat_completed
    )
    wrapped_nat = torch.where(
        nat_completed, torch.full_like(next_nat, nat_fail[len(nat_ids)]), next_nat
    )
    wrapped_end = torch.where(
        end_completed, torch.full_like(next_end, end_fail[len(end_ids)]), next_end
    )
    inc_count = torch.where(
        think_count < INT32_MAX,
        think_count + 1,
        torch.full_like(think_count, INT32_MAX),
    )
    out_in_think = torch.stack(
        [
            budget,
            torch.where(closed, zero, one),
            torch.where(closed, zero, inc_count),
            zero,
            torch.where(closed, zero, wrapped_nat),
            torch.where(closed, zero, wrapped_end),
            torch.where(closed, zero, torch.where(actively_forcing_seq, one, zero)),
        ],
        dim=-1,
    )

    untracked = torch.stack(
        [torch.full_like(budget, -1), zero, zero, zero, zero, zero, zero], dim=-1
    )
    active_out = torch.where(
        (in_think != 0).unsqueeze(-1), out_in_think, out_not_in_think
    )
    return torch.where((budget >= 0).unsqueeze(-1), active_out, untracked)


def _scatter_state_rows(
    state_table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    table_idx = torch.arange(
        state_table.shape[0], dtype=slot_ids.dtype, device=state_table.device
    ).unsqueeze(1)
    match = (slot_ids > 0).unsqueeze(0) & (table_idx == slot_ids.unsqueeze(0))
    cand = torch.where(match.unsqueeze(-1), rows.unsqueeze(0), _AMAX_SENTINEL).amax(
        dim=1
    )
    return torch.where(match.any(dim=1, keepdim=True), cand, state_table)


def _seed_states_impl(
    state_table: torch.Tensor, seed_slots: torch.Tensor, seed_values: torch.Tensor
) -> torch.Tensor:
    return _scatter_state_rows(state_table, seed_slots, seed_values)


def _override_and_advance_plain_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    selected_token_ids: torch.Tensor,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    forced_id = _query_forced_token_tensor(st, end_ids)
    out_tokens = torch.where(
        (active_slot & (forced_id >= 0)).unsqueeze(-1),
        forced_id.to(selected_token_ids.dtype).unsqueeze(-1),
        selected_token_ids,
    )
    next_st = _advance_state_tensor(
        st,
        out_tokens.squeeze(-1).to(torch.int32),
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return out_tokens, _scatter_state_rows(state_table, slot_ids, next_st)


def _prepare_spec_verify_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_reqs = slot_ids.shape[0]
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    no_req_tok = torch.full(
        (num_reqs,), NO_FORCED_TOKEN, dtype=torch.int32, device=state_table.device
    )

    draft_i32 = draft_token_ids.to(torch.int32)
    forced_target_ids = torch.full_like(draft_i32, NO_FORCED_TOKEN)

    if max_draft_tokens > 0:
        req_rows = torch.arange(
            num_reqs, dtype=segment_ids.dtype, device=state_table.device
        ).unsqueeze(1)
        seg_match = (
            (segment_ids.unsqueeze(0) == req_rows)
            & (group_indices >= 0).unsqueeze(0)
            & (
                group_indices.unsqueeze(0)
                < draft_lengths.to(group_indices.dtype).unsqueeze(1)
            )
        )
        for step_idx in range(max_draft_tokens):
            step_forced_req = torch.where(
                active_slot, _query_forced_token_tensor(st, end_ids), no_req_tok
            )
            pos_match = seg_match & (group_indices.unsqueeze(0) == step_idx)
            step_forced_tok = torch.where(
                pos_match, step_forced_req.unsqueeze(1), _AMAX_SENTINEL
            ).amax(dim=0)
            forced_target_ids = torch.where(
                pos_match.any(dim=0), step_forced_tok, forced_target_ids
            )
            d_tok = torch.where(
                pos_match.any(dim=1),
                torch.where(pos_match, draft_i32.unsqueeze(0), _AMAX_SENTINEL).amax(
                    dim=1
                ),
                no_req_tok,
            )
            step_active = active_slot & (step_idx < draft_lengths) & (d_tok >= 0)
            next_st = _advance_state_tensor(
                st,
                d_tok,
                start_ids,
                start_fail,
                nat_ids,
                nat_fail,
                end_ids,
                end_fail,
            )
            st = torch.where(step_active.unsqueeze(-1), next_st, st)

    forced_bonus_ids = torch.where(
        active_slot, _query_forced_token_tensor(st, end_ids), no_req_tok
    )
    effective_draft_ids = torch.where(
        (forced_target_ids >= 0) & (draft_i32 != forced_target_ids),
        torch.full_like(draft_i32, NO_FORCED_TOKEN),
        draft_i32,
    )
    return forced_target_ids, forced_bonus_ids, effective_draft_ids


def _force_logits_rows_impl(
    logits: torch.Tensor, forced_ids: torch.Tensor
) -> torch.Tensor:
    is_forced = (forced_ids >= 0).unsqueeze(-1)
    masked = torch.where(is_forced, torch.full_like(logits, float("-inf")), logits)
    safe_ids = forced_ids.clamp(min=0, max=logits.shape[-1] - 1).to(torch.int64)
    zero_src = torch.zeros(
        (logits.shape[0], 1), dtype=logits.dtype, device=logits.device
    )
    forced_val = torch.where(
        is_forced, zero_src, logits.gather(-1, safe_ids.unsqueeze(-1))
    )
    return masked.scatter(-1, safe_ids.unsqueeze(-1), forced_val)


def _force_token_ids_1d_impl(
    token_ids: torch.Tensor, forced_ids: torch.Tensor
) -> torch.Tensor:
    return torch.where(forced_ids >= 0, forced_ids.to(token_ids.dtype), token_ids)


def _advance_spec_states_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    next_tokens: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    for col_idx in range(max_draft_tokens + 1):
        tok = next_tokens[:, col_idx].to(torch.int32)
        next_st = _advance_state_tensor(
            st, tok, start_ids, start_fail, nat_ids, nat_fail, end_ids, end_fail
        )
        st = torch.where((active_slot & (tok >= 0)).unsqueeze(-1), next_st, st)
    return next_tokens + 0, _scatter_state_rows(state_table, slot_ids, st)


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_seed_states_tpu = _tpu_compile(_seed_states_impl)
_override_and_advance_plain_tpu = _tpu_compile(_override_and_advance_plain_impl)
_prepare_spec_verify_tpu = _tpu_compile(_prepare_spec_verify_impl)
_force_logits_rows_tpu = _tpu_compile(_force_logits_rows_impl)
_force_token_ids_1d_tpu = _tpu_compile(_force_token_ids_1d_impl)
_advance_spec_states_tpu = _tpu_compile(_advance_spec_states_impl)


def seed_states(
    state_table: torch.Tensor, seed_slots: torch.Tensor, seed_values: torch.Tensor
) -> torch.Tensor:
    fn = _seed_states_tpu if state_table.device.type == "tpu" else _seed_states_impl
    return fn(state_table, seed_slots, seed_values)


def override_and_advance_plain(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    selected_token_ids: torch.Tensor,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _override_and_advance_plain_tpu
        if state_table.device.type == "tpu"
        else _override_and_advance_plain_impl
    )
    return fn(state_table, slot_ids, selected_token_ids, *markers.kernel_args())


def prepare_spec_verify(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = (
        _prepare_spec_verify_tpu
        if state_table.device.type == "tpu"
        else _prepare_spec_verify_impl
    )
    return fn(
        state_table,
        slot_ids,
        draft_token_ids,
        draft_lengths,
        segment_ids,
        group_indices,
        max_draft_tokens,
        *markers.kernel_args(),
    )


def force_logits_rows(logits: torch.Tensor, forced_ids: torch.Tensor) -> torch.Tensor:
    fn = (
        _force_logits_rows_tpu
        if logits.device.type == "tpu"
        else _force_logits_rows_impl
    )
    return fn(logits, forced_ids)


def force_token_ids_1d(
    token_ids: torch.Tensor, forced_ids: torch.Tensor
) -> torch.Tensor:
    fn = (
        _force_token_ids_1d_tpu
        if token_ids.device.type == "tpu"
        else _force_token_ids_1d_impl
    )
    return fn(token_ids, forced_ids)


def advance_spec_states(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    next_tokens: torch.Tensor,
    max_draft_tokens: int,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _advance_spec_states_tpu
        if state_table.device.type == "tpu"
        else _advance_spec_states_impl
    )
    return fn(
        state_table, slot_ids, next_tokens, max_draft_tokens, *markers.kernel_args()
    )


def _same_target_device(actual: torch.device, requested: torch.device) -> bool:
    """Match e.g. ``device('tpu', 0)`` against unindexed ``device('tpu')``."""
    if actual.type != requested.type:
        return False
    if requested.index is not None:
        return actual.index == requested.index
    if actual.index is None:
        return True
    default_idx = (
        torch.accelerator.current_device_index()
        if torch.accelerator.is_available()
        and (acc := torch.accelerator.current_accelerator()) is not None
        and acc.type == requested.type
        else 0
    )
    return actual.index == default_idx


class ThinkingBudgetTracker:
    """Request-ID-keyed ``thinking_token_budget`` lifecycle and device state."""

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_reqs: int,
    ) -> None:
        self._markers = ThinkingBudgetMarkers.from_reasoning_config(reasoning_config)
        self._max_num_reqs = max(int(max_num_reqs), 1)
        # 2 * max_num_reqs usable slots (1..2*max_num_reqs) plus slot 0 as the
        # permanent untracked sentinel guarantees a free/evictable slot even
        # when an async batch is in flight alongside a disjoint new batch.
        self._capacity = 2 * self._max_num_reqs + 1
        self._budgets: dict[str, int] = {}
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._capacity - 1, 0, -1))
        self._dirty_states: dict[str, StateTuple] = {}
        self._state_table: torch.Tensor | None = None
        self._slot_upload_cache: dict[
            tuple[torch.device, int, int], tuple[bytes, torch.Tensor]
        ] = {}

    @property
    def enabled(self) -> bool:
        return self._markers is not None

    @property
    def markers(self) -> ThinkingBudgetMarkers | None:
        return self._markers

    @property
    def requests(self) -> Mapping[str, int]:
        return self._budgets

    def __bool__(self) -> bool:
        return self._markers is not None and bool(self._budgets)

    def commit_state_table(self, state_table: torch.Tensor) -> None:
        self._state_table = state_table

    def ensure_state_table(self, device: torch.device) -> torch.Tensor:
        if self._state_table is None or not _same_target_device(
            self._state_table.device, device
        ):
            if self._state_table is not None:
                self._slot_by_req_id.clear()
                self._free_slots = list(range(self._capacity - 1, 0, -1))
            table = torch.zeros(
                (self._capacity, NUM_STATE_COLS), dtype=torch.int32, device="cpu"
            )
            table[:, COL_BUDGET] = -1
            self._state_table = table.to(device)
            self._slot_upload_cache.clear()
        return self._state_table

    def _release_slot(self, req_id: str) -> None:
        slot = self._slot_by_req_id.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def discard(self, req_ids: Iterable[str]) -> None:
        if not self._budgets:
            return
        for req_id in req_ids:
            self._budgets.pop(req_id, None)
            self._dirty_states.pop(req_id, None)
            self._release_slot(req_id)

    def on_preempted(self, req_ids: Iterable[str] | None) -> None:
        if not req_ids or not self._budgets:
            return
        for req_id in req_ids:
            if req_id in self._budgets:
                self._dirty_states.pop(req_id, None)
                self._release_slot(req_id)

    def add(
        self,
        req_id: str,
        sampling_params: "SamplingParams | None",
        prompt_token_ids: Sequence[int] | None,
    ) -> None:
        if self._markers is None:
            return
        raw_budget = (
            None if sampling_params is None else sampling_params.thinking_token_budget
        )
        if raw_budget is None or raw_budget < 0:
            self.discard((req_id,))
            return
        clamped_budget = min(int(raw_budget), INT32_MAX)
        self._budgets[req_id] = clamped_budget
        self._dirty_states[req_id] = compute_sequence_state(
            prompt_token_ids or (), clamped_budget, self._markers
        )

    def on_resumed(
        self,
        resumed_req_ids: Iterable[str],
        requests: Mapping[str, "CachedRequestState"],
    ) -> None:
        if self._markers is None or not self._budgets:
            return
        for req_id in resumed_req_ids:
            budget = self._budgets.get(req_id)
            req_state = None if budget is None else requests.get(req_id)
            if budget is None or req_state is None:
                continue
            full_tokens = list(req_state.prompt_token_ids or ()) + [
                tok for tok in req_state.output_token_ids if tok >= 0
            ]
            self._dirty_states[req_id] = compute_sequence_state(
                full_tokens, budget, self._markers
            )

    def _allocate_slot(self, req_id: str, protected_req_ids: AbstractSet[str]) -> int:
        existing = self._slot_by_req_id.get(req_id)
        if existing is not None:
            return existing
        if not self._free_slots:
            for candidate_id in list(self._slot_by_req_id):
                if candidate_id not in protected_req_ids:
                    self._release_slot(candidate_id)
                    break
        if not self._free_slots:
            raise RuntimeError(
                f"ThinkingBudgetTracker slot pool exhausted (capacity={self._capacity})"
            )
        slot = self._free_slots.pop()
        self._slot_by_req_id[req_id] = slot
        return slot

    def prepare_step(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        requests: Mapping[str, "CachedRequestState"],
        unsettled_req_ids: AbstractSet[str] = frozenset(),
    ) -> torch.Tensor | None:
        """Ensure all scheduled budgeted requests own seeded slots on ``device``."""
        if self._markers is None or not self._budgets:
            return None
        batch_tracked = list(
            dict.fromkeys(
                req_id
                for req_id in batch_req_ids
                if req_id is not None and req_id in self._budgets
            )
        )
        if not batch_tracked:
            return None

        state_table = self.ensure_state_table(device)
        protected = set(batch_tracked) | set(unsettled_req_ids)
        to_seed: list[tuple[int, StateTuple]] = []
        for req_id in batch_tracked:
            had_slot = req_id in self._slot_by_req_id
            slot = self._allocate_slot(req_id, protected)
            dirty = self._dirty_states.pop(req_id, None)
            if dirty is None and not had_slot:
                req_state = requests.get(req_id)
                full_tokens = (
                    []
                    if req_state is None
                    else list(req_state.prompt_token_ids or ())
                    + [tok for tok in req_state.output_token_ids if tok >= 0]
                )
                dirty = compute_sequence_state(
                    full_tokens, self._budgets[req_id], self._markers
                )
            if dirty is not None:
                to_seed.append((slot, dirty))

        if to_seed:
            seed_width = self._max_num_reqs
            for start in range(0, len(to_seed), seed_width):
                chunk = to_seed[start : start + seed_width]
                slots_np = np.zeros(seed_width, dtype=np.int32)
                vals_np = np.zeros((seed_width, NUM_STATE_COLS), dtype=np.int32)
                vals_np[:, COL_BUDGET] = -1
                for idx, (slot, st_tuple) in enumerate(chunk):
                    slots_np[idx] = slot
                    vals_np[idx] = st_tuple
                # Synchronous .to(device): sample_tokens() runs after the runner's
                # input-staging fence, so async H2D from temporary CPU buffers
                # would risk reading freed host memory in TorchTPU.
                state_table = seed_states(
                    state_table,
                    torch.from_numpy(slots_np).to(device),
                    torch.from_numpy(vals_np).to(device),
                )
            self._state_table = state_table
        return state_table

    def build_chunk_slot_ids(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int,
        is_partial_prefill: Callable[[str], bool] | None = None,
    ) -> torch.Tensor | None:
        """Build ``[padded_num_reqs]`` int32 slot IDs for ``[start_idx, end_idx)``."""
        slots_np = np.zeros(padded_num_reqs, dtype=np.int32)
        any_active = False
        upper = min(end_idx, len(batch_req_ids))
        for batch_i in range(start_idx, upper):
            local_i = batch_i - start_idx
            if local_i >= padded_num_reqs:
                break
            req_id = batch_req_ids[batch_i]
            if req_id is None:
                continue
            slot = self._slot_by_req_id.get(req_id)
            if slot is None or (
                is_partial_prefill is not None and is_partial_prefill(req_id)
            ):
                continue
            slots_np[local_i] = slot
            any_active = True
        if not any_active:
            return None
        canon_dev = (
            self._state_table.device
            if self._state_table is not None
            and _same_target_device(self._state_table.device, device)
            else device
        )
        cache_key = (canon_dev, padded_num_reqs, start_idx)
        payload = slots_np.tobytes()
        cached = self._slot_upload_cache.get(cache_key)
        if cached is not None and cached[0] == payload:
            return cached[1]
        if len(self._slot_upload_cache) >= _MAX_SLOT_UPLOAD_CACHE:
            self._slot_upload_cache.clear()
        # Synchronous .to(device) avoids unfenced async H2D reads; caching
        # skips redundant H2D copies across steady-state decode steps.
        dev_slots = torch.from_numpy(slots_np).to(device)
        self._slot_upload_cache[cache_key] = (payload, dev_slots)
        return dev_slots

    def precompile(
        self,
        device: torch.device,
        num_reqs_paddings: Sequence[int],
        verify_logits_paddings: Sequence[int],
        spec_verify_pairs: Sequence[tuple[int, int]],
        max_draft_tokens: int,
        vocab_size: int,
        logits_dtype: torch.dtype,
        sync_fn: Callable[..., None],
    ) -> None:
        """Pre-warm all thinking-budget subgraphs across compiled bucket shapes."""
        if self._markers is None:
            return
        markers = self._markers
        state_table = self.ensure_state_table(device)

        # Allocate warmup inputs on CPU and upload via .to(device) (mirroring
        # TPUModelRunner._precompile_rejection_sampler) so compiled kernels are
        # traced on genuine H2D PJRT buffers rather than lazy device splats.
        seed_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(device)
        seed_vals = torch.zeros((self._max_num_reqs, NUM_STATE_COLS), dtype=torch.int32)
        seed_vals[:, COL_BUDGET] = -1
        state_table = seed_states(state_table, seed_slots, seed_vals.to(device))
        sync_fn(state_table)

        for num_reqs in num_reqs_paddings:
            slot_ids = torch.zeros(num_reqs, dtype=torch.int32).to(device)
            selected = torch.zeros((num_reqs, 1), dtype=torch.int64).to(device)
            out_plain, state_table = override_and_advance_plain(
                state_table, slot_ids, selected, markers
            )
            sync_fn([out_plain, state_table])
            logger.info("  -- plain num_seqs: %d", num_reqs)

        self._state_table = state_table
        if not spec_verify_pairs:
            return

        for num_tokens, num_reqs in spec_verify_pairs:
            forced_target, forced_bonus, eff_drafts = prepare_spec_verify(
                state_table,
                torch.zeros(num_reqs, dtype=torch.int32).to(device),
                torch.zeros(num_tokens, dtype=torch.int32).to(device),
                torch.zeros(num_reqs, dtype=torch.int32).to(device),
                torch.zeros(num_tokens, dtype=torch.int64).to(device),
                torch.zeros(num_tokens, dtype=torch.int32).to(device),
                max_draft_tokens,
                markers,
            )
            sync_fn([forced_target, forced_bonus, eff_drafts])

        for num_tokens in verify_logits_paddings:
            sync_fn(
                force_logits_rows(
                    torch.zeros((num_tokens, vocab_size), dtype=logits_dtype).to(
                        device
                    ),
                    torch.full((num_tokens,), NO_FORCED_TOKEN, dtype=torch.int32).to(
                        device
                    ),
                )
            )

        for length in sorted(set(num_reqs_paddings) | set(verify_logits_paddings)):
            sync_fn(
                force_token_ids_1d(
                    torch.zeros(length, dtype=torch.int64).to(device),
                    torch.full((length,), NO_FORCED_TOKEN, dtype=torch.int32).to(
                        device
                    ),
                )
            )

        for num_reqs in num_reqs_paddings:
            out_spec, state_table = advance_spec_states(
                state_table,
                torch.zeros(num_reqs, dtype=torch.int32).to(device),
                torch.full(
                    (num_reqs, max_draft_tokens + 1),
                    NO_FORCED_TOKEN,
                    dtype=torch.int32,
                ).to(device),
                max_draft_tokens,
                markers,
            )
            sync_fn([out_spec, state_table])
        self._state_table = state_table
