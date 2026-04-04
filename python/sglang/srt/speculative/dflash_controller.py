from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DFlashDifficultySignals:
    """Unified (paper-agnostic) difficulty signals for DFlash++ policies.

    These are deliberately *small* and stable so we can emit them into logs/bench JSON
    and also drive adaptive controllers (FailFast/DAWN/SSD/EAFT-style) without coupling
    to the exact internal tensor shapes.
    """

    verify_mode: str
    # Mean accept ratio emitted by the verifier debug path.
    # NOTE: This is not accept_len / spec_accept_length_mean (K), which is a different metric.
    accept_mean: Optional[float]
    tv_mean: Optional[float]
    frac_p_y_zero: Optional[float]
    frac_q_y_zero: Optional[float]
    p_entropy_mean: Optional[float]
    q_entropy_mean: Optional[float]
    p_max_mean: Optional[float]
    q_max_mean: Optional[float]

    @staticmethod
    def from_debug(
        *,
        verify_mode: str,
        dflash_debug: Optional[Dict[str, Any]],
        draft_conf_debug: Optional[Dict[str, Any]],
    ) -> "DFlashDifficultySignals":
        def _get(d: Optional[Dict[str, Any]], k: str) -> Optional[float]:
            if not d:
                return None
            v = d.get(k, None)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        # Prefer verify-side scalar stats (truth). Fall back to draft-side cheap stats.
        q_ent = _get(dflash_debug, "q_entropy_mean")
        q_max = _get(dflash_debug, "q_max_mean")
        if q_ent is None and draft_conf_debug:
            q_ent = _get(draft_conf_debug, "q_ent_mean_first")
        if q_max is None and draft_conf_debug:
            q_max = _get(draft_conf_debug, "q_max_mean_first")

        return DFlashDifficultySignals(
            verify_mode=str(verify_mode),
            accept_mean=_get(dflash_debug, "accept_ratio_mean") or _get(dflash_debug, "a_mean"),
            tv_mean=_get(dflash_debug, "tv_mean"),
            frac_p_y_zero=_get(dflash_debug, "frac_p_y_zero"),
            frac_q_y_zero=_get(dflash_debug, "frac_q_y_zero"),
            p_entropy_mean=_get(dflash_debug, "p_entropy_mean"),
            q_entropy_mean=q_ent,
            p_max_mean=_get(dflash_debug, "p_max_mean"),
            q_max_mean=q_max,
        )


@dataclass
class DFlashReqDifficultyState:
    """Per-request rolling state for DFlash++ policies.

    We keep this intentionally tiny so it can be attached dynamically to request objects
    without changing core request schemas.
    """

    accept_len_last: float = 0.0
    accept_len_ema: float = 0.0
    verify_ct_last: int = 0
    fanout_hard_latch_ct: int = 0
    q_entropy_mean_last: Optional[float] = None
    q_max_mean_last: Optional[float] = None
    tv_mean_last: Optional[float] = None

    def update(self, *, accept_len: int, verify_ct: int, ema_beta: float = 0.9) -> None:
        a = float(max(0, int(accept_len)))
        self.accept_len_last = a
        self.verify_ct_last = int(verify_ct)
        self.accept_len_ema = float(self.accept_len_ema) * float(ema_beta) + a * float(
            1.0 - float(ema_beta)
        )

    def update_verify_debug(self, sig: Optional[DFlashDifficultySignals]) -> None:
        if sig is None:
            return
        if sig.q_entropy_mean is not None and math.isfinite(float(sig.q_entropy_mean)):
            self.q_entropy_mean_last = float(sig.q_entropy_mean)
        if sig.q_max_mean is not None and math.isfinite(float(sig.q_max_mean)):
            self.q_max_mean_last = float(sig.q_max_mean)
        if sig.tv_mean is not None and math.isfinite(float(sig.tv_mean)):
            self.tv_mean_last = float(sig.tv_mean)

    def update_fanout_hard_phase(
        self, *, hard_now: bool, latch_rounds: int
    ) -> bool:
        latch_rounds = int(max(0, int(latch_rounds)))
        if latch_rounds <= 0:
            return bool(hard_now)
        if bool(hard_now):
            self.fanout_hard_latch_ct = int(latch_rounds)
            return True
        if int(self.fanout_hard_latch_ct) > 0:
            self.fanout_hard_latch_ct = max(0, int(self.fanout_hard_latch_ct) - 1)
            return True
        return False


def survival_should_force_target_only(
    req_states: list[DFlashReqDifficultyState], *, accept_ema_le: float
) -> bool:
    if not req_states:
        return False
    denom = 0
    s = 0.0
    for st in req_states:
        if st is None:
            continue
        denom += 1
        s += float(getattr(st, "accept_len_ema", 0.0))
    if denom <= 0:
        return False
    mean_ema = float(s) / float(denom)
    return mean_ema <= float(accept_ema_le)


def req_is_hard_enough_for_fanout(
    req_state: Optional[DFlashReqDifficultyState],
    *,
    accept_ema_le: float,
    accept_last_le: float,
    min_verify_ct: int,
) -> bool:
    """Return whether a request looks hard enough to spend SSD fan-out budget.

    This is a conservative gate:
    - if no threshold is configured, fan-out is allowed
    - if thresholds are configured but we have no request state yet, fan-out stays off
    - once enough history exists, low accept EMA or low last accept can enable fan-out
    """

    use_ema = math.isfinite(float(accept_ema_le)) and float(accept_ema_le) >= 0.0
    use_last = math.isfinite(float(accept_last_le)) and float(accept_last_le) >= 0.0
    if not use_ema and not use_last and int(min_verify_ct) <= 0:
        return True
    if req_state is None:
        return False
    if int(getattr(req_state, "verify_ct_last", 0)) < int(min_verify_ct):
        return False
    hard = False
    if use_ema:
        hard = hard or float(getattr(req_state, "accept_len_ema", 0.0)) <= float(
            accept_ema_le
        )
    if use_last:
        hard = hard or float(getattr(req_state, "accept_len_last", 0.0)) <= float(
            accept_last_le
        )
    return bool(hard)


def compute_adaptive_max_steps_for_req(
    req_state: Optional[DFlashReqDifficultyState],
    *,
    step_count: int,
    verify_ct_ge: int,
    last_verify_ct_ge: int,
    accept_ema_hard_le: float,
    accept_ema_medium_le: float,
    accept_last_hard_le: float,
    accept_last_medium_le: float,
    hard_cap_steps: int,
    medium_cap_steps: int,
    q_entropy_hard_le: float = -1.0,
    q_entropy_hard_ge: float = -1.0,
    q_max_hard_ge: float = -1.0,
    q_max_hard_le: float = -1.0,
    tv_hard_ge: float = -1.0,
) -> int:
    """Choose a logical speculative cap while keeping the physical block fixed.

    This is intentionally conservative:
    - without enough verify history, keep the full physical width
    - hard requests get a small logical cap
    - medium requests get a medium logical cap
    - easy / predictable requests keep the full width
    """

    step_count = max(1, int(step_count))
    hard_cap_steps = max(1, min(step_count, int(hard_cap_steps)))
    medium_cap_steps = max(hard_cap_steps, min(step_count, int(medium_cap_steps)))
    verify_ct_ge = max(0, int(verify_ct_ge))
    last_verify_ct_ge = max(0, int(last_verify_ct_ge))

    if req_state is None:
        return step_count
    verify_ct = int(getattr(req_state, "verify_ct_last", 0))
    if verify_ct < min(verify_ct_ge, last_verify_ct_ge):
        return step_count

    accept_ema = float(getattr(req_state, "accept_len_ema", 0.0))
    accept_last = float(getattr(req_state, "accept_len_last", 0.0))
    q_entropy = getattr(req_state, "q_entropy_mean_last", None)
    q_max = getattr(req_state, "q_max_mean_last", None)
    tv_mean = getattr(req_state, "tv_mean_last", None)

    if (
        accept_last_hard_le >= 0.0
        and verify_ct >= last_verify_ct_ge
        and accept_last <= float(accept_last_hard_le)
    ):
        return hard_cap_steps
    if (
        q_entropy_hard_le >= 0.0
        and q_entropy is not None
        and math.isfinite(float(q_entropy))
        and float(q_entropy) <= float(q_entropy_hard_le)
        and accept_ema <= float(accept_ema_medium_le)
    ):
        return hard_cap_steps
    if (
        q_entropy_hard_ge >= 0.0
        and q_entropy is not None
        and math.isfinite(float(q_entropy))
        and float(q_entropy) >= float(q_entropy_hard_ge)
        and accept_ema <= float(accept_ema_medium_le)
    ):
        return hard_cap_steps
    if (
        q_max_hard_ge >= 0.0
        and q_max is not None
        and math.isfinite(float(q_max))
        and float(q_max) >= float(q_max_hard_ge)
        and accept_ema <= float(accept_ema_medium_le)
    ):
        return hard_cap_steps
    if (
        q_max_hard_le >= 0.0
        and q_max is not None
        and math.isfinite(float(q_max))
        and float(q_max) <= float(q_max_hard_le)
        and accept_ema <= float(accept_ema_medium_le)
    ):
        return hard_cap_steps
    if (
        tv_hard_ge >= 0.0
        and tv_mean is not None
        and math.isfinite(float(tv_mean))
        and float(tv_mean) >= float(tv_hard_ge)
        and accept_ema <= float(accept_ema_medium_le)
    ):
        return hard_cap_steps

    if accept_ema <= float(accept_ema_hard_le):
        return hard_cap_steps
    if (
        accept_last_medium_le >= 0.0
        and verify_ct >= last_verify_ct_ge
        and accept_last <= float(accept_last_medium_le)
    ):
        return medium_cap_steps
    if accept_ema <= float(accept_ema_medium_le):
        return medium_cap_steps
    return step_count
