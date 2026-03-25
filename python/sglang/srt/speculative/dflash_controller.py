from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _env_flag(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v not in ("", "0", "false", "off", "no")


def _env_float(name: str, default: float) -> float:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


@dataclass
class DFlashDifficultySignals:
    """Unified (paper-agnostic) difficulty signals for DFlash++ policies.

    These are deliberately *small* and stable so we can emit them into logs/bench JSON
    and also drive adaptive controllers (FailFast/DAWN/SSD/EAFT-style) without coupling
    to the exact internal tensor shapes.
    """

    verify_mode: str
    # Mean accept ratio: E[min(1, p(y)/q(y))] across proposed positions (pq mode).
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
class DFlashAdaptivePqConfig:
    enabled: bool
    # Temperature multiplier applied to the draft *proposal* distribution q.
    temp_mul: float
    temp_min: float
    temp_max: float
    # Controller gain for entropy matching.
    ent_kp: float
    # Optional "FailFast" fallback: disable pq when it is clearly pathological.
    disable_pq_on_p_y_zero_ge: float
    disable_pq_on_tv_ge: float
    # EAFT-like "confident conflict" gate (draft is confident but acceptance is low).
    eaft_q_ent_le: float
    eaft_a_mean_le: float
    eaft_disable_rounds: int

    @staticmethod
    def from_env(*, default_temp_mul: float) -> "DFlashAdaptivePqConfig":
        enabled = _env_flag("SGLANG_DFLASH_ADAPTIVE_PQ")
        temp_min = _env_float("SGLANG_DFLASH_ADAPTIVE_PQ_TEMP_MIN", 0.5)
        temp_max = _env_float("SGLANG_DFLASH_ADAPTIVE_PQ_TEMP_MAX", 2.0)
        ent_kp = _env_float("SGLANG_DFLASH_ADAPTIVE_PQ_ENT_KP", 0.25)
        disable_p_y0 = _env_float("SGLANG_DFLASH_FAILFAST_PY0_GE", 0.35)
        disable_tv = _env_float("SGLANG_DFLASH_FAILFAST_TV_GE", 0.85)
        eaft_q_ent_le = _env_float("SGLANG_DFLASH_EAFT_QENT_LE", -1.0)
        eaft_a_mean_le = _env_float("SGLANG_DFLASH_EAFT_AMEAN_LE", -1.0)
        eaft_disable_rounds = _env_int("SGLANG_DFLASH_EAFT_DISABLE_ROUNDS", 2)
        return DFlashAdaptivePqConfig(
            enabled=bool(enabled),
            temp_mul=float(default_temp_mul),
            temp_min=float(temp_min),
            temp_max=float(temp_max),
            ent_kp=float(ent_kp),
            disable_pq_on_p_y_zero_ge=float(disable_p_y0),
            disable_pq_on_tv_ge=float(disable_tv),
            eaft_q_ent_le=float(eaft_q_ent_le),
            eaft_a_mean_le=float(eaft_a_mean_le),
            eaft_disable_rounds=int(eaft_disable_rounds),
        )


class DFlashAdaptivePqController:
    """FailFast/EAFT-inspired *policy layer* for pq verification.

    Correctness guarantee: this controller never changes verifier math; it only
    adjusts draft proposal shaping (q) and (optionally) turns pq off when it is
    clearly counterproductive.
    """

    def __init__(self, cfg: DFlashAdaptivePqConfig):
        self.cfg = cfg
        self._disabled_pq_rounds_left = 0

    def should_force_target_only(self) -> bool:
        return self._disabled_pq_rounds_left > 0

    def on_verify_end(self, sig: DFlashDifficultySignals) -> Tuple[float, Dict[str, Any]]:
        """Update temp_mul (and maybe failfast-disable pq) from verify-side stats.

        Returns: (new_temp_mul, debug_dict)
        """
        dbg: Dict[str, Any] = {"enabled": bool(self.cfg.enabled)}
        if not self.cfg.enabled:
            return float(self.cfg.temp_mul), dbg

        disabled_before = int(self._disabled_pq_rounds_left)

        # FailFast: if pq is clearly pathological, temporarily disable it.
        py0 = sig.frac_p_y_zero
        tv = sig.tv_mean
        if py0 is not None and py0 >= self.cfg.disable_pq_on_p_y_zero_ge:
            self._disabled_pq_rounds_left = max(self._disabled_pq_rounds_left, 2)
        if tv is not None and tv >= self.cfg.disable_pq_on_tv_ge:
            self._disabled_pq_rounds_left = max(self._disabled_pq_rounds_left, 2)

        # EAFT-style "confident conflict" gating:
        # if the draft is confident (low entropy) but acceptance is low, fail fast by
        # disabling pq briefly (or you can use this as a trigger to raise draft_temp_mul).
        if (
            sig.q_entropy_mean is not None
            and sig.accept_mean is not None
            and self.cfg.eaft_q_ent_le >= 0.0
            and self.cfg.eaft_a_mean_le >= 0.0
        ):
            if float(sig.q_entropy_mean) <= float(self.cfg.eaft_q_ent_le) and float(sig.accept_mean) <= float(self.cfg.eaft_a_mean_le):
                self._disabled_pq_rounds_left = max(
                    self._disabled_pq_rounds_left, int(self.cfg.eaft_disable_rounds)
                )
                dbg["eaft_conflict"] = True
            else:
                dbg["eaft_conflict"] = False

        disabled_set = int(self._disabled_pq_rounds_left)
        if self._disabled_pq_rounds_left > 0:
            self._disabled_pq_rounds_left -= 1
        disabled_after = int(self._disabled_pq_rounds_left)

        # EAFT-style entropy matching: move draft entropy towards target entropy.
        p_ent = sig.p_entropy_mean
        q_ent = sig.q_entropy_mean
        if p_ent is not None and q_ent is not None and math.isfinite(p_ent) and math.isfinite(q_ent):
            # If q entropy < p entropy => draft is over-confident => increase temp.
            # If q entropy > p entropy => draft is too diffuse => decrease temp.
            delta = float(p_ent - q_ent)
            scale = math.exp(self.cfg.ent_kp * delta)
            new_temp = float(self.cfg.temp_mul) * float(scale)
            new_temp = max(self.cfg.temp_min, min(self.cfg.temp_max, new_temp))
            self.cfg.temp_mul = float(new_temp)
            dbg.update(
                {
                    "p_ent": float(p_ent),
                    "q_ent": float(q_ent),
                    "delta_ent": float(delta),
                    "temp_mul": float(self.cfg.temp_mul),
                    "scale": float(scale),
                }
            )
        else:
            dbg.update({"temp_mul": float(self.cfg.temp_mul)})

        dbg["pq_disabled_rounds_left"] = int(self._disabled_pq_rounds_left)
        dbg["pq_disabled_before"] = int(disabled_before)
        dbg["pq_disabled_set"] = int(disabled_set)
        dbg["pq_disabled_after"] = int(disabled_after)
        dbg["pq_disabled_triggered"] = bool(disabled_set > disabled_before)
        return float(self.cfg.temp_mul), dbg


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

    def update(self, *, accept_len: int, verify_ct: int, ema_beta: float = 0.9) -> None:
        a = float(max(0, int(accept_len)))
        self.accept_len_last = a
        self.verify_ct_last = int(verify_ct)
        self.accept_len_ema = float(self.accept_len_ema) * float(ema_beta) + a * float(
            1.0 - float(ema_beta)
        )

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
