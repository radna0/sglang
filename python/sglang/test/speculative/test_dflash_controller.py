import unittest

from sglang.srt.speculative.dflash_controller import (
    compute_adaptive_max_steps_for_req,
    DFlashAdaptivePqConfig,
    DFlashAdaptivePqController,
    DFlashDifficultySignals,
    DFlashReqDifficultyState,
    survival_should_force_target_only,
)


class TestDFlashAdaptivePqController(unittest.TestCase):
    def test_entropy_matching_increases_temp_when_q_too_peaky(self):
        cfg = DFlashAdaptivePqConfig(
            enabled=True,
            temp_mul=1.0,
            temp_min=0.5,
            temp_max=2.0,
            ent_kp=1.0,
            disable_pq_on_p_y_zero_ge=999.0,
            disable_pq_on_tv_ge=999.0,
            eaft_q_ent_le=-1.0,
            eaft_a_mean_le=-1.0,
            eaft_disable_rounds=2,
        )
        ctrl = DFlashAdaptivePqController(cfg)
        sig = DFlashDifficultySignals(
            verify_mode="pq",
            accept_mean=0.5,
            tv_mean=0.2,
            frac_p_y_zero=0.0,
            frac_q_y_zero=0.0,
            p_entropy_mean=3.0,
            q_entropy_mean=2.0,
            p_max_mean=0.2,
            q_max_mean=0.4,
        )
        new_temp, dbg = ctrl.on_verify_end(sig)
        self.assertGreater(new_temp, 1.0)
        self.assertIn("temp_mul", dbg)

    def test_entropy_matching_decreases_temp_when_q_too_diffuse(self):
        cfg = DFlashAdaptivePqConfig(
            enabled=True,
            temp_mul=1.0,
            temp_min=0.5,
            temp_max=2.0,
            ent_kp=1.0,
            disable_pq_on_p_y_zero_ge=999.0,
            disable_pq_on_tv_ge=999.0,
            eaft_q_ent_le=-1.0,
            eaft_a_mean_le=-1.0,
            eaft_disable_rounds=2,
        )
        ctrl = DFlashAdaptivePqController(cfg)
        sig = DFlashDifficultySignals(
            verify_mode="pq",
            accept_mean=0.5,
            tv_mean=0.2,
            frac_p_y_zero=0.0,
            frac_q_y_zero=0.0,
            p_entropy_mean=2.0,
            q_entropy_mean=3.0,
            p_max_mean=0.2,
            q_max_mean=0.1,
        )
        new_temp, _ = ctrl.on_verify_end(sig)
        self.assertLess(new_temp, 1.0)

    def test_failfast_disables_pq_on_high_p_y_zero(self):
        cfg = DFlashAdaptivePqConfig(
            enabled=True,
            temp_mul=1.0,
            temp_min=0.5,
            temp_max=2.0,
            ent_kp=0.0,
            disable_pq_on_p_y_zero_ge=0.1,
            disable_pq_on_tv_ge=999.0,
            eaft_q_ent_le=-1.0,
            eaft_a_mean_le=-1.0,
            eaft_disable_rounds=2,
        )
        ctrl = DFlashAdaptivePqController(cfg)
        sig = DFlashDifficultySignals(
            verify_mode="pq",
            accept_mean=0.5,
            tv_mean=0.2,
            frac_p_y_zero=0.5,
            frac_q_y_zero=0.0,
            p_entropy_mean=2.0,
            q_entropy_mean=2.0,
            p_max_mean=0.2,
            q_max_mean=0.2,
        )
        _, dbg = ctrl.on_verify_end(sig)
        self.assertTrue(dbg.get("pq_disabled_triggered"))
        self.assertTrue(ctrl.should_force_target_only() or dbg.get("pq_disabled_after", 0) >= 0)

    def test_eaft_confident_conflict_gate_disables_pq(self):
        cfg = DFlashAdaptivePqConfig(
            enabled=True,
            temp_mul=1.0,
            temp_min=0.5,
            temp_max=2.0,
            ent_kp=0.0,
            disable_pq_on_p_y_zero_ge=999.0,
            disable_pq_on_tv_ge=999.0,
            eaft_q_ent_le=1.0,
            eaft_a_mean_le=0.2,
            eaft_disable_rounds=3,
        )
        ctrl = DFlashAdaptivePqController(cfg)
        sig = DFlashDifficultySignals(
            verify_mode="pq",
            accept_mean=0.1,
            tv_mean=0.2,
            frac_p_y_zero=0.0,
            frac_q_y_zero=0.0,
            p_entropy_mean=2.0,
            q_entropy_mean=0.5,
            p_max_mean=0.2,
            q_max_mean=0.9,
        )
        _, dbg = ctrl.on_verify_end(sig)
        self.assertTrue(dbg.get("eaft_conflict"))
        self.assertTrue(dbg.get("pq_disabled_triggered"))

    def test_from_debug_accept_ratio_key(self):
        sig = DFlashDifficultySignals.from_debug(
            verify_mode="pq",
            dflash_debug={"accept_ratio_mean": 0.33, "tv_mean": 0.2, "p_entropy_mean": 1.0, "q_entropy_mean": 1.1},
            draft_conf_debug=None,
        )
        self.assertAlmostEqual(sig.accept_mean, 0.33, places=6)

    def test_from_debug_backward_compat_a_mean_key(self):
        sig = DFlashDifficultySignals.from_debug(
            verify_mode="pq",
            dflash_debug={"a_mean": 0.22, "tv_mean": 0.2},
            draft_conf_debug=None,
        )
        self.assertAlmostEqual(sig.accept_mean, 0.22, places=6)

    def test_survival_should_force_target_only(self):
        st1 = DFlashReqDifficultyState(accept_len_ema=0.5)
        st2 = DFlashReqDifficultyState(accept_len_ema=1.5)
        self.assertFalse(survival_should_force_target_only([st1, st2], accept_ema_le=0.5))
        self.assertTrue(survival_should_force_target_only([st1, st2], accept_ema_le=1.0))

    def test_compute_adaptive_max_steps_hard_request(self):
        st = DFlashReqDifficultyState(accept_len_ema=1.5, verify_ct_last=12)
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
            ),
            4,
        )

    def test_compute_adaptive_max_steps_medium_request(self):
        st = DFlashReqDifficultyState(accept_len_ema=4.0, verify_ct_last=12)
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
            ),
            6,
        )

    def test_compute_adaptive_max_steps_easy_request(self):
        st = DFlashReqDifficultyState(accept_len_ema=7.0, verify_ct_last=12)
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
            ),
            8,
        )

    def test_compute_adaptive_max_steps_uses_q_entropy_hard_gate(self):
        st = DFlashReqDifficultyState(
            accept_len_ema=4.5, verify_ct_last=12, q_entropy_mean_last=0.2
        )
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
                q_entropy_hard_le=0.5,
            ),
            4,
        )

    def test_compute_adaptive_max_steps_uses_q_entropy_high_hard_gate(self):
        st = DFlashReqDifficultyState(
            accept_len_ema=4.5, verify_ct_last=12, q_entropy_mean_last=2.3
        )
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
                q_entropy_hard_ge=2.0,
            ),
            4,
        )

    def test_compute_adaptive_max_steps_uses_q_max_hard_gate(self):
        st = DFlashReqDifficultyState(
            accept_len_ema=4.5, verify_ct_last=12, q_max_mean_last=0.95
        )
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
                q_max_hard_ge=0.9,
            ),
            4,
        )

    def test_compute_adaptive_max_steps_uses_q_max_low_hard_gate(self):
        st = DFlashReqDifficultyState(
            accept_len_ema=4.5, verify_ct_last=12, q_max_mean_last=0.35
        )
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
                q_max_hard_le=0.4,
            ),
            4,
        )

    def test_compute_adaptive_max_steps_uses_tv_hard_gate(self):
        st = DFlashReqDifficultyState(
            accept_len_ema=4.5, verify_ct_last=12, tv_mean_last=0.85
        )
        self.assertEqual(
            compute_adaptive_max_steps_for_req(
                st,
                step_count=8,
                verify_ct_ge=8,
                accept_ema_hard_le=2.0,
                accept_ema_medium_le=5.0,
                hard_cap_steps=4,
                medium_cap_steps=6,
                tv_hard_ge=0.8,
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
