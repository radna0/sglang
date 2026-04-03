import unittest

import torch


class TestDflashPQMaximalCoupling(unittest.TestCase):
    def test_maximal_coupling_matches_target_distribution(self):
        # Single-step maximal coupling used by speculative sampling:
        #   1) y ~ q
        #   2) accept y with prob min(1, p(y)/q(y))
        #   3) else sample from (p - q)+
        #
        # This must produce exact x ~ p (up to sampling noise).
        torch.manual_seed(0)
        vocab = 8
        n = 50_000

        p = torch.rand(vocab, dtype=torch.float64)
        p = p / p.sum()
        q = torch.rand(vocab, dtype=torch.float64)
        q = q / q.sum()

        y = torch.multinomial(q, n, replacement=True)
        accept_prob = (p[y] / q[y]).clamp(max=1.0)
        u = torch.rand(n, dtype=torch.float64)
        accept = u < accept_prob

        out = y.clone()
        resid = (p - q).clamp(min=0.0)
        if float(resid.sum()) > 0:
            resid = resid / resid.sum()
        else:
            resid = p
        num_reject = int((~accept).sum().item())
        if num_reject:
            out[~accept] = torch.multinomial(resid, num_reject, replacement=True)

        hist = torch.bincount(out, minlength=vocab).to(torch.float64) / float(n)

        # Tight-ish tolerances: with n=50k, expected std ~ sqrt(p(1-p)/n) <= ~0.01.
        self.assertTrue(
            torch.allclose(hist, p, atol=0.01, rtol=0.05),
            msg=f"empirical hist {hist.tolist()} diverges from p {p.tolist()}",
        )

        # Acceptance probability under maximal coupling is sum_x min(p(x), q(x)).
        empirical_accept = float(accept.to(torch.float64).mean().item())
        expected_accept = float(torch.minimum(p, q).sum().item())
        self.assertAlmostEqual(empirical_accept, expected_accept, delta=0.02)

