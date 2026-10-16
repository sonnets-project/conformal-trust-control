"""Small synthetic tests of the paper's temporal and fallback contracts."""
from pathlib import Path
import runpy
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD = runpy.run_path(str(ROOT / "new-reward/warm0_batch_replay.py"))
deploy = METHOD["deploy"]


def ledger():
    n = 60
    starts = np.datetime64("2000-01-01") + np.repeat(np.arange(12) * 10, 5).astype("timedelta64[D]")
    outcomes = {lam: np.full(n, lam * 8.0) for lam in METHOD["LAM_GRID"]}
    return {"n": n, "ds": starts, "de": starts + np.timedelta64(5, "D"),
            "iv": np.full(n, .2), "order": np.arange(n),
            "dp_lam": outcomes, "d_lam": {k: v.copy() for k, v in outcomes.items()}}


class ControllerContractTests(unittest.TestCase):
    def test_zero_trust_is_exact_fallback(self):
        result = deploy(ledger(), 5.0, mode="fix0")
        np.testing.assert_array_equal(result["dpnl_bp"], 0.0)
        np.testing.assert_array_equal(result["lam"], 0.0)
        self.assertEqual(result["breach"], 0.0)

    def test_warmup_includes_all_episodes_in_denominator(self):
        data = ledger()
        data["dp_lam"][1.0][-5:] = -50.0
        result = deploy(data, 5.0)
        np.testing.assert_array_equal(result["lam"][:15], 0.0)
        self.assertEqual(int(result["informed"].sum()), 45)
        self.assertEqual(result["breach"], result["breach_indicator"].sum() / 60)

    def test_future_outcomes_do_not_affect_earlier_decisions(self):
        original, altered = ledger(), ledger()
        for lam in METHOD["LAM_GRID"][1:]:
            altered["dp_lam"][lam][40:] = -1e8
        first, second = deploy(original, 5.0), deploy(altered, 5.0)
        # None of those outcomes has completed when the batch at indices 40:45 starts.
        np.testing.assert_array_equal(first["lam"][:45], second["lam"][:45])

    def test_same_date_feedback_is_order_invariant(self):
        original, shuffled = ledger(), ledger()
        for data in (original, shuffled):
            for lam in METHOD["LAM_GRID"][1:]:
                data["dp_lam"][lam][20:30:2] = -20 * lam
        shuffled["order"] = np.arange(60).reshape(-1, 5)[:, ::-1].ravel()
        first, second = deploy(original, 5.0), deploy(shuffled, 5.0)
        np.testing.assert_array_equal(first["lam"], second["lam"])
        self.assertAlmostEqual(first["eps_eff_final"], second["eps_eff_final"], places=14)

    def test_largest_admissible_candidate_without_monotonicity(self):
        data = ledger()
        data["dp_lam"][1.0][:] = -100
        data["dp_lam"][.5][:] = -100
        result = deploy(data, 5.0)
        np.testing.assert_array_equal(result["lam"][result["informed"]], .75)

    def test_virtual_infinite_loss_rejects_insufficient_calibration(self):
        quantile = METHOD["conformal_lower_q"](np.ones(3), np.ones(3), .1)
        self.assertTrue(np.isneginf(quantile))


if __name__ == "__main__":
    unittest.main()
