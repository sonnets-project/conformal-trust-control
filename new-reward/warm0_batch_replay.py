"""Replay the deployment controller on saved counterfactual episode records.

The v1.3 protocol uses the BS fallback until 15 episodes have completed,
includes warm-up in the breach denominator, and applies completion feedback
as one ACI update per decision date. Outputs go to results/warm0_batch_v13/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


VERSION = "warm0_batch_v13"
REGIMES = (
    "crash_test2018",
    "crash_test2020",
    "mix_test2022",
    "mix_test2024",
)
KAPPAS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS_TARGET = 0.10
WARMUP = 15
GAMMA_EPS = 0.02
HALF_LIFE = 30.0
TAU_IV = 0.04
M_EMP = 0.3
BOOTSTRAP_REPS = 3000
BOOTSTRAP_START_BATCHES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cvar5(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    count = max(1, int(0.05 * len(values)))
    return float(np.sort(values)[:count].mean())


def conformal_lower_q(values: np.ndarray, weights: np.ndarray, eps: float) -> float:
    """Weighted lower quantile with a virtual -infinity atom of weight one."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    total = float(weights.sum() + 1.0)
    if total <= 0 or 1.0 / total >= eps:
        return -np.inf
    index = np.argsort(values)
    ordered_values = values[index]
    ordered_weights = weights[index]
    cumulative = (1.0 + np.cumsum(ordered_weights)) / total
    location = int(np.searchsorted(cumulative, eps, side="left"))
    return float(ordered_values[min(location, len(ordered_values) - 1)])


def load_ledger(path: Path) -> dict[tuple[int, int], dict]:
    archive = np.load(path)
    shared = {
        "ds": archive["ds"].astype("datetime64[ns]"),
        "de": archive["de"].astype("datetime64[ns]"),
        "iv": archive["iv"],
        "ss": archive["ss"],
        "order": archive["order"],
    }
    count = len(shared["ds"])
    store: dict[tuple[int, int], dict] = {}
    for key in archive.files:
        if key.count("|") != 3:
            continue
        kappa_key, seed_key, kind, lam_key = key.split("|")
        item = store.setdefault(
            (int(kappa_key[1:]), int(seed_key[1:])),
            {"n": count, "d_lam": {}, "dp_lam": {}, "vadp": {}, **shared},
        )
        target = {"d": item["d_lam"], "dp": item["dp_lam"], "vadp": item["vadp"]}[kind]
        target[float(lam_key)] = archive[key]
    expected = {(kappa, seed) for kappa in KAPPAS for seed in SEEDS}
    if set(store) != expected:
        raise RuntimeError(f"ledger keys differ from expected 15 policies: {path}")
    return store


def budget_from_validation(ledger: dict, floor_bp: float = 5.0) -> float:
    return float(max(floor_bp, -np.quantile(ledger["vadp"][1.0], EPS_TARGET)))


def validation_static_lambda(ledger: dict, budget_bp: float) -> float:
    for lam in sorted(LAM_GRID, reverse=True):
        if float(np.mean(ledger["vadp"][lam] < -budget_bp)) <= EPS_TARGET:
            return float(lam)
    return 0.0


def deploy(
    ledger: dict,
    budget_bp: float,
    mode: str = "crc_rwc_aci",
    fixed_lambda: float | None = None,
) -> dict:
    """Replay one policy seed with an empty test ledger and delayed feedback."""
    n = ledger["n"]
    ds, de = ledger["ds"], ledger["de"]
    iv, order = ledger["iv"], ledger["order"]
    d_lam, dp_lam = ledger["d_lam"], ledger["dp_lam"]
    used = np.full(n, np.nan)
    informed = np.zeros(n, dtype=bool)
    eps_eff = float(EPS_TARGET)
    pending: list[int] = []

    for step, episode in enumerate(order):
        past = order[:step]
        done = past[de[past] < ds[episode]]
        informed[episode] = len(done) >= WARMUP

        if mode == "crc_rwc_aci" and pending:
            completed = [j for j in pending if de[j] < ds[episode]]
            pending = [j for j in pending if de[j] >= ds[episode]]
            if completed:
                errors = np.asarray(
                    [float(dp_lam[used[j]][j] < -budget_bp) for j in completed]
                )
                # Equivalent to summing the original per-contract ACI increments,
                # but clipping only once makes the update invariant to row order.
                batch_step = GAMMA_EPS * float(np.sum(EPS_TARGET - errors))
                eps_eff = float(
                    np.clip(
                        eps_eff + batch_step,
                        0.25 * EPS_TARGET,
                        min(0.45, 4 * EPS_TARGET),
                    )
                )

        if mode == "fix0":
            lam = 0.0
        elif mode == "fix1":
            lam = 1.0
        elif mode == "fixed":
            if fixed_lambda is None:
                raise ValueError("fixed mode requires fixed_lambda")
            lam = float(fixed_lambda)
        elif not informed[episode]:
            lam = 0.0
        elif mode == "a2margin":
            threshold = cvar5(d_lam[0.0][done]) - M_EMP
            lam = 0.0
            for candidate in sorted(LAM_GRID, reverse=True):
                if cvar5(d_lam[candidate][done]) >= threshold:
                    lam = candidate
                    break
        elif mode == "emp":
            lam = 0.0
            for candidate in sorted(LAM_GRID, reverse=True):
                if float(np.mean(dp_lam[candidate][done] < -budget_bp)) <= EPS_TARGET:
                    lam = candidate
                    break
        elif mode == "ivrule":
            percentile = float(np.mean(iv[done] <= iv[episode]))
            lam = 1.0 if percentile < 0.5 else (0.5 if percentile < 0.8 else 0.25)
        elif mode in {"crc", "crc_rwc", "crc_rwc_aci"}:
            if mode in {"crc_rwc", "crc_rwc_aci"}:
                age_days = (ds[episode] - de[done]) / np.timedelta64(1, "D")
                weights = np.exp(-np.log(2) * np.maximum(age_days, 0.0) / HALF_LIFE)
                weights *= np.exp(-np.abs(iv[done] - iv[episode]) / TAU_IV)
            else:
                weights = np.ones(len(done))
            eps_used = eps_eff if mode == "crc_rwc_aci" else EPS_TARGET
            lam = 0.0
            # Every candidate is evaluated independently.  No monotonic-risk
            # theorem is assumed for the clipped, cost-aware rollout.
            for candidate in sorted(LAM_GRID, reverse=True):
                if conformal_lower_q(dp_lam[candidate][done], weights, eps_used) >= -budget_bp:
                    lam = candidate
                    break
        else:
            raise ValueError(f"unsupported replay mode: {mode}")

        used[episode] = float(lam)
        if mode == "crc_rwc_aci":
            pending.append(int(episode))

    dp_used = np.asarray([dp_lam[used[i]][i] for i in range(n)], float)
    d_used = np.asarray([d_lam[used[i]][i] for i in range(n)], float)
    return {
        "lam": used,
        "dpnl_bp": dp_used,
        "delta_reward": d_used,
        "breach_indicator": dp_used < -budget_bp,
        "breach": float(np.mean(dp_used < -budget_bp)),
        "breach_informed": (
            float(np.mean(dp_used[informed] < -budget_bp)) if informed.any() else np.nan
        ),
        "informed": informed,
        "eps_eff_final": eps_eff,
        "budget_bp": float(budget_bp),
    }


def block_bootstrap_ci(
    values: np.ndarray,
    dates: np.ndarray,
    statistic=np.mean,
    reps: int = BOOTSTRAP_REPS,
    block_batches: int = BOOTSTRAP_START_BATCHES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-sided CI using consecutive start-date batches as the block unit."""
    values = np.asarray(values, float)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    groups = [np.where(inverse == index)[0] for index in range(len(unique_dates))]
    rng = np.random.default_rng(seed)
    n_dates = len(unique_dates)
    n_blocks = int(np.ceil(n_dates / block_batches))
    draws = np.empty(reps)
    for rep in range(reps):
        starts = rng.integers(0, n_dates, n_blocks)
        date_index = np.concatenate(
            [(start + np.arange(block_batches)) % n_dates for start in starts]
        )[:n_dates]
        selection = np.concatenate([groups[index] for index in date_index])
        draws[rep] = statistic(values[selection])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(statistic(values)), float(low), float(high)


def cluster_upper_bound(
    indicator: np.ndarray,
    dates: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    block_batches: int = BOOTSTRAP_START_BATCHES,
    seed: int = 0,
) -> tuple[float, float, int]:
    """One-sided 95% moving-block-bootstrap bound over start-date clusters."""
    indicator = np.asarray(indicator, float)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    daily = np.asarray([indicator[inverse == i].mean() for i in range(len(unique_dates))])
    rng = np.random.default_rng(seed)
    n_dates = len(daily)
    n_blocks = int(np.ceil(n_dates / block_batches))
    draws = np.empty(reps)
    for rep in range(reps):
        starts = rng.integers(0, n_dates, n_blocks)
        index = np.concatenate(
            [(start + np.arange(block_batches)) % n_dates for start in starts]
        )[:n_dates]
        draws[rep] = daily[index].mean()
    return float(daily.mean()), float(np.percentile(draws, 95)), n_dates


def wilson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    """One-sided Wilson upper bound; reported only as an iid reference."""
    if total <= 0:
        return np.nan
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center + radius)


def summarize_seed_runs(runs: list[dict], dates: np.ndarray) -> dict:
    mean_path = np.mean([run["dpnl_bp"] for run in runs], axis=0)
    point, low, high = block_bootstrap_ci(mean_path, dates)
    return {
        "mean_bp": point,
        "mean_ci_low": low,
        "mean_ci_high": high,
        "breach": float(np.mean([run["breach"] for run in runs])),
        "breach_informed": float(np.mean([run["breach_informed"] for run in runs])),
        "mean_lambda": float(np.mean([run["lam"].mean() for run in runs])),
        "cvar5_bp": cvar5(mean_path),
        "mean_path": mean_path,
    }


def ledger_paths(root: Path, regime: str) -> tuple[Path, Path]:
    directory = root / "new-reward" / "checkpoints" / f"conformal_controls_v12r_{regime}"
    aggressive = directory / f"ledger_arm00_{regime}.npz"
    conservative = directory / f"ledger_base_{regime}.npz"
    if not aggressive.exists() or not conservative.exists():
        raise FileNotFoundError(f"sealed v1.2R ledgers missing for {regime}: {directory}")
    return aggressive, conservative


def run(root: Path) -> None:
    output = root / "results" / VERSION
    output.mkdir(parents=True, exist_ok=True)
    panel_rows = []
    common_rows = []
    stress_rows = []
    seed_rows = []
    breach_rows = []
    source_hashes = {}

    for regime in REGIMES:
        aggressive_path, conservative_path = ledger_paths(root, regime)
        source_hashes[str(aggressive_path.relative_to(root))] = sha256(aggressive_path)
        source_hashes[str(conservative_path.relative_to(root))] = sha256(conservative_path)
        aggressive = load_ledger(aggressive_path)
        conservative = load_ledger(conservative_path)
        dates = aggressive[(1, 0)]["ds"]

        for kappa in KAPPAS:
            aggressive_variable = []
            aggressive_common = []
            conservative_common = []
            variable_budgets = []

            for seed in SEEDS:
                aggr_ledger = aggressive[(kappa, seed)]
                cons_ledger = conservative[(kappa, seed)]
                variable_budget = budget_from_validation(aggr_ledger)
                variable_budgets.append(variable_budget)
                aggr_variable = deploy(aggr_ledger, variable_budget)
                aggr_c5 = deploy(aggr_ledger, 5.0)
                cons_c5 = deploy(cons_ledger, 5.0)
                aggressive_variable.append(aggr_variable)
                aggressive_common.append(aggr_c5)
                conservative_common.append(cons_c5)

                for arm, budget_kind, result in (
                    ("aggressive", "validation", aggr_variable),
                    ("aggressive", "common5", aggr_c5),
                    ("conservative", "common5", cons_c5),
                ):
                    seed_rows.append(
                        {
                            "regime": regime,
                            "kappa": kappa,
                            "arm": arm,
                            "budget_kind": budget_kind,
                            "seed": seed,
                            "budget_bp": result["budget_bp"],
                            "mean_bp": float(result["dpnl_bp"].mean()),
                            "breach": result["breach"],
                            "breach_informed": result["breach_informed"],
                            "mean_lambda": float(result["lam"].mean()),
                            "cvar5_bp": cvar5(result["dpnl_bp"]),
                            "eps_eff_final": result["eps_eff_final"],
                        }
                    )

            variable_summary = summarize_seed_runs(aggressive_variable, dates)
            panel_rows.append(
                {
                    "regime": regime,
                    "kappa": kappa,
                    "budget_mean_bp": float(np.mean(variable_budgets)),
                    "budget_min_bp": float(np.min(variable_budgets)),
                    "budget_max_bp": float(np.max(variable_budgets)),
                    **{key: value for key, value in variable_summary.items() if key != "mean_path"},
                }
            )

            aggr_summary = summarize_seed_runs(aggressive_common, dates)
            cons_summary = summarize_seed_runs(conservative_common, dates)
            difference = aggr_summary["mean_path"] - cons_summary["mean_path"]
            difference_point, difference_low, difference_high = block_bootstrap_ci(difference, dates)
            common_rows.append(
                {
                    "regime": regime,
                    "kappa": kappa,
                    "conservative_mean_bp": cons_summary["mean_bp"],
                    "aggressive_mean_bp": aggr_summary["mean_bp"],
                    "aggressive_minus_conservative_bp": difference_point,
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                    "conservative_breach": cons_summary["breach"],
                    "aggressive_breach": aggr_summary["breach"],
                    "conservative_cvar5_bp": cons_summary["cvar5_bp"],
                    "aggressive_cvar5_bp": aggr_summary["cvar5_bp"],
                    "conservative_mean_lambda": cons_summary["mean_lambda"],
                    "aggressive_mean_lambda": aggr_summary["mean_lambda"],
                }
            )

            for arm, budget_kind, runs in (
                ("aggressive", "validation", aggressive_variable),
                ("aggressive", "common5", aggressive_common),
                ("conservative", "common5", conservative_common),
            ):
                stacked = np.stack([run["breach_indicator"] for run in runs])
                mean_indicator = stacked.mean(axis=0)
                cluster_point, cluster_upper, date_count = cluster_upper_bound(mean_indicator, dates)
                total_successes = int(stacked.sum())
                total_count = int(stacked.size)
                breach_rows.append(
                    {
                        "regime": regime,
                        "kappa": kappa,
                        "arm": arm,
                        "budget_kind": budget_kind,
                        "contract_point": float(stacked.mean()),
                        "worst_seed": float(max(run["breach"] for run in runs)),
                        "date_equal_weight_point": cluster_point,
                        "date_block_one_sided_95_upper": cluster_upper,
                        "iid_wilson_one_sided_95_upper_reference": wilson_upper(
                            total_successes, total_count
                        ),
                        "start_date_batches": date_count,
                    }
                )

        if regime == "mix_test2022":
            kappa = 2
            mode_specs = (
                ("fix1", "fix1", None),
                ("fix75", "fixed", 0.75),
                ("valstatic", "valstatic", None),
                ("ivrule", "ivrule", None),
                ("fix50", "fixed", 0.50),
                ("fix25", "fixed", 0.25),
                ("emp", "emp", None),
                ("crc", "crc", None),
                ("crc_rwc", "crc_rwc", None),
                ("fix0", "fix0", None),
                ("a2margin", "a2margin", None),
                ("crc_rwc_aci", "crc_rwc_aci", None),
            )
            for label, mode, fixed_lambda in mode_specs:
                runs = []
                budgets = []
                for seed in SEEDS:
                    ledger = aggressive[(kappa, seed)]
                    budget = budget_from_validation(ledger)
                    budgets.append(budget)
                    if mode == "valstatic":
                        lam = validation_static_lambda(ledger, budget)
                        result = deploy(ledger, budget, mode="fixed", fixed_lambda=lam)
                    else:
                        result = deploy(ledger, budget, mode=mode, fixed_lambda=fixed_lambda)
                    runs.append(result)
                summary = summarize_seed_runs(runs, dates)
                stress_rows.append(
                    {
                        "rule": label,
                        "budget_mean_bp": float(np.mean(budgets)),
                        "mean_bp": summary["mean_bp"],
                        "mean_ci_low": summary["mean_ci_low"],
                        "mean_ci_high": summary["mean_ci_high"],
                        "breach": summary["breach"],
                        "mean_lambda": summary["mean_lambda"],
                        "cvar5_bp": summary["cvar5_bp"],
                        "source": "seed-level grid ledger replay",
                    }
                )

            # aci_open deploys off the lambda grid.  Its post-warmup path is
            # unchanged by the corrected fallback because the open-loop update
            # does not ingest deployed warmup outcomes.  Therefore the exact
            # corrected aggregate is obtained by setting the 26 uninformed
            # entries of the historical mean path and lambda path to zero.
            historical_dir = (
                root / "new-reward" / "checkpoints" / "conformal_controls_mix_test2022"
            )
            dpm_path = historical_dir / "dpm_arm00.npz"
            lambda_path = historical_dir / "lampath_arm00.npz"
            controls_path = historical_dir / "controls_arms_mix_test2022.csv"
            for path in (dpm_path, lambda_path, controls_path):
                source_hashes[str(path.relative_to(root))] = sha256(path)
            representative = aggressive[(kappa, 0)]
            informed = np.zeros(representative["n"], dtype=bool)
            for step, episode in enumerate(representative["order"]):
                past = representative["order"][:step]
                informed[episode] = int(
                    np.sum(representative["de"][past] < representative["ds"][episode])
                ) >= WARMUP
            aci_path = np.load(dpm_path)[f"k{kappa}|aci_open"].astype(float).copy()
            aci_lambda = np.load(lambda_path)[f"k{kappa}|aci_open"].astype(float).copy()
            aci_path[~informed] = 0.0
            aci_lambda[~informed] = 0.0
            old_controls = pd.read_csv(controls_path)
            old_breach = float(
                old_controls.loc[
                    (old_controls["arm"] == "arm00")
                    & (old_controls["kappa"] == kappa)
                    & (old_controls["mode"] == "aci_open"),
                    "breach",
                ].iloc[0]
            )
            aci_point, aci_low, aci_high = block_bootstrap_ci(aci_path, dates)
            stress_rows.append(
                {
                    "rule": "aci_open",
                    "budget_mean_bp": float(
                        np.mean(
                            [
                                budget_from_validation(aggressive[(kappa, seed)])
                                for seed in SEEDS
                            ]
                        )
                    ),
                    "mean_bp": aci_point,
                    "mean_ci_low": aci_low,
                    "mean_ci_high": aci_high,
                    "breach": old_breach * float(informed.mean()),
                    "mean_lambda": float(aci_lambda.mean()),
                    "cvar5_bp": cvar5(aci_path),
                    "source": "exact aggregate warmup correction of sealed off-grid path",
                }
            )

    pd.DataFrame(panel_rows).to_csv(output / "table5_aggressive_validation_budget.csv", index=False)
    pd.DataFrame(common_rows).to_csv(output / "table6_common_budget.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "table4_stress_rules.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_level_results.csv", index=False)
    pd.DataFrame(breach_rows).to_csv(output / "breach_uncertainty.csv", index=False)

    manifest = {
        "version": VERSION,
        "description": "warmup BS fallback, all-episode breach, order-invariant batched ACI",
        "parameters": {
            "eps_target": EPS_TARGET,
            "warmup_completed_episodes": WARMUP,
            "warmup_lambda": 0.0,
            "gamma_eps_per_contract": GAMMA_EPS,
            "half_life_days": HALF_LIFE,
            "tau_iv": TAU_IV,
            "lambda_grid": list(LAM_GRID),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_block_unit": "consecutive start-date batches",
            "bootstrap_block_batches": BOOTSTRAP_START_BATCHES,
        },
        "source_ledgers_sha256": source_hashes,
        "outputs_sha256": {},
    }
    for path in sorted(output.glob("*.csv")):
        manifest["outputs_sha256"][path.name] = sha256(path)
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"PASS: wrote mechanism-only replay to {output}")
    print(pd.DataFrame(panel_rows).to_string(index=False))
    print(pd.DataFrame(common_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ICAIF package root",
    )
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
