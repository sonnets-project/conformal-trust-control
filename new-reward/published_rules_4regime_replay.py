"""Compare six deployment rules across four test regimes.

Use the aggressive policy (kappa=2, beta=0) with policy-specific validation
budgets. Five grid rules are evaluated from the counterfactual ledgers;
continuous ACI uses its recorded path with the v1.3 warm-up correction.
Outputs go to results/published_rules_4regime_v13/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "published_rules_4regime_v13"
KAPPA = 2
BETA = 0.0
EPS_TARGET = 0.10
REGIMES = (
    "crash_test2018",
    "crash_test2020",
    "mix_test2022",
    "mix_test2024",
)
REGIME_LABELS = {
    "crash_test2018": "2018 flash-crash",
    "crash_test2020": "2020 COVID",
    "mix_test2022": "2022 bear",
    "mix_test2024": "2024 bull",
}
REGIME_SHORT = {
    "crash_test2018": "2018",
    "crash_test2020": "2020",
    "mix_test2022": "2022",
    "mix_test2024": "2024",
}
SEEDS = (0, 1, 2, 3, 4)

# Five published-method-inspired baselines followed by the proposed rule.
RULE_SPECS = (
    ("fix1", "Full", "fix1"),
    ("ivrule", "IV", "ivrule"),
    ("aci_open", "ACI", "aci_open"),
    ("crc", "CRC", "crc"),
    ("crc_rwc", "W-CRC", "crc_rwc"),
    ("crc_rwc_aci", "CTC", "crc_rwc_aci"),
)
GRID_RULES = {"fix1", "ivrule", "crc", "crc_rwc", "crc_rwc_aci"}
PUBLISHED_BASELINES = {"fix1", "ivrule", "aci_open", "crc", "crc_rwc"}

# Regression anchor: the already accepted 2022 warm0_batch_v13 stress results.
REFERENCE_2022 = {
    "fix1": (44.481487045413814, 0.27259259259259255),
    "ivrule": (39.657668319349696, 0.19762962962962963),
    "aci_open": (30.115281470568366, 0.11437037037037037),
    "crc": (19.486293031928923, 0.06696296296296296),
    "crc_rwc": (9.604217967458275, 0.034074074074074076),
    "crc_rwc_aci": (22.45418642854525, 0.0957037037037037),
}


def _load_v13_helpers() -> dict:
    """Load the sealed replay helpers without invoking their main routine."""
    helper_path = Path(__file__).with_name("warm0_batch_replay.py")
    if not helper_path.exists():
        raise FileNotFoundError(f"Required v1.3 replay helper missing: {helper_path}")
    return runpy.run_path(str(helper_path))


V13 = _load_v13_helpers()
sha256 = V13["sha256"]
cvar5 = V13["cvar5"]
load_ledger = V13["load_ledger"]
budget_from_validation = V13["budget_from_validation"]
deploy = V13["deploy"]
summarize_seed_runs = V13["summarize_seed_runs"]
block_bootstrap_ci = V13["block_bootstrap_ci"]
WARMUP = int(V13["WARMUP"])


def _v12r_ledger_path(root: Path, regime: str) -> Path:
    return (
        root
        / "new-reward"
        / "checkpoints"
        / f"conformal_controls_v12r_{regime}"
        / f"ledger_arm00_{regime}.npz"
    )


def _historical_aci_paths(root: Path, regime: str) -> tuple[Path, Path, Path]:
    directory = root / "new-reward" / "checkpoints" / f"conformal_controls_{regime}"
    return (
        directory / "dpm_arm00.npz",
        directory / "lampath_arm00.npz",
        directory / f"controls_arms_{regime}.csv",
    )


def _preflight(root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Validate every frozen input before producing any result."""
    ledgers: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    expected_keys = {(KAPPA, seed) for seed in SEEDS}

    for regime in REGIMES:
        ledger_path = _v12r_ledger_path(root, regime)
        if not ledger_path.exists():
            raise FileNotFoundError(f"Missing aggressive v1.2R ledger: {ledger_path}")
        ledger = load_ledger(ledger_path)
        missing = expected_keys - set(ledger)
        if missing:
            raise RuntimeError(f"{regime}: missing policy-seed keys {sorted(missing)}")

        representative = ledger[(KAPPA, SEEDS[0])]
        n = int(representative["n"])
        order = representative["order"]
        dates = representative["ds"]
        if n <= 0 or len(order) != n or len(dates) != n:
            raise RuntimeError(f"{regime}: inconsistent episode arrays")

        for seed in SEEDS:
            item = ledger[(KAPPA, seed)]
            if int(item["n"]) != n:
                raise RuntimeError(f"{regime}, seed {seed}: episode count differs")
            for store_name in ("d_lam", "dp_lam", "vadp"):
                if set(map(float, item[store_name])) != set(V13["LAM_GRID"]):
                    raise RuntimeError(
                        f"{regime}, seed {seed}: {store_name} has wrong lambda grid"
                    )

        dpm_path, lambda_path, controls_path = _historical_aci_paths(root, regime)
        for path in (dpm_path, lambda_path, controls_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing sealed ACI input: {path}")
        key = f"k{KAPPA}|aci_open"
        if key not in np.load(dpm_path).files or key not in np.load(lambda_path).files:
            raise RuntimeError(f"{regime}: sealed ACI path lacks {key}")

        ledgers[regime] = ledger
        for path in (ledger_path, dpm_path, lambda_path, controls_path):
            hashes[str(path.relative_to(root))] = sha256(path)

    return ledgers, hashes


def _informed_mask(representative: dict) -> np.ndarray:
    """Episodes with at least WARMUP strictly earlier completed episodes."""
    informed = np.zeros(int(representative["n"]), dtype=bool)
    order = representative["order"]
    ds, de = representative["ds"], representative["de"]
    for step, episode in enumerate(order):
        past = order[:step]
        done = past[de[past] < ds[episode]]
        informed[episode] = len(done) >= WARMUP
    return informed


def _corrected_aci_open(root: Path, regime: str, ledger: dict, dates: np.ndarray) -> dict:
    """Apply the accepted v1.3 correction to a sealed continuous ACI path."""
    dpm_path, lambda_path, controls_path = _historical_aci_paths(root, regime)
    key = f"k{KAPPA}|aci_open"
    path = np.load(dpm_path)[key].astype(float).copy()
    lambda_values = np.load(lambda_path)[key].astype(float).copy()
    representative = ledger[(KAPPA, SEEDS[0])]
    informed = _informed_mask(representative)
    if len(path) != representative["n"] or len(lambda_values) != representative["n"]:
        raise RuntimeError(f"{regime}: ACI path length differs from the ledger")

    # Exact BS fallback during the uninformed prefix: relative PnL and trust are zero.
    path[~informed] = 0.0
    lambda_values[~informed] = 0.0

    controls = pd.read_csv(controls_path)
    selected = controls.loc[
        (controls["arm"] == "arm00")
        & (controls["kappa"] == KAPPA)
        & (controls["mode"] == "aci_open")
    ]
    if len(selected) != 1:
        raise RuntimeError(f"{regime}: expected one aggregate aci_open control row")

    # Historical v1.2 breach was computed on the informed subset.  Multiplying
    # by its share puts the zero-breach fallback prefix back into the denominator.
    breach_all = float(selected["breach"].iloc[0]) * float(informed.mean())
    point, low, high = block_bootstrap_ci(path, dates)
    return {
        "mean_bp": point,
        "mean_ci_low": low,
        "mean_ci_high": high,
        "breach": breach_all,
        "mean_lambda": float(lambda_values.mean()),
        "cvar5_bp": cvar5(path),
        "mean_path": path,
        "informed_share": float(informed.mean()),
    }


def _aggregate_grid_rule(
    regime: str,
    rule: str,
    ledger: dict,
    dates: np.ndarray,
) -> tuple[dict, list[dict], list[float]]:
    runs = []
    seed_rows = []
    budgets = []
    for seed in SEEDS:
        item = ledger[(KAPPA, seed)]
        budget = budget_from_validation(item)
        result = deploy(item, budget, mode=rule)
        budgets.append(float(budget))
        runs.append(result)
        seed_rows.append(
            {
                "regime": REGIME_LABELS[regime],
                "regime_key": regime,
                "kappa": KAPPA,
                "beta": BETA,
                "rule": rule,
                "seed": seed,
                "aggregation_level": "seed",
                "budget_bp": float(budget),
                "n_episodes": int(item["n"]),
                "mean_dpnl_bp": float(result["dpnl_bp"].mean()),
                "breach": float(result["breach"]),
                "breach_informed": float(result["breach_informed"]),
                "mean_lambda": float(result["lam"].mean()),
                "cvar5_bp": cvar5(result["dpnl_bp"]),
                "source": "seed-level v1.2R grid-ledger replay",
            }
        )
    return summarize_seed_runs(runs, dates), seed_rows, budgets


def _panel(root: Path, ledgers: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict] = []
    seed_rows: list[dict] = []

    for regime in REGIMES:
        ledger = ledgers[regime]
        dates = ledger[(KAPPA, SEEDS[0])]["ds"]
        budgets = [budget_from_validation(ledger[(KAPPA, seed)]) for seed in SEEDS]

        for rule, display_name, mode in RULE_SPECS:
            if rule in GRID_RULES:
                summary, rows, rule_budgets = _aggregate_grid_rule(
                    regime, mode, ledger, dates
                )
                seed_rows.extend(rows)
                source = "five-seed v1.2R grid-ledger replay"
                budgets_used = rule_budgets
            else:
                summary = _corrected_aci_open(root, regime, ledger, dates)
                source = "sealed off-grid path with v1.3 warm-up correction"
                budgets_used = budgets

            aggregate_rows.append(
                {
                    "regime": REGIME_LABELS[regime],
                    "regime_key": regime,
                    "regime_year": int(REGIME_SHORT[regime]),
                    "kappa": KAPPA,
                    "beta": BETA,
                    "rule": rule,
                    "display_name": display_name,
                    "budget_mean_bp": float(np.mean(budgets_used)),
                    "budget_min_bp": float(np.min(budgets_used)),
                    "budget_max_bp": float(np.max(budgets_used)),
                    "n_episodes": int(len(dates)),
                    "mean_dpnl_bp": float(summary["mean_bp"]),
                    "mean_ci_low": float(summary["mean_ci_low"]),
                    "mean_ci_high": float(summary["mean_ci_high"]),
                    "breach": float(summary["breach"]),
                    "feasible_point_estimate": bool(summary["breach"] <= EPS_TARGET),
                    "mean_lambda": float(summary["mean_lambda"]),
                    "cvar5_bp": float(summary["cvar5_bp"]),
                    "source": source,
                }
            )

    panel = pd.DataFrame(aggregate_rows)
    seed_level = pd.DataFrame(seed_rows)
    return panel, seed_level


def _best_published(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        sub = panel[panel["regime_key"] == regime]
        baselines = sub[
            sub["rule"].isin(PUBLISHED_BASELINES)
            & sub["feasible_point_estimate"]
        ]
        ctc = sub[sub["rule"] == "crc_rwc_aci"].iloc[0]
        if baselines.empty:
            rows.append(
                {
                    "regime": REGIME_LABELS[regime],
                    "regime_key": regime,
                    "best_feasible_rule": "none",
                    "baseline_mean_bp": np.nan,
                    "baseline_breach": np.nan,
                    "ctc_mean_bp": float(ctc["mean_dpnl_bp"]),
                    "ctc_breach": float(ctc["breach"]),
                    "ctc_minus_baseline_bp": np.nan,
                    "relative_improvement_pct": np.nan,
                }
            )
            continue
        best = baselines.sort_values("mean_dpnl_bp", ascending=False).iloc[0]
        difference = float(ctc["mean_dpnl_bp"] - best["mean_dpnl_bp"])
        relative = (
            100.0 * difference / float(best["mean_dpnl_bp"])
            if float(best["mean_dpnl_bp"]) > 0
            else np.nan
        )
        rows.append(
            {
                "regime": REGIME_LABELS[regime],
                "regime_key": regime,
                "best_feasible_rule": best["display_name"],
                "baseline_mean_bp": float(best["mean_dpnl_bp"]),
                "baseline_breach": float(best["breach"]),
                "ctc_mean_bp": float(ctc["mean_dpnl_bp"]),
                "ctc_breach": float(ctc["breach"]),
                "ctc_minus_baseline_bp": difference,
                "relative_improvement_pct": relative,
            }
        )
    return pd.DataFrame(rows)


def _write_plot(panel: pd.DataFrame, output: Path) -> list[Path]:
    """Create an optional four-panel trade-off figure from the same CSV."""
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "icaif_mpl_cache"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - core CSV output must still work
        print(f"WARNING: matplotlib unavailable; skipping plot ({exc})")
        return []

    colors = {
        "fix1": "#7f7f7f",
        "ivrule": "#e69f00",
        "aci_open": "#cc79a7",
        "crc": "#56b4e9",
        "crc_rwc": "#009e73",
        "crc_rwc_aci": "#0072b2",
    }
    markers = {
        "fix1": "o",
        "ivrule": "s",
        "aci_open": "D",
        "crc": "^",
        "crc_rwc": "v",
        "crc_rwc_aci": "*",
    }
    offsets = {
        "fix1": (-28, 7),
        "ivrule": (6, -11),
        "aci_open": (6, 7),
        "crc": (6, 8),
        "crc_rwc": (6, 7),
        "crc_rwc_aci": (6, -15),
    }
    offset_overrides = {
        ("crash_test2018", "ivrule"): (6, -14),
        ("crash_test2018", "crc"): (6, 10),
        ("crash_test2020", "crc"): (6, -14),
        ("mix_test2022", "crc"): (-28, 8),
        ("mix_test2024", "crc"): (-28, 10),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    for ax, regime in zip(axes.flat, REGIMES):
        sub = panel[panel["regime_key"] == regime]
        ax.axvline(EPS_TARGET, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        for _, row in sub.iterrows():
            rule = str(row["rule"])
            size = 90 if rule == "crc_rwc_aci" else 48
            ax.scatter(
                row["breach"],
                row["mean_dpnl_bp"],
                s=size,
                marker=markers[rule],
                color=colors[rule],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            dx, dy = offset_overrides.get((regime, rule), offsets[rule])
            ax.annotate(
                str(row["display_name"]),
                (row["breach"], row["mean_dpnl_bp"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_title(REGIME_LABELS[regime], fontsize=9)
        ax.margins(x=0.08, y=0.12)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Realized breach frequency")
        ax.set_ylabel(r"Mean $\Delta$PnL vs. Black-Scholes (bp)")
    png = output / "published_rules_tradeoff.png"
    pdf = output / "published_rules_tradeoff.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _regression_gate(panel: pd.DataFrame) -> None:
    sub = panel[panel["regime_key"] == "mix_test2022"].set_index("rule")
    for rule, (expected_mean, expected_breach) in REFERENCE_2022.items():
        actual_mean = float(sub.loc[rule, "mean_dpnl_bp"])
        actual_breach = float(sub.loc[rule, "breach"])
        if not math.isclose(actual_mean, expected_mean, rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(
                f"2022 regression failed for {rule} mean: {actual_mean} != {expected_mean}"
            )
        if not math.isclose(actual_breach, expected_breach, rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(
                f"2022 regression failed for {rule} breach: "
                f"{actual_breach} != {expected_breach}"
            )


def _general_gates(panel: pd.DataFrame, seed_level: pd.DataFrame) -> None:
    if len(panel) != len(REGIMES) * len(RULE_SPECS):
        raise AssertionError(f"Expected 24 aggregate rows, found {len(panel)}")
    if len(seed_level) != len(REGIMES) * len(GRID_RULES) * len(SEEDS):
        raise AssertionError(f"Expected 100 seed-level grid rows, found {len(seed_level)}")
    if not panel["breach"].between(0.0, 1.0).all():
        raise AssertionError("A breach frequency lies outside [0,1]")
    if not (panel["budget_min_bp"] >= 5.0).all():
        raise AssertionError("A validation-selected budget is below the 5 bp floor")
    full = panel[panel["rule"] == "fix1"]
    if not np.allclose(full["mean_lambda"].to_numpy(), 1.0):
        raise AssertionError("fix1 does not have mean lambda exactly one")
    for regime in REGIMES:
        sub = panel[panel["regime_key"] == regime]
        if sub["rule"].nunique() != len(RULE_SPECS):
            raise AssertionError(f"{regime}: duplicate or missing deployment rule")
        if not np.allclose(sub["budget_mean_bp"], sub["budget_mean_bp"].iloc[0]):
            raise AssertionError(f"{regime}: rules do not share the same seed budgets")


def run(root: Path, *, check_reference: bool = False) -> dict[str, Path]:
    root = Path(root).resolve()
    required = root / "new-reward" / "warm0_batch_replay.py"
    if not required.exists():
        raise FileNotFoundError(f"ICAIF package root is incorrect: {root}")

    output = root / "results" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ICAIF four-regime published-rule replay")
    print(f"root: {root}")
    print("NO TRAINING: frozen beta=0 policies, kappa=2, ledger replay only")
    print("=" * 72)

    ledgers, hashes_before = _preflight(root)
    print("PASS: all four v1.2R ledgers and sealed ACI paths are present")

    panel, seed_level = _panel(root, ledgers)
    _general_gates(panel, seed_level)
    if check_reference:
        _regression_gate(panel)
        print("PASS: 2022 results match the paper reference")

    best = _best_published(panel)
    panel_path = output / "published_rules_panel.csv"
    seed_path = output / "published_rules_seed_level.csv"
    best_path = output / "best_published_feasible.csv"
    panel.to_csv(panel_path, index=False)
    seed_level.to_csv(seed_path, index=False)
    best.to_csv(best_path, index=False)
    plot_paths = _write_plot(panel, output)

    # Re-hash every frozen input after the run: the experiment is replay-only.
    _, hashes_after = _preflight(root)
    if hashes_before != hashes_after:
        raise AssertionError("A frozen input changed during the replay")

    output_files = [panel_path, seed_path, best_path, *plot_paths]
    manifest = {
        "version": VERSION,
        "description": (
            "Four regimes, kappa=2, aggressive beta=0, six deployment rules; "
            "frozen-policy replay only"
        ),
        "training_performed": False,
        "parameters": {
            "regimes": list(REGIMES),
            "kappa": KAPPA,
            "beta": BETA,
            "seeds": list(SEEDS),
            "rules": [spec[0] for spec in RULE_SPECS],
            "eps_target": EPS_TARGET,
            "warmup_completed_episodes": WARMUP,
            "warmup_lambda": 0.0,
            "lambda_grid": list(V13["LAM_GRID"]),
            "gamma_eps_per_contract_ctc": float(V13["GAMMA_EPS"]),
            "half_life_days": float(V13["HALF_LIFE"]),
            "tau_iv": float(V13["TAU_IV"]),
            "bootstrap_reps": int(V13["BOOTSTRAP_REPS"]),
            "bootstrap_block_batches": int(V13["BOOTSTRAP_START_BATCHES"]),
            "budget": "per-policy-seed validation 10% quantile with 5 bp floor",
        },
        "source_sha256": hashes_before,
        "output_sha256": {
            path.name: sha256(path) for path in output_files
        },
        "acceptance_gates": {
            "aggregate_rows": int(len(panel)),
            "seed_level_grid_rows": int(len(seed_level)),
            "2022_regression": "pass" if check_reference else "not requested",
            "frozen_inputs_unchanged": True,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n24-row aggregate panel:")
    print(
        panel[
            ["regime", "display_name", "mean_dpnl_bp", "breach", "mean_lambda"]
        ].to_string(index=False)
    )
    print("\nBest feasible published-method-inspired baseline by regime:")
    print(best.to_string(index=False))
    print(f"\nPASS: all outputs written to {output}")

    return {
        "output_dir": output,
        "panel_csv": panel_path,
        "seed_csv": seed_path,
        "best_csv": best_path,
        "manifest": manifest_path,
        "plot_png": output / "published_rules_tradeoff.png",
        "plot_pdf": output / "published_rules_tradeoff.pdf",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ICAIF package root",
    )
    parser.add_argument("--check-reference", action="store_true", help="Require equality to the published 2022 results")
    args = parser.parse_args()
    run(args.root, check_reference=args.check_reference)


if __name__ == "__main__":
    main()
