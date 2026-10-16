"""Evaluate controller sensitivity using saved aggressive-policy episodes.

At kappa=2, vary the breach target, warm-up size, recency half-life, IV scale
and ACI step size one at a time across four regimes. The default configuration
is compared with the validation-budget panel before the other settings run.
Outputs go to results/v13_sensitivity/.
"""

from __future__ import annotations

import argparse
import json
import runpy
import time
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


VERSION = "v13_sensitivity"
REGIMES = (
    "crash_test2018",
    "crash_test2020",
    "mix_test2022",
    "mix_test2024",
)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_KAPPA = 2
DEFAULT_BOOTSTRAP_REPS = 3000
DEFAULT_BOOTSTRAP_BLOCK_BATCHES = 10
REGRESSION_ATOL = 1e-12


@dataclass(frozen=True)
class ControllerConfig:
    """All deployment parameters that may affect the v1.3 trust path."""

    eps_target: float = 0.10
    warmup: int = 15
    gamma_eps: float = 0.02
    half_life_days: float = 30.0
    tau_iv: float = 0.04
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    m_emp: float = 0.3

    def validate(self) -> None:
        if not 0.0 < self.eps_target < 1.0:
            raise ValueError("eps_target must lie strictly between zero and one")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.gamma_eps < 0.0:
            raise ValueError("gamma_eps must be non-negative")
        if self.half_life_days <= 0.0:
            raise ValueError("half_life_days must be positive")
        if self.tau_iv <= 0.0:
            raise ValueError("tau_iv must be positive")
        if self.m_emp < 0.0:
            raise ValueError("m_emp must be non-negative")
        grid = tuple(float(value) for value in self.lambda_grid)
        if not grid:
            raise ValueError("lambda_grid must not be empty")
        if grid != tuple(sorted(set(grid))):
            raise ValueError("lambda_grid must be unique and sorted")
        if grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("lambda_grid must contain exact fallback 0 and full trust 1")
        if any(value < 0.0 or value > 1.0 for value in grid):
            raise ValueError("lambda_grid values must lie in [0, 1]")


@dataclass(frozen=True)
class SensitivitySpec:
    config_id: str
    parameter: str
    parameter_value: float | None
    config: ControllerConfig


def _load_sealed_v13(root: Path) -> dict:
    path = root / "new-reward" / "warm0_batch_replay.py"
    if not path.is_file():
        raise FileNotFoundError(f"sealed v1.3 replay is missing: {path}")
    namespace = runpy.run_path(str(path))
    required = {
        "sha256",
        "cvar5",
        "load_ledger",
        "deploy",
        "block_bootstrap_ci",
        "cluster_upper_bound",
    }
    missing = required - set(namespace)
    if missing:
        raise RuntimeError(f"sealed v1.3 replay lacks helpers: {sorted(missing)}")
    return namespace


def configured_deploy(base_deploy: Callable, config: ControllerConfig) -> Callable:
    """Clone the sealed deploy bytecode with a private configured global state.

    ``warm0_batch_replay.py`` is hash-bound into the saved lineage.  Reimplementing
    the controller here would create a second algorithm that could silently drift.
    Cloning the existing function preserves its code object while giving each
    sensitivity run an isolated set of constants.
    """

    config.validate()
    namespace = dict(base_deploy.__globals__)
    namespace.update(
        {
            "EPS_TARGET": float(config.eps_target),
            "WARMUP": int(config.warmup),
            "GAMMA_EPS": float(config.gamma_eps),
            "HALF_LIFE": float(config.half_life_days),
            "TAU_IV": float(config.tau_iv),
            "LAM_GRID": tuple(float(value) for value in config.lambda_grid),
            "M_EMP": float(config.m_emp),
        }
    )
    cloned = types.FunctionType(
        base_deploy.__code__,
        namespace,
        name=f"deploy_{_config_token(config)}",
        argdefs=base_deploy.__defaults__,
        closure=base_deploy.__closure__,
    )
    cloned.__kwdefaults__ = base_deploy.__kwdefaults__
    cloned.__doc__ = base_deploy.__doc__
    return cloned


def validation_budget(
    ledger: dict,
    config: ControllerConfig,
    floor_bp: float = 5.0,
) -> float:
    """Recalibrate the validation budget at the configured target probability."""

    full_trust = ledger["vadp"].get(1.0)
    if full_trust is None:
        raise RuntimeError("validation ledger has no full-trust candidate")
    return float(max(floor_bp, -np.quantile(full_trust, config.eps_target)))


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _config_token(config: ControllerConfig) -> str:
    return "_".join(
        (
            f"e{_float_token(config.eps_target)}",
            f"w{config.warmup}",
            f"g{_float_token(config.gamma_eps)}",
            f"h{_float_token(config.half_life_days)}",
            f"t{_float_token(config.tau_iv)}",
        )
    )


def sensitivity_specs(default: ControllerConfig) -> list[SensitivitySpec]:
    """Return the pre-specified one-factor-at-a-time experiment grid."""

    specs = [SensitivitySpec("baseline", "baseline", None, default)]
    grids = (
        ("eps_target", (0.05, 0.10, 0.20)),
        ("warmup", (10, 15, 30)),
        ("half_life_days", (15.0, 30.0, 60.0)),
        ("tau_iv", (0.02, 0.04, 0.08)),
        ("gamma_eps", (0.01, 0.02, 0.04)),
    )
    for parameter, values in grids:
        for value in values:
            typed_value = int(value) if parameter == "warmup" else float(value)
            if typed_value == getattr(default, parameter):
                # The baseline is evaluated once and reused as the centre point
                # of every one-factor panel; do not rerun identical deployments.
                continue
            config = replace(default, **{parameter: typed_value})
            specs.append(
                SensitivitySpec(
                    config_id=f"{parameter}_{_float_token(float(value))}",
                    parameter=parameter,
                    parameter_value=float(value),
                    config=config,
                )
            )
    if len({spec.config_id for spec in specs}) != len(specs):
        raise RuntimeError("sensitivity configuration identifiers are not unique")
    return specs


def aggressive_ledger_path(root: Path, regime: str) -> Path:
    path = (
        root
        / "new-reward"
        / "checkpoints"
        / f"conformal_controls_v12r_{regime}"
        / f"ledger_arm00_{regime}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"sealed aggressive ledger is missing: {path}")
    return path


def _summarize_runs(
    runs: list[dict],
    dates: np.ndarray,
    cvar5: Callable,
    block_bootstrap_ci: Callable,
    cluster_upper_bound: Callable,
    bootstrap_reps: int,
    bootstrap_block_batches: int,
) -> dict:
    mean_path = np.mean([run["dpnl_bp"] for run in runs], axis=0)
    point, low, high = block_bootstrap_ci(
        mean_path,
        dates,
        reps=bootstrap_reps,
        block_batches=bootstrap_block_batches,
        seed=0,
    )
    mean_indicator = np.mean([run["breach_indicator"] for run in runs], axis=0)
    date_point, date_upper, date_count = cluster_upper_bound(
        mean_indicator,
        dates,
        reps=bootstrap_reps,
        block_batches=bootstrap_block_batches,
        seed=0,
    )
    return {
        "mean_bp": point,
        "mean_ci_low": low,
        "mean_ci_high": high,
        "breach": float(np.mean([run["breach"] for run in runs])),
        "breach_informed": float(np.mean([run["breach_informed"] for run in runs])),
        "worst_seed_breach": float(max(run["breach"] for run in runs)),
        "date_equal_weight_breach": date_point,
        "date_block_one_sided_95_upper": date_upper,
        "start_date_batches": int(date_count),
        "mean_lambda": float(np.mean([run["lam"].mean() for run in runs])),
        "cvar5_bp": float(cvar5(mean_path)),
    }


def _add_baseline_differences(panel: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        panel.loc[panel["config_id"].eq("baseline")]
        .set_index("regime")
        .sort_index()
    )
    if len(baseline) != len(REGIMES):
        raise RuntimeError("baseline panel does not contain exactly four regimes")
    panel = panel.copy()
    panel["mean_change_vs_baseline_bp"] = panel.apply(
        lambda row: row["mean_bp"] - baseline.loc[row["regime"], "mean_bp"], axis=1
    )
    panel["breach_change_vs_baseline"] = panel.apply(
        lambda row: row["breach"] - baseline.loc[row["regime"], "breach"], axis=1
    )
    panel["cvar5_change_vs_baseline_bp"] = panel.apply(
        lambda row: row["cvar5_bp"] - baseline.loc[row["regime"], "cvar5_bp"], axis=1
    )
    return panel


def _configuration_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, part in panel.groupby("config_id", sort=False):
        first = part.iloc[0]
        rows.append(
            {
                "config_id": config_id,
                "parameter": first["parameter"],
                "parameter_value": first["parameter_value"],
                "eps_target": first["eps_target"],
                "warmup": int(first["warmup"]),
                "gamma_eps": first["gamma_eps"],
                "half_life_days": first["half_life_days"],
                "tau_iv": first["tau_iv"],
                "mean_across_regimes_bp": float(part["mean_bp"].mean()),
                "minimum_regime_mean_bp": float(part["mean_bp"].min()),
                "average_breach": float(part["breach"].mean()),
                "maximum_regime_breach": float(part["breach"].max()),
                "maximum_breach_target_gap": float(part["breach_target_gap"].max()),
                "maximum_date_block_one_sided_95_upper": float(
                    part["date_block_one_sided_95_upper"].max()
                ),
                "maximum_date_block_upper_target_gap": float(
                    part["date_block_upper_target_gap"].max()
                ),
                "minimum_regime_cvar5_bp": float(part["cvar5_bp"].min()),
                "average_mean_lambda": float(part["mean_lambda"].mean()),
                "all_regimes_point_feasible": bool(part["point_feasible"].all()),
                "all_regimes_upper_bound_feasible": bool(
                    part["upper_bound_feasible"].all()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    baseline_mean = float(
        summary.loc[summary["config_id"].eq("baseline"), "mean_across_regimes_bp"].iloc[0]
    )
    baseline_worst_mean = float(
        summary.loc[
            summary["config_id"].eq("baseline"), "minimum_regime_mean_bp"
        ].iloc[0]
    )
    summary["mean_change_vs_baseline_bp"] = (
        summary["mean_across_regimes_bp"] - baseline_mean
    )
    summary["minimum_mean_change_vs_baseline_bp"] = (
        summary["minimum_regime_mean_bp"] - baseline_worst_mean
    )
    return summary


def _assert_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=REGRESSION_ATOL):
        raise AssertionError(
            f"default v1.3 regression failed for {label}: "
            f"actual={actual:.17g}, expected={expected:.17g}"
        )


def verify_default_regression(root: Path, panel: pd.DataFrame, kappa: int) -> dict:
    """Require the configured default to match the accepted v1.3 panel."""

    reference_path = (
        root
        / "results"
        / "warm0_batch_v13"
        / "table5_aggressive_validation_budget.csv"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"accepted v1.3 reference panel is missing: {reference_path}")
    reference = pd.read_csv(reference_path)
    reference = reference.loc[reference["kappa"].eq(kappa)].set_index("regime")
    baseline = panel.loc[panel["config_id"].eq("baseline")].set_index("regime")
    if set(reference.index) != set(baseline.index):
        raise AssertionError("default regression regime set differs from v1.3 reference")

    column_map = {
        "budget_mean_bp": "budget_mean_bp",
        "budget_min_bp": "budget_min_bp",
        "budget_max_bp": "budget_max_bp",
        "mean_bp": "mean_bp",
        "mean_ci_low": "mean_ci_low",
        "mean_ci_high": "mean_ci_high",
        "breach": "breach",
        "breach_informed": "breach_informed",
        "mean_lambda": "mean_lambda",
        "cvar5_bp": "cvar5_bp",
    }
    checks = 0
    for regime in REGIMES:
        for actual_column, expected_column in column_map.items():
            _assert_close(
                f"{regime}.{actual_column}",
                float(baseline.loc[regime, actual_column]),
                float(reference.loc[regime, expected_column]),
            )
            checks += 1
    return {
        "status": "PASS",
        "reference": str(reference_path.relative_to(root)),
        "kappa": kappa,
        "regimes": len(REGIMES),
        "numeric_checks": checks,
        "absolute_tolerance": REGRESSION_ATOL,
    }


def _plot_summary(summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    """Create a compact return/risk sensitivity view for Chapter 6."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parameters = (
        "eps_target",
        "warmup",
        "half_life_days",
        "tau_iv",
        "gamma_eps",
    )
    labels = {
        "eps_target": r"Target $\epsilon$",
        "warmup": "Warm-up episodes",
        "half_life_days": "Recency half-life (days)",
        "tau_iv": r"IV scale $\tau_{IV}$",
        "gamma_eps": r"ACI step $\gamma$",
    }
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    default = ControllerConfig()
    default_values = asdict(default)
    baseline = summary.loc[summary["config_id"].eq("baseline")].iloc[0]

    for axis, parameter in zip(axes.flat, parameters):
        baseline_point = baseline.copy()
        baseline_point["parameter_value"] = float(default_values[parameter])
        part = pd.concat(
            [
                summary.loc[summary["parameter"].eq(parameter)],
                baseline_point.to_frame().T,
            ],
            ignore_index=True,
        ).sort_values(
            "parameter_value"
        )
        x = part["parameter_value"].to_numpy(float)
        reward = part["minimum_regime_mean_bp"].to_numpy(float)
        point_gap = 100.0 * part["maximum_breach_target_gap"].to_numpy(float)
        upper_gap = 100.0 * part["maximum_date_block_upper_target_gap"].to_numpy(float)
        risk_axis = axis.twinx()
        axis.plot(x, reward, "o-", color="#1f77b4", linewidth=1.5, markersize=5)
        risk_axis.plot(x, point_gap, "s-", color="#ff7f0e", linewidth=1.3, markersize=4)
        risk_axis.plot(
            x,
            upper_gap,
            "^--",
            color="#9467bd",
            linewidth=1.2,
            markersize=4,
        )
        risk_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        default_value = float(default_values[parameter])
        axis.axvline(default_value, color="#d62728", linestyle=":", linewidth=1.4)
        axis.set_title(labels[parameter])
        axis.set_xlabel(labels[parameter])
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:g}" for value in x])
        axis.set_ylabel("Minimum regime mean improvement (bp)")
        risk_axis.set_ylabel("Worst-regime target gap (percentage points)")
        axis.grid(alpha=0.22)

    axes.flat[-1].axis("off")
    legend_handles = (
        Line2D([0], [0], color="#1f77b4", marker="o", label="minimum regime mean"),
        Line2D([0], [0], color="#ff7f0e", marker="s", label="maximum point-estimate gap"),
        Line2D(
            [0],
            [0],
            color="#9467bd",
            marker="^",
            linestyle="--",
            label="maximum one-sided 95% upper-gap",
        ),
        Line2D([0], [0], color="#d62728", linestyle=":", label="reported default"),
    )
    axes.flat[-1].legend(handles=legend_handles, loc="center", frameon=False)
    figure.suptitle(
        "Final v1.3 controller: one-factor-at-a-time sensitivity",
        fontsize=14,
    )
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    root: Path,
    output: Path,
    kappa: int,
    bootstrap_reps: int,
    bootstrap_block_batches: int,
    make_plot: bool,
) -> None:
    if kappa not in (1, 2, 3):
        raise ValueError("kappa must be one of 1, 2, or 3")
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    if bootstrap_block_batches <= 0:
        raise ValueError("bootstrap_block_batches must be positive")

    started = time.perf_counter()
    v13 = _load_sealed_v13(root)
    sha256 = v13["sha256"]
    cvar5 = v13["cvar5"]
    load_ledger = v13["load_ledger"]
    base_deploy = v13["deploy"]
    block_bootstrap_ci = v13["block_bootstrap_ci"]
    cluster_upper_bound = v13["cluster_upper_bound"]

    default = ControllerConfig()
    specs = sensitivity_specs(default)
    ledgers = {}
    source_hashes = {
        "new-reward/v13_sensitivity_replay.py": sha256(
            root / "new-reward" / "v13_sensitivity_replay.py"
        ),
        "new-reward/warm0_batch_replay.py": sha256(
            root / "new-reward" / "warm0_batch_replay.py"
        )
    }
    for regime in REGIMES:
        path = aggressive_ledger_path(root, regime)
        ledgers[regime] = load_ledger(path)
        source_hashes[str(path.relative_to(root))] = sha256(path)

    seed_rows: list[dict] = []
    panel_rows: list[dict] = []
    for spec in specs:
        deploy = configured_deploy(base_deploy, spec.config)
        config_values = asdict(spec.config)
        config_values["lambda_grid"] = list(spec.config.lambda_grid)

        for regime in REGIMES:
            runs = []
            budgets = []
            dates = ledgers[regime][(kappa, SEEDS[0])]["ds"]
            start_date_batches = int(len(np.unique(dates)))
            for seed in SEEDS:
                ledger = ledgers[regime][(kappa, seed)]
                budget = validation_budget(ledger, spec.config)
                result = deploy(ledger, budget, mode="crc_rwc_aci")
                budgets.append(budget)
                runs.append(result)
                seed_rows.append(
                    {
                        "config_id": spec.config_id,
                        "parameter": spec.parameter,
                        "parameter_value": spec.parameter_value,
                        "regime": regime,
                        "kappa": kappa,
                        "seed": seed,
                        "budget_bp": budget,
                        "n_episodes": int(ledger["n"]),
                        "start_date_batches": start_date_batches,
                        "mean_bp": float(result["dpnl_bp"].mean()),
                        "breach": float(result["breach"]),
                        "breach_informed": float(result["breach_informed"]),
                        "cvar5_bp": float(cvar5(result["dpnl_bp"])),
                        "mean_lambda": float(result["lam"].mean()),
                        "eps_eff_final": float(result["eps_eff_final"]),
                        **{
                            key: value
                            for key, value in config_values.items()
                            if key != "lambda_grid"
                        },
                    }
                )

            summary = _summarize_runs(
                runs,
                dates,
                cvar5,
                block_bootstrap_ci,
                cluster_upper_bound,
                bootstrap_reps,
                bootstrap_block_batches,
            )
            panel_rows.append(
                {
                    "config_id": spec.config_id,
                    "parameter": spec.parameter,
                    "parameter_value": spec.parameter_value,
                    "regime": regime,
                    "kappa": kappa,
                    "budget_mean_bp": float(np.mean(budgets)),
                    "budget_min_bp": float(np.min(budgets)),
                    "budget_max_bp": float(np.max(budgets)),
                    **summary,
                    **{
                        key: value
                        for key, value in config_values.items()
                        if key != "lambda_grid"
                    },
                }
            )

    panel = pd.DataFrame(panel_rows)
    panel["breach_target_gap"] = panel["breach"] - panel["eps_target"]
    panel["date_block_upper_target_gap"] = (
        panel["date_block_one_sided_95_upper"] - panel["eps_target"]
    )
    panel["point_feasible"] = panel["breach_target_gap"] <= REGRESSION_ATOL
    panel["upper_bound_feasible"] = (
        panel["date_block_upper_target_gap"] <= REGRESSION_ATOL
    )
    panel = _add_baseline_differences(panel)
    seed_level = pd.DataFrame(seed_rows)
    summary = _configuration_summary(panel)
    regression = verify_default_regression(root, panel, kappa)
    reference_path = root / regression["reference"]
    source_hashes[regression["reference"]] = sha256(reference_path)

    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "sensitivity_seed_level.csv"
    panel_path = output / "sensitivity_panel.csv"
    summary_path = output / "sensitivity_summary.csv"
    seed_level.to_csv(seed_path, index=False)
    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)

    plotted = []
    if make_plot:
        pdf_path = output / "sensitivity_worst_regime.pdf"
        png_path = output / "sensitivity_worst_regime.png"
        _plot_summary(summary, pdf_path, png_path)
        plotted = [pdf_path, png_path]

    elapsed = time.perf_counter() - started
    output_files = [seed_path, panel_path, summary_path, *plotted]
    manifest = {
        "version": VERSION,
        "description": (
            "One-factor-at-a-time replay of the final warm0, all-episode, "
            "batched-ACI v1.3 controller; frozen aggressive policies; no training"
        ),
        "training_performed": False,
        "kappa": kappa,
        "regimes": list(REGIMES),
        "seeds": list(SEEDS),
        "baseline_config": asdict(default),
        "configurations": [
            {
                "config_id": spec.config_id,
                "parameter": spec.parameter,
                "parameter_value": spec.parameter_value,
                "config": asdict(spec.config),
            }
            for spec in specs
        ],
        "bootstrap": {
            "reps": bootstrap_reps,
            "block_unit": "consecutive start-date batches",
            "block_batches": bootstrap_block_batches,
            "seed": 0,
        },
        "default_regression": regression,
        "source_sha256": source_hashes,
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "elapsed_seconds": elapsed,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(
        f"PASS: {len(specs)} configurations x {len(REGIMES)} regimes x "
        f"{len(SEEDS)} seeds; default v1.3 regression matched {regression['numeric_checks']} "
        f"values; wrote {output} in {elapsed:.2f}s"
    )
    print(
        summary[
            [
                "config_id",
                "minimum_regime_mean_bp",
                "maximum_regime_breach",
                "maximum_breach_target_gap",
                "all_regimes_point_feasible",
                "maximum_date_block_upper_target_gap",
                "all_regimes_upper_bound_feasible",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="final_method package root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory; default: <root>/results/v13_sensitivity",
    )
    parser.add_argument("--kappa", type=int, default=DEFAULT_KAPPA)
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )
    parser.add_argument(
        "--bootstrap-block-batches",
        type=int,
        default=DEFAULT_BOOTSTRAP_BLOCK_BATCHES,
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="write CSV and manifest outputs without the PDF/PNG summary figure",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "results" / VERSION
    )
    run(
        root=root,
        output=output,
        kappa=args.kappa,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_block_batches=args.bootstrap_block_batches,
        make_plot=not args.skip_plot,
    )


if __name__ == "__main__":
    main()
