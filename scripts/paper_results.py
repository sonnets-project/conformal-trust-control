"""Current ICAIF paper result contract, separate from historical table numbering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
REGIMES = ("crash_test2018", "crash_test2020", "mix_test2022", "mix_test2024")
PAPER_PARAMETERS = {"baseline", "half_life_days", "tau_iv", "gamma_eps"}
KEYS = {
    "figure3_layer1.csv": ["regime", "kappa"],
    "table1_classical_baselines.csv": ["regime", "comparison"],
    "table2_three_arm.csv": ["regime", "kappa"],
    "table3_deployment_rules.csv": ["regime_key", "rule"],
    "table4_aggressive.csv": ["regime", "kappa"],
    "table5_common_budget.csv": ["regime", "kappa"],
    "breach_uncertainty.csv": ["regime", "kappa", "arm", "budget_kind"],
    "sensitivity.csv": ["config_id", "regime"],
    "robustness_tail.csv": ["regime", "kappa"],
    "transaction_costs_2022.csv": ["kappa", "tc_bp", "kind"],
    "turnover_2022.csv": ["kappa"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_references() -> dict[str, pd.DataFrame]:
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    require(set(manifest["files"]) == set(KEYS), "Unexpected reference inventory")
    frames = {}
    for name, spec in manifest["files"].items():
        path = REFERENCE / name
        require(path.is_file(), f"Reference missing: {name}")
        require(sha256(path) == spec["sha256"], f"Reference hash mismatch: {name}")
        frame = pd.read_csv(path)
        require(len(frame) == spec["rows"], f"Row count changed: {name}")
        require(list(frame.columns) == spec["columns"], f"Schema changed: {name}")
        require(not frame.duplicated(KEYS[name]).any(), f"Duplicate result keys: {name}")
        frames[name] = frame
    check_paper_claims(frames)
    return frames


def check_paper_claims(frames: dict[str, pd.DataFrame]) -> None:
    layer = frames["figure3_layer1.csv"]
    require(len(layer) == 12 and (layer.full_lo > 0).all(), "Figure 3: expected 12 positive CIs")
    expected_beta = ([8, 16, 8], [8, 8, 8], [16, 16, 16], [16, 16, 16])
    for regime, beta in zip(REGIMES, expected_beta):
        values = layer.loc[layer.regime.eq(regime)].sort_values("kappa").beta_star.tolist()
        require(values == beta, f"Layer-1 beta selection changed for {regime}")
    require(len(frames["table1_classical_baselines.csv"]) == 16, "Table 1: expected 16 paired rows")
    require(len(frames["table2_three_arm.csv"]) == 12, "Table 2: expected 12 cells")
    rules = frames["table3_deployment_rules.csv"]
    require(len(rules) == 24, "Table 3: expected four regimes and six rules")
    stress = rules[rules.regime_key.eq("mix_test2022")].set_index("rule")
    for rule, mean, breach in (("fix1", 44.48, .273), ("crc", 19.49, .067), ("crc_rwc_aci", 22.45, .096)):
        require(abs(stress.loc[rule, "mean_dpnl_bp"] - mean) < .0051, f"Table 3 {rule}: mean differs")
        require(abs(stress.loc[rule, "breach"] - breach) < .00051, f"Table 3 {rule}: breach differs")
    aggressive = frames["table4_aggressive.csv"]
    require(len(aggressive) == 12, "Table 4: expected 12 cells")
    require(round(aggressive.breach.min(), 3) == .078 and round(aggressive.breach.max(), 3) == .102,
            "Table 4: corrected all-episode breach range differs")
    bounds = frames["breach_uncertainty.csv"]
    require((bounds.date_block_one_sided_95_upper > .10).all(), "Breach bounds do not match paper")
    common = frames["table5_common_budget.csv"]
    require(len(common) == 12 and (common.aggressive_minus_conservative_bp > 0).all(),
            "Table 5: expected 12 positive paired means")
    require(int((common.difference_ci_low > 0).sum()) == 11, "Table 5: expected 11 positive CIs")
    sens = frames["sensitivity.csv"]
    require(len(sens) == 28 and set(sens.parameter) == PAPER_PARAMETERS,
            "Section 6.6 requires exactly seven configurations across four regimes")
    require((sens.mean_ci_low > 0).all(), "Section 6.6: expected 28 positive CIs")


def compare_frames(name: str, actual: pd.DataFrame, expected: pd.DataFrame) -> dict:
    """Compare numeric values, not serialized float spellings or CSV line endings."""
    keys = KEYS[name]
    require(set(expected.columns) <= set(actual.columns), f"Missing output columns: {name}")
    require(not actual.duplicated(keys).any(), f"Duplicate replay keys: {name}")
    actual = actual[expected.columns].sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    require(len(actual) == len(expected), f"Replay row count differs: {name}")
    maximum = 0.0
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
            a = pd.to_numeric(actual[column]).to_numpy(dtype=float)
            b = expected[column].to_numpy(dtype=float)
            require(np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True),
                    f"Numerical mismatch: {name}/{column}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(a[finite] - b[finite]))))
        else:
            a = actual[column].fillna("").astype(str).tolist()
            b = expected[column].fillna("").astype(str).tolist()
            require(a == b, f"Categorical mismatch: {name}/{column}")
    return {"rows": len(actual), "max_abs_difference": maximum, "status": "PASS"}


def classical_baselines(checkpoints: Path) -> pd.DataFrame:
    source = pd.read_csv(checkpoints / "baselines_v3/baselines_v3.csv")
    rows = []
    for regime in REGIMES:
        frame = source[source.regime.eq(regime)]
        selections = {
            "HW-MV vs BS": frame[frame.arm.str.startswith("HW-MV")],
            "WW vs BS (kappa=2)": frame[frame.arm.str.startswith("WW") & frame.kappa.eq(2)],
            "Residual-HW (kappa=2)": frame[frame.arm.str.contains("resid.HW", regex=True)],
            "Residual-WW (kappa=2)": frame[frame.arm.str.contains("resid.WW", regex=True)],
        }
        for label, selected in selections.items():
            require(len(selected) == 1, f"Ambiguous classical baseline: {regime}/{label}")
            row = selected.iloc[0]
            rows.append(dict(regime=regime, comparison=label, mean=row.full,
                             lo=row.full_lo, hi=row.full_hi, p_le0=row.full_p))
    return pd.DataFrame(rows)


def render_layer1(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = frame.sort_values(["regime", "kappa"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    y = np.arange(len(ordered))[::-1]
    values = ordered.full.to_numpy()
    ax.errorbar(values, y, xerr=np.vstack([values - ordered.full_lo, ordered.full_hi - values]),
                fmt="o", color="#2474a8", capsize=3, markersize=4)
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.set_yticks(y, [f"{r[-4:]}  /  kappa={k}" for r, k in zip(ordered.regime, ordered.kappa)])
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="#dddddd", linewidth=.7)
    ax.set_xlabel("Paired mean Delta PnL versus Black-Scholes (bp), 95% CI")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "figure3_layer1.png", dpi=180)
    fig.savefig(output / "figure3_layer1.pdf")
    plt.close(fig)
