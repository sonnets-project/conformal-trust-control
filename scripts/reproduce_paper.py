#!/usr/bin/env python3
"""Export paper results or rerun deployment analyses from saved episode records."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from paper_results import (ROOT, REFERENCE, PAPER_PARAMETERS, REGIMES, KEYS,
                           classical_baselines, compare_frames, render_layer1,
                           require, sha256, verify_references)


def inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def input_files(artifacts: Path) -> list[Path]:
    ckpt = artifacts / "new-reward/checkpoints"
    paths = [ckpt / "unified_v3/panel_v3_summary.csv", ckpt / "e2_summary.csv",
             ckpt / "baselines_v3/baselines_v3.csv",
             ckpt / "conformal_controls_mix_test2022/tc_sweep_mix_test2022.csv",
             ckpt / "conformal_controls_mix_test2022/turnover_mix_test2022.csv"]
    for regime in REGIMES:
        directory = ckpt / f"conformal_controls_v12r_{regime}"
        paths.extend(directory / f"ledger_{arm}_{regime}.npz" for arm in ("arm00", "base"))
        old = ckpt / f"conformal_controls_{regime}"
        # Continuous ACI uses the archived path, as in the accepted table generator.
        paths.extend(old / name for name in
                     ("dpm_arm00.npz", "lampath_arm00.npz",
                      f"controls_arms_{regime}.csv"))
        paths.extend(ckpt / f"unified_v3/perep_dp_{regime}_k{k}.csv" for k in (1, 2, 3))
    return paths


def run_stage(command: list[str], log: Path, environment: dict) -> None:
    print(f"Running {log.stem} ...", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"{log.stem} failed; see {log}\n{tail}")
    print(f"  {log.stem}: PASS", flush=True)


def replay(artifacts: Path, output: Path, reference: dict, compare: bool = True) -> tuple[dict, dict]:
    paths = input_files(artifacts)
    missing = [str(p.relative_to(artifacts)) for p in paths if not p.is_file()]
    require(not missing, "Private replay inputs missing:\n  " + "\n  ".join(missing)
            + "\nSee docs/reproduction.md. No data is downloaded by this command.")
    hashes_before = {str(p.relative_to(artifacts)): sha256(p) for p in paths}
    ckpt = artifacts / "new-reward/checkpoints"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ctc-replay-") as temporary:
        stage = Path(temporary)
        (stage / "new-reward").mkdir()
        # Scripts write only to the temporary root. The original archive is read-only.
        (stage / "new-reward/checkpoints").symlink_to(ckpt, target_is_directory=True)
        for name in ("warm0_batch_replay.py", "published_rules_4regime_replay.py",
                     "v13_sensitivity_replay.py", "robust_reanalysis.py"):
            shutil.copy2(ROOT / "new-reward" / name, stage / "new-reward" / name)
        env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(stage / ".matplotlib"),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for name in ("warm0_batch_replay", "published_rules_4regime_replay"):
            run_stage([sys.executable, str(stage / "new-reward" / f"{name}.py"),
                       "--root", str(stage)] + (["--check-reference"]
                        if compare and name == "published_rules_4regime_replay" else []),
                      logs / f"{name}.log", env)
        run_stage([sys.executable, str(stage / "new-reward/v13_sensitivity_replay.py"),
                   "--root", str(stage), "--skip-plot"], logs / "sensitivity.log", env)
        run_stage([sys.executable, str(stage / "new-reward/robust_reanalysis.py"),
                   "--ckpt", str(ckpt), "--out", str(stage / "results/robustness")],
                  logs / "robustness.log", env)

        frames = {
            "figure3_layer1.csv": pd.read_csv(ckpt / "unified_v3/panel_v3_summary.csv"),
            "table1_classical_baselines.csv": classical_baselines(ckpt),
            "table2_three_arm.csv": pd.read_csv(ckpt / "e2_summary.csv"),
            "table3_deployment_rules.csv": pd.read_csv(stage / "results/published_rules_4regime_v13/published_rules_panel.csv"),
            "table4_aggressive.csv": pd.read_csv(stage / "results/warm0_batch_v13/table5_aggressive_validation_budget.csv"),
            "table5_common_budget.csv": pd.read_csv(stage / "results/warm0_batch_v13/table6_common_budget.csv"),
            "breach_uncertainty.csv": pd.read_csv(stage / "results/warm0_batch_v13/breach_uncertainty.csv"),
            "sensitivity.csv": pd.read_csv(stage / "results/v13_sensitivity/sensitivity_panel.csv"),
            "robustness_tail.csv": pd.read_csv(stage / "results/robustness/robust_data.csv"),
            "transaction_costs_2022.csv": pd.read_csv(ckpt / "conformal_controls_mix_test2022/tc_sweep_mix_test2022.csv"),
            "turnover_2022.csv": pd.read_csv(ckpt / "conformal_controls_mix_test2022/turnover_mix_test2022.csv"),
        }
        bounds = frames["breach_uncertainty.csv"]
        frames["breach_uncertainty.csv"] = bounds[bounds.arm.eq("aggressive") & bounds.budget_kind.eq("validation")]
        frames["sensitivity.csv"] = frames["sensitivity.csv"].loc[lambda df: df.parameter.isin(PAPER_PARAMETERS)]
        comparisons = ({name: compare_frames(name, frames[name], expected) for name, expected in reference.items()}
                       if compare else {"status": "not requested for this new training run"})
        # Export aggregate tables and run metadata.
        for name in ("robust_tail_alpha.csv", "robust_block.csv"):
            shutil.copy2(stage / "results/robustness" / name, output / name)
        for dirname in ("warm0_batch_v13", "published_rules_4regime_v13", "v13_sensitivity"):
            shutil.copy2(stage / "results" / dirname / "manifest.json", output / f"{dirname}_manifest.json")

    hashes_after = {str(p.relative_to(artifacts)): sha256(p) for p in paths}
    require(hashes_before == hashes_after, "A private input changed during replay")
    return frames, {"numeric_comparisons": comparisons, "input_sha256": hashes_before,
                    "private_inputs_unchanged": True,
                    "recomputed": ["Tables 3-5 (grid rules)", "breach bounds", "controller sensitivity", "tail and block robustness"],
                    "loaded_saved_statistics": ["Figure 3", "Tables 1-2", "transaction-cost and turnover summaries"],
                    "aci_baseline": "Archived continuous ACI path with the documented warm-up/denominator correction; not newly redeployed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "replay"), default="reference")
    parser.add_argument("--artifacts", type=Path, help="Experiment root containing new-reward/checkpoints; required for replay")
    parser.add_argument("--output", type=Path, help="New output directory (never overwrites an existing directory)")
    parser.add_argument("--skip-comparison", action="store_true", help="Export a newly trained experiment without requiring equality to the published reference numbers")
    args = parser.parse_args()
    reference = verify_references()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / "results/generated" / f"{args.mode}-{stamp}").expanduser().resolve()
    require(not output.exists(), f"Output already exists; choose a new --output directory: {output}")
    require(not inside(output, REFERENCE) and not inside(output, ROOT / "new-reward")
            and not inside(output, ROOT / ".git"), "Output must not overlap source/reference files")
    artifacts = args.artifacts.expanduser().resolve() if args.artifacts else None
    if args.mode == "replay":
        require(artifacts is not None, "--mode replay requires --artifacts /path/to/private/archive")
        require(not inside(output, artifacts / "new-reward/checkpoints")
                and not inside(artifacts, output), "Output must not overlap the input checkpoints")
        missing = [p for p in input_files(artifacts) if not p.is_file()]
        require(not missing, "Private replay inputs missing:\n  " + "\n  ".join(str(p.relative_to(artifacts)) for p in missing))
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    started = time.perf_counter()
    try:
        if args.mode == "replay":
            frames, detail = replay(artifacts, output, reference, compare=not args.skip_comparison)
        else:
            frames = reference
            detail = {"recomputed": [], "note": "Verified saved aggregate results and rendered Figure 3. No policy, controller or bootstrap experiment was rerun."}
        for name, frame in frames.items():
            frame[reference[name].columns].sort_values(KEYS[name]).to_csv(output / name, index=False)
        render_layer1(frames["figure3_layer1.csv"], output)
        report = {"status": "PASS", "mode": args.mode, "training_performed": False,
                  "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "python": platform.python_version(),
                  "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")},
                  "reference_manifest_sha256": sha256(REFERENCE / "manifest.json"),
                  "outputs_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
                  **detail}
        (output / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "PASS.txt").write_text(f"CTC {args.mode}: PASS\nTraining performed: false\n")
        print(f"CTC {args.mode}: PASS\nOutputs: {output}")
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error) + "\n")
        raise


if __name__ == "__main__":
    main()
