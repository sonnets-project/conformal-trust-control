"""Train and save residual policies for the common-budget comparison.

Invoked by notebooks/07_common_budget_all_regimes.ipynb. Policies are reloaded
from checkpoints before constructing counterfactual episode records. Existing
runs are checked against their saved source, checkpoint and input hashes.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v1.2R"
RUN_REGIMES = [
    "mix_test2022",
    "crash_test2018",
    "crash_test2020",
    "mix_test2024",
]
SOURCE_NOTEBOOK = (
    "06_common_budget_setup.ipynb"
)


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: os.PathLike[str] | str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(torch, payload, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".partial")
    assert not target.exists(), f"refusing to overwrite sealed checkpoint: {target}"
    assert not partial.exists(), f"incomplete prior write exists: {partial}"
    torch.save(payload, partial)
    os.replace(partial, target)


def _atomic_npy_save(target: Path, array: np.ndarray) -> None:
    partial = Path(str(target) + ".partial.npy")
    assert not target.exists(), f"refusing to overwrite sealed reference: {target}"
    assert not partial.exists(), f"incomplete prior write exists: {partial}"
    np.save(partial, array)
    os.replace(partial, target)


def _load_state(torch, policy_class, model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    assert checkpoint["version"] == VERSION
    policy = policy_class().to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    policy.eval()
    return policy, checkpoint


def _state_equal(torch, left, right) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key].cpu(), b[key].cpu()) for key in a)


def _find_kit() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_cells(source_path: Path):
    with open(source_path) as handle:
        notebook = json.load(handle)
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    def cell(marker: str) -> str:
        hits = [text for text in code if text.lstrip().startswith(marker)]
        assert len(hits) == 1, f"source marker not unique/found: {marker!r}"
        return hits[0]

    h0_position = next(
        index
        for index, text in enumerate(code)
        if text.lstrip().startswith("# ── H0.")
    )
    selected = {
        "H0": cell("# ── H0."),
        "G2": cell("# ── G2."),
        "V1": cell("# ── V1 (v1.2)."),
        "V3": cell("# ── V3 (v1.2)."),
        "V4": cell("# ── V4 (v1.2"),
        "V5": cell("# ── V5 (v1.2)."),
    }
    return code[:h0_position], selected


def _run_cell(ipython, label: str, source: str) -> None:
    start = time.time()
    print(f"\n┏━━ v1.2R ▶ {label} " + "━" * 42)
    result = ipython.run_cell(source)
    if not result.success:
        print(f"┗━━ failed after {(time.time() - start) / 60:.1f} min")
        result.raise_error()
    print(f"┗━━ done in {(time.time() - start) / 60:.1f} min")


def _lineage_paths(kit: Path, regime: str):
    root = kit / "new-reward" / "checkpoints" / "v12r_lineage"
    policy_dir = root / "policies" / regime
    reference_dir = root / "references"
    diagnostic_dir = root / "diagnostics"
    policy_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = policy_dir / f"v12r_policy_manifest_{regime}.json"
    return root, policy_dir, reference_dir, diagnostic_dir, manifest


def _seal_or_load_lineage(ns, kit: Path, source_path: Path, runner_path: Path):
    torch = ns["torch"]
    assert str(ns["DEVICE"]) == "cpu", (
        "strict v1.2R requires a CPU runtime; use the CPU device in the setup cell"
    )
    assert int(torch.get_num_threads()) == 1, (
        "strict v1.2R requires torch threads=1; rerun the audited setup cells"
    )
    regime = ns["PANEL_REGIME"]
    root, policy_dir, reference_dir, diagnostic_dir, manifest_path = _lineage_paths(
        kit, regime
    )
    source_hashes = {
        "p3_v12_source": _sha256(source_path),
        "v12r_runner": _sha256(runner_path),
    }
    rollout_md5 = {split: _md5(ns["ROLLF"][split]) for split in ("train", "validation", "test")}
    arm_beta = {
        "cons": {int(k): float(ns["BEST_BETA_BASE"][k]) for k in ns["KAPPAS"]},
        "aggr": {int(k): 0.0 for k in ns["KAPPAS"]},
    }

    if manifest_path.exists():
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        assert manifest["version"] == VERSION
        assert manifest["regime"] == regime
        assert manifest["source_sha256"] == source_hashes, (
            "sealed v1.2R source changed; use a new lineage name instead of overwriting"
        )
        assert manifest["rollout_md5"] == rollout_md5, (
            "rollout fingerprint differs from sealed v1.2R lineage"
        )
        assert manifest["arm_beta"] == {
            arm: {str(k): value for k, value in values.items()}
            for arm, values in arm_beta.items()
        }
        for relative, digest in manifest["files_sha256"].items():
            target = root / relative
            assert target.exists() and _sha256(target) == digest, (
                f"sealed v1.2R file failed SHA-256: {target}"
            )
        create_new = False
        print(f"v1.2R lineage found and verified: {manifest_path}")
    else:
        unexpected = list(policy_dir.glob("*.pt")) + list(
            reference_dir.glob(f"*{regime}*.npy")
        )
        assert not unexpected, (
            "incomplete unmanifested v1.2R lineage exists; preserve it for diagnosis "
            "and start with a clean/new lineage name"
        )
        create_new = True
        manifest = None
        print(f"creating new immutable v1.2R lineage: {regime}")

    policies = {"cons": {}, "aggr": {}}
    files_sha256 = {}
    reload_checks = 0
    for arm in ("cons", "aggr"):
        for kappa in ns["KAPPAS"]:
            for seed in ns["POLICY_SEEDS"]:
                model_path = policy_dir / f"{arm}_k{kappa}_seed{seed}.pt"
                beta = arm_beta[arm][kappa]
                if create_new:
                    trained = ns["train_residual"](
                        ns["pack_tr"],
                        kappa,
                        beta,
                        ns["ALPHA"],
                        ns["EPOCHS"],
                        ns["LR"],
                        seed=seed,
                    )
                    payload = {
                        "version": VERSION,
                        "regime": regime,
                        "arm": arm,
                        "kappa": int(kappa),
                        "seed": int(seed),
                        "beta": float(beta),
                        "epochs": int(ns["EPOCHS"]),
                        "lr": float(ns["LR"]),
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in trained.state_dict().items()
                        },
                    }
                    _atomic_torch_save(torch, payload, model_path)
                    reloaded, checkpoint = _load_state(
                        torch, ns["ResidualPolicy"], model_path, ns["DEVICE"]
                    )
                    assert _state_equal(torch, trained, reloaded), (
                        f"checkpoint reload changed tensors: {model_path}"
                    )
                    reload_checks += len(trained.state_dict())
                    del trained
                else:
                    reloaded, checkpoint = _load_state(
                        torch, ns["ResidualPolicy"], model_path, ns["DEVICE"]
                    )
                assert (
                    checkpoint["arm"] == arm
                    and int(checkpoint["kappa"]) == kappa
                    and int(checkpoint["seed"]) == seed
                    and float(checkpoint["beta"]) == beta
                )
                policies[arm][(kappa, seed)] = reloaded
                files_sha256[str(model_path.relative_to(root))] = _sha256(model_path)

    drift_rows = []
    for kappa in ns["KAPPAS"]:
        paths = []
        for seed in ns["POLICY_SEEDS"]:
            policy = policies["cons"][(kappa, seed)]
            position = ns["_positions"](ns["pack_te"], policy)
            native_pnl = (
                ns["_episode_pnl"](ns["pack_te"], position)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            native_path = (
                reference_dir
                / f"base_pnl_{regime}_k{kappa}_seed{seed}.npy"
            )
            if create_new:
                _atomic_npy_save(native_path, native_pnl)
            assert np.array_equal(native_pnl, np.load(native_path)), (
                f"v1.2R per-seed native PnL replay failed: {regime} "
                f"k={kappa} seed={seed}"
            )
            files_sha256[str(native_path.relative_to(root))] = _sha256(native_path)
            dp = (native_pnl - ns["BS_PNL_TE"]) / ns["SP_TE"] * 1e4
            paths.append(dp)
        current = np.mean(paths, axis=0)
        reference_path = reference_dir / f"dp_{regime}_k{kappa}.npy"
        if create_new:
            _atomic_npy_save(reference_path, current)
        reference = np.load(reference_path)
        assert np.array_equal(current, reference), (
            f"v1.2R base replay is not bitwise identical: {regime} k={kappa}"
        )
        files_sha256[str(reference_path.relative_to(root))] = _sha256(reference_path)

        old_path = kit / "new-reward" / "checkpoints" / "unified_v3" / f"dp_{regime}_k{kappa}.npy"
        if old_path.exists():
            old = np.load(old_path)
            delta = current - old
            drift_rows.append(
                {
                    "regime": regime,
                    "kappa": kappa,
                    "max_abs_bp": float(np.max(np.abs(delta))),
                    "mean_abs_bp": float(np.mean(np.abs(delta))),
                    "rmse_bp": float(np.sqrt(np.mean(delta**2))),
                    "mean_v12r": float(np.mean(current)),
                    "mean_v12_reference": float(np.mean(old)),
                    "bitwise": bool(np.array_equal(current, old)),
                }
            )

    diagnostic_path = diagnostic_dir / f"v12r_vs_v12_drift_{regime}.csv"
    if create_new:
        pd.DataFrame(drift_rows).to_csv(diagnostic_path, index=False)
    else:
        assert diagnostic_path.exists(), f"missing sealed drift diagnostic: {diagnostic_path}"
    files_sha256[str(diagnostic_path.relative_to(root))] = _sha256(diagnostic_path)

    if create_new:
        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION,
            "regime": regime,
            "status": "sealed",
            "lineage_gate": "checkpoint SHA-256 + reloaded-policy bitwise replay",
            "historical_v12_role": "diagnostic only; never an admission gate",
            "source_sha256": source_hashes,
            "rollout_md5": rollout_md5,
            "arm_beta": {
                arm: {str(k): value for k, value in values.items()}
                for arm, values in arm_beta.items()
            },
            "kappas": list(ns["KAPPAS"]),
            "policy_seeds": list(ns["POLICY_SEEDS"]),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "device": str(ns["DEVICE"]),
                "torch_threads": int(torch.get_num_threads()),
            },
            "gates": {
                "train_save_reload_tensor_bitwise": True,
                "reload_tensor_comparisons": int(reload_checks),
                "base_reference_bitwise": True,
                "old_v12_required": False,
            },
            "files_sha256": files_sha256,
        }
        with open(manifest_path, "x") as handle:
            json.dump(manifest, handle, indent=1)
        print(f"★ v1.2R checkpoint lineage sealed: {manifest_path}")
    return policies, manifest_path, root


def _prepare_policy_stores(ns, policies):
    base = {}
    for kappa in ns["KAPPAS"]:
        base[kappa] = [
            {"seed": seed, "pol": policies["cons"][(kappa, seed)]}
            for seed in ns["POLICY_SEEDS"]
        ]
    budgets = {
        (kappa, seed): ns["calibrate_budget_bp"](
            ns["pack_va"],
            ns["EPS_VA"],
            policies["cons"][(kappa, seed)],
            ns["EPS_T"],
        )
        for kappa in ns["KAPPAS"]
        for seed in ns["POLICY_SEEDS"]
    }
    lamval_base = {
        (kappa, seed): ns["lam_val_static"](
            policies["cons"][(kappa, seed)], kappa, budgets[(kappa, seed)]
        )
        for kappa in ns["KAPPAS"]
        for seed in ns["POLICY_SEEDS"]
    }
    aggr_policies = dict(policies["aggr"])
    aggr_grids = {
        (kappa, seed): ns["precompute_grids"](
            ns["EPS_TE"], ns["pack_te"], aggr_policies[(kappa, seed)], kappa
        )
        for kappa in ns["KAPPAS"]
        for seed in ns["POLICY_SEEDS"]
    }
    aggr_budgets = {
        (kappa, seed): ns["calibrate_budget_bp"](
            ns["pack_va"],
            ns["EPS_VA"],
            aggr_policies[(kappa, seed)],
            ns["EPS_T"],
        )
        for kappa in ns["KAPPAS"]
        for seed in ns["POLICY_SEEDS"]
    }
    aggr_lamval = {
        (kappa, seed): ns["lam_val_static"](
            aggr_policies[(kappa, seed)], kappa, aggr_budgets[(kappa, seed)]
        )
        for kappa in ns["KAPPAS"]
        for seed in ns["POLICY_SEEDS"]
    }
    ns.update(
        {
            "BASE": base,
            "budgets": budgets,
            "LAMVAL_BASE": lamval_base,
            "ARMS_TO_RUN": [(0.0, "[β=0]", "arm00")],
            "BUDGETS_ARM": {"[β=0]": aggr_budgets},
            "ARM_STORE": {
                "[β=0]": {
                    "pols": aggr_policies,
                    "grids": aggr_grids,
                    "lamval": aggr_lamval,
                    "budgets": aggr_budgets,
                }
            },
            "DO_MILD": False,
            "HAS_V3_SIXARMS": False,
        }
    )


def _disk_ledger_gate(ns):
    ns["_BUD"] = {
        "base": dict(ns["budgets"]),
        "arm00": dict(ns["ARM_STORE"]["[β=0]"]["budgets"]),
    }
    ns["_LAMV"] = {
        "base": dict(ns["LAMVAL_BASE"]),
        "arm00": dict(ns["ARM_STORE"]["[β=0]"]["lamval"]),
    }
    checks = 0
    for arm, memory_store in ns["LEDGERS"].items():
        disk_store = ns["load_ledger"](arm)
        assert set(memory_store) == set(disk_store)
        for key, expected in memory_store.items():
            actual = disk_store[key]
            for shared in ("ds", "de", "iv", "ss", "order"):
                assert np.array_equal(expected[shared], actual[shared]), (
                    f"v1.2R ledger disk gate failed: {arm} {key} {shared}"
                )
                checks += 1
            for kind in ("d_lam", "dp_lam", "vadp"):
                for lam in ns["LAM_GRID"]:
                    assert np.array_equal(expected[kind][lam], actual[kind][lam]), (
                        f"v1.2R ledger disk gate failed: {arm} {key} {kind} {lam}"
                    )
                    checks += 1
    ns.update(
        {
            "GATE4": True,
            "GATE4_KIND": "v12r_disk_ledger_bitwise",
            "GATE4_MAXD": 0.0,
            "_g4_n": checks,
        }
    )
    print(f"★ v1.2R ledger gate passed: {checks} arrays bitwise identical after reload")


def _patch_v3(source: str) -> str:
    old = "if (not LOAD_FROM_LEDGER) and all(abs(budgets[ks] - C_COMMON) < 1e-12 for ks in budgets):"
    new = "if False and all(abs(budgets[ks] - C_COMMON) < 1e-12 for ks in budgets):"
    assert old in source
    return source.replace(old, new, 1)


def _patch_h0(source: str, kit: Path) -> str:
    start = source.index("UNI_DIR =")
    resume = source.index("print(f'H0 ready")
    replacement = (
        f"UNI_DIR = r'{kit / 'new-reward' / 'checkpoints' / 'v12r_lineage' / 'references'}'\n"
        "V3_DIR = ''\n"
        "HAS_V3_SIXARMS = False\n"
    )
    return source[:start] + replacement + source[resume:]


def _patch_g2(source: str) -> str:
    assert "EXP_NAME = 'conformal_controls'" in source
    return source.replace(
        "EXP_NAME = 'conformal_controls'",
        "EXP_NAME = 'conformal_controls_v12r'",
        1,
    )


def _patch_v5(source: str) -> str:
    patched = source.replace(
        "conformal_controls_{RG}", "conformal_controls_v12r_{RG}"
    )
    patched = patched.replace(
        "controls_panel_summary_{_c5tag5}.csv",
        "controls_panel_summary_c5_v12r.csv",
    )
    patched = patched.replace(
        "controls_panel_manifest_{_c5tag5}.json",
        "controls_panel_manifest_c5_v12r.json",
    )
    patched = patched.replace(
        "'budget_mode': 'common',",
        "'version': 'v1.2R', 'budget_mode': 'common',",
        1,
    )
    patched = patched.replace(
        "'unified-budget evaluation (supervisor request on §7.6):",
        "'v1.2R checkpoint-sealed unified-budget evaluation:",
        1,
    )
    return patched


def _finalize_regime_manifest(ns, policy_manifest: Path, lineage_root: Path):
    manifest_path = (
        Path(ns["EXP_DIR"])
        / f"controls_manifest_{ns['PANEL_REGIME']}_c5.json"
    )
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["version"] = VERSION
    manifest["base_policy_lineage"] = "checkpoint-sealed v1.2R"
    manifest["policy_manifest"] = str(policy_manifest)
    manifest["policy_manifest_sha256"] = _sha256(policy_manifest)
    manifest["lineage_root"] = str(lineage_root)
    manifest["historical_v12_admission_gate"] = False
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1)


def _finalize_panel_manifest(kit: Path, source_path: Path, runner_path: Path):
    checkpoint_root = kit / "new-reward" / "checkpoints"
    panel_path = checkpoint_root / "controls_panel_summary_c5_v12r.csv"
    manifest_path = checkpoint_root / "controls_panel_manifest_c5_v12r.json"
    assert panel_path.exists() and manifest_path.exists()
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["version"] = VERSION
    manifest["lineage"] = "checkpoint-sealed"
    manifest["source_sha256"] = {
        "p3_v12_source": _sha256(source_path),
        "v12r_runner": _sha256(runner_path),
        "panel": _sha256(panel_path),
        "regime_manifests": {
            regime: _sha256(
                checkpoint_root
                / f"conformal_controls_v12r_{regime}"
                / f"controls_manifest_{regime}_c5.json"
            )
            for regime in RUN_REGIMES
        },
        "policy_manifests": {
            regime: _sha256(
                checkpoint_root
                / "v12r_lineage"
                / "policies"
                / regime
                / f"v12r_policy_manifest_{regime}.json"
            )
            for regime in RUN_REGIMES
        },
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1)


def main() -> None:
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell
    kit = _find_kit()
    source_path = kit / "notebooks" / SOURCE_NOTEBOOK
    runner_path = kit / "new-reward" / "v12r_lineage_autorun.py"
    assert source_path.exists() and runner_path.exists()
    setup_sources, source = _source_cells(source_path)
    ipython = get_ipython() or InteractiveShell.instance()
    ns = ipython.user_ns

    for index, setup in enumerate(setup_sources, 1):
        _run_cell(ipython, f"setup {index}/{len(setup_sources)}", setup)

    h0_pattern = re.compile(r"(?m)^PANEL_REGIME\s*=\s*'[^']+'")
    start_all = time.time()
    for regime in RUN_REGIMES:
        for key in (
            "BASE",
            "ARM_STORE",
            "BUDGETS_ARM",
            "LEDGERS",
            "POLS",
            "GATE4",
            "DPM_C5",
            "LPATH_C5",
            "CTRL_C5",
        ):
            ns.pop(key, None)
        gc.collect()
        h0 = h0_pattern.sub(f"PANEL_REGIME = '{regime}'", source["H0"], count=1)
        h0 = _patch_h0(h0, kit)
        assert f"PANEL_REGIME = '{regime}'" in h0
        _run_cell(ipython, f"{regime} / context", h0)
        _run_cell(
            ipython,
            f"{regime} / audited control mechanism",
            _patch_g2(source["G2"]),
        )

        policies, policy_manifest, lineage_root = _seal_or_load_lineage(
            ns, kit, source_path, runner_path
        )
        ns["UNI_DIR"] = str(lineage_root / "references")
        ns["EXP_NAME"] = "conformal_controls_v12r"
        ns["EXP_DIR"] = str(
            kit / "new-reward" / "checkpoints" / f"conformal_controls_v12r_{regime}"
        )
        Path(ns["EXP_DIR"]).mkdir(parents=True, exist_ok=True)
        _prepare_policy_stores(ns, policies)

        _run_cell(ipython, f"{regime} / v1.2R ledger construction", source["V1"])
        _disk_ledger_gate(ns)
        _run_cell(
            ipython,
            f"{regime} / common c=5 evaluation",
            _patch_v3(source["V3"]),
        )
        _finalize_regime_manifest(ns, policy_manifest, lineage_root)
        _run_cell(ipython, f"{regime} / budget sweep", source["V4"])
        print(
            f"★★ {regime} v1.2R complete — elapsed {(time.time() - start_all) / 3600:.2f} h"
        )

    _run_cell(ipython, "v1.2R four-regime aggregation", _patch_v5(source["V5"]))
    _finalize_panel_manifest(kit, source_path, runner_path)
    print(
        f"\n━━ v1.2R AUTORUN COMPLETE in {(time.time() - start_all) / 3600:.2f} h ━━"
    )
    print(
        "deliverable → new-reward/checkpoints/controls_panel_summary_c5_v12r.csv"
    )


if __name__ == "__main__":
    main()
