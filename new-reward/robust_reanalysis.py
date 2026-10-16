#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalculate tail and bootstrap statistics from saved episode improvements.

Inputs are unified_v3/perep_dp_<regime>_k<k>.csv, with date_start and dp_bp
columns. Outputs include CVaR at 1%, 5% and 10%, mean-improvement intervals
for block lengths 5, 10 and 20, and the panel summary.

Example:
    python robust_reanalysis.py --ckpt /path/to/checkpoints --out /path/to/results
"""
import os, argparse
import numpy as np
import pandas as pd

REGIMES = ["crash_test2018", "crash_test2020", "mix_test2022", "mix_test2024"]
RLAB = {"crash_test2018": "2018 flash-crash", "crash_test2020": "2020 COVID",
        "mix_test2022": "2022 bear", "mix_test2024": "2024 bull"}
KAPPAS = [1, 2, 3]
ALPHAS = [0.01, 0.05, 0.10]
BLOCKS = [5, 10, 20]
B = 3000


def cvar(a, alpha):
    a = np.sort(np.asarray(a, float))
    k = max(1, int(alpha * len(a)))
    return float(a[:k].mean())

def block_ci(x, dates, stat, block_days=10, B=B, seed=0):
    """Daily circular block bootstrap (resampling unit = trading day), matching
    the E0 estimator used in the paper."""
    x = np.asarray(x, float)
    d_uniq, inv = np.unique(dates, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(d_uniq))]
    n = len(d_uniq); nb = int(np.ceil(n / block_days))
    rng = np.random.default_rng(seed)
    point = stat(x)
    out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, nb)
        day_idx = np.concatenate([(s + np.arange(block_days)) % n for s in starts])[:n]
        sel = np.concatenate([groups[j] for j in day_idx])
        out[b] = stat(x[sel])
    lo, hi = np.percentile(out, [2.5, 97.5])
    p_le0 = float(np.mean(out <= 0.0))
    return float(point), float(lo), float(hi), p_le0


def main():
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(package_dir, "new-reward", "checkpoints"))
    ap.add_argument("--out", default=os.path.join(package_dir, "new-reward", "report_out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows_tail, rows_block, audit = [], [], []
    print(f"{'cell':22s} {'mean(block10)':>22s} | cvar5 (panel check)")
    for rg in REGIMES:
        for k in KAPPAS:
            p = f"{a.ckpt}/unified_v3/perep_dp_{rg}_k{k}.csv"
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"  [skip] {rg} k{k}: {type(e).__name__}")
                continue
            x = df["dp_bp"].to_numpy(float)
            dts = pd.to_datetime(df["date_start"]).to_numpy()

            # ---- R-tail: CVaR_alpha with daily-block CI ----
            cvar_str = {}
            for al in ALPHAS:
                pt, lo, hi, p0 = block_ci(x, dts, lambda v, al=al: cvar(v, al))
                rows_tail.append(dict(regime=rg, kappa=k, alpha=al, cvar=pt,
                                      lo=lo, hi=hi, p_le0=p0))
                cvar_str[al] = pt

            # ---- R-block: mean edge CI at block in {5,10,20} ----
            for bd in BLOCKS:
                pt, lo, hi, p0 = block_ci(x, dts, np.mean, block_days=bd)
                rows_block.append(dict(regime=rg, kappa=k, block_days=bd, mean=pt,
                                       lo=lo, hi=hi, p_le0=p0))

            audit.append(dict(regime=rg, kappa=k, n_ep=len(x), mean=float(np.mean(x)),
                              cvar1=cvar_str[0.01], cvar5=cvar_str[0.05], cvar10=cvar_str[0.10]))
            print(f"{rg+' k'+str(k):22s} {f'{np.mean(x):+.3f}':>22s} | cvar5={cvar_str[0.05]:+.3f}")

    pd.DataFrame(rows_tail).to_csv(f"{a.out}/robust_tail_alpha.csv", index=False)
    pd.DataFrame(rows_block).to_csv(f"{a.out}/robust_block.csv", index=False)
    pd.DataFrame(audit).to_csv(f"{a.out}/robust_data.csv", index=False)
    print(f"\nwrote {a.out}/robust_tail_alpha.csv, robust_block.csv, robust_data.csv")


if __name__ == "__main__":
    main()
