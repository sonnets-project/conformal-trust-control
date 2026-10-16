#!/usr/bin/env python3
"""Evaluate the 120 supplied policies on locally prepared option data."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'new-reward'))

import numpy as np
import pandas as pd
import torch

import ctc_runtime as runtime
import episode_evaluation as evaluation
import warm0_batch_replay as controller
from checkpoint_catalog import PRETRAINED, REGIMES, ARMS, KAPPAS, SEEDS, sha256, verify_catalog


def load_policy(entry, directory=PRETRAINED):
    """Load only tensors and scalar training metadata; require the published digest."""
    path = Path(directory) / entry['path']
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {entry["path"]}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    fields = {'version', 'regime', 'arm', 'kappa', 'seed', 'beta', 'epochs', 'lr', 'state_dict'}
    if set(payload) != fields:
        raise ValueError(f'Unexpected checkpoint fields: {entry["path"]}')
    for key in fields - {'state_dict'}:
        if payload[key] != entry[key]:
            raise ValueError(f'Checkpoint metadata mismatch: {entry["path"]}: {key}')
    if not all(torch.is_tensor(value) and bool(torch.isfinite(value).all())
               for value in payload['state_dict'].values()):
        raise ValueError(f'Invalid checkpoint tensors: {entry["path"]}')
    policy = evaluation.ResidualPolicy()
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def prepare_split(panel, year, split, regime, work):
    frame = panel.loc[panel['date'].dt.year.eq(year)].sort_values(
        ['expiration', 'strike', 'date']).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'{regime}: no quotes for {split} year {year}')
    group = frame.groupby(['expiration', 'strike'], sort=False)
    frame['nbr_next_steps'] = (group['date'].transform('count') - group.cumcount() - 1).astype('int32')
    columns = ['quote_datetime', 'ticker', 'expiration', 'strike', 'underlying_bid',
               'underlying_ask', 'bid', 'ask', 'nbr_next_steps']
    frame[columns].to_csv(work / 'data' / f'{split}.csv', index=False)
    # Use the same CSV round-trip and episode ordering as the training notebooks.
    path = runtime.generate_rollout(split, regime, str(work))
    return evaluation._episodes_from_csv(path), Path(path)


def prepare_rates(panel, source, destination):
    """Retain supplied rates and fill missing dates from the cleaned panel."""
    panel_rates = (panel[['date', 'r']].drop_duplicates('date').sort_values('date')
                   .rename(columns={'date': 'Date', 'r': '1y'}))
    panel_rates['Date'] = pd.to_datetime(panel_rates['Date']).dt.strftime('%Y-%m-%d')
    rates = pd.concat([pd.read_csv(source), panel_rates]).drop_duplicates('Date').sort_values('Date')
    rates.to_csv(destination, index=False)


def evaluate_policy(policy, kappa, validation, test, pack_validation, pack_test):
    ledger = evaluation.precompute_grids(test, pack_test, policy, kappa)
    start_spot = np.array([float(row['ep_step_df']['S-1'].iloc[0]) for _, row in validation.iterrows()])
    ledger['vadp'] = {lam: evaluation.episode_dpnl_bp_lambda(
        pack_validation, policy, np.full(len(validation), lam), start_spot) for lam in controller.LAM_GRID}
    return ledger


def summarize(runs, dates, bootstrap_reps):
    mean_path = np.mean([run['dpnl_bp'] for run in runs], axis=0)
    point, low, high = controller.block_bootstrap_ci(mean_path, dates, reps=bootstrap_reps)
    return dict(mean_bp=point, mean_ci_low=low, mean_ci_high=high,
                breach=float(np.mean([run['breach'] for run in runs])),
                breach_informed=float(np.mean([run['breach_informed'] for run in runs])),
                mean_lambda=float(np.mean([run['lam'].mean() for run in runs])),
                cvar5_bp=controller.cvar5(mean_path), mean_path=mean_path)


def seed_row(regime, arm, entry, mode, result):
    return dict(regime=regime, arm=arm, kappa=entry['kappa'], seed=entry['seed'], beta=entry['beta'],
                evaluation=mode, budget_bp=result['budget_bp'],
                mean_bp=float(np.mean(result['dpnl_bp'])), breach=result['breach'],
                mean_lambda=float(np.mean(result['lam'])), cvar5_bp=controller.cvar5(result['dpnl_bp']))


def run(regimes, output, bootstrap_reps=3000):
    catalog = verify_catalog()
    entries = {(item['regime'], item['arm'], item['kappa'], item['seed']): item
               for item in catalog['checkpoints']}
    config = runtime.initialize()
    panel_path = Path(config['CLEAN_PARQUET'])
    rate_path = Path(config['LOCAL_REPO']) / 'data/1yr_treasury.csv'
    inputs_before = {'cleaned_panel_sha256': sha256(panel_path), 'treasury_sha256': sha256(rate_path),
                     'checkpoint_manifest_sha256': sha256(PRETRAINED / 'manifest.json')}
    panel = pd.read_parquet(panel_path)
    panel['date'] = pd.to_datetime(panel['date'])
    rows, summaries, aggressive_rows, common_rows, breach_rows = [], [], [], [], []
    selected_digests, episode_counts = {}, {}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING', training_performed=False, checkpoint_count=0,
                  regimes=list(regimes), bootstrap_replicates=bootstrap_reps,
                  python=platform.python_version(),
                  packages={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'torch')},
                  inputs=inputs_before)
    report_path = output / 'evaluation_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        effective_rates = output / '1yr_treasury.csv'
        prepare_rates(panel, rate_path, effective_rates)
        evaluation.set_rates(effective_rates)
        report['evaluation_rates_sha256'] = sha256(effective_rates)
        for regime in regimes:
            print(f'{regime}: preparing validation and test episodes', flush=True)
            years = catalog['splits'][regime]
            work = output / 'episodes' / regime
            (work / 'data').mkdir(parents=True)
            shutil.copyfile(effective_rates, work / 'data/1yr_treasury.csv')
            validation, _ = prepare_split(panel, years['validation'], 'validation', regime, work)
            test, _ = prepare_split(panel, years['test'], 'test', regime, work)
            pack_va = evaluation.pack_episodes(validation)
            pack_te = evaluation.pack_episodes(test)
            episode_counts[regime] = {'validation': len(validation), 'test': len(test)}
            dates = pd.to_datetime(test['date_start']).to_numpy()
            stores = {arm: {} for arm in ARMS}
            results = {}
            for arm in ARMS:
                for kappa in KAPPAS:
                    grouped = {key: [] for key in ('full', 'guarded_validation', 'guarded_common5')}
                    budgets = []
                    for seed in SEEDS:
                        entry = entries[regime, arm, kappa, seed]
                        policy = load_policy(entry)
                        ledger = evaluate_policy(policy, kappa, validation, test, pack_va, pack_te)
                        stores[arm][kappa, seed] = ledger
                        budget = controller.budget_from_validation(ledger)
                        budgets.append(budget)
                        runs = dict(full=controller.deploy(ledger, budget, mode='fix1'),
                                    guarded_validation=controller.deploy(ledger, budget),
                                    guarded_common5=controller.deploy(ledger, 5.0))
                        for mode, result in runs.items():
                            grouped[mode].append(result)
                            rows.append(seed_row(regime, arm, entry, mode, result))
                        selected_digests[entry['path']] = entry['sha256']
                    for mode, runs in grouped.items():
                        summary = summarize(runs, dates, bootstrap_reps)
                        results[arm, kappa, mode] = summary
                        summaries.append(dict(regime=regime, arm=arm, kappa=kappa, evaluation=mode,
                                              test_episodes=len(test), **{k:v for k,v in summary.items() if k!='mean_path'}))
                    if arm == 'aggr':
                        aggr = results[arm, kappa, 'guarded_validation']
                        aggressive_rows.append(dict(regime=regime, kappa=kappa,
                                                    budget_mean_bp=float(np.mean(budgets)),
                                                    budget_min_bp=float(np.min(budgets)), budget_max_bp=float(np.max(budgets)),
                                                    **{k:v for k,v in aggr.items() if k!='mean_path'}))
                        indicators = np.stack([item['breach_indicator'] for item in grouped['guarded_validation']])
                        point, upper, count = controller.cluster_upper_bound(indicators.mean(axis=0), dates, reps=bootstrap_reps)
                        breach_rows.append(dict(regime=regime, kappa=kappa, contract_point=float(indicators.mean()),
                                                arm='aggressive', budget_kind='validation',
                                                worst_seed=float(max(item['breach'] for item in grouped['guarded_validation'])),
                                                date_equal_weight_point=point, date_block_one_sided_95_upper=upper,
                                                iid_wilson_one_sided_95_upper_reference=controller.wilson_upper(
                                                    int(indicators.sum()), int(indicators.size)),
                                                start_date_batches=count))
                    print(f'{regime}: {arm}, kappa={kappa}, five checkpoints evaluated', flush=True)
            for kappa in KAPPAS:
                cons = results['cons', kappa, 'guarded_common5']
                aggr = results['aggr', kappa, 'guarded_common5']
                mean, low, high = controller.block_bootstrap_ci(aggr['mean_path']-cons['mean_path'], dates, reps=bootstrap_reps)
                common_rows.append(dict(regime=regime, kappa=kappa,
                                        conservative_mean_bp=cons['mean_bp'], aggressive_mean_bp=aggr['mean_bp'],
                                        aggressive_minus_conservative_bp=mean, difference_ci_low=low, difference_ci_high=high,
                                        conservative_breach=cons['breach'], aggressive_breach=aggr['breach'],
                                        conservative_cvar5_bp=cons['cvar5_bp'], aggressive_cvar5_bp=aggr['cvar5_bp'],
                                        conservative_mean_lambda=cons['mean_lambda'], aggressive_mean_lambda=aggr['mean_lambda']))
            # Counterfactual records are generated locally, under the ignored output directory.
            for arm, store in stores.items():
                representative = store[1, 0]
                arrays = {key: representative[key] for key in ('ds', 'de', 'iv', 'ss', 'order')}
                for (kappa, seed), ledger in store.items():
                    for kind, member in (('d', 'd_lam'), ('dp', 'dp_lam'), ('vadp', 'vadp')):
                        arrays.update({f'k{kappa}|s{seed}|{kind}|{lam}': values for lam, values in ledger[member].items()})
                np.savez_compressed(work / f'ledger_{arm}.npz', **arrays)
        frames = {'seed_results.csv': rows, 'policy_summary.csv': summaries,
                  'table4_aggressive.csv': aggressive_rows, 'table5_common_budget.csv': common_rows,
                  'breach_uncertainty.csv': breach_rows}
        for name, values in frames.items():
            pd.DataFrame(values).to_csv(output / name, index=False)
        inputs_after = {'cleaned_panel_sha256': sha256(panel_path), 'treasury_sha256': sha256(rate_path),
                        'checkpoint_manifest_sha256': sha256(PRETRAINED / 'manifest.json')}
        if inputs_after != inputs_before:
            raise RuntimeError('An input file changed during evaluation.')
        verify_catalog()
        report.update(status='PASS', checkpoint_count=len(selected_digests), checkpoint_sha256=selected_digests,
                      episode_counts=episode_counts, outputs_sha256={name:sha256(output/name) for name in frames},
                      inputs_unchanged=True)
    except BaseException as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Pretrained evaluation: PASS ({len(selected_digests)} checkpoints; no training)\nOutputs: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime', choices=('all', *REGIMES), default='all')
    parser.add_argument('--output', type=Path, help='New directory for local evaluation outputs')
    parser.add_argument('--bootstrap-reps', type=int, default=3000, help='Bootstrap repetitions; paper setting: 3000')
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error('--bootstrap-reps must be positive')
    output = (args.output or ROOT/'results/generated'/time.strftime('pretrained-%Y%m%d-%H%M%S')).expanduser().resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}')
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'results/generated'):
        parser.error('Within the repository, write evaluation outputs under results/generated/')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(REGIMES if args.regime=='all' else (args.regime,), output, args.bootstrap_reps)


if __name__ == '__main__':
    main()
