#!/usr/bin/env python3
"""Exercise the complete workflow on synthetic quotes in a separate local copy.

Training uses two epochs and bootstrap calculations use 20 replicates. These
settings check execution and file dependencies; they are not paper experiments.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctc_runtime import call_price
from verify_release import release_files
from checkpoint_catalog import verify_catalog


def make_market(raw):
    raw.mkdir(parents=True)
    rows, spots, rates = [], [], []
    for year in range(2007, 2025):
        days = pd.bdate_range(f'{year}-01-03', periods=54)
        for index, day in enumerate(days):
            spot = 3000 * (1 + .018 * np.sin(index * .5) + .0005 * index)
            spots.append({'date': day, 'spot': spot})
            rates.append({'observation_date': day, 'DGS1': 4.})
            for strike_factor in (.85, .925, 1., 1.075, 1.15):
                strike = 3000 * strike_factor
                for tenor in (30, 60, 85):
                    expiry = days[0] + pd.Timedelta(days=tenor)
                    maturity = (expiry - day).days
                    if maturity < 10:
                        continue
                    midpoint = call_price(spot, strike, .04, .01, .25, maturity / 365.)
                    rows.append(dict(date=day, exdate=expiry, cp_flag='C',
                                     strike_price=strike * 1000, best_bid=max(midpoint-.01, .001),
                                     best_offer=midpoint+.01, volume=10, open_interest=100,
                                     impl_volatility=.25, delta=.5, gamma=.01, vega=.1,
                                     theta=-.01, forward_price=spot, root='SPXW', am_settlement=0))
    pd.DataFrame(rows).to_parquet(raw / 'SPX0025.parquet', index=False)
    pd.DataFrame(spots).to_csv(raw / 'spot.csv', index=False)
    pd.DataFrame(rates).to_csv(raw / 'DGS1.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory for the test copy and logs')
    args = parser.parse_args()
    output = (args.output or ROOT / 'results/generated' / time.strftime('smoke-%Y%m%d-%H%M%S')).resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}')
    files = release_files()
    output.mkdir(parents=True)
    checkout = output / 'repository'
    for source in files:
        if source.is_symlink():
            raise RuntimeError(f'Release source must not be a symlink: {source}')
        target = checkout / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for directory, pattern in (('notebooks', '*.ipynb'), ('new-reward', '*.py')):
        for path in (checkout / directory).glob(pattern):
            source = path.read_text()
            source = re.sub(r'\b400\b', '2', source)
            source = re.sub(r'\b3000\b', '20', source)
            path.write_text(source)
    make_market(checkout / 'data/raw')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg',
               IPYTHONDIR=str(checkout / '.runtime/ipython'))
    for key in ('CTC_DATA', 'CTC_RATES', 'CTC_WORKSPACE', 'SPX0025_PATH',
                'CTC_SPOT_CSV', 'CTC_TREASURY_PATH'):
        env.pop(key, None)
    started = time.perf_counter()
    report = {'status': 'RUNNING', 'data': 'synthetic', 'epochs': 2, 'bootstrap_replicates': 20}
    original_catalog = verify_catalog(checkout / 'pretrained')
    try:
        for script, arguments in (('prepare_data', []),
                                  ('evaluate_pretrained', ['--bootstrap-reps', '20']),
                                  ('run_experiments', [])):
            log = output / f'{script}.log'
            print(f'{script}: running; log {log}', flush=True)
            with log.open('w') as handle:
                result = subprocess.run([sys.executable, '-u', f'scripts/{script}.py', *arguments], cwd=checkout,
                                        env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{script} failed; see {log}')
        if verify_catalog(checkout / 'pretrained') != original_catalog:
            raise RuntimeError('A supplied checkpoint changed during the workflow.')
        report['pretrained_checkpoints_unchanged'] = 120
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        (output / 'smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Complete workflow smoke test passed: {output}')


if __name__ == '__main__':
    main()
