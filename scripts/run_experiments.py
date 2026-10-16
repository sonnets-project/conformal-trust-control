#!/usr/bin/env python3
"""Train policies and run the paper experiments from local market data."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('layer1', 'controls', 'three-arm', 'common-budget', 'robustness', 'analysis')
REGIMES = ('crash_test2018', 'crash_test2020', 'mix_test2022', 'mix_test2024')
NOTEBOOKS = {
    'layer1': '01_train_policies_and_baselines.ipynb',
    'controls': '03_evaluate_policy_arms.ipynb',
    'three-arm': '05_three_arm_all_regimes.ipynb',
    'common-budget': '07_common_budget_all_regimes.ipynb',
    'robustness': '08_robustness_2022.ipynb',
}


def commands(stage):
    if stage == 'analysis':
        return [(stage, [sys.executable, str(ROOT / 'scripts/reproduce_paper.py'),
                         '--mode', 'replay', '--artifacts', str(ROOT), '--skip-comparison'])]
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_notebook.py'),
               str(ROOT / 'notebooks' / NOTEBOOKS[stage])]
    if stage == 'controls':
        return [(f'{stage}-{regime}', command + ['--regime', regime]) for regime in REGIMES]
    return [(stage, command)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', *STAGES), default='all', help='Run all stages or one stage after its prerequisites')
    args = parser.parse_args()
    if args.stage != 'analysis':
        sys.path.insert(0, str(ROOT))
        from ctc_runtime import initialize
        initialize()
    run_dir = ROOT / 'results/generated' / time.strftime('run-%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MPLCONFIGDIR=str(ROOT / '.runtime/matplotlib'))
    report = {'status': 'RUNNING', 'stages': [], 'packages': {name: importlib.metadata.version(name)
              for name in ('numpy', 'pandas', 'scipy', 'torch')}}
    report_path = run_dir / 'run.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        for stage in (STAGES if args.stage == 'all' else (args.stage,)):
            for name, command in commands(stage):
                log = run_dir / f'{name}.log'
                print(f'{name}: running; log {log.relative_to(ROOT)}', flush=True)
                with log.open('w') as handle:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    tail = '\n'.join(log.read_text(errors='replace').splitlines()[-25:])
                    raise RuntimeError(f'{name} failed. See {log}\n{tail}')
                report['stages'].append(name)
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                print(f'{name}: complete', flush=True)
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Run complete: {run_dir}')


if __name__ == '__main__':
    main()
