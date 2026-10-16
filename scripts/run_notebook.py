#!/usr/bin/env python3
"""Execute a notebook in a fresh IPython process, without saving outputs in its source."""
import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ('crash_test2018', 'crash_test2020', 'mix_test2022', 'mix_test2024')


def execute(path, regime=None):
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    os.chdir(ROOT)
    for index, cell in enumerate(json.loads(path.read_text())['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if regime:
            source = re.sub(r"(?m)^PANEL_REGIME\s*=\s*'[^']+'", f"PANEL_REGIME = '{regime}'", source)
        print(f'\n{path.name}: cell {index + 1}', flush=True)
        result = shell.run_cell(source, store_history=True)
        result.raise_error()
    return shell.user_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--regime', choices=REGIMES)
    args = parser.parse_args()
    path = args.notebook.expanduser().resolve()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    execute(path, args.regime)


if __name__ == '__main__':
    main()
