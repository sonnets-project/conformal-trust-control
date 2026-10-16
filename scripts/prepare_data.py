#!/usr/bin/env python3
"""Run the data-cleaning notebook from the repository root."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--options', type=Path, help='OptionMetrics parquet; default data/raw/SPX0025.parquet')
    parser.add_argument('--rates', type=Path, help='Treasury CSV; default data/raw/DGS1.csv')
    parser.add_argument('--spot', type=Path, help='Index closes CSV; default data/raw/spot.csv, otherwise download ^GSPC')
    args = parser.parse_args()
    for value, variable in ((args.options, 'SPX0025_PATH'), (args.rates, 'CTC_TREASURY_PATH'), (args.spot, 'CTC_SPOT_CSV')):
        if value is not None:
            os.environ[variable] = str(value.expanduser().resolve())
    os.chdir(ROOT)
    notebook = ROOT / 'notebooks/00_prepare_data.ipynb'
    namespace = {'__name__': '__main__'}
    for index, cell in enumerate(json.loads(notebook.read_text())['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'{notebook.name}:cell{index}', 'exec'), namespace)


if __name__ == '__main__':
    main()
