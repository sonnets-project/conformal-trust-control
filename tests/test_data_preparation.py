"""Exercise the cleaning notebook with constructed quotes and local input series."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_inputs(directory: Path, *, fred: bool = False) -> dict[str, str]:
    dates = pd.bdate_range('2020-01-02', periods=6)
    base = dict(date=dates[0], exdate=pd.Timestamp('2020-02-28'), cp_flag='C',
                strike_price=100000, best_bid=5., best_offer=5.2, volume=10,
                open_interest=100, impl_volatility=.2, delta=.5, gamma=.01,
                vega=.1, theta=-.01, forward_price=100., root='SPXW', am_settlement=0)
    rows = [dict(base, date=date) for date in dates]
    invalid = [dict(am_settlement=1), dict(best_bid=0), dict(best_offer=4),
               dict(best_bid=1, best_offer=3), dict(impl_volatility=np.nan),
               dict(strike_price=90000), dict(exdate=pd.Timestamp('2020-01-05')),
               dict(strike_price=130000), dict(cp_flag='P'),
               dict(date=pd.Timestamp('2020-01-10'))]
    rows.extend(dict(base, **values) for values in invalid)
    pd.DataFrame(rows).to_parquet(directory / 'options.parquet', index=False)
    pd.DataFrame({'date': dates, 'spot': 100.}).to_csv(directory / 'spot.csv', index=False)
    if fred:
        rates = pd.DataFrame({'observation_date': dates, 'DGS1': [5., np.nan, 5.1, 5.2, 5.3, 5.4]})
    else:
        rates = pd.DataFrame({'Date': dates, '1y': [.05, np.nan, .051, .052, .053, .054]})
    rates.to_csv(directory / 'rates.csv', index=False)
    return dict(CTC_WORKSPACE=str(directory / 'experiment'),
                SPX0025_PATH=str(directory / 'options.parquet'),
                CTC_TREASURY_PATH=str(directory / 'rates.csv'),
                CTC_SPOT_CSV=str(directory / 'spot.csv'))


def run_notebook(environment: dict, notebook: Path | None = None) -> dict:
    notebook = notebook or ROOT / 'notebooks/00_prepare_data.ipynb'
    namespace = {}
    with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
        for i, cell in enumerate(json.loads(notebook.read_text())['cells']):
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), f'{notebook.name}:cell{i}', 'exec'), namespace)
    return namespace


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'Data preparation tests require pyarrow')
class DataPreparationTests(unittest.TestCase):
    def test_filters_units_rates_and_episode_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = make_inputs(Path(temporary))
            run_notebook(env)
            root = Path(env['CTC_WORKSPACE'])
            panel = pd.read_parquet(root / 'data/SPX_2007_2024_clean_SPXW.parquet')
            self.assertEqual(len(panel), 6)
            np.testing.assert_array_equal(panel['strike'], 100.)
            np.testing.assert_array_equal(panel['nbr_next_steps'], [5, 4, 3, 2, 1, 0])
            np.testing.assert_allclose(panel['r'], [.05, .05, .051, .052, .053, .054])
            self.assertTrue(panel['dtm'].between(10, 90).all())
            self.assertTrue(panel['quote_datetime'].str.endswith('09:31:00').all())
            self.assertTrue((root / 'new-reward/data/1yr_treasury.csv').is_file())

    def test_fred_percent_and_decimal_inputs_produce_same_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / 'decimal'
            second = root / 'fred'
            first.mkdir()
            second.mkdir()
            decimal = run_notebook(make_inputs(first))['edh']
            fred = run_notebook(make_inputs(second, fred=True))['edh']
            pd.testing.assert_frame_equal(decimal, fred, check_exact=True)
            rates = pd.read_csv(second / 'experiment/new-reward/data/1yr_treasury.csv')
            np.testing.assert_allclose(rates['1y'], [.05, .051, .052, .053, .054])

    def test_duplicate_spot_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = make_inputs(Path(temporary))
            spot = pd.read_csv(env['CTC_SPOT_CSV'])
            pd.concat([spot, spot.iloc[:1]]).to_csv(env['CTC_SPOT_CSV'], index=False)
            with self.assertRaisesRegex(ValueError, 'one row per date'):
                run_notebook(env)


if __name__ == '__main__':
    unittest.main()
