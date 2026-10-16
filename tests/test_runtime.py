"""Check empirical path selection, benchmark accounting and cache invalidation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctc_runtime import call_price, implied_volatility, select_contract_paths, replay_benchmark, generate_rollout


def quotes():
    dates = pd.bdate_range('2022-01-03', periods=12)
    expiry = pd.Timestamp('2022-02-02')
    rows = []
    for index, date in enumerate(dates):
        spot = 100. + .2 * index
        price = call_price(spot, 100., .04, .01, .2, (expiry - date).days / 365.)
        rows.append(dict(quote_datetime=date.strftime('%Y-%m-%d 09:31:00'),
                         expiration='2022-02-02', ticker='SPX', strike=100.,
                         underlying_bid=spot, underlying_ask=spot,
                         bid=price - .01, ask=price + .01, nbr_next_steps=11-index))
    return pd.DataFrame(rows), pd.DataFrame({'Date': dates.strftime('%Y-%m-%d'), '1y': .04})


class RuntimeTests(unittest.TestCase):
    def test_iv_solver_and_benchmark_accounting(self):
        frame, rates = quotes()
        paths = select_contract_paths(frame)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0]['quote_datetime'].iloc[0][:10], '2022-01-03')
        self.assertEqual(paths[1]['quote_datetime'].iloc[0][:10], '2022-01-10')
        self.assertTrue(all(len(path) == 6 for path in paths))
        record = replay_benchmark(paths, rates)
        self.assertEqual(len(record), 10)
        np.testing.assert_allclose(record['v'], .2, atol=1e-10)
        for _, episode in record.groupby('episode', sort=False):
            previous = 0.
            for _, row in episode.iterrows():
                strike = row['S-1'] / row['S/K']
                years = row['T'] / 365.
                d1 = (np.log(row['S-1'] / strike) + (.04-.01+.5*.2**2)*years) / (.2*np.sqrt(years))
                self.assertAlmostEqual(row['B Pos'], -norm.cdf(d1), places=9)
                cost = -abs(row['B Pos']-previous)*row['S-1']*.001
                pnl = row['B Pos']*(row['S0']-row['S-1']) + row['P0']-row['P-1'] + cost
                self.assertAlmostEqual(row['B TC'], cost)
                self.assertAlmostEqual(row['B PnL'], pnl)
                self.assertAlmostEqual(row['B Reward'], (pnl*100-2*max(-pnl*100,0))*10)
                previous = row['B Pos']

    def test_cache_changes_when_quotes_or_rates_change(self):
        frame, rates = quotes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data').mkdir()
            frame.to_csv(root/'data/test.csv', index=False)
            rates.to_csv(root/'data/1yr_treasury.csv', index=False)
            first = generate_rollout('test', 'example', temporary)
            self.assertEqual(first, generate_rollout('test', 'example', temporary))
            rates['1y'] += .001
            rates.to_csv(root/'data/1yr_treasury.csv', index=False)
            second = generate_rollout('test', 'example', temporary)
            self.assertNotEqual(first, second)
            frame['ask'] += .001
            frame.to_csv(root/'data/test.csv', index=False)
            self.assertNotEqual(second, generate_rollout('test', 'example', temporary))

    def test_incomplete_split_is_rejected(self):
        frame, _ = quotes()
        frame['nbr_next_steps'] = 0
        with self.assertRaisesRegex(ValueError, 'No complete'):
            select_contract_paths(frame)


if __name__ == '__main__':
    unittest.main()
