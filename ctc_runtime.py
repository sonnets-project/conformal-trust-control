"""Repository paths and empirical episode construction for the paper notebooks."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
STEPS = 5
DIVIDEND_YIELD = 0.01
TRANSACTION_COST = 0.001
STRIKE_TARGETS = (0.85, 0.925, 1.0, 1.075, 1.15)
EXPIRY_TARGETS = (10, 30, 60)


def project_root() -> Path:
    return ROOT


def cleaned_panel() -> Path:
    return Path(os.environ.get('CTC_DATA', ROOT / 'data/SPX_2007_2024_clean_SPXW.parquet')).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize() -> dict:
    """Create local working directories and initialize the rate input."""
    panel = cleaned_panel()
    if not panel.is_file():
        raise FileNotFoundError(f'Cleaned panel not found: {panel}. Run python scripts/prepare_data.py first.')
    runtime = ROOT / '.runtime/hedging'
    for name in ('data', 'results', 'settings'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    source_rates = Path(os.environ.get('CTC_RATES', ROOT / 'new-reward/data/1yr_treasury.csv')).expanduser().resolve()
    if not source_rates.is_file():
        raise FileNotFoundError(f'Treasury series not found: {source_rates}. Run python scripts/prepare_data.py first.')
    rates = pd.read_csv(source_rates, usecols=['Date', '1y'])
    rates.to_csv(runtime / 'data/1yr_treasury.csv', index=False)
    checkpoints = ROOT / 'new-reward/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    return dict(PROJECT_ROOT=str(ROOT), DRIVE_REPO=str(ROOT / 'new-reward'),
                LOCAL_REPO=str(runtime), CLEAN_PARQUET=str(panel),
                OUT_DIR=str(checkpoints / 'walkforward'))


def call_price(spot, strike, rate, dividend, volatility, maturity):
    if maturity <= 0:
        return max(spot - strike, 0)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return spot * np.exp(-dividend * maturity) * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def implied_volatility(spot, strike, rate, dividend, maturity, midpoint):
    # Preserve the price perturbation sequence used to generate the original paths.
    offsets = [0., 0.] + [sign * n * .0001 for n in range(1, 20) for sign in (-1, 1)]
    for offset in offsets:
        try:
            return brentq(lambda sigma: midpoint + offset - call_price(
                spot, strike, rate, dividend, sigma, maturity), .001, 100., maxiter=1000)
        except ValueError:
            continue
    return None


def select_contract_paths(frame: pd.DataFrame, steps: int = STEPS) -> list[pd.DataFrame]:
    """Select the historical strike/expiry grid at five-observation intervals."""
    data = frame.copy()
    data['scaled K'] = data['strike'] / data['underlying_bid']
    data['dtm'] = ((pd.to_datetime(data['expiration']) + pd.DateOffset(hours=15, minutes=31)
                    - pd.to_datetime(data['quote_datetime'])) / np.timedelta64(1, 'D'))
    eligible = data[(data['nbr_next_steps'] >= steps)
                    & data['quote_datetime'].str.slice(11, 16).eq('09:31')
                    & (data['dtm'] > steps + 1)]
    days = sorted(eligible['quote_datetime'].unique())
    by_day = {day: group for day, group in eligible.groupby('quote_datetime', sort=False)}
    paths = []
    for day_index in range(0, len(days), steps):
        candidates = by_day[days[day_index]]
        for target_index, (strike_target, expiry_target) in enumerate(
                (k, t) for k in STRIKE_TARGETS for t in EXPIRY_TARGETS):
            # The historical iterator stops at this boundary before returning a path.
            if target_index == len(STRIKE_TARGETS) * len(EXPIRY_TARGETS) - 1 and day_index + steps >= len(days):
                break
            for radius in range(1, 11):
                pool = candidates[candidates['scaled K'].between(strike_target - radius * .005,
                                                                  strike_target + radius * .005)
                                  & candidates['dtm'].between(expiry_target - radius, expiry_target + radius)]
                if pool.empty:
                    continue
                distance = abs(pool['scaled K'] / strike_target - 1) + abs(pool['dtm'] / expiry_target - 1)
                start = distance.sort_values().index[0]
                path = data.loc[start:start + steps].copy().reset_index(drop=True)
                if len(path) != steps + 1 or path['strike'].nunique() != 1 or path['expiration'].nunique() != 1:
                    raise ValueError('Contract rows must be sorted by expiration, strike and date.')
                divisor = path.loc[0, 'underlying_bid']
                path = path.drop(columns=['scaled K', 'dtm'])
                for column in ('underlying_bid', 'underlying_ask', 'bid', 'ask', 'strike'):
                    path[column] = path[column] / divisor
                paths.append(path)
                break
    if not paths:
        raise ValueError('No complete five-step episodes were found in the selected data split.')
    return paths


def replay_benchmark(paths: list[pd.DataFrame], rates: pd.DataFrame) -> pd.DataFrame:
    """Build market and BS step records without a learned actor or scaler."""
    rate_lookup = dict(zip(rates['Date'].astype(str), rates['1y']))
    records = []
    previous_iv = 0.0
    for episode, path in enumerate(paths):
        quotes = []
        for _, row in path.iterrows():
            day = row['quote_datetime'][:10]
            maturity = (pd.Timestamp(row['expiration'][:10]) - pd.Timestamp(day)).days
            if day not in rate_lookup:
                raise ValueError(f'Missing Treasury rate for {day}')
            spot, strike = row['underlying_bid'], float(row['strike'])
            midpoint = .5 * (row['bid'] + row['ask'])
            rate = rate_lookup[day]
            solved = implied_volatility(spot, strike, rate, DIVIDEND_YIELD, maturity / 365., midpoint)
            if solved:
                previous_iv = solved
            quotes.append((day, maturity, spot, strike, midpoint, rate, previous_iv, row['ticker'], row['expiration'][:10]))
        previous_position = 0.0
        for step, (old, new) in enumerate(zip(quotes[:-1], quotes[1:])):
            day, maturity, spot, strike, midpoint, rate, volatility, ticker, expiry = old
            years = maturity / 365.
            effective_spot = (spot / strike) * strike
            d1 = (np.log(effective_spot / strike) + (rate - DIVIDEND_YIELD + .5 * volatility**2) * years) / (volatility * np.sqrt(years))
            position = -norm.cdf(d1)
            cost = -abs(position - previous_position) * spot * TRANSACTION_COST
            pnl = position * (new[2] - spot)
            pnl += (new[4] - midpoint) + cost
            scaled = pnl * 100
            reward = (scaled - 2 * max(0., -scaled)) * 10
            records.append({'T': maturity, 'S/K': spot / strike, 'Date': day, 'DateStep': 0,
                            'B Reward': reward, 'A Reward': reward, 'A PnL': pnl, 'B PnL': pnl,
                            'P0': new[4], 'P-1': midpoint, 'S0': new[2], 'S-1': spot,
                            'A Pos': position, 'B Pos': position, 'A TC': cost, 'B TC': cost,
                            'A PnL - TC': pnl - cost, 'B PnL - TC': pnl - cost,
                            'Expiry': expiry, 'v': new[6], 'episode': episode, 'ticker': ticker, 'gstep': step})
            previous_position = position
    return pd.DataFrame(records)


def generate_rollout(split: str, regime: str, runtime: str) -> str:
    runtime = Path(runtime)
    input_path = runtime / 'data' / f'{split}.csv'
    rate_path = runtime / 'data/1yr_treasury.csv'
    fingerprint = hashlib.sha256((sha256(input_path) + sha256(rate_path) + sha256(Path(__file__))).encode()).hexdigest()
    destination = runtime / 'results' / f'{regime}_{split}'
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f'bs_reference__{fingerprint[:12]}.csv'
    if output.is_file():
        return str(output)
    frame = pd.read_csv(input_path)
    rates = pd.read_csv(rate_path)
    paths = select_contract_paths(frame)
    records = replay_benchmark(paths, rates)
    partial = output.with_suffix('.partial.csv')
    records.to_csv(partial, index=False)
    partial.replace(output)
    print(f'{regime} {split}: {len(paths)} episodes')
    return str(output)
