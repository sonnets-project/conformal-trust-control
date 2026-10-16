"""Episode features and evaluation shared with the original experiment notebook.

This module preserves the published checkpoint's state and ledger conventions.
It does not train policies or read saved experiment records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from residual_policy import ResidualPolicy, _episode_reward, _seg_sum

DEVICE = torch.device('cpu')
TC_RATE = 0.001
Q_DIV = 0.0
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
RATES = None


def set_rates(path):
    global RATES
    RATES = pd.read_csv(path, usecols=['Date', '1y'])
    RATES['Date'] = pd.to_datetime(RATES['Date'])


def bs_greeks_inner(S, K, r, q, vol, T_years):
    if T_years <= 0 or vol <= 1e-8 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    sqT = np.sqrt(T_years)
    d1 = (np.log(S/K) + (r - q + 0.5*vol*vol) * T_years) / (vol * sqT)
    d2 = d1 - vol * sqT
    n_d1 = norm.pdf(d1)
    return (float(norm.cdf(d1)),
            float(n_d1 / (S * vol * sqT)),
            float(S * n_d1 * sqT),
            float(-(S * n_d1 * vol) / (2 * sqT) - r * K * np.exp(-r * T_years) * norm.cdf(d2)))

def get_r(date):
    rows = RATES[RATES['Date']==date]
    if len(rows): return float(rows['1y'].iloc[0])
    rows = RATES[RATES['Date']<=date].sort_values('Date')
    return float(rows['1y'].iloc[-1]) if len(rows) else 0.04

def build_state(row, position_before_action):
    T_days = float(row['T'])
    spot   = float(row['S-1'])
    K      = spot / float(row['S/K'])
    IV     = float(row['v'])
    date   = pd.to_datetime(row['Date'])
    r      = get_r(date)
    d, g, vega, theta = bs_greeks_inner(spot, K, r, Q_DIV, IV, max(T_days, 0)/365.0)
    return np.array([float(row['S/K']), T_days/30.0, position_before_action, IV,
                     d, g, vega, theta], dtype=np.float32)

def _episodes_from_csv(p):
    csv = pd.read_csv(p); csv['Date'] = pd.to_datetime(csv['Date'])
    eps = []
    for eid, edf in csv.groupby('episode', sort=False):
        edf = edf.sort_values('gstep').reset_index(drop=True)
        eps.append({'episode': int(eid), 'date_start': edf['Date'].iloc[0], 'ep_step_df': edf})
    return pd.DataFrame(eps).sort_values('date_start').reset_index(drop=True)

def pack_episodes(eps_df):
    """Pack contiguous episodes using the BS position as a state feature."""
    states, b_pos, S_old, S_new, P_old, P_new, ep_id, is_first = [], [], [], [], [], [], [], []
    for ei, (_, erow) in enumerate(eps_df.iterrows()):
        df = erow['ep_step_df']; prev_b = 0.0
        for t in range(len(df)):
            r = df.iloc[t]
            states.append(build_state(r, prev_b))
            b = float(r['B Pos'])
            b_pos.append(b); S_old.append(float(r['S-1'])); S_new.append(float(r['S0']))
            P_old.append(float(r['P-1'])); P_new.append(float(r['P0']))
            ep_id.append(ei); is_first.append(1.0 if t == 0 else 0.0)
            prev_b = b
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt, device=DEVICE)
    return dict(states=t(states), b_pos=t(b_pos), S_old=t(S_old), S_new=t(S_new),
                P_old=t(P_old), P_new=t(P_new), ep_id=t(ep_id, torch.long),
                is_first=t(is_first), n_ep=int(eps_df.shape[0]))

def _episode_pnl(pack, pos):
    p_prev = torch.cat([torch.zeros(1, device=pos.device), pos[:-1]]) * (1.0 - pack['is_first'])
    tc  = -torch.abs(pos - p_prev) * pack['S_old'] * TC_RATE
    pnl = pos * (pack['S_new'] - pack['S_old']) + (pack['P_new'] - pack['P_old']) + tc
    return _seg_sum(pnl, pack['ep_id'], pack['n_ep'])

@torch.no_grad()
def episode_delta_lambda(pack, policy, kappa, lam_vec):
    """Evaluate benchmark-relative downside-shaped rewards."""
    adj = policy(pack['states'])
    lam_step = torch.as_tensor(np.asarray(lam_vec, dtype=np.float32), device=DEVICE)[pack['ep_id']]
    pos = torch.clamp(pack['b_pos'] + lam_step * adj, -1.5, 1.5)
    r_pol = _episode_reward(pack, pos, kappa)
    r_bs  = _episode_reward(pack, pack['b_pos'], kappa)
    return (r_pol - r_bs).cpu().numpy(), r_pol.cpu().numpy(), r_bs.cpu().numpy()

@torch.no_grad()
def episode_dpnl_bp_lambda(pack, policy, lam_vec, start_spot):
    """Evaluate benchmark-relative PnL in basis points of entry notional."""
    adj = policy(pack['states'])
    lam_step = torch.as_tensor(np.asarray(lam_vec, dtype=np.float32), device=DEVICE)[pack['ep_id']]
    pos = torch.clamp(pack['b_pos'] + lam_step * adj, -1.5, 1.5)
    dp = (_episode_pnl(pack, pos) - _episode_pnl(pack, pack['b_pos'])).cpu().numpy()
    return dp / start_spot * 1e4

@torch.no_grad()
def precompute_grids(eps_df, pack, policy, kappa):
    """Construct counterfactual outcomes on the fixed trust grid."""
    n  = len(eps_df)
    ds = pd.to_datetime(eps_df['date_start']).to_numpy()
    de = np.array([pd.to_datetime(e['ep_step_df']['Date']).max()
                   for _, e in eps_df.iterrows()], dtype='datetime64[ns]')
    ss = np.array([float(e['ep_step_df']['S-1'].iloc[0]) for _, e in eps_df.iterrows()])
    iv = np.array([float(e['ep_step_df']['v'].iloc[0])  for _, e in eps_df.iterrows()])
    order = np.argsort(ds)
    d_lam  = {la: episode_delta_lambda(pack, policy, kappa, np.full(n, la))[0] for la in LAM_GRID}
    dp_lam = {la: episode_dpnl_bp_lambda(pack, policy, np.full(n, la), ss)     for la in LAM_GRID}
    return dict(n=n, ds=ds, de=de, ss=ss, iv=iv, order=order, d_lam=d_lam, dp_lam=dp_lam)
