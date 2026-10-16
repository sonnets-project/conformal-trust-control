"""Black–Scholes residual policy and mean–CVaR training.

The model accepts episode tensors on CPU. State construction and the tensor
schema are described in docs/reproduction.md.
"""
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cpu")
TC_RATE = 0.001
COLLAPSE_TH, N_RETRY = 1e-3, 1

class ResidualPolicy(nn.Module):
    def __init__(self, dim=8, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # initial≈BS
    def forward(self, s): return self.net(s).squeeze(-1)

def _seg_sum(vals, ep_id, n_ep):
    out = torch.zeros(n_ep, device=vals.device, dtype=vals.dtype)
    return out.index_add_(0, ep_id, vals)

def _episode_reward(pack, pos, kappa):
    p_prev = torch.cat([torch.zeros(1, device=pos.device), pos[:-1]]) * (1.0 - pack['is_first'])
    tc  = -torch.abs(pos - p_prev) * pack['S_old'] * TC_RATE
    pnl = pos * (pack['S_new'] - pack['S_old']) + (pack['P_new'] - pack['P_old']) + tc
    q = pnl * 100.0
    step_r = (q - kappa * torch.relu(-q)) * 10.0          # new downside reward, differentiable (relu)
    return _seg_sum(step_r, pack['ep_id'], pack['n_ep'])

def _positions(pack, policy):
    adj = policy(pack['states'])
    return torch.clamp(pack['b_pos'] + adj, -1.5, 1.5)     # position = BS delta + residual (B Pos=-delta∈[-1,0])

def _train_residual_once(pack_tr, kappa, beta, alpha=0.05, epochs=400, lr=1e-3, seed=0,
                         init_std=0.01, warm_frac=0.25):
    """single training run: small random init of the last layer + pure mean for the first warm_frac(β=0) + β ramp linearly to the target (v9)."""
    torch.manual_seed(seed); np.random.seed(seed)
    policy = ResidualPolicy().to(DEVICE)
    torch.nn.init.normal_(policy.net[-1].weight, std=init_std)
    torch.nn.init.zeros_(policy.net[-1].bias)
    eta = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam(list(policy.parameters()) + [eta], lr=lr)
    bs_r = _episode_reward(pack_tr, pack_tr['b_pos'], kappa).detach()
    for ep in range(epochs):
        bt = 0.0 if ep < warm_frac * epochs else \
             beta * min(1.0, (ep - warm_frac*epochs) / max(1.0, warm_frac*epochs))
        opt.zero_grad()
        delta = _episode_reward(pack_tr, _positions(pack_tr, policy), kappa) - bs_r
        cvar = eta - (1.0 / alpha) * torch.relu(eta - delta).mean()
        loss = -(delta.mean() + bt * cvar)
        loss.backward(); opt.step()
    return policy

def _adj_scale(policy, pack):
    with torch.no_grad():
        return policy(pack['states']).abs().mean().item()

def train_residual(pack_tr, kappa, beta, alpha=0.05, epochs=400, lr=1e-3, seed=0, **kw):
    """v9 = warmup training + auto-restart on collapse (the criterion looks only at the train side |adj|, does not touch test; seed+1000 retry)."""
    policy = _train_residual_once(pack_tr, kappa, beta, alpha, epochs, lr, seed, **kw)
    a = _adj_scale(policy, pack_tr); tries = 0
    while a < COLLAPSE_TH and tries < N_RETRY:
        tries += 1
        print(f'  ↻ collapse-retry{tries}: κ={kappa} β={beta} seed {seed}→{seed+1000*tries} (|adj|={a:.5f})')
        p2 = _train_residual_once(pack_tr, kappa, beta, alpha, epochs, lr, seed + 1000*tries, **kw)
        a2 = _adj_scale(p2, pack_tr)
        if a2 > a: policy, a = p2, a2
    if a < COLLAPSE_TH:
        print(f'  ⚠ collapse persists: κ={kappa} β={beta} seed={seed} (|adj|={a:.5f})')
    return policy

@torch.no_grad()
def eval_delta(pack, policy, kappa):
    bs_r = _episode_reward(pack, pack['b_pos'], kappa)
    return (_episode_reward(pack, _positions(pack, policy), kappa) - bs_r).cpu().numpy()

@torch.no_grad()
def eval_rewards(pack, policy, kappa):
    r_pol = _episode_reward(pack, _positions(pack, policy), kappa).cpu().numpy()
    r_bs  = _episode_reward(pack, pack['b_pos'], kappa).cpu().numpy()
    return r_pol, r_bs
