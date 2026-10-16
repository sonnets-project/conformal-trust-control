#!/usr/bin/env python3
"""Train the original Layer-1 model on synthetic tensors; not a paper experiment."""
from pathlib import Path
import runpy

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    method = runpy.run_path(str(ROOT / "new-reward/residual_policy.py"))
    torch.set_num_threads(1)
    rng = np.random.default_rng(23)
    episodes, steps = 24, 5
    n = episodes * steps
    spot = np.concatenate([np.ones((episodes, 1)),
                           np.exp(np.cumsum(rng.normal(0, .01, (episodes, steps)), axis=1))], axis=1)
    option = .05 + .5 * (spot - 1) + rng.normal(0, .0003, spot.shape)
    first = np.tile([1., 0., 0., 0., 0.], episodes)
    states = np.column_stack([spot[:, :-1].ravel(), np.ones(n), -.5 * (1 - first),
                              np.full(n, .2), np.full(n, .5), np.full(n, .01),
                              np.full(n, .1), np.full(n, -.01)])
    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    pack = dict(states=tensor(states), b_pos=tensor(np.full(n, -.5)),
                S_old=tensor(spot[:, :-1].ravel()), S_new=tensor(spot[:, 1:].ravel()),
                P_old=tensor(option[:, :-1].ravel()), P_new=tensor(option[:, 1:].ravel()),
                is_first=tensor(first), ep_id=torch.arange(episodes).repeat_interleave(steps),
                n_ep=episodes)
    initial = method["ResidualPolicy"]()
    np.testing.assert_array_equal(method["eval_delta"](pack, initial, 2), np.zeros(episodes))
    fitted = method["train_residual"](pack, kappa=2, beta=1, epochs=8, seed=7)
    delta = method["eval_delta"](pack, fitted, 2)
    if delta.shape != (episodes,) or not np.isfinite(delta).all():
        raise AssertionError("Layer-1 training produced invalid episode rewards")
    pos = method["_positions"](pack, fitted)
    if not bool(((pos >= -1.5) & (pos <= 1.5)).all()):
        raise AssertionError("Position bounds violated")
    print(f"Synthetic Layer-1 training: PASS ({episodes} episodes, 8 epochs, torch {torch.__version__})")
    print("This smoke test uses no market data and does not reproduce empirical paper results.")


if __name__ == "__main__":
    main()
