# Pretrained residual policies

These are the 120 saved policies from the original v1.2R common-budget runs.
The files are unchanged from the experiment archive. Each model has an
8–64–64–1 ReLU network with 4,801 parameters.

| Component | Values |
|---|---|
| Test regime | `crash_test2018`, `crash_test2020`, `mix_test2022`, `mix_test2024` |
| Policy arm | `cons`: validation-selected beta; `aggr`: beta = 0 |
| Downside penalty | kappa = 1, 2, 3 |
| Training seed | 0, 1, 2, 3, 4 |

Files follow `<regime>/<arm>_k<kappa>_seed<seed>.pt`. The full set is 2.56 MiB.
[manifest.json](manifest.json) records each file's SHA-256, beta, epoch count,
learning rate, split years and recorded training environment.

After installing the training requirements and preparing the market data:

```bash
python scripts/evaluate_pretrained.py
```

Run this command from the repository root. It verifies the checkpoints,
loads them on the CPU with `weights_only=True`, and evaluates validation and
test paths constructed from the local data. Validation determines the loss
budget; the saved policy weights and beta are fixed.

The evaluation exports full-trust and controlled performance, Tables 4–5
and breach-rate uncertainty. The model files contain no optimizer state,
market quotes or saved episode paths. Classical baselines and the additional
three-arm policies are fitted by `scripts/run_experiments.py`.

Fresh training writes to `new-reward/checkpoints/`. It does not replace these
weights. Code and weights are available under the repository's [MIT license](../LICENSE).
