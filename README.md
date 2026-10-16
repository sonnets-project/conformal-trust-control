# Conformal Trust Control

Code and experiments for **Conformal Trust Control for Learned Sequential
Policies under Regime Shift**, by Jiajun Li, Ce Guo and Wayne Luk, ICAIF '26.
[Paper](https://doi.org/10.1145/3858063.3858769) · [Citation](CITATION.cff)

The method combines a Black–Scholes residual hedging policy with an
online conformal controller that adjusts how much of the learned policy
is deployed. Experiments use SPXW options with held-out test years 2018,
2020, 2022 and 2024.

## Installation

Use Python 3.12 and run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

The requirements retain the NumPy, pandas, SciPy and PyTorch versions recorded
by the original training notebooks. Training runs on the CPU with one thread.

## Data

The experiments use OptionMetrics IvyDB SPXW quotes accessed through Imperial
College London, daily S&P 500 closes from Yahoo Finance, and one-year U.S.
Treasury yields. The institutional data license does not allow redistribution.

Download the inputs described in [data/README.md](data/README.md) and place them here:

```text
data/raw/SPX0025.parquet   # OptionMetrics quotes, 2007–2024
data/raw/DGS1.csv         # Treasury yields from FRED
data/raw/spot.csv         # Daily index closes: date,spot
```

If `spot.csv` is absent, the cleaning notebook downloads `^GSPC` closes.

## Running the experiments

Prepare the data once:

```bash
python scripts/prepare_data.py
```

This runs [the cleaning notebook](notebooks/00_prepare_data.ipynb). Then choose:

### Evaluate the pretrained policies

```bash
python scripts/evaluate_pretrained.py
```

Loads the 120 supplied checkpoints and evaluates both policy arms across all
four test years. It computes full-trust performance and the controlled results
for Tables 4–5, including breach rates and bootstrap intervals. No training is
needed. Results go to `results/generated/pretrained-<timestamp>/`.
For one test year, add `--regime mix_test2022`.

### Train from scratch

```bash
python scripts/run_experiments.py
```

Runs validation selection, policy training, all four test regimes, baselines,
the three-arm comparison and robustness experiments, then exports the paper
tables and Figure 3. A full run includes hundreds of policy fits; allow several
hours on a CPU. New models are saved under `new-reward/checkpoints/`, separately
from the supplied weights in `pretrained/`. Logs and tables go to `results/generated/`.

The supplied checkpoints cover the conservative and aggressive policies used
in the common-budget experiment. Other fitted comparisons require the training
entry above. Results from a different data extract or software environment
can differ from the published numbers.
The [validation notes](docs/validation.md#pretrained-checkpoints) describe the
measured differences when rebuilding paths for the supplied checkpoints.

To run individual stages or use Jupyter, see the
[experiment guide](docs/reproduction.md) and [notebook index](notebooks/README.md).

To view the supplied paper results without downloading data or training:

```bash
python -m pip install -r requirements-reproduce.txt
bash run_reproduction.sh --mode reference
```

## Repository contents

| Directory | Contents |
|---|---|
| `notebooks/` | Data cleaning, policy training and experiment notebooks |
| `new-reward/` | Residual policy, deployment controller and analysis code |
| `scripts/` | Data preparation and experiment runners |
| `pretrained/` | 120 original residual-policy checkpoints and their SHA-256 catalog |
| `results/reference/` | Aggregate results for Figure 3, Tables 1–5 and Section 6.6 |
| `tests/` | Controller and reproduction tests |

See [tests and numerical checks](docs/validation.md) for test commands.
The code and supplied model weights are licensed under [MIT](LICENSE); [data terms](DATA_LICENSE.md)
apply separately.
