# Experiments

The experiments have three stages: prepare the option panel, train the
residual policies and evaluate the deployment rules. See
[data preparation](../data/README.md) for the inputs and cleaning notebook,
and the [notebook index](../notebooks/README.md) for execution order.
The two entry points below share the same data preparation. Pretrained
evaluation loads the supplied weights; policy training fits new models.

## Pretrained evaluation

Install `requirements-training.txt` with Python 3.12, download the inputs in
[data/README.md](../data/README.md), then run from the repository root:

```bash
python scripts/prepare_data.py
python scripts/evaluate_pretrained.py
```

The evaluator loads all 120 files listed in [the checkpoint catalog](../pretrained/manifest.json).
It constructs the validation and test episodes from the cleaned panel,
rebuilds the BS benchmark and evaluates the fixed policies on the five-point
trust grid. It then applies the v1.3 deployment controller. The validation
year sets the loss budget; test observations never change the policy weights.
Training data and previous experiment outputs are not required for this step.

To evaluate the 30 checkpoints for one test year:

```bash
python scripts/evaluate_pretrained.py --regime mix_test2022
```

Outputs are written to a new `results/generated/pretrained-<timestamp>/`:

| File | Contents |
|---|---|
| `seed_results.csv` | Each policy seed at full trust, with a validation budget, and with a common 5 bp budget |
| `policy_summary.csv` | Five-seed means, CVaR, breach rates, mean trust and bootstrap intervals |
| `table4_aggressive.csv` | Aggressive policies with validation-calibrated budgets |
| `table5_common_budget.csv` | Conservative and aggressive policies at the common 5 bp budget |
| `breach_uncertainty.csv` | Contract-weighted and date-weighted breach rates and date-block upper bounds |
| `evaluation_report.json` | Input and checkpoint hashes, package versions, episode counts and run status |
| `episodes/` | Locally generated splits, BS paths and counterfactual ledgers |

`--output` selects a new output directory. `--bootstrap-reps` defaults to the
paper's 3,000 replicates; reducing it is useful for a quick execution check.
The output directory and all generated market records are ignored by Git when
using the default location. The evaluator never modifies the supplied weights.

These checkpoints are the conservative and aggressive models from the v1.2R
common-budget runs. They support Tables 4–5 and full-trust diagnostics; the
classical baselines, three-arm comparison and other fitted experiments are
run through the training entry. The reference tables below preserve the
published results. Newly evaluated numbers depend on the data extract and
numerical environment, and are saved separately.

## Paper results

`results/reference/` contains the aggregate results below. The source files,
selected columns and checksums are recorded in `manifest.json`.
In the deployment comparison, `CTC` denotes `crc_rwc_aci`, the method labeled
"Ours" in paper Table 3. The numerical entries are unchanged.

| Paper item | Result file | Rows |
|---|---|---:|
| Figure 3: Layer-1 panel | `figure3_layer1.csv` | 12 |
| Table 1: classical baselines | `table1_classical_baselines.csv` | 16 |
| Table 2: three-arm comparison | `table2_three_arm.csv` | 12 |
| Table 3: deployment rules | `table3_deployment_rules.csv` | 24 |
| Table 4: aggressive policy | `table4_aggressive.csv` | 12 |
| Table 5: common loss budget | `table5_common_budget.csv` | 12 |
| Breach-rate uncertainty | `breach_uncertainty.csv` | 12 |
| Section 6.6: controller sensitivity | `sensitivity.csv` | 28 |
| Section 6.6: tail robustness | `robustness_tail.csv` | 12 |
| Section 6.6: transaction costs | `transaction_costs_2022.csv` | 9 |
| Section 6.6: turnover | `turnover_2022.csv` | 3 |

To export these tables and draw Figure 3:

```bash
bash run_reproduction.sh --mode reference
```

This command reads the supplied aggregate results. Figure 3 uses the paper's
values with a new plotting layout. Outputs are saved to
`results/generated/reference-<timestamp>/`; `--output` selects another new
directory.

## Deployment replay

Given saved episode records from an experiment:

```bash
bash run_reproduction.sh --mode replay --artifacts /path/to/experiment
```

Replay recalculates the grid-rule comparisons for Tables 3–5, breach-rate
bounds, controller sensitivity and tail/block-bootstrap statistics. The
continuous ACI comparison uses its recorded path with the warm-up correction.
Figure 3, Tables 1–2 and cost/turnover results are read from saved summaries.

The runner compares each exported table against the reference values and
writes CSVs, figures, logs and `verification_report.json` to
`results/generated/replay-<timestamp>/`. It reads the input directory without
modifying it. A mismatch raises an error and is recorded in the output directory.

The input directory must contain:

```text
new-reward/checkpoints/
  unified_v3/
    panel_v3_summary.csv
    perep_dp_<regime>_k<k>.csv
  e2_summary.csv
  baselines_v3/baselines_v3.csv
  conformal_controls_v12r_<regime>/
    ledger_arm00_<regime>.npz
    ledger_base_<regime>.npz
  conformal_controls_<regime>/
    dpm_arm00.npz
    lampath_arm00.npz
    controls_arms_<regime>.csv
  conformal_controls_mix_test2022/
    tc_sweep_mix_test2022.csv
    turnover_mix_test2022.csv
```

Here `<regime>` is `crash_test2018`, `crash_test2020`, `mix_test2022` or
`mix_test2024`, and `<k>` is 1, 2 or 3. These episode records are derived from
the licensed data and are not distributed in the repository.

The sensitivity notebook evaluates 11 settings, including additional epsilon
and warm-up diagnostics. The paper reports seven settings: the baseline and
variations in half-life, IV scale and ACI step size. The export therefore
contains 28 regime/setting rows. Replay also writes `robust_tail_alpha.csv`
and `robust_block.csv`.

## Training from scratch

Install `requirements-training.txt` with Python 3.12. The recorded versions are
NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3 and PyTorch 2.11.0. The notebooks use
CPU training with one thread. The requirements also retain the dependencies
from the previous training environment.

After [preparing the data](../data/README.md), run:

```bash
python scripts/run_experiments.py
```

The runner executes these stages in order, in separate Python processes:

| Stage | Work |
|---|---|
| `layer1` | Validation search for beta, four-regime residual-policy panel, classical baselines |
| `controls` | Deployment rules and policy arms for each of the four regimes |
| `three-arm` | Out-of-distribution, in-distribution and small-sample comparisons |
| `common-budget` | Saved residual policies and counterfactual ledgers for the common budget |
| `robustness` | Epsilon, transaction-cost and turnover experiments for 2022 |
| `analysis` | v1.3 deployment replay, sensitivity, bootstrap statistics and table export |

To run one stage after its prerequisites have completed:

```bash
python scripts/run_experiments.py --stage controls
```

Each invocation creates a dated directory under `results/generated/`, with
one log per notebook or regime and a `run.json` status record. The runner stops
on the first failed cell. It does not skip failed gates or overwrite notebook
source with execution outputs. It also does not automatically skip a partially
completed stage; rerunning `layer1` or `controls` repeats its training.
The common-budget stage verifies existing checkpoint hashes before reusing them.
On a fresh download it trains all policies from scratch. It does not read
`pretrained/`; newly fitted models are saved in `new-reward/checkpoints/`.

For interactive work:

```bash
python -m ipykernel install --user --name conformal-trust-control
jupyter lab notebooks
```

Select the `conformal-trust-control` kernel and follow the
[notebook execution order](../notebooks/README.md). Start Jupyter in the
repository root. Paths are resolved from that root even when a notebook
changes its working directory.

`ctc_runtime.py` constructs five-step market paths and the BS benchmark directly
from the cleaned panel. The residual-policy state uses the BS position path,
so constructing it does not require a pretrained actor, scaler or a separate
hedging-environment installation. The runtime retains the original contract
selection, normalization, implied-volatility solver and transaction-cost
conventions. For compatibility with the paper code, the recorded `v` field is
the implied volatility after the step, as in the original environment.

Local files are stored as follows:

| Path | Contents |
|---|---|
| `data/raw/` | Downloaded inputs |
| `data/SPX_2007_2024_clean_SPXW.parquet` | Cleaned option panel |
| `new-reward/data/1yr_treasury.csv` | Normalized Treasury series |
| `.runtime/hedging/` | Split CSVs and cached BS paths |
| `new-reward/checkpoints/` | Trained policies, ledgers and intermediate results |
| `results/generated/` | Logs, exported tables and figures |

These local inputs and outputs are ignored by Git. For an existing cleaned
panel stored elsewhere, `CTC_DATA` and `CTC_RATES` can select the panel and
normalized Treasury file without copying them.

The analysis stage exports the newly trained results without requiring them
to equal the published reference numbers. Different data extracts, library
versions and CPUs can change training results. To request the strict numerical
comparison separately:

```bash
bash run_reproduction.sh --mode replay --artifacts .
```

Use `--skip-comparison` to export a new experiment without this equality check.
The supplied reference CSVs are never replaced by a new run.

## Standalone residual policy

`new-reward/residual_policy.py` contains the 8–64–64–1 ReLU policy and the
mean–CVaR training objective. It adds the residual to the BS position and
clips the result to [-1.5, 1.5]. It can be used with caller-supplied tensors
without importing the market environment.

`train_residual(pack, kappa, beta)` takes a dictionary with:

| Key | Type and shape | Contents |
|---|---|---|
| `states` | float tensor `(N, 8)` | State features constructed by the training notebook |
| `b_pos` | float tensor `(N,)` | BS hedge position |
| `S_old`, `S_new` | float tensors `(N,)` | Consecutive normalized underlying prices |
| `P_old`, `P_new` | float tensors `(N,)` | Consecutive normalized option prices |
| `is_first` | float tensor `(N,)` | 1 on the first row of each episode, otherwise 0 |
| `ep_id` | int64 tensor `(N,)` | Episode index, starting at zero |
| `n_ep` | integer | Number of episodes |

Rows must be chronological within contiguous episodes. Use CPU tensors.
`eval_delta` returns the change in the downside-shaped training reward;
monetary delta-PnL in bp is calculated separately in the experiment notebooks.
Run `python scripts/smoke_training.py` for an example with synthetic inputs.

## Evaluation settings

| Test year | Training years | Validation year | Test episodes |
|---|---|---|---:|
| 2018 | 2008–2016 | 2017 | 602 |
| 2020 | 2008–2018 | 2019 | 663 |
| 2022 | 2008–2020 | 2021 | 675 |
| 2024 | 2008–2022 | 2023 | 630 |

The panel uses kappa in {1, 2, 3} and five seeds. Trust is selected from
{0, 0.25, 0.5, 0.75, 1}, checking every candidate separately. The controller
uses the BS fallback until 15 completed episodes are available. Calibration
includes only episodes completed strictly before the decision date, and
same-date completion feedback is processed in one ACI update.

Defaults are epsilon=0.1, ACI step size=0.02, recency half-life=30 days and
IV scale=0.04. Breach frequencies include warm-up episodes. The uncertainty
analysis uses 3,000 bootstrap replicates with blocks of 10 start-date batches.
The one-sided breach bounds exceed 0.1, so the observed point rates should
not be interpreted as a 95% guarantee of meeting the target.
