# Tests and numerical checks

Run the tests from the repository root:

```bash
python -m pip install -r requirements-reproduce.txt pyarrow
python scripts/verify_release.py
python -m unittest discover -s tests -v
bash run_reproduction.sh --mode reference
```

The tests cover controller timing, warm-up, the BS fallback, nonmonotone
candidate feasibility and batched feedback. They also check data cleaning
on constructed quotes, numeric comparisons, output-directory handling, contract selection, BS accounting
and cache invalidation.
`verify_release.py` checks source/result hashes and excludes data files and
saved notebook outputs from the repository.
It also verifies all 120 published checkpoints against their catalog and
rejects unlisted model files. With PyTorch installed, the tests load every
model and check inference, BS fallback accounting and unchanged weights.

The Layer-1 training test uses synthetic episodes and requires PyTorch:

```bash
python scripts/smoke_training.py
```

## Pretrained checkpoints

The [pretrained validation report](pretrained_validation.json) records a full
evaluation of all 120 supplied checkpoints on the original local cleaned
panel, with 3,000 bootstrap replicates. The test years contain 602, 663, 675
and 630 episodes, matching the paper. All checkpoint files match the original
training archive, and their hashes remain unchanged after evaluation.
The test suite passes all 22 tests when PyTorch is installed, including the
check that rejects notebook execution metadata even when outputs are empty.

This run recalculates results from market data and model weights. It does not
exactly reproduce every archived number. Two 2022 test selections have tied
distance scores; the local sorting order chooses different contracts from
the recorded experiment. The original ranking algorithm is retained. Other
differences arise in the reconstructed numerical arrays under the local
environment; the checkpoint weights themselves are identical.

Across the 12 rows, the maximum absolute difference in mean PnL is 0.29625 bp
for Table 4 and 0.05259 bp for the aggressive arm in Table 5. The largest
Table 4 breach-rate difference is 0.000297 (about 0.030 percentage points).
The report lists differences for every numerical column. Published reference
tables remain unchanged.

The separate-copy workflow test also passes data preparation, all 120
pretrained evaluations and the complete training workflow on synthetic data.
It uses two training epochs to check execution. Full empirical retraining
was not performed as part of this check.

## Agreement with the paper

The [paper-data check](paper_data_validation.json) compares the cleaned panel
with the data specification in Section 5 and the reference tables with the
paper PDF. All 5,068,896 cleaned rows satisfy the documented quote, volatility,
intrinsic-value, maturity and moneyness filters. The four test sample sizes
are listed in [the data guide](../data/README.md#cleaned-panel).

All 384 numerical entries in Tables 1–5 match the reference CSVs when rounded
to the precision printed in the paper. In Table 3, the repository uses `CTC`
as the display name for `crc_rwc_aci`. This is a label change only.

## Replay from saved episode records

The [saved replay report](replay_validation.json) compares 11 exported tables
with the paper references. All comparisons have a maximum absolute difference
of 0.0. Controller, sensitivity and bootstrap calculations are rerun from the
saved episode records; training summaries and the continuous ACI path are
loaded from the experiment archive.

The comparison used Python 3.12.10, NumPy 2.4.6, pandas 3.0.3, SciPy 1.16.2
and Matplotlib 3.10.6. These direct dependencies are pinned in
`requirements-reproduce-tested.txt`. The synthetic training test used PyTorch
2.8.0 on CPU. The original training versions are in
`requirements-training.txt`; full empirical retraining was not part of these
checks.

## Local workflow

Run the complete workflow on synthetic quotes:

```bash
python scripts/smoke_workflow.py
```

This requires the training dependencies. It creates a separate copy under
`results/generated/`, builds synthetic input files, and runs data preparation,
all four training regimes, the three-arm experiment, common-budget policies,
robustness and result export. It also evaluates all 120 pretrained policies
on those synthetic quotes and checks that both entry points leave the supplied
weights unchanged. That copy uses two training epochs and 20
bootstrap replicates. The original notebooks retain 400 epochs and 3,000
replicates. Logs and a status report remain in the test directory.

The [local-runtime validation report](portability_validation.json) records:

- 16 passing tests and syntax checks for 121 notebook code cells.
- Exact agreement with the previous environment on all 675 selected 2022
  contract paths and 3,375 BS steps, including positions, volatility,
  transaction costs and rewards.
- Construction of the 2018 split from the local licensed panel, followed by
  the reward-replay check and an eight-epoch fit on 24 training episodes.
- A successful synthetic run through every stage of the command-line workflow.

These checks used PyTorch 2.8.0 and the package versions recorded in the report.
They test execution and the runtime migration. They do not substitute for
full empirical training with the recorded training environment.
