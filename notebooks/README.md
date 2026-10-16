# Notebooks

Start Jupyter from the repository root and select the environment installed
from `requirements-training.txt`. Data and outputs use paths within this
repository. `python scripts/run_experiments.py` runs the training and analysis
sequence without editing any notebook.

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `00_prepare_data.ipynb` | Clean option quotes and join spot prices and Treasury yields |
| 2 | `01_train_policies_and_baselines.ipynb` | Layer-1 panel, beta search and classical baselines |
| 3 | `03_evaluate_policy_arms.ipynb` | Deployment rules; run once per `PANEL_REGIME` in a fresh kernel |
| 4 | `05_three_arm_all_regimes.ipynb` | Three-arm comparison across all regimes |
| 5 | `07_common_budget_all_regimes.ipynb` | Train and save common-budget policies and ledgers |
| 6 | `08_robustness_2022.ipynb` | Epsilon, cost and turnover analysis for 2022 |
| 7 | `09_replay_deployment.ipynb` | Completed-history controller with BS warm-up and batched ACI updates |
| 8 | `10_compare_published_rules.ipynb` | Six deployment rules across four regimes |
| 9 | `11_controller_sensitivity.ipynb` | Controller-parameter sensitivity |

The four `PANEL_REGIME` values are `crash_test2018`, `crash_test2020`,
`mix_test2022` and `mix_test2024`. The command-line runner selects each one
in turn; interactive runs change the assignment in the H0 cell before running.

`04_three_arm_setup.ipynb` defines the three-arm experiment used by
`05_three_arm_all_regimes.ipynb`. Likewise, `06_common_budget_setup.ipynb`
defines functions used by `07_common_budget_all_regimes.ipynb`; run the
all-regime notebooks to execute those experiments.

`02_compare_deployment_rules_2022.ipynb` runs the optional single-regime
comparison. The main sequence includes 2022 through
`03_evaluate_policy_arms.ipynb`. Original source names and file hashes are
recorded in `docs/source_manifest.json`.

After all stages finish, export the tables with:

```bash
bash run_reproduction.sh --mode replay --artifacts . --skip-comparison
```

For parameters, output locations and numerical comparisons, see
[the experiment guide](../docs/reproduction.md).
