# Data

The experiments use daily S&P 500 option quotes from
[OptionMetrics IvyDB](https://optionmetrics.com/data-products/), obtained through
Imperial College London. The input covers 2007–2024; the expanding-window
experiments start in 2008. The cleaned panel contains PM-settled SPXW calls.

Index levels are daily S&P 500 closes, downloaded from Yahoo Finance as
`^GSPC`. The risk-free input is the one-year U.S. Treasury yield.
[FRED DGS1](https://fred.stlouisfed.org/series/DGS1) provides a daily series for
rebuilding this input. The notebook accepts either a FRED CSV in percent
or the experiment's `Date,1y` CSV in decimal units.

OptionMetrics quotes are not included because the institutional subscription
does not permit redistribution. Obtain the quotes through your institution
or directly from the provider.

## Prepare the inputs

Export daily SPX option quotes for 2007-01-01 through 2024-12-31 to a parquet
file. `SPX0025.parquet` is the filename used in the original experiment, not a
vendor product identifier. The export must include:

```text
date, exdate, cp_flag, strike_price, best_bid, best_offer,
volume, open_interest, impl_volatility, delta, gamma, vega,
theta, forward_price, root, am_settlement
```

`date` and `exdate` are dates or timestamps. `cp_flag` identifies calls as `C`;
`am_settlement=0` selects PM settlement. Keep `strike_price` in the vendor's
original units: the notebook divides it by 1,000. Quotes are in index points
and implied volatility is a decimal.

For the Treasury input, download the daily CSV from the DGS1 page above,
covering the same period. The notebook recognizes `observation_date,DGS1`
(or `DATE,DGS1`) and converts percent to decimal. An existing `Date,1y` file
must already use decimals: a 5% yield is stored as `0.05`.

For the spot input, either supply a CSV with `date,spot` columns or let the
notebook download `^GSPC` closes through `yfinance`. A saved copy is useful
when repeating an experiment because providers can revise historical data.

## Run the cleaning notebook

Place the files in `data/raw/`:

```text
data/raw/SPX0025.parquet
data/raw/DGS1.csv
data/raw/spot.csv
```

Then, from the repository root:

```bash
python scripts/prepare_data.py
```

The script executes `notebooks/00_prepare_data.ipynb`. To inspect each cleaning
step interactively, run `jupyter lab notebooks/00_prepare_data.ipynb` and select
the Python environment used for installation. Both entry points use the same
code and write:

```text
data/SPX_2007_2024_clean_SPXW.parquet
new-reward/data/1yr_treasury.csv
```

The Treasury output uses decimal rates. These files feed directly into
both `python scripts/evaluate_pretrained.py` and
`python scripts/run_experiments.py`; no files need to be moved afterwards.
Downloaded data, locally trained checkpoints and runtime caches are excluded
by `.gitignore`. The 120 supplied policy weights are stored in `pretrained/`.

For inputs stored elsewhere, pass their paths explicitly:

```bash
python scripts/prepare_data.py --options /path/to/options.parquet \
  --rates /path/to/DGS1.csv --spot /path/to/spot.csv
```

Omit `--spot` and `data/raw/spot.csv` to download index closes with `yfinance`.
The notebook also accepts `SPX0025_PATH`, `CTC_TREASURY_PATH` and `CTC_SPOT_CSV`
environment variables. `CTC_WORKSPACE` changes the output root for a separate
experiment; ordinary use requires none of these variables.

## Cleaning procedure

The notebook applies the following filters in order:

1. Retain PM-settled quotes (`am_settlement=0`).
2. Require positive bid and ask prices, with ask at least bid.
3. Remove quotes whose bid–ask spread exceeds 50% of the midpoint.
4. Require a finite, positive implied volatility.
5. Join index closes by date and remove unmatched observations.
6. Require the option midpoint to be at least its intrinsic value.
7. Retain 10–90 calendar days to expiry and spot/strike in [0.83, 1.17].
8. Retain calls and join the Treasury series, carrying the last available
   rate forward across missing dates.

Each stage reports the number of retained observations. Rows are then sorted
by expiration, strike and date. `nbr_next_steps` counts the observations
remaining in each contract path; episode construction and train/validation/test
splits take place in the training notebooks.

## Cleaned panel

The paper's cleaned panel contains 5,068,896 rows, spanning 2007-02-21 to
2024-12-31. Section 5 rounds this to 5.07 million rows. The chronological
experiments use these splits:

| Training years | Validation year | Test year | Test episodes |
|---|---|---|---:|
| 2008–2016 | 2017 | 2018 | 602 |
| 2008–2018 | 2019 | 2020 | 663 |
| 2008–2020 | 2021 | 2022 | 675 |
| 2008–2022 | 2023 | 2024 | 630 |

Each episode has five trading steps. Start dates are spaced five trading days
apart, with up to 15 contracts selected from the strike/expiry grid described
in Section 5. Missing grid cells are omitted.

| Columns | Description |
|---|---|
| `date`, `quote_datetime` | Observation date and environment-compatible timestamp |
| `ticker`, `expiration`, `strike` | Underlying and contract identifiers |
| `underlying_bid`, `underlying_ask` | Both set to the daily index close |
| `bid`, `ask` | Option quote in index points |
| `dtm` | Calendar days to expiry |
| `impl_volatility`, `delta` | Vendor implied volatility and delta |
| `r` | One-year Treasury yield, in decimal units |
| `nbr_next_steps` | Remaining observations in the contract path |

The timestamp uses `09:31:00` to match the environment's input format; the
source observations are daily quotes, not intraday observations at that time.
The archived cleaned panel has MD5 `d955fc81bdd441021ffbc4715e4d124d`.
A new vendor extract or revised spot/rate series may produce a different file.
