# mlgw-2026: UPI Fraud Rings

ML Gradient Wars 2026. Read `research.md` for the plan and `work/README.md` for who does what.

## Setup (every machine)

1. Clone the repo.
2. Copy the five data files (`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`, `sample_submission.csv`) into the repo root or into `data/`. They are not in git.
3. Create the environment:

Windows:
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS:
```
brew install libomp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Build the shared files. These are deterministic, so every machine gets identical output:

```
python -m src.folds
python -m src.base_features
python -m src.windows
```

`src.folds` must print `sha256=6a366df1a48b8439`. If yours differs, your data files differ: stop and tell the team.
`src.windows` is a self-test: at `half_s0` the train months should show about 141 other payments per receiver in 24 hours, close to the test month's 143.

## Train a model

```
python -m src.train --config configs/L1_lgbm_base1_s0.yaml --mode sep
python -m src.train --config configs/L1_lgbm_base1_s0.yaml
python -m src.postprocess --run L1_lgbm_base1_s0
```

- `--mode sep` gives only the honest score (train Jul+Aug, validate Sep). Use it while iterating.
- Without it you also get `oof/<run_id>.parquet`, `preds/<run_id>.parquet` and `runs/<run_id>.json`.
- `src.postprocess` fits the class weights for macro-F1 and writes `submissions/<run_id>.csv`.

Commit your `runs/<run_id>.json` and your config. Sync `features/`, `oof/` and `preds/` through the shared bucket.

## Writing a feature group

Use `src/windows.py` for anything that looks at other rows. It applies the half-density rule for you.

```python
from datetime import timedelta
import polars as pl
from src.common import load_raw, save_features
from src.windows import build_events, window_agg, gap_features, attach

VIEW = "half_s0"
raw = load_raw()
ev = build_events(raw, VIEW)                       # look-backs up to 7 days by default
f1 = window_agg(ev, "receiver_id", timedelta(hours=1), "past",
                [pl.len().alias("rcv_n_past_1h"),
                 pl.col("sender_id").n_unique().alias("rcv_senders_past_1h")])
f2 = gap_features(ev, "device_id", "dev")
save_features(attach(raw, f1, f2), "flow_v1", VIEW)
```

Rules the code enforces: one row per `txn_id`, all rows present, no `label`, no raw `app_version`, no raw `ip_risk_score`. Prefix integer category columns with `cat_`. Feature names must be unique across groups, so prefix them (`rcv_`, `snd_`, `dev_`, `pair_`, `mer_`).

## Layout

```
src/common.py         constants, paths, loaders, label order
src/folds.py          folds.csv (month folds + half-density flags)
src/metric.py         macro-F1 and the per-class table
src/base_features.py  single-row features (group base_v1)
src/windows.py        shared window library with the density rule
src/train.py          config-driven LightGBM trainer
src/postprocess.py    weighted argmax, submission writer
configs/              one YAML per run
runs/                 one JSON per run (committed)
work/                 per-machine task files
```
