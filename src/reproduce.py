"""Rebuild the best public upload from a clean clone and compare it row by row (work/AWS task 8).

    python -m src.reproduce                      # run every step, one at a time
    python -m src.reproduce --compare-only       # just compare the newest blend file with the uploaded one

Steps (each a separate process, run in order; nothing runs in parallel):
  folds -> base_v1 -> labels_clean (from the committed change list) -> flow/behaviour features in fw_s0 and
  fw_s1 -> 4 LightGBM + 4 XGBoost -> blend -> phone-id zones (src.device_block) -> zones and linked rows
  applied to the blend (src.zone_apply) -> compare with submissions/<TARGET>.
Data: the five CSV files in the repo root or data/. Writes reproduce_report.md.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TARGET = "1114-0359__L2_blend8_zones2.csv"           # best public upload, 0.5403
NAME = "REPRO_blend8_beh3"                           # the blend underneath (0.5019 on its own)
ZONE_NAME = "REPRO_blend8_zones2"
ZONE_RULES = "R1,R2"                                 # link rules the uploaded file used
RUNS = ["L2_lgbm_fw-clean_b1-f1-beh12_s0", "L2_lgbm_fw-clean_b1-f1-beh12_s1", "L2_lgbm_fw-clean_b1-f1-beh12_s2",
        "L2_lgbm_fw1-clean_b1-f1-beh12_s3"]
XGB = [("configs/L2_lgbm_fw-clean_b1-f1-beh12_s0.yaml", "L2_xgb_fw-clean_b1-f1-beh12_s0", 0),
       *[("configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_s0.yaml", f"L2_xgb_fw-clean_b1-f1-beh123-lh2_s{s}", s) for s in (0, 1, 2)]]
STEPS = [
    ["-m", "src.folds"],
    ["-m", "src.base_features"],
    ["-m", "src.labels_clean", "--from-changes"],
    *[["-m", f"src.{m}", "--view", v] for v in ("fw_s0", "fw_s1") for m in ("flow_features", "behaviour_features", "behaviour2_features")],
    ["-m", "src.behaviour3_features", "--view", "fw_s0"],
    ["-m", "src.labelhist_features", "--labels", "clean"],
    *[["-m", "src.train", "--config", f"configs/{r}.yaml"] for r in RUNS],
    *[["-m", "src.train_xgb", "--config", c, "--run-id", r, "--seed", str(s)] for c, r, s in XGB],
    ["-m", "src.blend", "--name", NAME, "--runs", *RUNS, *[r for _, r, _ in XGB]],
    ["-m", "src.device_block", "--runs", *RUNS, *[r for _, r, _ in XGB]],
    ["-m", "src.zone_apply", "--propagate", "--rules", ZONE_RULES, "--name", ZONE_NAME, "--base", "{BLEND}", "--runs", *RUNS, *[r for _, r, _ in XGB]],
]


def newest(name: str) -> str:
    found = sorted(glob.glob(str(ROOT / "submissions" / f"*{name}.csv")))
    return found[-1] if found else ""


def compare() -> str:
    new = sorted(glob.glob(str(ROOT / "submissions" / f"*{ZONE_NAME}.csv")))
    if not new:
        return "no reproduced file found"
    a, b = pl.read_csv(new[-1]), pl.read_csv(ROOT / "submissions" / TARGET)
    j = a.join(b, on="id", suffix="_up")
    diff = j.filter(pl.col("label") != pl.col("label_up"))
    lines = [f"reproduced: {Path(new[-1]).name}", f"uploaded:   {TARGET}",
             f"rows: {a.height} vs {b.height}; same ids: {set(a['id']) == set(b['id'])}", f"rows with a different label: {diff.height}"]
    if diff.height:
        lines.append("differences (uploaded -> reproduced): " + ", ".join(f"{x} -> {y}: {n}" for x, y, n in
                     diff.group_by("label_up", "label").agg(n=pl.len()).sort("n", descending=True).head(10).rows()))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    log = []
    if not args.compare_only:
        for step in STEPS:
            t0 = time.time()
            step = [newest(NAME) if a == "{BLEND}" else a for a in step]
            r = subprocess.run([sys.executable, *step], cwd=ROOT, capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
            log.append(f"- `{' '.join(step[1:])}`: {status}, {time.time() - t0:.0f} s")
            print(log[-1], flush=True)
            if r.returncode != 0:
                log.append("```\n" + r.stderr[-2000:] + "\n```")
                break
    result = compare()
    print(result)
    if args.compare_only:
        return
    (ROOT / "reproduce_report.md").write_text("# Reproduce report\n\n## Steps\n\n" + "\n".join(log) + "\n\n## Comparison\n\n```\n" + result + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
