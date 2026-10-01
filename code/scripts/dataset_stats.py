#!/usr/bin/env python3
"""
Dataset composition from PTB-XL metadata alone -- no waveforms needed.

Every count the manuscript states about the corpus (records, patients,
labelled records, split sizes, per-class counts) is produced here from
ptbxl_database.csv and scp_statements.csv, using the *same* label
aggregation that preprocess_ptbxl.py applies before training. Because it
needs only the two metadata files (~6.6 MB), anyone can check the
manuscript's dataset numbers in seconds without downloading the signals.

Writes results/dataset.json, which aggregate_results.py turns into the
composition table and the RecordCount / PatientCount / LabelledCount
macros.

    python scripts/dataset_stats.py --raw data/ptbxl_raw --out results/dataset.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_ptbxl import aggregate_labels  # noqa: E402  same logic as training


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/ptbxl_raw")
    ap.add_argument("--task", default="superdiagnostic")
    ap.add_argument("--out", default="results/dataset.json")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.raw, "ptbxl_database.csv"),
                     index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(os.path.join(args.raw, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    n_raw = int(len(df))
    p_raw = int(df.patient_id.nunique())

    df["labels"] = aggregate_labels(df, agg, args.task)
    lab = df[df["labels"].apply(len) > 0]

    classes = sorted({c for row in lab["labels"] for c in row})
    counts = {c: int(sum(c in row for row in lab["labels"])) for c in classes}
    folds = lab.strat_fold.values
    splits = {"train": int(np.isin(folds, range(1, 9)).sum()),
              "val": int((folds == 9).sum()),
              "test": int((folds == 10).sum())}
    multi = float((lab["labels"].apply(len) > 1).mean())

    # patient-disjointness of the official folds, checked rather than assumed
    pid_by_split = {
        "train": set(lab.patient_id[np.isin(folds, range(1, 9))]),
        "val": set(lab.patient_id[folds == 9]),
        "test": set(lab.patient_id[folds == 10]),
    }
    overlap = (len(pid_by_split["train"] & pid_by_split["val"])
               + len(pid_by_split["train"] & pid_by_split["test"])
               + len(pid_by_split["val"] & pid_by_split["test"]))

    out = {
        "source": "PTB-XL metadata (ptbxl_database.csv, scp_statements.csv)",
        "task": args.task,
        "n_records_raw": n_raw,
        "n_patients_raw": p_raw,
        "n_records_labelled": int(len(lab)),
        "n_patients_labelled": int(lab.patient_id.nunique()),
        "splits": splits,
        "class_counts": counts,
        "multi_label_fraction": multi,
        "patient_overlap_between_splits": int(overlap),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"records {n_raw:,}  patients {p_raw:,}  "
          f"labelled {len(lab):,}  multi-label {100*multi:.1f}%")
    print("splits  " + "  ".join(f"{k} {v:,}" for k, v in splits.items()))
    print("classes " + "  ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"patient overlap between splits: {overlap}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
