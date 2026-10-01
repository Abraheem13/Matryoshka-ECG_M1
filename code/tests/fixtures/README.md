# Test fixtures

## `results/` — SYNTHETIC, NOT EXPERIMENTAL DATA

Built by `build_fixture.py`. It has the exact schema the pipeline writes
(`runs/*/results.json`, `probing.json`, `transfer.json`,
`benchmark.json`, `quantization.json`, `stats.json`), with per-seed
values constructed so that their mean and sample standard deviation
equal what the manuscript reports.

Its only use is the golden-file test, `tests/test_reproduce_tables.py`,
which proves that `scripts/aggregate_results.py` turns outputs of this
schema into exactly the manuscript's tables, macros and figure data.

It must never be presented as, or mixed with, experimental results. The
experimental outputs are the authors' `results/` directory from the
original run, published separately.

Two details of the construction are worth knowing:

- Where the manuscript reports a single-run value for seed 0 (the
  checkpoint used for probing, benchmarking and quantisation), seed 0
  takes that value and the other seeds absorb the remaining variance.
- Three rows of the robustness table print a Δ that differs by 1e-4 from
  the difference of the printed AUCs, because Δ was computed before
  rounding. The fixture uses unrounded values that round to both, which
  is the only way a real run could have produced that table.

## `dataset.json` — REAL

Computed by `scripts/dataset_stats.py` from the public PTB-XL v1.0.3
metadata (`ptbxl_database.csv`, `scp_statements.csv`) with the same label
aggregation used for training: 21,799 records, 18,869 patients, 21,388
labelled, splits 17,084 / 2,146 / 2,158, 24.1 % multi-label, zero patient
overlap between splits.

## `expected/`

The generator's output on the fixture: the snapshot the golden test
compares against. Its values were separately checked to be identical to
the tables of the original experimental run (`tests/compare_values.py`).
Regenerate it only after a deliberate formatting change, and re-run that
value comparison when you do.
