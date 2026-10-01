# Runbook

Three ways to reproduce, cheapest first. Each is one command.

| Path | Needs | Time | Reproduces |
|---|---|---|---|
| **A. Verify** | Python | ~1 min | that every published table, macro and figure-data file is the generator's output |
| **B. From run outputs** | a `results/` directory | ~1 min | every table, macro and figure data file, exactly |
| **C. From scratch** | PTB-XL, one GPU | ~45–55 GPU-h | the measurements themselves, within seed noise |

---

## Environment

```bash
conda env create -f environment.yml && conda activate mecg
# or
pip install -r code/requirements.txt
```

Python 3.10 or newer. Path A needs only `numpy`, `scipy`, `scikit-learn`
and `pyyaml`; paths B and C additionally need `torch`.

## A. Verify (no data, no GPU)

```bash
make verify                                           # Linux / macOS
powershell -ExecutionPolicy Bypass -File verify.ps1   # Windows
```

In order: the statistics unit tests (DeLong, paired bootstrap and AUC
against `sklearn`); regeneration of the derived quantities (link budget,
configuration table); and the golden-file check, which regenerates every
file in `paper/generated/` from the fixture and fails if one character
differs.

Read `code/tests/fixtures/README.md` before interpreting the result: the
fixture is synthetic, so path A establishes that the published numbers
are the generator's output, not that the measurements are correct.

## B. From run outputs, without retraining

```bash
# place the run outputs so that code/results/runs/<run>/results.json exists
make tables
```

This regenerates every table, macro and figure data file in
`paper/generated/` from measured output. Expect small, legitimate
differences against the committed files where the article printed a
value rounded from a longer one.

The run outputs are not yet published in this repository; see **Status**
in `README.md`.

## C. From scratch

```bash
code/scripts/download_data.sh                      # PTB-XL v1.0.3, ~1.7 GB
python code/scripts/preprocess_ptbxl.py \
    --raw data/ptbxl_raw --out data/processed
cd code
SEEDS="0" ./run_all.sh     # one seed, whole matrix, ~4 h
./run_all.sh               # full matrix, ~45-55 GPU-h
```

`run_all.sh` has ten stages: dataset statistics, training, fixed-rate
baselines, ablations, reduced-lead training, robustness, probing,
hardware benchmarking, quantisation, and aggregation. Stage 10 writes
`paper/generated/` and runs the checks.

It is idempotent: a run whose `results.json` already exists is skipped,
so an interrupted job resumes. Delete a run directory to force a redo.

Before spending GPU hours:

```bash
make smoke      # the whole pipeline on synthetic signals, ~20 min CPU
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `test_reproduce_tables` fails after you edit a generator | intended: the golden snapshot no longer matches. Review the diff it prints, then refresh `code/tests/fixtures/expected/`. |
| `make tables` output differs from the committed files | expected when regenerating from real run outputs; see path B. |
| ONNX export warning in `benchmark_hardware.py` | `onnxscript` missing; the benchmark records the failure and continues. |
| NeuroKit2 import error in `analysis/features.py` | only the probing stage needs it: `pip install neurokit2`. |
