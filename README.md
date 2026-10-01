# Nested Semantic Encoding for Bandwidth-Adaptive Mission-Critical Biosignal Telemetry in 6G Networks

Code, configurations and analysis for the article of the same name,
submitted to the *IEEE Open Journal of the Communications Society*.

A single ECG encoder is trained so that **every prefix of its
512-dimensional embedding is independently usable**. Prefix length then
acts as a transmission-rate knob: one artefact serves a 64-byte
narrowband uplink and a 2-kilobyte clinical link with no retraining, no
model swap and no revalidation.

```bash
git clone https://github.com/Abraheem13/Matryoshka-ECG_M1.git
cd Matryoshka-ECG_M1
pip install -r code/requirements.txt

make verify                                           # Linux / macOS
powershell -ExecutionPolicy Bypass -File verify.ps1   # Windows
```

No GPU, no dataset, no network, no LaTeX. About a minute.

---

## What `make verify` checks

Every number, table and figure-data file the article reports lives in
[`paper/generated/`](paper/generated). None of it is written by hand:
all of it is emitted by `code/scripts/aggregate_results.py` (measured
results), `link_budget.py` (declared arithmetic) and
`emit_config_table.py` (the configuration, read from the YAML).

`make verify` runs the generator and fails if a single character of
`paper/generated/` differs from what it produces. That is the guarantee
this repository makes about itself: **every published number is
generator output, and no measured result was typed by hand — including
in this README, which quotes none.** (The dataset counts below are
properties of the public corpus, recomputable by `dataset_stats.py`.)

It also runs the statistics unit tests, which check the DeLong test, the
paired bootstrap and the AUC implementation against `sklearn` and
against their own invariants.

**What it does not prove.** It does not re-run the experiments. The
inputs it feeds the generator are the synthetic fixture in
[`code/tests/fixtures/`](code/tests/fixtures/README.md), whose per-seed
values are constructed to have the summary statistics the article
reports. It therefore proves that the table-generating path is
deterministic and matches what was published; it is not independent
evidence for the measurements themselves. Only path C below is that.
The fixture is labelled synthetic in its own README and must never be
mixed with experimental output.

## Reproduction paths

| Path | Needs | Time | Reproduces |
|---|---|---|---|
| **A. Verify** | Python | ~1 min | that every published table, macro and figure-data file is the generator's output |
| **B. From run outputs** | a `results/` directory | ~1 min | every table, macro and figure data file, exactly |
| **C. From scratch** | PTB-XL, one GPU | ~45–55 GPU-h | the measurements themselves, within seed noise |

```bash
# A
make verify

# B  (place the run outputs at code/results/, then)
make tables

# C
code/scripts/download_data.sh
python code/scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out data/processed
cd code && ./run_all.sh          # SEEDS="0" ./run_all.sh for a ~4 h single seed
```

`run_all.sh` is idempotent: a run whose `results.json` already exists is
skipped, so an interrupted job resumes cleanly. Its stage 10 regenerates
everything in `paper/generated/`.

To check the corpus statistics without training anything, run
`python code/scripts/dataset_stats.py --raw <dir with the two PTB-XL
CSVs>`; it recomputes every dataset count from ~6.6 MB of public
metadata.

## Layout

```
code/
  mecg/
    data/dataset.py        normalisation modes, lead subsets, artefact injection
    models/backbones.py    XResNet1D, Inception1D (+SE, configurable width)
    models/heads.py        MRL, MRL-E, linear
    losses.py              multi-granularity objective
    analysis/stats.py      DeLong, paired bootstrap, TOST equivalence, Holm
    analysis/probing.py    prefix/slab probes, CKA, effective rank
    analysis/features.py   physiological descriptors (NeuroKit2)
    analysis/metrics.py    AUC / F1 / AP with tuned thresholds
  scripts/
    preprocess_ptbxl.py    PTB-XL -> raw mV .npy + metadata
    train.py               one training run
    eval_transfer.py       svd | robustness | leads | external
    run_probing.py         the representation-hierarchy test
    benchmark_hardware.py  measured latency, energy and peak memory
    quantize.py            fp16 / INT8, differential penalty
    dataset_stats.py       corpus composition from PTB-XL metadata alone
    link_budget.py         declared airtime and energy arithmetic
    emit_config_table.py   configuration table, generated from the YAML
    aggregate_results.py   results -> tables, macros and figure data
    smoke_test.py          33 checks on synthetic signals, no PTB-XL needed
  tests/                   unit tests, golden-file test, synthetic fixture
  run_all.sh               the full experiment matrix, 10 stages
paper/generated/           every published table, macro and figure data file
```

## Provenance

Results flow one way:

```
experiments -> results/*.json -> aggregate_results.py -> paper/generated/*.{tex,dat} -> article
```

Measured and derived quantities are kept apart by construction.
`link_budget.py` writes to `generated/linkbudget.tex` under an `LB` macro
prefix, and the article labels that table as derived arithmetic over
measured payload sizes. No link-budget number is presented as a network
measurement.

## What this code does and does not establish

**Does:** that diagnostic information in twelve-lead ECG representations
concentrates into a short prefix; that nesting reaches that accuracy at
parity with training a separate model per rate; that inference latency is
flat in prefix length; and that the trailing coordinates carry physiology
the five-class task does not need.

**Does not:** any network measurement, any execution on wearable or
microcontroller hardware, or any evaluation on wearable-acquired data.

## Relation to the earlier code in this repository

This replaces the original contents, which accompanied an earlier version
of the work. **Do not use the old code or its results.** Six defects were
found in it, and the current results depend on all six being fixed.

| Defect | Effect |
|---|---|
| Per-record z-scoring baked into the `.npy` files | destroyed absolute voltage, the diagnostic criterion for hypertrophy |
| SVD basis *and* classifier fitted on the test split | leakage; the curve was then plotted against honestly evaluated models |
| Weight decay grouped by `'bn' in name` | missed every BatchNorm nested inside `ConvBlock1d` |
| F1 at a fixed 0.5 threshold | understates imbalanced multi-label F1 |
| Model selection on the largest head | biased the checkpoint toward one operating point, which is the one thing a nested model must avoid |
| SE attention described but absent from the code | description/implementation mismatch |

Three further bugs surfaced only by *running* the code, the worst an
`nn.ModuleDict` integer-versus-string key lookup that made every forward
pass raise. That is the argument for running `smoke_test.py` before
committing GPU hours.

## Status

- The experimental run outputs (`results/`) and trained checkpoints are
  **not yet published here**. Until they are, path B cannot be run and
  path A is the available check. They will be attached as a release.
- No licence is attached yet, so default copyright applies; contact the
  authors about reuse.

## Data and citation

PTB-XL v1.0.3 (PhysioNet, CC BY 4.0). Expect 21,799 records and 18,869
patients; v1.0.1 and v1.0.2 differ. External corpora: CPSC-2018,
Chapman–Shaoxing/Ningbo and Georgia from PhysioNet/CinC-2020.

```bibtex
@article{rashid2026nested,
  author  = {Rashid, Abraheem and Iradat, Faisal and Iqbal, Waseem and
             Bangash, Yawar Abbas and Kumail, Muhammad},
  title   = {Nested Semantic Encoding for Bandwidth-Adaptive
             Mission-Critical Biosignal Telemetry in {6G} Networks},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026}
}
```
