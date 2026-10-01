#!/usr/bin/env python3
"""
Build the golden-test fixture: a results/ tree with the real schema whose
summary statistics equal the values printed in the manuscript.

READ THIS BEFORE USING THE OUTPUT
---------------------------------
This is NOT experimental data. The per-seed values are synthetic,
constructed so that their mean and sample standard deviation equal what
the manuscript reports (and, where the manuscript reports a single-run
value for seed 0, so that seed 0 takes it). Its only purpose is to let
tests/test_reproduce_tables.py prove that scripts/aggregate_results.py
regenerates the manuscript's tables and macros from outputs of this
schema. The authors' experimental outputs are the results/ directory of
the original run.

The single exception is dataset.json, which is copied from the real
computation over the PTB-XL v1.0.3 metadata (scripts/dataset_stats.py).

A side effect worth recording: the construction exposes the manuscript's
numbers to internal-consistency constraints. For example, the seed-0
cross-backbone differences in the statistics table must be compatible with
the Inception1D seed-0 AUCs and the XResNet1D-101 mean and SD. They are:
taking the upper XResNet seed as seed 0 reproduces the reported paired
differences to within 1e-4 at d=16 and d=512.

    python tests/fixtures/build_fixture.py --out tests/fixtures/results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil

DIMS = [16, 32, 64, 128, 256, 512]
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))

# ---- values as printed in the manuscript ------------------------------------
HEAD_MEAN = {16: .9269, 32: .9268, 64: .9267, 128: .9266, 256: .9266, 512: .9263}
HEAD_SD = {16: .0018, 32: .0020, 64: .0019, 128: .0020, 256: .0018, 512: .0021}
HEAD_SEED0 = {16: .9264, 512: .9251}      # the probed / benchmarked checkpoint
XRES_MEAN = {16: .9104, 32: .9101, 64: .9099, 128: .9095, 256: .9090, 512: .9086}
XRES_SD = {16: .0019, 32: .0022, 64: .0021, 128: .0017, 256: .0014, 512: .0019}
FIXED_INC = {16: .9264, 32: .9241, 64: .9238, 128: .9247, 256: .9241, 512: .9268}
PARAMS_FIXED = {16: 4080645, 32: 4097157, 64: 4130181, 128: 4196229,
                256: 4328325, 512: 4592517}
PERCLASS = {  # CD HYP MI NORM STTC
    16: [.9206, .9046, .9292, .9456, .9344], 32: [.9209, .9036, .9292, .9457, .9344],
    64: [.9207, .9035, .9292, .9457, .9345], 128: [.9206, .9032, .9289, .9458, .9344],
    256: [.9204, .9033, .9288, .9458, .9345], 512: [.9199, .9036, .9285, .9456, .9341],
}
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

# (condition, backbone, head, lead_subset, norm_mode, {dim: auc})
SINGLE_SEED = [
    ("inception1d_mrle", "inception1d", "mrl-e", "12lead", "dataset",
     {16: .9212, 512: .9190}),
    ("inception1d_mrl_perrecord", "inception1d", "mrl", "12lead", "per_record",
     {16: .9102}),
    ("inception1d_mrl_ls01", "inception1d", "mrl", "12lead", "dataset", {16: .9257}),
    ("inception1d_mrl_wlinear", "inception1d", "mrl", "12lead", "dataset",
     {16: .9237}),
    ("inception1d_mrl_wexponential", "inception1d", "mrl", "12lead", "dataset",
     {16: .9259}),
    ("inception1d_mrl_winverse", "inception1d", "mrl", "12lead", "dataset",
     {16: .9244}),
    ("inception1d_constant", "inception1d", "mrl", "12lead", "dataset", {16: .9205}),
    ("xresnet1d50_mrl", "xresnet1d50", "mrl", "12lead", "dataset",
     {16: .9153, 512: .9126}),
    ("inception1d_mrl_3lead", "inception1d", "mrl", "3lead", "dataset",
     {16: .8924, 512: .8913}),
    ("inception1d_mrl_lead_I", "inception1d", "mrl", "lead_I", "dataset",
     {16: .8315, 512: .8270}),
]

PROBE = {  # descriptor: (R2 by dim, s_j, c_j)
    "hr": ([.373, .554, .657, .735, .789, .826], 256, 16),
    "rr_mean": ([.306, .456, .566, .634, .679, .710], 256, 16),
    "rr_sd": ([.068, .077, .099, .129, .170, .191], 512, 16),
    "rr_rmssd": ([.066, .077, .096, .119, .158, .179], 512, 16),
    "qrs_duration": ([.102, .139, .159, .201, .247, .296], 512, 32),
    "qt_interval": ([.143, .188, .227, .263, .289, .330], 512, 32),
    "st_level": ([.297, .321, .368, .398, .421, .440], 256, 64),
    "t_amplitude": ([.550, .595, .607, .646, .675, .713], 512, 128),
    "r_amplitude": ([.444, .582, .660, .742, .783, .830], 512, 256),
    "p_amplitude": ([.113, .165, .201, .296, .402, .502], 512, 256),
    "qrs_axis": ([.135, .238, .342, .461, .514, .583], 512, 128),
}
ERANK = {16: 5.67, 32: 6.51, 64: 7.79, 128: 9.67, 256: 12.63, 512: 16.74}
NU = {16: .045, 32: .084, 64: .152, 128: .275, 256: .514, 512: 1.000}
CKA_FULL = [.883, .915, .951, .963, .990, 1.000]
SVD = {16: (.9269, .9993), 32: (.9278, .9996), 64: (.9277, .9998),
       128: (.9278, .9999), 256: (.9276, 1.0000), 512: (.9276, 1.0000)}
ROB = {
    "clean": (.9264, .9251),
    "baseline_wander": {20: (.9262, .9248), 10: (.9249, .9232), 0: (.9057, .9036)},
    # The manuscript's Delta for these rows was computed before rounding,
    # so the printed Delta differs by 1e-4 from the difference of the
    # printed AUCs. Unrounded values are chosen that round to BOTH the
    # printed AUCs and the printed Delta, which is the only way a real run
    # could have produced that table.
    "muscle": {20: (.925649, .924251), 10: (.920251, .918549),
               0: (.8659, .8645)},
    "electrode_motion": {20: (.9262, .9249), 10: (.9235, .9222), 0: (.8907, .8903)},
    "powerline": {20: (.9253, .9239), 10: (.9164, .9146), 0: (.8380, .8374)},
    "mixed": {20: (.9257, .9244), 10: (.9213, .9198), 0: (.886549, .885851)},
}
LEADS = {"12lead": (.9264, .9251), "8lead": (.9001, .8986),
         "6lead": (.7715, .7717), "3lead": (.6636, .6698),
         "2lead": (.6442, .6530), "lead_II": (.5753, .5832),
         "lead_I": (.5450, .5461)}
GPU_MS = {16: 4.658, 32: 4.655, 64: 4.654, 128: 4.662, 256: 4.658, 512: 4.680}
CROSS = {16: (.0146, .0100, .0192), 32: (.0144, .0099, .0190),
         64: (.0147, .0100, .0193), 128: (.0152, .0105, .0199),
         256: (.0157, .0110, .0204), 512: (.0151, .0104, .0197)}


def seeds_with(mu, sd, n, first=None):
    """n values with sample mean `mu` and sample SD `sd`; values[0]=`first`."""
    if n == 1:
        return [mu]
    if n == 2:
        a = sd / math.sqrt(2)
        return [mu + a, mu - a] if first is None else [first, 2 * mu - first]
    if first is None:
        a = sd * math.sqrt(3) / 2 if n == 4 else sd
        return [mu - a if i % 2 == 0 else mu + a for i in range(n)]
    rest_mean = (n * mu - first) / (n - 1)
    budget = ((n - 1) * sd ** 2 - (first - mu) ** 2
              - (n - 1) * (rest_mean - mu) ** 2)
    t = math.sqrt(max(budget, 0.0) / 2)
    tail = [rest_mean + t, rest_mean - t] + [rest_mean] * (n - 3)
    return [first] + tail


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline="\n" keeps the fixture byte-identical on every platform.
    # Without it a run on Windows rewrites all 28 files with CRLF and
    # leaves a clean checkout looking modified.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)


def run(out, name, seed, backbone, head, leads, norm, dims, test, n_params):
    write(os.path.join(out, "runs", name, "results.json"), {
        "run_name": name, "seed": seed, "dims": dims, "backbone": backbone,
        "head": head, "lead_subset": leads, "norm_mode": norm,
        "n_params": n_params, "class_names": CLASSES,
        "_fixture": "synthetic; see tests/fixtures/build_fixture.py", "test": test,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)

    # headline: four seeds, seed 0 pinned where the manuscript reports it
    cols = {d: seeds_with(HEAD_MEAN[d], HEAD_SD[d], 4, HEAD_SEED0.get(d))
            for d in DIMS}
    for s in range(4):
        test = {str(d): {"macro_auc": cols[d][s],
                         "per_class_auc": {str(i): v for i, v in
                                           enumerate(PERCLASS[d])}}
                for d in DIMS}
        run(out, f"inception1d_mrl_s{s}", s, "inception1d", "mrl", "12lead",
            "dataset", DIMS, test, 4595022)

    # XResNet1D-101: two seeds; seed 0 is the upper value (see docstring)
    xcols = {d: seeds_with(XRES_MEAN[d], XRES_SD[d], 2) for d in DIMS}
    for s in range(2):
        test = {str(d): {"macro_auc": xcols[d][s], "per_class_auc": {}}
                for d in DIMS}
        run(out, f"xresnet1d101_mrl_s{s}", s, "xresnet1d101", "mrl", "12lead",
            "dataset", DIMS, test, 34071630)

    for d in DIMS:
        run(out, f"inception1d_fixed_d{d}_s0", 0, "inception1d", "linear",
            "12lead", "dataset", [d],
            {str(d): {"macro_auc": FIXED_INC[d], "per_class_auc": {}}},
            PARAMS_FIXED[d])

    for cond, bb, head, leads, norm, vals in SINGLE_SEED:
        test = {str(d): {"macro_auc": v, "per_class_auc": {}}
                for d, v in vals.items()}
        run(out, f"{cond}_s0", 0, bb, head, leads, norm, DIMS, test, None)

    names = list(PROBE)
    cka = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    for i, v in enumerate(CKA_FULL):
        cka[i][5] = cka[5][i] = v
    write(os.path.join(out, "probing.json"), {
        "dims": DIMS, "feature_names": names,
        "prefix_probe": {n: {"r2_by_dim": {str(d): r for d, r in zip(DIMS, v[0])},
                             "best_r2": max(v[0]), "saturation_dim": v[1],
                             "claimed_dim": v[2]} for n, v in PROBE.items()},
        "hierarchy_verdict": {"n": 11, "rho": 0.399, "p": 0.224,
                              "supported": False},
        "geometry": {"cka_matrix": cka, "cka_dims": DIMS,
                     "effective_rank_full": ERANK[512],
                     "effective_rank_by_prefix": {str(d): v for d, v in ERANK.items()},
                     "variance_cumfrac_by_dim": {str(d): v for d, v in NU.items()}},
    })

    rob = {"clean": {"16": ROB["clean"][0], "512": ROB["clean"][1]}}
    for kind, by_snr in ROB.items():
        if kind == "clean":
            continue
        for snr, (a, b) in by_snr.items():
            rob[f"{kind}_snr{snr}"] = {"16": a, "512": b}
    write(os.path.join(out, "transfer.json"), {
        "svd_no_leakage": {str(d): {"macro_auc": a, "explained_variance": e}
                           for d, (a, e) in SVD.items()},
        "robustness": rob,
        "lead_ablation": {k: {"16": a, "512": b} for k, (a, b) in LEADS.items()},
    })

    write(os.path.join(out, "benchmark.json"), {
        "dims": DIMS,
        "environment": {"python": "3.12.11", "torch": "2.11.0+cu128",
                        "gpu": "NVIDIA L4"},
        "params": {"total": 4595022, "fp32_mb": 4595022 * 4 / 1e6},
        "macs": {"macs": 1529.7e6},
        "devices": {
            "cuda": {"per_dim": {str(d): {"end_to_end": {"median_ms": v}}
                                 for d, v in GPU_MS.items()},
                     "energy": {"energy_per_inference_mj": 3.5,
                                "avg_power_w": 46.9, "idle_power_w": 46.1},
                     "peak_memory_mb": 41.3},
            "cpu": {"per_dim": {"512": {"end_to_end": {"median_ms": 16.7}}}},
        },
    })

    write(os.path.join(out, "quantization.json"), {
        "dims": DIMS,
        "modes": {
            "fp32": {"per_dim": {"16": {"macro_auc": .9264}}, "size_mb": 18.6,
                     "latency_ms": 4.51, "device": "cuda"},
            "fp16": {"per_dim": {"16": {"macro_auc": .9264}}, "size_mb": 18.6,
                     "latency_ms": 4.56, "device": "cuda",
                     "small_prefix_penalty": 6e-6},
            "dynamic": {"per_dim": {"16": {"macro_auc": .9263}}, "size_mb": 15.7,
                        "latency_ms": 17.13, "device": "cpu",
                        "small_prefix_penalty": -8e-5},
            "static": {"error": "NotImplementedError: quantized::conv1d"},
        },
    })

    write(os.path.join(out, "stats.json"), {
        "equivalence": {"diff": .0013, "ci_low": .0007, "ci_high": .0019,
                        "p": 0.0, "equivalent": True, "margin": 0.01,
                        "dim_small": 16, "dim_large": 512, "seed": 0},
        "cross_backbone": {str(d): {"diff": a, "ci_low": lo, "ci_high": hi,
                                    "p": 0.0, "p_holm": 0.0, "significant": True}
                           for d, (a, lo, hi) in CROSS.items()},
    })

    # Real, not synthetic: computed from the PTB-XL v1.0.3 metadata by
    # scripts/dataset_stats.py and committed beside this file.
    shutil.copy2(os.path.join(HERE, "dataset.json"),
                 os.path.join(out, "dataset.json"))

    with open(os.path.join(out, "README.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("SYNTHETIC FIXTURE -- NOT EXPERIMENTAL DATA.\n\nBuilt by "
                "build_fixture.py so that summary statistics equal the "
                "manuscript's printed values. Used only by "
                "tests/test_reproduce_tables.py. dataset.json is the exception: "
                "it is computed from the real PTB-XL v1.0.3 metadata.\n")
    print(f"fixture written to {out}")


if __name__ == "__main__":
    main()
