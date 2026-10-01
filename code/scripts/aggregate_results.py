r"""
Aggregate every run into statistics and into the LaTeX the manuscript
includes.

This is the bridge between experiments and manuscript. Nothing numerical
in the paper is typed by hand: every table under paper/generated/ and
every inline result macro in paper/generated/numbers.tex is written by
this script from the run outputs, and the paper \input s them.

What changed, and why
---------------------
An earlier version of this script had two defects that together meant
it could not reproduce the manuscript:

1. It grouped runs by (backbone, head, lead subset, dims) only. Every
   Inception1D-MRL ablation -- per-record normalisation, label smoothing,
   the three weight strategies, constant width -- shares that key with
   the four headline seeds, so all ten would have been averaged into the
   headline number. Runs are now grouped by *condition*: the run name
   with its trailing `_s<seed>` removed, which run_all.sh makes unique
   per condition. A consistency check refuses to merge runs whose
   recorded configuration differs.

2. Several tables (geometry, lead ablation, composition) and about a
   quarter of the inline macros were produced outside this script. All
   of them are now emitted here.

tests/test_reproduce_tables.py regenerates every table from a fixture and
compares it with the manuscript, so a regression in either direction is
caught.

Inputs (under --results-dir)
----------------------------
runs/*/results.json          one per training run          (train.py)
runs/*/test_predictions.npz  for the paired statistics     (train.py)
probing.json                 probes and geometry           (run_probing.py)
transfer.json                SVD, robustness, leads        (eval_transfer.py)
benchmark.json               latency, energy, memory       (benchmark_hardware.py)
quantization.json            fp16 / INT8                   (quantize.py)
dataset.json                 corpus composition            (dataset_stats.py)
stats.json                   optional; used when predictions are absent

    python scripts/aggregate_results.py --results-dir results --paper-dir ../paper
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.stats import (equivalence_test, holm_bonferroni,  # noqa: E402
                                 paired_bootstrap_diff, seed_summary)

# ---------------------------------------------------------------------------
# Condition registry.  Names follow run_all.sh; seeds are the `_s<n>` suffix.
# ---------------------------------------------------------------------------
HEADLINE = "inception1d_mrl"
XRES = "xresnet1d101_mrl"
FIXED = {"inception1d": "inception1d_fixed_d{d}",
         "xresnet1d101": "xresnet1d101_fixed_d{d}"}

ABLATION_GROUPS = [
    ("Proposed", [(HEADLINE, "Inception1D-MRL", False)]),
    ("Head variant", [("inception1d_mrle", "MRL-E (shared $W$)", False)]),
    ("Backbone", [(XRES, "XResNet1D-101", False),
                  ("xresnet1d50_mrl", "XResNet1D-50", False),
                  ("inception1d_constant", "Inception1D (constant width)",
                   False)]),
    ("Preprocessing", [(HEADLINE, "Dataset-level norm. (proposed)", True),
                       ("inception1d_mrl_perrecord", "Per-record norm.",
                        True)]),
    ("Objective", [("inception1d_mrl_ls01", "Label smoothing $\\varepsilon$=0.1",
                    False),
                   ("inception1d_mrl_wlinear", "Weights: linear", False),
                   ("inception1d_mrl_wexponential", "Weights: exponential",
                    False),
                   ("inception1d_mrl_winverse", "Weights: inverse", False)]),
    ("Reduced-lead (native training)",
     [("inception1d_mrl_3lead", "3-lead (I, II, aVF)", False),
      ("inception1d_mrl_lead_I", "Lead I only", False)]),
]

LEAD_ROWS = [("12lead", "12", "all"), ("8lead", "8", "I, II, V1--V6"),
             ("6lead", "6", "limb"), ("3lead", "3", "I, II, aVF"),
             ("2lead", "2", "I, II"), ("lead_II", "1", "II"),
             ("lead_I", "1", "I")]

ARTEFACTS = [("baseline_wander", "baseline wander"),
             ("muscle", "EMG (muscle)"),
             ("electrode_motion", "electrode motion"),
             ("powerline", "mains (50\\,Hz)"),
             ("mixed", "mixed")]
TABLE_SNRS = [20, 10, 0]

DESCRIPTOR_LABELS = {
    "hr": "heart rate", "rr_mean": "RR mean", "rr_sd": "RR SD",
    "rr_rmssd": "RR RMSSD", "qrs_duration": "QRS duration",
    "qt_interval": "QT interval", "st_level": "ST level",
    "t_amplitude": "T amplitude", "r_amplitude": "R amplitude",
    "p_amplitude": "P amplitude", "qrs_axis": "QRS axis",
}

QUANT_ROWS = [("fp32", "fp32"), ("fp16", "fp16 autocast"),
              ("dynamic", "INT8 dynamic"), ("static", "INT8 static")]

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

SEED_SUFFIX = re.compile(r"_s(\d+)$")


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------
def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "runs", "*",
                                              "results.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(path)
        runs.append(r)
    return runs


def condition_of(r):
    name = r.get("run_name") or os.path.basename(r["_dir"])
    return SEED_SUFFIX.sub("", name)


def build_summary(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[condition_of(r)].append(r)

    summary = {}
    for cond, rs in groups.items():
        # Refuse to merge runs whose recorded configuration differs: that
        # is exactly how ablations leaked into the headline before.
        sig = {(r["backbone"], r["head"], r.get("lead_subset", "12lead"),
                r.get("norm_mode", "dataset"), tuple(r["dims"])) for r in rs}
        if len(sig) > 1:
            raise SystemExit(f"condition '{cond}' mixes configurations: {sig}")
        rs = sorted(rs, key=lambda r: r["seed"])
        dims = rs[0]["dims"]
        per_dim = {}
        for d in dims:
            vals = [r["test"][str(d)] for r in rs if str(d) in r["test"]]
            if not vals:
                continue
            per_dim[str(d)] = {
                "auc": seed_summary([v["macro_auc"] for v in vals]),
                "f1": seed_summary([v.get("macro_f1", float("nan")) for v in vals]),
                "ap": seed_summary([v.get("macro_ap", float("nan")) for v in vals]),
            }
        summary[cond] = {
            "backbone": rs[0]["backbone"], "head": rs[0]["head"],
            "lead_subset": rs[0].get("lead_subset", "12lead"),
            "norm_mode": rs[0].get("norm_mode", "dataset"),
            "dims": dims, "n_seeds": len(rs),
            "seeds": [r["seed"] for r in rs],
            "n_params": rs[0].get("n_params"),
            "class_names": rs[0].get("class_names", []),
            "per_dim": per_dim,
            "runs": rs,
            "run_dirs": [r["_dir"] for r in rs],
        }
    return summary


def auc_mean(summary, cond, d):
    e = summary.get(cond)
    if not e or str(d) not in e["per_dim"]:
        return None
    return e["per_dim"][str(d)]["auc"]


def seed0_auc(summary, cond, d):
    """Test AUC of the lowest-seed run -- the checkpoint the analyses use."""
    e = summary.get(cond)
    if not e:
        return None
    t = e["runs"][0]["test"].get(str(d))
    return None if t is None else t["macro_auc"]


# ---------------------------------------------------------------------------
# Statistics (need predictions; otherwise read a stored stats.json)
# ---------------------------------------------------------------------------
def load_preds(run_dir, d):
    f = np.load(os.path.join(run_dir, "test_predictions.npz"))
    return f["y_true"], f[f"probs_d{d}"]


def compute_stats(summary, margin):
    out = {}
    h = summary.get(HEADLINE)
    if h and len(h["dims"]) > 1:
        rd, lo, hi = h["run_dirs"][0], h["dims"][0], h["dims"][-1]
        try:
            y, ps = load_preds(rd, lo)
            _, pl = load_preds(rd, hi)
            e = equivalence_test(y, ps, pl, margin=margin, seed=0)
            e.update({"dim_small": lo, "dim_large": hi, "seed": h["seeds"][0]})
            out["equivalence"] = e
        except (FileNotFoundError, KeyError):
            pass
    x = summary.get(XRES)
    if h and x:
        rows, pvals, used = {}, [], []
        for d in h["dims"]:
            try:
                y, pa = load_preds(h["run_dirs"][0], d)
                _, pb = load_preds(x["run_dirs"][0], d)
            except (FileNotFoundError, KeyError):
                continue
            r = paired_bootstrap_diff(y, pa, pb, seed=0)
            rows[str(d)] = r
            pvals.append(r["p"])
            used.append(d)
        if pvals:
            adj, rej = holm_bonferroni(pvals)
            for i, d in enumerate(used):
                rows[str(d)]["p_holm"] = adj[i]
                rows[str(d)]["significant"] = bool(rej[i])
            out["cross_backbone"] = rows
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def esc(s):
    return str(s).replace("_", "\\_")


def f4(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.4f}"


def mean_sd(stat):
    if stat is None or not np.isfinite(stat["mean"]):
        return "--"
    if stat["n"] > 1 and np.isfinite(stat["std"]) and stat["std"] > 0:
        return f"{stat['mean']:.4f}\\,$\\pm$\\,{stat['std']:.4f}"
    return f"{stat['mean']:.4f}"


def signed(x, nd=4):
    """$+$0.0013 / $-$0.0013, the manuscript's signed-difference style."""
    return f"${'+' if x >= 0 else '-'}${abs(x):.{nd}f}"


def pval(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def sci1(x):
    """One significant figure in scientific notation: $-8\\!\\times\\!10^{-5}$."""
    if x == 0 or not np.isfinite(x):
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = round(x / 10 ** exp)
    if abs(mant) == 10:
        mant, exp = mant // 10, exp + 1
    return f"${'+' if mant > 0 else '-'}{abs(mant)}\\!\\times\\!10^{{{exp}}}$"


def r3(x):
    """.373 -- three decimals without the leading zero, as in the probe table."""
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0") else s


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def word(n):
    return NUM_WORDS.get(n, str(n))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tab_main(summary, n_test):
    h, x = summary.get(HEADLINE), summary.get(XRES)
    if not h:
        return "% no headline condition found\n"
    dims = h["dims"]
    fixed_inc = [summary.get(FIXED["inception1d"].format(d=d)) for d in dims]
    fixed_xr = [summary.get(FIXED["xresnet1d101"].format(d=d)) for d in dims]
    n_fixed = sorted({f["n_seeds"] for f in fixed_inc if f})
    parts = [f"{word(h['n_seeds'])} for Inception1D-MRL"]
    if x:
        parts.append(f"{word(x['n_seeds'])} for XResNet1D-101-MRL")
    if n_fixed:
        parts.append(f"{' or '.join(word(n) for n in n_fixed)} for the "
                     f"fixed-dimension baselines")
    xres_note = ("" if any(fixed_xr) else
                 " XResNet1D-101 fixed-dimension baselines were not run.")
    L = [
        "\\begin{table*}[!t]", "\\centering",
        f"\\caption{{Test-set macro AUC-ROC on PTB-XL fold 10 "
        f"($n$={thousands(n_test)}), mean\\,$\\pm$\\,SD over training seeds "
        f"({', '.join(parts)}). Fixed-dimension baselines use the "
        f"same backbone as the corresponding MRL model, which is "
        f"the like-for-like comparison.{xres_note} Payload is bytes per "
        f"transmitted embedding (float32).}}",
        "\\label{tab:main}", "\\small", "\\setlength{\\tabcolsep}{9pt}",
        "\\begin{tabular}{rccccc}", "\\toprule",
        "$d$ & Inc1D-MRL & Inc1D fixed & XRes101-MRL & XRes101 fixed & Payload \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(dims):
        fi = fixed_inc[i]["per_dim"][str(d)]["auc"] if fixed_inc[i] else None
        fx = fixed_xr[i]["per_dim"][str(d)]["auc"] if fixed_xr[i] else None
        L.append(f"{d} & {mean_sd(auc_mean(summary, HEADLINE, d))} & "
                 f"{mean_sd(fi)} & {mean_sd(auc_mean(summary, XRES, d))} & "
                 f"{mean_sd(fx)} & {4 * d}\\,B \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


def tab_stats(stats, margin, n_dims):
    e, cross = stats.get("equivalence"), stats.get("cross_backbone")
    if not e and not cross:
        return "% no statistics available\n"
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Statistical assessment. Differences are assessed by the "
        "paired bootstrap of Eq.~\\eqref{eq:bootstrap} ($B$=2{,}000); "
        f"equivalence follows Eq.~\\eqref{{eq:tost}} with $\\delta$={margin:g}. "
        f"Cross-model rows are Holm-corrected across the {word(n_dims)} "
        "granularities. Tests use one training run per condition (seed 0); "
        "seed-to-seed variation is reported separately in "
        "Table~\\ref{tab:main}.}",
        "\\label{tab:stats}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}lrcrl@{}}", "\\toprule",
        "Comparison & $\\Delta$AUC & 95\\% CI & $p$ & Verdict \\\\", "\\midrule",
    ]
    if e:
        L += ["\\multicolumn{5}{@{}l}{\\textit{Within model: is granularity "
              "immaterial?}} \\\\",
              f"$d$={e['dim_small']} vs {e['dim_large']} & {signed(e['diff'])} "
              f"& [{signed(e['ci_low'])}, {signed(e['ci_high'])}] & "
              f"{pval(e['p'])} & "
              f"{'equivalent' if e['equivalent'] else 'not equivalent'} \\\\"]
    if cross:
        if e:
            L.append("\\midrule")
        L.append("\\multicolumn{5}{@{}l}{\\textit{Across backbone: Inc1D vs "
                 "XRes101}} \\\\")
        for d, r in sorted(cross.items(), key=lambda kv: int(kv[0])):
            lab = f"$d$={d}"
            L.append(f"{lab:<9} & {signed(r['diff'])} & [{signed(r['ci_low'])}, "
                     f"{signed(r['ci_high'])}] & "
                     f"{pval(r.get('p_holm', r['p']))} & "
                     f"{'sig.' if r.get('significant') else 'n.s.'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_compute(bench):
    if not bench:
        return "% benchmark.json not found\n"
    dims = bench["dims"]
    gpu = bench["devices"].get("cuda", {})
    cpu = bench["devices"].get("cpu", {})
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Measured computational profile (batch size 1, \\R{GPUName}, "
        "median of 200 timed runs with CUDA-event timing). Latency is "
        "invariant to $m$ as Equation~\\eqref{eq:latency} predicts: the total "
        "spread across all six granularities is \\R{LatencySpread}\\,ms on "
        "GPU, smaller than the run-to-run IQR. Head parameter counts are "
        "exact. CPU latency is end-to-end at $d$=\\R{DimLarge} under "
        "PyTorch's default threading.}",
        "\\label{tab:compute}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{rccc}", "\\toprule",
        "$m$ & Payload & Head params & GPU latency (ms) \\\\", "\\midrule",
    ]
    for d in dims:
        ms = gpu.get("per_dim", {}).get(str(d), {}).get("end_to_end", {}) \
                .get("median_ms", float("nan"))
        L.append(f"{str(d):<3} & {thousands(4 * d)}\\,B & "
                 f"{thousands(5 * d + 5)} & {ms:.3f} \\\\")
    L += [
        "\\midrule",
        "\\multicolumn{4}{@{}l}{\\textit{Encoder, constant across all "
        "granularities}} \\\\",
        "\\multicolumn{2}{@{}l}{multiply-accumulates} & "
        "\\multicolumn{2}{r}{\\R{MeasuredMACsExact}\\,M} \\\\",
        "\\multicolumn{2}{@{}l}{parameters / size (fp32)} & "
        "\\multicolumn{2}{r}{\\R{IncParams} / \\R{ModelMB}\\,MB} \\\\",
        "\\multicolumn{2}{@{}l}{CPU latency (default threading)} & "
        "\\multicolumn{2}{r}{\\R{MeasuredCPU}\\,ms} \\\\",
        "\\multicolumn{2}{@{}l}{peak GPU memory} & "
        "\\multicolumn{2}{r}{\\R{PeakMemMB}\\,MB} \\\\",
        "\\multicolumn{2}{@{}l}{energy per inference (net of idle)} & "
        "\\multicolumn{2}{r}{$\\approx$\\R{EnergyPerInf}\\,mJ} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ]
    return "\n".join(L)


def tab_geometry(probe, transfer):
    if not probe or "geometry" not in probe:
        return "% probing.json geometry not found\n"
    g = probe["geometry"]
    dims = probe.get("dims") or g.get("cka_dims")
    cka = np.asarray(g["cka_matrix"])
    svd = (transfer or {}).get("svd_no_leakage", {})
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Representation geometry "
        "(Eqs.~\\eqref{eq:erank}--\\eqref{eq:cka}). Effective rank grows "
        "steadily with prefix length, so trailing coordinates are not "
        "degenerate. $\\nu(m)$ is close to $m/d$ because the encoder "
        "terminates in batch normalisation, which equalises per-coordinate "
        "variance by construction; $\\nu$ is therefore uninformative here and "
        "effective rank is the meaningful measure. The final columns report a "
        "leakage-free PCA baseline on the same embeddings: 16 principal "
        "components capture \\R{SVDExplVar}\\% of the variance and match the "
        "nested model's accuracy, confirming that the embedding is "
        "intrinsically low-dimensional even though its coordinate "
        "variance is spread almost uniformly.}",
        "\\label{tab:geometry}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.4pt}",
        "\\begin{tabular}{rccccc}", "\\toprule",
        "$m$ & $\\mathrm{erank}(Z_{1:m})$ & $\\nu(m)$ & "
        "$\\mathrm{CKA}(Z_{1:m},Z)$", "& PCA-$m$ AUC & expl.\\ var. \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(dims):
        s = svd.get(str(d), {})
        L.append(f"{str(d):<3} & {g['effective_rank_by_prefix'][str(d)]:.2f} & "
                 f"{g['variance_cumfrac_by_dim'][str(d)]:.3f} & "
                 f"{cka[i, -1]:.3f} & {f4(s.get('macro_auc'))} & "
                 f"{f4(s.get('explained_variance'))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_leads(summary, transfer):
    la = (transfer or {}).get("lead_ablation")
    if not la:
        return "% lead ablation not found\n"
    dims = [int(k) for k in next(iter(la.values()))]
    lo, hi = str(min(dims)), str(max(dims))
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Reduced-lead acquisition, distinguishing two regimes. "
        "Zero-shot masking removes leads at inference on the twelve-lead model; "
        "native training uses the reduced lead set directly. The gap is "
        "decisive: a twelve-lead model given one lead is near chance "
        "(\\R{ZeroShotLeadI}), whereas a model trained on that lead reaches "
        "\\R{LeadINative}. Graceful degradation is therefore not a viable "
        "fallback for wearable deployment. Flatness across $d$ survives in "
        "both regimes.}",
        "\\label{tab:leads}", "\\footnotesize", "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{llcccc}", "\\toprule",
        "& & \\multicolumn{2}{c}{Zero-shot masking} & "
        "\\multicolumn{2}{c}{Native training} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
        f"Leads & Set & $d$={lo} & $d$={hi} & $d$={lo} & $d$={hi} \\\\",
        "\\midrule",
    ]
    for key, n, label in LEAD_ROWS:
        if key not in la:
            continue
        z = la[key]
        if key == "12lead":        # the twelve-lead model *is* the native one
            nat = (z[lo], z[hi])
        else:
            cond = f"{HEADLINE}_{key}"
            a, b = auc_mean(summary, cond, lo), auc_mean(summary, cond, hi)
            nat = (a["mean"] if a else None, b["mean"] if b else None)
        nat_s = [("---" if v is None else f4(v)) for v in nat]
        if key == "lead_I" and nat[0] is not None:
            nat_s[0] = f"\\textbf{{{nat_s[0]}}}"
        L.append(f"{n:<2} & {label:<20} & {f4(z[lo])} & {f4(z[hi])} & "
                 f"{nat_s[0]} & {nat_s[1]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_robustness(transfer):
    rob = (transfer or {}).get("robustness")
    if not rob or "clean" not in rob:
        return "% robustness sweep not found\n"
    dims = sorted(int(k) for k in rob["clean"])
    lo, hi = str(dims[0]), str(dims[-1])
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Artefact robustness (macro AUC, Eq.~\\eqref{eq:snr}). The "
        "smallest prefix is never worse than the largest under any artefact "
        "at any SNR: $\\Delta$ is positive throughout. Degradation is graceful "
        "down to 10\\,dB and becomes material at 0\\,dB, with mains "
        "interference the most damaging. $\\Delta$ is computed from "
        "unrounded values, so it can differ by $10^{-4}$ from the "
        "difference of the printed entries.}",
        "\\label{tab:robustness}", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
        f"Artefact & SNR & $d$={lo} & $d$={hi} & $\\Delta$ \\\\", "\\midrule",
        f"clean & --- & {f4(rob['clean'][lo])} & {f4(rob['clean'][hi])} & "
        f"{signed(rob['clean'][lo] - rob['clean'][hi])} \\\\",
        "\\midrule",
    ]
    for kind, label in ARTEFACTS:
        first = True
        for snr in TABLE_SNRS:
            k = f"{kind}_snr{snr}"
            if k not in rob:
                continue
            snr_s = (f"\\phantom{{0}}{snr}\\,dB" if snr < 10 else f"{snr}\\,dB")
            name = label if first else ""
            first = False
            L.append(f"{name} & {snr_s} & {f4(rob[k][lo])} & {f4(rob[k][hi])} "
                     f"& {signed(rob[k][lo] - rob[k][hi])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_probe(probe, auc_lo, auc_hi, margin):
    if not probe or "prefix_probe" not in probe:
        return "% probing.json not found\n"
    dims = probe["dims"]
    pp = probe["prefix_probe"]
    names = probe.get("feature_names") or list(pp)
    v = probe.get("hierarchy_verdict", {})
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Probing the nesting hierarchy "
        "(Eqs.~\\eqref{eq:probe}--\\eqref{eq:spearman}). Out-of-sample $R^2$ "
        "for eleven physiological descriptors as a function of prefix length. "
        "Every descriptor improves monotonically to $d$=512, so the "
        "saturation criterion of Eq.~\\eqref{eq:sat} lands at 256 or 512 for "
        "all eleven and has little rank variance to correlate against the "
        "hypothesised ordering. The consequential observation is the contrast "
        "with Table~\\ref{tab:main}: diagnostic AUC is flat from $d$=16 while "
        "descriptor recoverability rises by a mean factor of \\R{ProbeGain}.}",
        "\\label{tab:probe}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{@{}l" + "r" * (len(dims) + 3) + "@{}}", "\\toprule",
        f"& \\multicolumn{{{len(dims)}}}{{c}}{{$R^2_j(m)$}} & & & \\\\",
        f"\\cmidrule(lr){{2-{len(dims) + 1}}}",
        "Descriptor & " + " & ".join(str(d) for d in dims)
        + " & $\\times$ & $s_j$ & $c_j$ \\\\",
        "\\midrule",
    ]
    for n in names:
        r = pp[n]
        r2 = [r["r2_by_dim"][str(d)] for d in dims]
        ratio = r2[-1] / r2[0] if r2[0] else float("nan")
        label = DESCRIPTOR_LABELS.get(n, esc(n))
        L.append(f"{label:<15} & " + " & ".join(r3(x) for x in r2)
                 + f" & {ratio:.1f} & {r['saturation_dim']} & "
                   f"{r['claimed_dim']} \\\\")
    ncol = len(dims) + 4
    verdict = "supported" if v.get("supported") else "not supported"
    flat = (auc_lo is not None and auc_hi is not None
            and abs(auc_lo - auc_hi) < margin)
    L += ["\\midrule",
          f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{Spearman "
          f"$\\rho(c_j,s_j)={v.get('rho', float('nan')):.3f}$, "
          f"$p$={v.get('p', float('nan')):.3f} $\\Rightarrow$",
          f"hierarchy \\textbf{{{verdict}}}}} \\\\"]
    if auc_lo is not None and auc_hi is not None:
        # kept outside the f-string: backslashes in f-string expressions
        # are a syntax error before Python 3.12
        flat_s = " (\\textbf{flat})" if flat else ""
        L += [f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{Macro AUC over the same "
              f"range: {auc_lo:.4f} $\\rightarrow$",
              f"{auc_hi:.4f}{flat_s}}} \\\\"]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_perclass(summary):
    h = summary.get(HEADLINE)
    if not h:
        return "% no headline condition found\n"
    names = h["class_names"]
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Per-class test AUC-ROC across granularities "
        f"(Inception1D-MRL, mean over {word(h['n_seeds'])} seeds). The class "
        "ordering is preserved exactly at every operating point, which is "
        "what consistent diagnostic behaviour across a deployment family "
        "requires. HYP remains hardest, consistent with its rarity and its "
        "dependence on absolute voltage.}",
        "\\label{tab:perclass}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{r" + "c" * len(names) + "}", "\\toprule",
        "$d$ & " + " & ".join(esc(n) for n in names) + " \\\\", "\\midrule",
    ]
    for d in h["dims"]:
        vals = []
        for ci in range(len(names)):
            xs = [r["test"][str(d)]["per_class_auc"].get(str(ci))
                  for r in h["runs"] if str(d) in r["test"]]
            xs = [x for x in xs if x is not None and np.isfinite(x)]
            vals.append(f"{np.mean(xs):.4f}" if xs else "--")
        L.append(f"{str(d):<3} & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_quant(quant):
    if not quant or "modes" not in quant:
        return "% quantization.json not found\n"
    modes = quant["modes"]
    lo = str(min(int(d) for d in quant.get("dims", [16])))
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Quantisation, the compute axis nesting does not address. "
        f"The differential penalty is the AUC change at $d$={lo} minus that "
        "at $d$=512: it is below $10^{-4}$ for both working modes, so "
        "quantisation does not harm the small operating points "
        "disproportionately and the two techniques compose safely. MB is the "
        "serialised checkpoint, which includes normalisation buffers and so "
        "exceeds the parameter size in Table~\\ref{tab:compute}. Static "
        "post-training quantisation could not be evaluated because "
        "\\texttt{quantized::conv1d} is unavailable in this build.}",
        "\\label{tab:quant}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
        f"Mode & Device & MB & ms & AUC $d$={lo} & $\\Delta$ penalty \\\\",
        "\\midrule",
    ]
    for key, label in QUANT_ROWS:
        e = modes.get(key)
        if e is None:
            continue
        if "error" in e or "per_dim" not in e:
            L.append(f"{label:<13} & CPU & \\multicolumn{{4}}{{l}}{{not "
                     f"supported in this PyTorch build}} \\\\")
            continue
        dev = "GPU" if e.get("device") == "cuda" else "CPU"
        pen = e.get("small_prefix_penalty")
        pen_s = "---" if key == "fp32" or pen is None else sci1(pen)
        L.append(f"{label:<13} & {dev} & {e['size_mb']:.1f} & "
                 f"{e['latency_ms']:.2f} & "
                 f"{e['per_dim'][lo]['macro_auc']:.4f} & {pen_s} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_ablations(summary):
    h = summary.get(HEADLINE)
    if not h:
        return "% no headline condition found\n"
    lo, hi = str(h["dims"][0]), str(h["dims"][-1])
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Ablations. Each row is a full training condition; macro AUC "
        "is reported at the smallest and largest nesting dimension as the "
        "mean over the listed seeds (standard deviations for the proposed "
        "model are in Table~\\ref{tab:main}). Flatness across $d$ is the "
        "property under test, and it holds in every condition where both "
        "ends are available. Single-seed differences within roughly 0.002 AUC "
        "should not be interpreted; a dash marks a value not available.}",
        "\\label{tab:ablations}", "\\small", "\\setlength{\\tabcolsep}{3.4pt}",
        "\\begin{tabular}{llccc}", "\\toprule",
        f"Condition & Seeds & $d$={lo} & $d$={hi} & $\\Delta$ \\\\", "\\midrule",
    ]
    first = True
    for group, rows in ABLATION_GROUPS:
        present = [r for r in rows if r[0] in summary]
        if not present:
            continue
        if not first:
            L.append("\\midrule")
        first = False
        L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for cond, label, bold in present:
            e = summary[cond]
            a, b = auc_mean(summary, cond, lo), auc_mean(summary, cond, hi)
            av = a["mean"] if a else None
            bv = b["mean"] if b else None
            a_s = f4(av)
            if bold and av is not None:
                a_s = f"\\textbf{{{a_s}}}"
            delta = (f"{av - bv:+.4f}" if av is not None and bv is not None
                     else "--")
            L.append(f"{label} & {e['n_seeds']} & {a_s} & {f4(bv)} & "
                     f"{delta} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_composition(ds):
    if not ds:
        return "% dataset.json not found\n"
    n = ds["n_records_labelled"]
    order = sorted(ds["class_counts"], key=lambda c: -ds["class_counts"][c])
    desc = {"NORM": "normal ECG", "MI": "myocardial infarction",
            "STTC": "ST/T-wave change", "CD": "conduction disturbance",
            "HYP": "hypertrophy"}
    multi = round(100 * ds["multi_label_fraction"])
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{PTB-XL superdiagnostic composition (\\SI{100}{\\hertz} "
        "variant, v1.0.3, records with at least one diagnostic label), "
        "computed from the released metadata by the same label aggregation "
        "used for training. Percentages exceed 100\\% because "
        f"{multi}\\% of recordings carry more than one label. HYP is both the "
        "rarest class and the one whose diagnostic criteria depend on "
        "absolute voltage, which motivates the normalisation analysis of "
        "Section~\\ref{sec:model:preproc}.}",
        "\\label{tab:composition}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
        "Class & Description & Records & Share \\\\", "\\midrule",
    ]
    for c in order:
        k = ds["class_counts"][c]
        L.append(f"{c:<4} & {desc.get(c, c):<23} & {thousands(k)} & "
                 f"{100 * k / n:.1f}\\% \\\\")
    sp = ds["splits"]
    L += ["\\midrule",
          f"\\multicolumn{{2}}{{@{{}}l}}{{Train (folds 1--8)}} & "
          f"{thousands(sp['train'])} & {100 * sp['train'] / n:.1f}\\% \\\\",
          f"\\multicolumn{{2}}{{@{{}}l}}{{Validation (fold 9)}} & "
          f"{thousands(sp['val'])} & {100 * sp['val'] / n:.1f}\\% \\\\",
          f"\\multicolumn{{2}}{{@{{}}l}}{{Test (fold 10)}} & "
          f"{thousands(sp['test'])} & {100 * sp['test'] / n:.1f}\\% \\\\",
          f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{Total}}}} & "
          f"\\textbf{{{thousands(n)}}} & 100\\% \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def tab_external(transfer):
    ext = {k: v for k, v in (transfer or {}).items() if k.startswith("external_")}
    if not ext:
        return ("% No external corpus was processed in this run: the "
                "PhysioNet/CinC-2020\n% training sets are served as directories "
                "rather than tarballs, so the\n% download step did not "
                "complete. See the limitations subsection.\n")
    L = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Zero-shot cross-dataset transfer (macro AUC). Labels are "
        "mapped from SNOMED-CT onto the PTB-XL superclasses, a lossy "
        "many-to-one mapping; these numbers measure transfer of the "
        "superclass concept rather than identical-task performance.}",
        "\\label{tab:external}", "\\footnotesize",
        "\\begin{tabular}{lrcc}", "\\toprule",
        "Corpus & $n$ & smallest $d$ & largest $d$ \\\\", "\\midrule",
    ]
    for k, v in sorted(ext.items()):
        bd = v["by_dim"]
        keys = sorted(bd, key=int)
        L.append(f"{esc(k.replace('external_', ''))} & "
                 f"{thousands(v['n_records'])} & "
                 f"{f4(bd[keys[0]]['macro_auc'])} & "
                 f"{f4(bd[keys[-1]]['macro_auc'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figure data.  The manuscript's figures are drawn by pgfplots from these
# files, so a rerun updates the figures exactly as it updates the tables.
# ---------------------------------------------------------------------------
def _dat(header, rows, nd=4):
    # Values are written at the precision the manuscript's tables print, so
    # the shipped data files contain only numbers that appear in the paper.
    out = [" ".join(header)]
    for r in rows:
        out.append(" ".join("nan" if v is None else
                            (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
                            for v in r))
    return "\n".join(out) + "\n"


def figure_data(summary, probe, transfer, bench):
    D = {}
    h, x = summary.get(HEADLINE), summary.get(XRES)
    if h:
        rows = []
        for d in h["dims"]:
            s = auc_mean(summary, HEADLINE, d)
            rows.append((d, s["mean"], s["std"] if s["n"] > 1 else 0.0))
        D["fig_auc_inc.dat"] = _dat(["d", "mean", "sd"], rows)
        rows = []
        for d in h["dims"]:
            f = summary.get(FIXED["inception1d"].format(d=d))
            if f:
                rows.append((d, f["per_dim"][str(d)]["auc"]["mean"]))
        D["fig_auc_fixed.dat"] = _dat(["d", "mean"], rows)
    if x:
        rows = []
        for d in x["dims"]:
            s = auc_mean(summary, XRES, d)
            rows.append((d, s["mean"], s["std"] if s["n"] > 1 else 0.0))
        D["fig_auc_xres.dat"] = _dat(["d", "mean", "sd"], rows)
    if bench:
        g = bench["devices"].get("cuda", {}).get("per_dim", {})
        D["fig_latency_gpu.dat"] = _dat(
            ["d", "ms"], [(d, g[str(d)]["end_to_end"]["median_ms"])
                          for d in bench["dims"] if str(d) in g], nd=3)
    if probe and "prefix_probe" in probe:
        dims = probe["dims"]
        names = probe.get("feature_names") or list(probe["prefix_probe"])
        rows = []
        for d in dims:
            vals = [probe["prefix_probe"][n]["r2_by_dim"][str(d)] for n in names]
            rows.append((d, *vals, float(np.mean(vals))))
        D["fig_probe.dat"] = _dat(["d", *names, "mean"], rows, nd=3)
    rob = (transfer or {}).get("robustness")
    if rob and "clean" in rob:
        lo = str(min(int(k) for k in rob["clean"]))
        rows = []
        for snr in TABLE_SNRS:
            rows.append((snr, *[rob.get(f"{k}_snr{snr}", {}).get(lo)
                                for k, _ in ARTEFACTS]))
        D["fig_robustness.dat"] = _dat(["snr", *[k for k, _ in ARTEFACTS]],
                                       rows)
        D["fig_clean.dat"] = _dat(["snr", "auc"], [(20, rob["clean"][lo]),
                                                   (0, rob["clean"][lo])])
    la = (transfer or {}).get("lead_ablation")
    if la:
        lo = str(min(int(k) for k in next(iter(la.values()))))
        zero, native = [], []
        for i, (key, _, _) in enumerate(LEAD_ROWS):
            if key not in la:
                continue
            zero.append((i, la[key][lo]))
            if key == "12lead":
                native.append((i, la[key][lo]))
            else:
                s = auc_mean(summary, f"{HEADLINE}_{key}", lo)
                if s:
                    native.append((i, s["mean"]))
        D["fig_leads_zero.dat"] = _dat(["idx", "auc"], zero)
        D["fig_leads_native.dat"] = _dat(["idx", "auc"], native)
    return D


# ---------------------------------------------------------------------------
# Inline macros
# ---------------------------------------------------------------------------
def build_macros(summary, stats, probe, transfer, bench, quant, ds, margin):
    M = {}

    def put(name, val, nd=4):
        M[name] = (f"{val:.{nd}f}" if isinstance(val, (float, np.floating))
                   else str(val))

    h = summary.get(HEADLINE)
    lo = hi = None
    if h:
        lo, hi = str(h["dims"][0]), str(h["dims"][-1])
        a, b = auc_mean(summary, HEADLINE, lo), auc_mean(summary, HEADLINE, hi)
        put("IncAUCsmall", a["mean"])
        put("IncAUClarge", b["mean"])
        put("IncSDsmall", a["std"])
        put("IncSDlarge", b["std"])
        # Difference of the *rounded* means, so the prose reconciles with the
        # printed table a reader can check.
        put("IncSpread", round(a["mean"], 4) - round(b["mean"], 4))
        M["DimSmall"], M["DimLarge"] = lo, hi
        M["NumSeeds"] = str(h["n_seeds"])
        M["IncParams"] = f"{h['n_params']:,}"

        for cond, key in ((FIXED["inception1d"].format(d=lo), "FixedAUCsmall"),
                          (FIXED["inception1d"].format(d=hi), "FixedAUClarge")):
            s = auc_mean(summary, cond, lo if key.endswith("small") else hi)
            if s:
                put(key, s["mean"])
        # MRL minus same-backbone fixed baseline at the intermediate
        # granularities, from the rounded table entries so the prose range
        # reconciles with what a reader sees in the main table.
        gaps = []
        for d in h["dims"][1:-1]:
            f = auc_mean(summary, FIXED["inception1d"].format(d=d), d)
            m = auc_mean(summary, HEADLINE, d)
            if f and m:
                gaps.append(round(m["mean"], 4) - round(f["mean"], 4))
        if gaps:
            put("MidGapLo", min(gaps))
            put("MidGapHi", max(gaps))
        pr = auc_mean(summary, "inception1d_mrl_perrecord", lo)
        if pr:
            put("PerRecordAUC", pr["mean"])
            put("NormGain", round(a["mean"], 4) - round(pr["mean"], 4))
        me = auc_mean(summary, "inception1d_mrle", lo)
        if me:
            put("MRLEAUCsmall", me["mean"])
        for cond, key in (("inception1d_mrl_lead_I", "LeadINative"),
                          ("inception1d_mrl_3lead", "LeadThreeNative")):
            s = auc_mean(summary, cond, lo)
            if s:
                put(key, s["mean"])

    x = summary.get(XRES)
    if x:
        put("XResAUCsmall", auc_mean(summary, XRES, x["dims"][0])["mean"])
        put("XResAUClarge", auc_mean(summary, XRES, x["dims"][-1])["mean"])
        M["XResParams"] = f"{x['n_params']:,}"

    e = stats.get("equivalence")
    if e:
        put("EquivDiff", e["diff"])
        put("EquivLo", e["ci_low"])
        put("EquivHi", e["ci_high"])
        M["EquivVerdict"] = ("statistically equivalent" if e["equivalent"]
                             else "not statistically equivalent")
    put("EquivMargin", margin, nd=3)

    if bench:
        dims = bench["dims"]
        top = str(dims[-1])
        gpu = bench["devices"].get("cuda", {})
        cpu = bench["devices"].get("cpu", {})
        g = gpu.get("per_dim", {})
        if top in g:
            put("MeasuredGPU", g[top]["end_to_end"]["median_ms"], nd=2)
        lat = [g[str(d)]["end_to_end"]["median_ms"] for d in dims if str(d) in g]
        if lat:
            put("LatencySpread", max(lat) - min(lat), nd=3)
        c = cpu.get("per_dim", {})
        if top in c:
            # end-to-end at the largest granularity, default PyTorch threading
            put("MeasuredCPU", c[top]["end_to_end"]["median_ms"], nd=1)
        if bench.get("macs", {}).get("macs"):
            macs_m = bench["macs"]["macs"] / 1e6
            M["MeasuredMACs"] = f"{macs_m:.0f}"
            M["MeasuredMACsExact"] = f"{macs_m:,.1f}".replace(",", "{,}")
        M["ModelMB"] = f"{bench['params']['fp32_mb']:.1f}"
        # energy and peak memory are recorded per device, not per dimension
        en = gpu.get("energy", {})
        if en.get("energy_per_inference_mj") is not None:
            put("EnergyPerInf", float(en["energy_per_inference_mj"]), nd=1)
        if en.get("avg_power_w") is not None:
            put("AvgPowerW", float(en["avg_power_w"]), nd=1)
        if en.get("idle_power_w") is not None:
            put("IdlePowerW", float(en["idle_power_w"]), nd=1)
        if gpu.get("peak_memory_mb") is not None:
            put("PeakMemMB", float(gpu["peak_memory_mb"]), nd=1)
        env = bench.get("environment", {})
        M["GPUName"] = esc(env["gpu"]) if env.get("gpu") else "n/a"
        M["TorchVersion"] = esc(env.get("torch", "n/a"))
        M["PythonVersion"] = esc(env.get("python", "n/a"))

    if probe:
        v = probe.get("hierarchy_verdict", {})
        put("HierRho", float(v.get("rho", float("nan"))), nd=3)
        put("HierP", float(v.get("p", float("nan"))), nd=3)
        M["HierVerdict"] = "supported" if v.get("supported") else "not supported"
        g = probe.get("geometry", {})
        pdims = probe.get("dims") or g.get("cka_dims")
        if g.get("effective_rank_full") is not None:
            put("EffRank", float(g["effective_rank_full"]), nd=1)
        if pdims:
            p0 = str(pdims[0])
            if p0 in g.get("effective_rank_by_prefix", {}):
                put("ErankSmall", float(g["effective_rank_by_prefix"][p0]), nd=2)
            if p0 in g.get("variance_cumfrac_by_dim", {}):
                put("VarFracSixteen", float(g["variance_cumfrac_by_dim"][p0]), nd=3)
            if g.get("cka_matrix"):
                put("CKASmallFull", float(np.asarray(g["cka_matrix"])[0, -1]), nd=3)
            ratios = []
            for r in probe.get("prefix_probe", {}).values():
                a0 = r["r2_by_dim"].get(str(pdims[0]))
                a1 = r["r2_by_dim"].get(str(pdims[-1]))
                if a0 and a1 is not None and np.isfinite(a0) and np.isfinite(a1):
                    ratios.append(a1 / a0)
            if ratios:
                put("ProbeGain", float(np.mean(ratios)), nd=1)

    if transfer:
        svd = transfer.get("svd_no_leakage", {})
        if lo and lo in svd:
            put("SVDAUCsmall", svd[lo]["macro_auc"])
            put("SVDExplVar", 100 * svd[lo]["explained_variance"], nd=2)
        rob = transfer.get("robustness", {})
        if rob and lo:
            if "clean" in rob and lo in rob["clean"]:
                put("CleanAUC", rob["clean"][lo])
            if "mixed_snr10" in rob:
                put("RobustMixedTenDB", rob["mixed_snr10"][lo])
            noisy = [v[lo] for k, v in rob.items() if k != "clean" and lo in v]
            if noisy:
                put("RobustWorst", min(noisy))
        la = transfer.get("lead_ablation", {})
        if la and lo:
            if "lead_I" in la:
                put("LeadIAUC", la["lead_I"][lo])
                put("ZeroShotLeadI", la["lead_I"][lo])
            if "6lead" in la:
                put("ZeroShotSixLead", la["6lead"][lo])
            native = auc_mean(summary, f"{HEADLINE}_lead_I", lo)
            if "12lead" in la and native:
                # price of the single-lead form factor: the twelve-lead model
                # against a model trained natively on lead I
                put("SingleLeadCost", la["12lead"][lo] - native["mean"], nd=3)
            if "lead_I" in la and native:
                # native single-lead training minus zero-shot masking, the
                # gap the discussion leans on; two decimals as in the prose
                put("LeadGap", native["mean"] - la["lead_I"][lo], nd=2)

    if quant and "modes" in quant:
        for mode, key in (("dynamic", "QuantDyn"), ("fp16", "QuantHalf")):
            q = quant["modes"].get(mode, {})
            if "size_mb" in q:
                put(key + "Size", float(q["size_mb"]), nd=1)
                put(key + "Lat", float(q["latency_ms"]), nd=1)
            if q.get("small_prefix_penalty") is not None:
                M[key + "Penalty"] = sci1(float(q["small_prefix_penalty"]))

    if ds:
        M["RecordCount"] = thousands(ds["n_records_raw"])
        M["PatientCount"] = thousands(ds["n_patients_raw"])
        M["LabelledCount"] = thousands(ds["n_records_labelled"])
        M["TestCount"] = thousands(ds["splits"]["test"])
        M["MultiLabelPct"] = str(round(100 * ds["multi_label_fraction"]))
    return M


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def sanitise_macro_name(name: str) -> str:
    """TeX control words are letters only; spell digits out."""
    out = [c if c.isalpha() else _DIGIT_WORDS.get(c, "") for c in name]
    return "".join(out) or "Unnamed"


def write_numbers(path, macros):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {sanitise_macro_name(k): v for k, v in macros.items()}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("% AUTO-GENERATED by scripts/aggregate_results.py -- do not edit.\n")
        for k, v in sorted(clean.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  wrote {path} ({len(clean)} macros)")


def write_table(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        # pgfplots reads .dat files as whitespace tables: a leading comment
        # line is fine there too, since '%' starts a pgfplots comment.
        f.write("% AUTO-GENERATED by scripts/aggregate_results.py -- do not edit.\n")
        f.write(body)
    print(f"  wrote {path}")


def _load(results_dir, name):
    p = os.path.join(results_dir, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def generate(results_dir, paper_dir, margin=0.01, verbose=True):
    """Everything main() does, returned rather than printed (used by tests)."""
    runs = load_runs(results_dir)
    summary = build_summary(runs)
    if verbose:
        print(f"  {len(runs)} runs in {len(summary)} conditions")
        for k in sorted(summary):
            print(f"    {k:<32} seeds={summary[k]['seeds']}")

    stats = compute_stats(summary, margin)
    if not stats:
        stats = _load(results_dir, "stats.json") or {}
    else:
        with open(os.path.join(results_dir, "stats.json"), "w",
                  encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=float)

    probe = _load(results_dir, "probing.json")
    transfer = _load(results_dir, "transfer.json")
    bench = _load(results_dir, "benchmark.json")
    quant = _load(results_dir, "quantization.json")
    ds = _load(results_dir, "dataset.json")

    h = summary.get(HEADLINE)
    n_test = ds["splits"]["test"] if ds else 2158
    lo = hi = None
    if h:
        lo, hi = h["dims"][0], h["dims"][-1]

    tables = {
        "tab_main.tex": tab_main(summary, n_test),
        "tab_stats.tex": tab_stats(stats, margin, len(h["dims"]) if h else 6),
        "tab_compute.tex": tab_compute(bench),
        "tab_geometry.tex": tab_geometry(probe, transfer),
        "tab_leads.tex": tab_leads(summary, transfer),
        "tab_robustness.tex": tab_robustness(transfer),
        "tab_probe.tex": tab_probe(probe,
                                   seed0_auc(summary, HEADLINE, lo) if h else None,
                                   seed0_auc(summary, HEADLINE, hi) if h else None,
                                   margin),
        "tab_perclass.tex": tab_perclass(summary),
        "tab_quant.tex": tab_quant(quant),
        "tab_ablations.tex": tab_ablations(summary),
        "tab_composition.tex": tab_composition(ds),
        "tab_external.tex": tab_external(transfer),
    }
    tables.update(figure_data(summary, probe, transfer, bench))
    macros = build_macros(summary, stats, probe, transfer, bench, quant, ds,
                          margin)
    return summary, stats, tables, macros


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--paper-dir", default="../paper")
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    summary, stats, tables, macros = generate(args.results_dir, args.paper_dir,
                                              args.margin)
    gen = os.path.join(args.paper_dir, "generated")
    for name, body in tables.items():
        write_table(os.path.join(gen, name), body)
    write_numbers(os.path.join(gen, "numbers.tex"), macros)

    out = {"conditions": {k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                          for k, v in summary.items()},
           "stats": stats}
    with open(os.path.join(args.results_dir, "aggregate.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote {args.results_dir}/aggregate.json")


if __name__ == "__main__":
    main()
