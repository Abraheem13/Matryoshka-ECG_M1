#!/usr/bin/env python3
"""
Link-budget arithmetic over measured payload sizes.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This script computes airtime and transmit energy for each nesting
granularity by dividing *measured* payload sizes by *declared* link
rates. It is arithmetic. It is not a network measurement, and nothing
it emits should be described as one.

The distinction matters here for a specific reason. An earlier version
of this work contained a device-tier latency table produced by scaling
one GPU measurement by a ratio of nominal device throughputs, and
presented it alongside measured quantities. That table has been
removed. The lesson is not "never extrapolate" -- a deployment
paper that refuses to do arithmetic over its own measurements is not
useful -- but "never let arithmetic sit in a table of measurements
without saying which is which".

So: outputs land in generated/linkbudget.tex, separate from
generated/numbers.tex, every macro is prefixed LB, and the manuscript
labels the resulting table as derived.

The ratios this produces (32x between granularities, 750x against the
raw waveform) are parameter-free: they depend only on payload sizes,
not on the link rates. The absolute airtimes and energies depend on the
declared parameters and move with them. The crossover rate below which
payload energy exceeds inference energy is the one number here that a
practitioner should recompute for their own hardware, so it is emitted
explicitly.

Usage
-----
    python scripts/link_budget.py --paper-dir ../paper
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

# --------------------------------------------------------------------
# Measured inputs.  These come from the experiment run, not from here.
# --------------------------------------------------------------------
NESTING_DIMS: List[int] = [16, 32, 64, 128, 256, 512]
BYTES_PER_COORD = 4                      # float32 embedding storage

# Raw acquisition: 12 leads x 1000 samples (10 s @ 100 Hz) x float32.
RAW_LEADS, RAW_SAMPLES = 12, 1000
RAW_BYTES = RAW_LEADS * RAW_SAMPLES * BYTES_PER_COORD   # 48,000 B

# Measured on NVIDIA L4 by scripts/benchmark_hardware.py. Reported in
# the manuscript with an explicit caveat: it is the difference of two
# nearly equal large numbers (46.9 W under load vs 46.1 W idle) and is
# an order-of-magnitude estimate, not a precise value.
ENERGY_PER_INFERENCE_J = 3.5e-3

ACQUISITION_WINDOW_S = 10.0              # one PTB-XL record

# --------------------------------------------------------------------
# Declared link parameters.  Every one of these is a parameter, not a
# result; they are printed in the manuscript so a reader can substitute
# their own.  Rates are effective uplink goodput, not PHY peak.
# --------------------------------------------------------------------
LINKS: List[Tuple[str, str, float, float, str]] = [
    # (key,    label,                      rate_bps,  tx_power_W, IMT-2030)
    ("nbiot",  "NB-IoT (coverage-limited)",   20_000.0, 0.50,
     "Ubiquitous connectivity"),
    ("ltem",   "LTE-M / massive comm.",      375_000.0, 0.35,
     "Massive communication"),
    ("urllc",  "5G/6G URLLC slice",       10_000_000.0, 0.20,
     "HRLLC"),
    ("mmwave", "mmWave / fixed backhaul", 100_000_000.0, 0.20,
     "Immersive comm."),
]


def payload_bytes(dim: int) -> int:
    return dim * BYTES_PER_COORD


def airtime_s(nbytes: int, rate_bps: float) -> float:
    return (nbytes * 8.0) / rate_bps


def tx_energy_j(nbytes: int, rate_bps: float, tx_power_w: float) -> float:
    return tx_power_w * airtime_s(nbytes, rate_bps)


def crossover_rate_bps(nbytes: int, tx_power_w: float,
                       inference_energy_j: float) -> float:
    """
    Uplink rate at which transmit energy for `nbytes` equals the energy
    of one encoder invocation.

        P_tx * (8*nbytes / R) = E_inf   =>   R = 8*nbytes*P_tx / E_inf

    Below this rate the payload dominates the per-round energy budget
    and shortening it is the effective lever. Above it the encoder
    dominates and nesting cannot help -- quantisation or a smaller
    encoder is the only remaining axis. This is the single most
    actionable number the analysis produces.
    """
    return (8.0 * nbytes * tx_power_w) / inference_energy_j


def _fmt_time(seconds: float) -> str:
    """Airtime with a unit chosen for legibility rather than uniformity."""
    if seconds >= 1.0:
        return f"{seconds:.2f}\\,s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}\\,ms"
    return f"{seconds * 1e6:.1f}\\,$\\mu$s"


def _fmt_energy(joules: float) -> str:
    if joules >= 1.0:
        return f"{joules:.2f}\\,J"
    if joules >= 1e-3:
        return f"{joules * 1e3:.1f}\\,mJ"
    return f"{joules * 1e6:.1f}\\,$\\mu$J"


def build() -> Dict[str, object]:
    small, large = NESTING_DIMS[0], NESTING_DIMS[-1]
    b_small, b_large = payload_bytes(small), payload_bytes(large)

    rows = []
    for key, label, rate, p_tx, scenario in LINKS:
        row = {"key": key, "label": label, "rate_bps": rate,
               "tx_power_w": p_tx, "scenario": scenario}
        for tag, nb in (("small", b_small), ("large", b_large),
                        ("raw", RAW_BYTES)):
            t = airtime_s(nb, rate)
            e = tx_energy_j(nb, rate, p_tx)
            row[f"airtime_{tag}_s"] = t
            row[f"txenergy_{tag}_j"] = e
            row[f"duty_{tag}"] = t / ACQUISITION_WINDOW_S
            # Per-round energy = one encoder pass + one uplink.
            row[f"round_{tag}_j"] = ENERGY_PER_INFERENCE_J + e
        row["round_ratio"] = row["round_large_j"] / row["round_small_j"]
        row["crossover_bps"] = crossover_rate_bps(b_large, p_tx,
                                                  ENERGY_PER_INFERENCE_J)
        rows.append(row)

    return {
        "raw_bytes": RAW_BYTES,
        "bytes_small": b_small,
        "bytes_large": b_large,
        "ratio_nesting": b_large / b_small,
        "ratio_vs_raw_small": RAW_BYTES / b_small,
        "ratio_vs_raw_large": RAW_BYTES / b_large,
        "energy_inference_j": ENERGY_PER_INFERENCE_J,
        "rows": rows,
    }


def emit_macros(d: Dict[str, object]) -> str:
    by = {r["key"]: r for r in d["rows"]}
    nb, ur = by["nbiot"], by["urllc"]

    L = [
        "% AUTO-GENERATED by scripts/link_budget.py -- do not edit.",
        "% DERIVED ARITHMETIC over measured payload sizes; NOT measurements.",
        f"\\newcommand{{\\LBRawKB}}{{{d['raw_bytes'] / 1000:.0f}}}",
        f"\\newcommand{{\\LBRatioNesting}}{{{d['ratio_nesting']:.0f}}}",
        f"\\newcommand{{\\LBRatioRawSmall}}{{{d['ratio_vs_raw_small']:.0f}}}",
        f"\\newcommand{{\\LBRatioRawLarge}}{{{d['ratio_vs_raw_large']:.0f}}}",
        f"\\newcommand{{\\LBNbRate}}{{{by['nbiot']['rate_bps'] / 1e3:.0f}}}",
        f"\\newcommand{{\\LBNbAirSmall}}{{{_fmt_time(nb['airtime_small_s'])}}}",
        f"\\newcommand{{\\LBNbAirLarge}}{{{_fmt_time(nb['airtime_large_s'])}}}",
        f"\\newcommand{{\\LBNbAirRaw}}{{{_fmt_time(nb['airtime_raw_s'])}}}",
        f"\\newcommand{{\\LBNbDutyRaw}}{{{nb['duty_raw'] * 100:.0f}}}",
        f"\\newcommand{{\\LBNbDutyLarge}}{{{nb['duty_large'] * 100:.1f}}}",
        f"\\newcommand{{\\LBNbDutySmall}}{{{nb['duty_small'] * 100:.2f}}}",
        f"\\newcommand{{\\LBNbTxSmall}}{{{_fmt_energy(nb['txenergy_small_j'])}}}",
        f"\\newcommand{{\\LBNbTxLarge}}{{{_fmt_energy(nb['txenergy_large_j'])}}}",
        f"\\newcommand{{\\LBNbRoundSmall}}{{{_fmt_energy(nb['round_small_j'])}}}",
        f"\\newcommand{{\\LBNbRoundLarge}}{{{_fmt_energy(nb['round_large_j'])}}}",
        f"\\newcommand{{\\LBNbRoundRatio}}{{{nb['round_ratio']:.1f}}}",
        f"\\newcommand{{\\LBUrRoundRatio}}{{{ur['round_ratio']:.2f}}}",
        f"\\newcommand{{\\LBCrossMbps}}{{{nb['crossover_bps'] / 1e6:.1f}}}",
        f"\\newcommand{{\\LBEnergyInf}}{{{d['energy_inference_j'] * 1e3:.1f}}}",
    ]
    # Plain numbers for the per-round energy figure, so its curves, tier
    # markers and crossover line are drawn from the same constants this
    # script computes with rather than retyped into the figure.
    nb_p = next(r for r in d["rows"] if r["key"] == "nbiot")["tx_power_w"]
    L += [
        f"\\newcommand{{\\LBPlotPtx}}{{{nb_p:g}}}",
        f"\\newcommand{{\\LBBitsSmall}}{{{8 * d['bytes_small']}}}",
        f"\\newcommand{{\\LBBitsLarge}}{{{8 * d['bytes_large']}}}",
        f"\\newcommand{{\\LBCrossBps}}{{{nb['crossover_bps']:.0f}}}",
    ]
    for r in d["rows"]:
        L.append(f"\\newcommand{{\\LBBps{r['key'].capitalize()}}}"
                 f"{{{r['rate_bps']:.0f}}}")
    # Per-tier declared parameters, so the parameter table in the paper
    # is generated from the same constants this script computes with and
    # cannot drift from them.
    for r in d["rows"]:
        k = r["key"].capitalize()
        rate = r["rate_bps"]
        rate_s = (f"{rate/1e6:g}\\,Mbit/s" if rate >= 1e6
                  else f"{rate/1e3:g}\\,kbit/s")
        L.append(f"\\newcommand{{\\LBRate{k}}}{{{rate_s}}}")
        L.append(f"\\newcommand{{\\LBPwr{k}}}{{{r['tx_power_w']:.2f}}}")
    return "\n".join(L) + "\n"


def emit_linkparams(d: Dict[str, object]) -> str:
    head = r"""% AUTO-GENERATED by scripts/link_budget.py -- do not edit.
\begin{table}[!t]
\centering
\caption{Declared uplink parameters. These are \textbf{inputs}, not
findings: rates are representative effective uplink goodput rather than
PHY peak, and transmit powers are representative active-transmit figures
drawn from the cited literature~\cite{nbiot2020,popovski2019,itu2160}.
They parameterise the uplink cost analysis of
Section~\ref{sec:results:link} and nothing else; the payload ratios that
carry our argument do not depend on them. The
IMT-2030 column names the usage scenario of~\cite{itu2160} that each
tier most closely corresponds to.}
\label{tab:linkparams}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrr@{}}
\toprule
Tier & IMT-2030 scenario & $R_\tau$ & $P_{\mathrm{tx}}$ \\
\midrule
"""
    body = []
    for r in d["rows"]:
        rate = r["rate_bps"]
        rate_s = (f"\\SI{{{rate/1e6:g}}}{{\\mega\\bit\\per\\second}}"
                  if rate >= 1e6 else
                  f"\\SI{{{rate/1e3:g}}}{{\\kilo\\bit\\per\\second}}")
        body.append(
            f"{r['label']} & {r['scenario']} & {rate_s} & "
            f"\\SI{{{r['tx_power_w']:.2f}}}{{\\watt}} \\\\")
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n" \
                                    "\\end{table}\n"


def emit_table(d: Dict[str, object]) -> str:
    head = r"""% AUTO-GENERATED by scripts/link_budget.py -- do not edit.
\begin{table}[!t]
\centering
\caption{Uplink cost of one monitoring round, \textbf{derived} by
dividing measured payload sizes by the declared effective goodput of
each tier. These are not network measurements. Transmit energy uses the
stated $P_{\mathrm{tx}}$; the per-round figure adds the measured
\R{LBEnergyInf}\,mJ encoder pass. Duty is airtime as a fraction of the
10\,s acquisition window: a value above 100\,\% means the tier cannot
sustain continuous monitoring at that payload.}
\label{tab:linkbudget}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Tier & Payload & Airtime & Duty & $E_{\mathrm{tx}}$ & Round \\
\midrule
"""
    body = []
    for r in d["rows"]:
        body.append(
            rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{r['label']}}}, "
            rf"$R$={r['rate_bps'] / 1e6:g}\,Mbit/s, "
            rf"$P_{{\mathrm{{tx}}}}$={r['tx_power_w']:g}\,W}} \\"
        )
        for tag, name in (("small", r"$d$=16 (64\,B)"),
                          ("large", r"$d$=512 (2\,KB)"),
                          ("raw", r"raw (48\,kB)")):
            duty = r[f"duty_{tag}"] * 100
            duty_s = (rf"\textbf{{{duty:.0f}\,\%}}" if duty > 100
                      else f"{duty:.2f}\\,\\%")
            body.append(
                f"\\quad {name} & & {_fmt_time(r[f'airtime_{tag}_s'])} & "
                f"{duty_s} & {_fmt_energy(r[f'txenergy_{tag}_j'])} & "
                f"{_fmt_energy(r[f'round_{tag}_j'])} \\\\"
            )
        body.append(r"\addlinespace[2pt]")
    tail = r"""\bottomrule
\end{tabular}
\end{table}
"""
    return head + "\n".join(body) + "\n" + tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-dir", default="../paper")
    ap.add_argument("--json-out", default=None,
                    help="optionally also dump the raw numbers as JSON")
    args = ap.parse_args()

    d = build()
    gen = os.path.join(args.paper_dir, "generated")
    os.makedirs(gen, exist_ok=True)

    with open(os.path.join(gen, "linkbudget.tex"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(emit_macros(d))
    with open(os.path.join(gen, "tab_linkbudget.tex"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(emit_table(d))
    with open(os.path.join(gen, "tab_linkparams.tex"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(emit_linkparams(d))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    nb = next(r for r in d["rows"] if r["key"] == "nbiot")
    print("link budget written to", gen)
    print(f"  payload {d['bytes_small']} B .. {d['bytes_large']} B "
          f"({d['ratio_nesting']:.0f}x), raw {d['raw_bytes']} B "
          f"({d['ratio_vs_raw_small']:.0f}x vs d=16)")
    print(f"  NB-IoT raw duty {nb['duty_raw'] * 100:.0f}% "
          f"-> continuous monitoring infeasible at raw payload")
    print(f"  NB-IoT per-round energy ratio d=512/d=16 = "
          f"{nb['round_ratio']:.1f}x")
    print(f"  crossover rate = {nb['crossover_bps'] / 1e6:.2f} Mbit/s")


if __name__ == "__main__":
    main()
