#!/usr/bin/env python3
"""
Golden-file test for the manuscript's generated files.

Rebuilds the fixture (tests/fixtures/build_fixture.py), runs
scripts/aggregate_results.py on it into a scratch directory, and checks:

1. REGRESSION (must pass). Every regenerated table, macro file and
   figure-data file is byte-identical, after its one-line header, to the
   snapshot in tests/fixtures/expected/. Any change to the generator that
   alters a single character of manuscript output fails here.

2. GROUPING (must pass). The four headline seeds form one condition and
   no ablation is merged into it -- the invariant whose violation meant
   the previous aggregator could not reproduce the manuscript.

3. PROVENANCE. Whether the published paper/generated/ files still equal
   that snapshot. They do until you regenerate from your own results/
   directory; after that they legitimately differ (unrounded values, the
   ablation dashes filled), so this is reported rather than failed. Pass
   --require-published-match to make it a hard failure, which is what the
   repository's own `make verify` does: it is the check that the numbers
   published here are the generator's output and were not edited by hand.

The snapshot values were separately shown to be identical to the tables
of the experimental run as published (tests/compare_values.py against the
previously generated files).

    python tests/test_reproduce_tables.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
EXPECTED = os.path.join(HERE, "fixtures", "expected")
FIXTURE = os.path.join(HERE, "fixtures", "results")
PAPER_GEN = os.path.join(os.path.dirname(CODE), "paper", "generated")

sys.path.insert(0, os.path.join(CODE, "scripts"))


def body(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(l.rstrip() for l in lines
                     if not l.startswith("% AUTO-GENERATED")).strip()


def first_diff(a_path, b_path):
    a, b = body(a_path).splitlines(), body(b_path).splitlines()
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
             min(len(a), len(b)))
    return (i + 1, a[i] if i < len(a) else "<eof>",
            b[i] if i < len(b) else "<eof>")


def main() -> int:
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable,
                    os.path.join(HERE, "fixtures", "build_fixture.py")],
                   check=True, capture_output=True, env=env)
    import aggregate_results as ag  # noqa: E402

    failures = []

    # ---- 2. grouping ---------------------------------------------------
    runs = ag.load_runs(FIXTURE)
    summary = ag.build_summary(runs)
    head = summary.get(ag.HEADLINE)
    if not head or head["seeds"] != [0, 1, 2, 3]:
        failures.append(f"headline must have seeds [0,1,2,3]; got "
                        f"{head and head['seeds']}")
    for cond in ("inception1d_mrl_perrecord", "inception1d_mrl_ls01",
                 "inception1d_mrl_wlinear", "inception1d_mrl_wexponential",
                 "inception1d_mrl_winverse", "inception1d_constant"):
        if cond not in summary or summary[cond]["n_seeds"] != 1:
            failures.append(f"ablation '{cond}' must be its own 1-seed group")
    print(f"  grouping : {len(runs)} runs -> {len(summary)} conditions; "
          f"headline seeds {head and head['seeds']}")

    # ---- 1. regression -------------------------------------------------
    manuscript_equal = True
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable,
                        os.path.join(CODE, "scripts", "aggregate_results.py"),
                        "--results-dir", FIXTURE, "--paper-dir", tmp],
                       check=True, capture_output=True, env=env)
        gen = os.path.join(tmp, "generated")
        names = sorted(n for n in os.listdir(gen)
                       if n.endswith((".tex", ".dat")))
        expected = set(os.listdir(EXPECTED))
        missing = sorted(expected - set(names))
        if missing:
            failures.append(f"generator no longer produces: {missing}")
        for n in names:
            exp = os.path.join(EXPECTED, n)
            if not os.path.isfile(exp):
                failures.append(f"{n}: produced but absent from the snapshot")
                continue
            if body(exp) != body(os.path.join(gen, n)):
                ln, a, b = first_diff(exp, os.path.join(gen, n))
                failures.append(f"{n}: differs from snapshot at line {ln}\n"
                                f"      expected   : {a}\n"
                                f"      regenerated: {b}")
            # ---- 3. provenance (reported only) ---------------------------
            ref = os.path.join(PAPER_GEN, n)
            if not os.path.isfile(ref) or body(ref) != body(exp):
                manuscript_equal = False
        print(f"  regression: {len(names)} generated files checked "
              f"against the snapshot")

    strict = "--require-published-match" in sys.argv
    if manuscript_equal:
        print("  provenance: paper/generated/ equals the pipeline's output "
              "(the values the article reports)")
    elif strict:
        failures.append(
            "paper/generated/ differs from the pipeline's output on this "
            "fixture. Either the generator changed without the published "
            "files being regenerated, or those files were edited by hand.")
    else:
        print("  provenance: paper/generated/ differs from the fixture "
              "snapshot -- expected after regenerating from real results/")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
