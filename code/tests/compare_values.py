#!/usr/bin/env python3
"""
Value-level comparison between regenerated tables and a reference set.

Captions, labels and layout may legitimately change; the numbers may
not. For every table this extracts the numeric tokens from the tabular
rows (lines containing '&'), in order, and requires the two sequences to
be identical.

    python tests/compare_values.py REF_DIR NEW_DIR
"""

from __future__ import annotations

import os
import re
import sys

NUM = re.compile(r"(?<![A-Za-z0-9_{])(\d*\.\d+|\d+)(?![A-Za-z}])")
THOUSANDS = re.compile(r"(?<=\d)(?:\{,\}|,)(?=\d{3}\b)")


def row_numbers(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith("%") or "&" not in line:
            continue
        line = re.sub(r"\\(?:R|cite|ref|eqref|label)\{[^}]*\}", " ", line)
        line = re.sub(r"\\(?:multicolumn|cmidrule)(?:\([^)]*\))?\{[^}]*\}"
                      r"(?:\{[^}]*\})?", " ", line)
        line = re.sub(r"\\(?:phantom)\{[^}]*\}", " ", line)
        line = re.sub(r"\\(?:SI)\{([^}]*)\}\{[^}]*\}", r" \1 ", line)
        line = re.sub(r"\\(?:mathcal|mathrm)\{[^}]*\}", " ", line)
        line = re.sub(r"10\^\{-?\d+\}", lambda m: m.group(0).replace("^", "E"),
                      line)
        line = THOUSANDS.sub("", line)
        for tok in NUM.findall(line):
            t = tok if not tok.startswith(".") else "0" + tok
            out.append(t)
    return out


def main():
    ref, new = sys.argv[1], sys.argv[2]
    names = sorted(n for n in os.listdir(new)
                   if n.startswith("tab_") and n.endswith(".tex"))
    bad = 0
    for n in names:
        rp = os.path.join(ref, n)
        if not os.path.isfile(rp):
            print(f"  --    {n:<22} (no reference)")
            continue
        a, b = row_numbers(rp), row_numbers(os.path.join(new, n))
        if a == b:
            print(f"  OK    {n:<22} {len(a):3d} values identical")
        else:
            bad += 1
            print(f"  DIFF  {n:<22} ref={len(a)} new={len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"        first difference at #{i}: ref {x} new {y}")
                    break
            if len(a) != len(b):
                print(f"        ref tail {a[len(b):][:8]}  new tail {b[len(a):][:8]}")
    print("FAIL" if bad else "ALL VALUES IDENTICAL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
