#!/usr/bin/env python3
"""
Compare two numbers.tex files macro by macro.

    python tests/compare_macros.py REF NEW [--allow NAME ...]

Reports macros whose values differ, macros only in REF (the paper uses a
value the pipeline no longer emits) and macros only in NEW. Exit code is
non-zero on any difference not listed with --allow.
"""

from __future__ import annotations

import argparse
import re
import sys

NEWC = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = NEWC.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--allow", nargs="*", default=[])
    a = ap.parse_args()
    ref, new = load(a.ref), load(a.new)
    allow = set(a.allow)
    bad = 0
    same = sorted(k for k in ref if k in new and ref[k] == new[k])
    for k in sorted(ref):
        if k in new and ref[k] != new[k]:
            tag = "allowed" if k in allow else "DIFF"
            bad += tag == "DIFF"
            print(f"  {tag:<8}{k:<20} ref={ref[k]!r:<28} new={new[k]!r}")
    for k in sorted(set(ref) - set(new)):
        tag = "allowed" if k in allow else "MISSING"
        bad += tag == "MISSING"
        print(f"  {tag:<8}{k:<20} only in reference ({ref[k]!r})")
    for k in sorted(set(new) - set(ref)):
        print(f"  new     {k:<20} {new[k]!r}")
    print(f"\n  {len(same)} identical, {bad} unexplained difference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
