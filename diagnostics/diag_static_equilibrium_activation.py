#!/usr/bin/env python3
"""Deprecated root-only diagnostic: NOT an equilibrium certificate.

The old implementation optimized only qacc[:6], declared equilibrium from relative
residual reduction, and classified release by its endpoint. None certifies full-state
equilibrium or stability. It also assumed nu=na and omitted consistent ctrl at the
instantaneous evaluation. That implementation is intentionally no longer executable.
Use diag_full_state_equilibrium_r28.py with explicit --xml and --out instead.
"""
from pathlib import Path
import sys


def main():
    replacement = Path(__file__).resolve().with_name("diag_full_state_equilibrium_r28.py")
    print("ROOT_ONLY_NOT_EQUILIBRIUM_CERTIFICATE: legacy diagnostic retired.", file=sys.stderr)
    print(f"Run {sys.executable} {replacement} --xml /absolute/path/model.xml "
          "--out /absolute/path/result.json", file=sys.stderr)
    print("Old output schema is not preserved: use the new separated instantaneous, activation-dynamics, "
          "and whole-interval release assessments.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
