"""engine.checks -- the verification harness.

Every layer in this project has to justify itself numerically.  The kinds of
evidence that are actually available here, in decreasing strength, are:

  analytic      -- a closed-form solution exists (diffusion kernel, Langmuir
                   equilibrium, exponential growth).  Strongest available.
  conservation  -- a quantity must be conserved to roundoff (mass, site count,
                   cell inventory).  Strong.
  determinism   -- the same seed must reproduce bit-for-bit.  Strong, and cheap.
  convergence   -- the answer must not change under refinement of dt or dx.
                   Strong for the numerical claim, silent about biology.
  invariant     -- a structural property that must hold (no duplicate ids, no
                   cycle in a lineage, no overlapping cells).
  order         -- measured convergence order vs the expected order.

What is NOT in this list, and cannot be: agreement with an experiment.  The
project has exactly one quantitative experimental series (a single control
wound calcium-radius trace).  A passing check suite therefore means "the code
does what it claims numerically", never "the model is biologically right".
`CheckSuite.summary()` says so in the emitted text so a downstream reader
cannot mistake one for the other.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Check", "CheckResult", "CheckSuite", "ChecksFailed"]

KINDS = ("analytic", "conservation", "determinism", "convergence", "invariant",
         "order")

DISCLAIMER = (
    "通过本检查只说明「代码在数值上做到了它声称的事」，"
    "**不说明模型在生物学上正确**。本项目的实验对照仅有一条钙波曲线。"
)


class ChecksFailed(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    kind: str
    fn: Callable[[], tuple]
    description: str = ""
    tolerance: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown check kind {self.kind!r}; use one of {KINDS}")


@dataclass
class CheckResult:
    name: str
    kind: str
    error: float | None
    tolerance: float | None
    passed: bool
    description: str = ""
    detail: str = ""

    def to_dict(self):
        return {"name": self.name, "kind": self.kind,
                "error": self.error, "tolerance": self.tolerance,
                "passed": bool(self.passed), "description": self.description,
                "detail": self.detail}


class CheckSuite:
    def __init__(self, name="checks"):
        self.name = name
        self.checks: list[Check] = []

    def add(self, name, kind, fn, description="", tolerance=None):
        self.checks.append(Check(name, kind, fn, description, tolerance))
        return self

    # convenience constructors -------------------------------------------
    def analytic(self, name, fn, description="", tolerance=1e-6):
        return self.add(name, "analytic", fn, description, tolerance)

    def conservation(self, name, fn, description="", tolerance=1e-12):
        return self.add(name, "conservation", fn, description, tolerance)

    def determinism(self, name, fn, description="", tolerance=0.0):
        return self.add(name, "determinism", fn, description, tolerance)

    def invariant(self, name, fn, description="", tolerance=0.0):
        return self.add(name, "invariant", fn, description, tolerance)

    # ------------------------------------------------------------------
    def run(self, stop_on_error=False):
        results = []
        for c in self.checks:
            try:
                out = c.fn()
                if isinstance(out, tuple):
                    err, tol = float(out[0]), float(out[1] if len(out) > 1 else c.tolerance)
                    detail = str(out[2]) if len(out) > 2 else ""
                else:
                    err, tol, detail = float(out), c.tolerance, ""
                if tol is None:
                    tol = c.tolerance if c.tolerance is not None else 1e-12
                results.append(CheckResult(c.name, c.kind, err, tol, err <= tol,
                                           c.description, detail))
            except Exception as exc:
                results.append(CheckResult(
                    c.name, c.kind, None, c.tolerance, False, c.description,
                    detail=f"EXCEPTION {type(exc).__name__}: {exc}\n"
                           + traceback.format_exc(limit=3)))
                if stop_on_error:
                    raise
        return results

    # ------------------------------------------------------------------
    def summary(self, results=None):
        results = results if results is not None else self.run()
        n = len(results)
        npass = sum(1 for r in results if r.passed)
        by_kind = {}
        for r in results:
            d = by_kind.setdefault(r.kind, {"n": 0, "passed": 0})
            d["n"] += 1
            d["passed"] += int(r.passed)
        return {"suite": self.name, "n_checks": n, "n_passed": npass,
                "n_failed": n - npass, "all_passed": npass == n and n > 0,
                "by_kind": by_kind, "disclaimer": DISCLAIMER}

    def require_all_passed(self, results=None):
        results = results if results is not None else self.run()
        bad = [r for r in results if not r.passed]
        if bad:
            raise ChecksFailed(
                "; ".join(f"{r.name}: error={r.error} tol={r.tolerance} {r.detail[:120]}"
                          for r in bad))
        return True

    # ------------------------------------------------------------------
    def markdown(self, results=None):
        results = results if results is not None else self.run()
        s = self.summary(results)
        rows = [f"## 检查套件 `{self.name}`", "",
                f"共 {s['n_checks']} 项，通过 {s['n_passed']}，失败 {s['n_failed']}。", "",
                "| 检查 | 类型 | 误差 | 容差 | 通过 | 说明 |",
                "|---|---|---|---|---|---|"]
        for r in results:
            err = "—" if r.error is None else f"{r.error:.3e}"
            tol = "—" if r.tolerance is None else f"{r.tolerance:.3e}"
            rows.append(f"| `{r.name}` | {r.kind} | {err} | {tol} | "
                        f"{'✅' if r.passed else '❌'} | {r.description} |")
        rows += ["", f"> {DISCLAIMER}"]
        return "\n".join(rows)

    def to_json(self, results=None):
        results = results if results is not None else self.run()
        return json.dumps({"summary": self.summary(results),
                           "results": [r.to_dict() for r in results]},
                          indent=2, ensure_ascii=False)
