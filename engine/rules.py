"""engine.rules -- a small declarative rule grammar for cell behaviour.

WHY THIS EXISTS
---------------
The strongest engineering lesson from physicell.org is not any particular
model, it is this design goal, quoted from their site:

    "Models can be completely expressed in human-readable rules with
     well-defined built-in behavior models, rather than one-off hand-written
     C++ ... This makes models better documented (including auto-generated
     annotations) and more reproducible."

In this project, every behavioural threshold currently lives as a magic number
inside Python.  That is how the following happened: an "illustrative" threshold
was silently read as a conclusion, and a chemotaxis term that never fired
produced bit-identical results that looked like a null result rather than a bug.

So rules here are data, not code:

  * a rule declares its condition, target, effect and value in text;
  * the rule's condition is checked against the pipeline's DECLARED ports, and
    the units must be compatible -- so a rule referring to a port that does not
    exist, or comparing mmHg to seconds, is rejected at compile time;
  * every rule carries its own provenance, and a rule claiming `measured`
    without a citation cannot be created;
  * `RuleSet.annotate()` auto-generates the documentation table, so the model
    description cannot drift away from the model.

GRAMMAR
-------
    condition := <port> <op> <number> <unit>
    op        := < | <= | > | >= | ==
    effect    := set | scale | add
    value     := <number> <unit>

Example:
    Rule("cell_o2 < 5 mmHg", "cellstate.death_rate", "set", "0.05 1/h",
         basis="illustrative", note="hypoxic death threshold, chosen by us")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .units import Quantity, UnitError, unit_of

__all__ = ["Rule", "RuleSet", "RuleError"]

_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b}
_COND = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*"
                   r"(-?[0-9.eE+-]+)\s*([A-Za-z0-9_^/%*]*)\s*$")
_VALUE = re.compile(r"^\s*(-?[0-9.eE+-]+)\s*([A-Za-z0-9_^/%*]*)\s*$")


class RuleError(ValueError):
    pass


@dataclass
class Rule:
    condition: str
    target: str
    effect: str
    value: str
    basis: str = "illustrative"
    source: str | None = None
    note: str = ""

    def __post_init__(self):
        if self.effect not in ("set", "scale", "add"):
            raise RuleError(f"effect must be set/scale/add, got {self.effect!r}")
        if self.basis not in ("measured", "derived", "illustrative", "assumed"):
            raise RuleError(f"bad basis {self.basis!r}")
        if self.basis == "measured" and not self.source:
            raise RuleError(
                f"rule {self.condition!r}: basis 'measured' requires a source. "
                f"Without a citation this rule is illustrative.")
        m = _COND.match(self.condition)
        if not m:
            raise RuleError(
                f"cannot parse condition {self.condition!r}; expected "
                f"'<port> <op> <number> <unit>'")
        self.port, self.op, self.threshold, self.cond_unit = (
            m.group(1), m.group(2), float(m.group(3)), m.group(4) or "1")
        unit_of(self.cond_unit)          # raises on an unknown unit
        mv = _VALUE.match(self.value)
        if not mv:
            raise RuleError(f"cannot parse value {self.value!r}")
        self.value_number, self.value_unit = float(mv.group(1)), mv.group(2) or "1"
        unit_of(self.value_unit)
        # filled in by RuleSet.compile from the providing layer's declaration
        self.port_unit = self.cond_unit

    # ------------------------------------------------------------------
    def test(self, state):
        v = state.get(self.port)
        if v is None:
            return False
        # the state value is in the PORT's unit (declared by the providing
        # layer), not in the rule's unit; convert before comparing
        q = Quantity(float(v), self.port_unit).to(self.cond_unit)
        return _OPS[self.op](q.value, self.threshold)

    def as_text(self):
        return f"{self.condition}  =>  {self.target} {self.effect} {self.value}"

    def to_dict(self):
        return {"condition": self.condition, "port": self.port, "op": self.op,
                "threshold": self.threshold, "cond_unit": self.cond_unit,
                "target": self.target, "effect": self.effect,
                "value_number": self.value_number, "value_unit": self.value_unit,
                "basis": self.basis, "source": self.source, "note": self.note}


class RuleSet:
    def __init__(self, name="rules"):
        self.name = name
        self.rules: list[Rule] = []
        self._compiled = None

    def add(self, condition, target, effect, value, **kw):
        self.rules.append(Rule(condition, target, effect, value, **kw))
        return self

    def __len__(self):
        return len(self.rules)

    # ------------------------------------------------------------------
    def compile(self, ports):
        """Validate every rule against the pipeline's declared ports.

        `ports` maps port name -> unit string.  A rule referring to a port that
        no layer provides, or comparing units that are not compatible, is a
        hard error: that rule can never fire, and a rule that can never fire is
        exactly the silent-failure mode this project has already been burned by.
        """
        problems = []
        for r in self.rules:
            if r.port not in ports:
                problems.append(
                    f"rule {r.as_text()!r} refers to port {r.port!r}, which no "
                    f"layer provides (available: {sorted(ports)})")
                continue
            provided = ports[r.port]
            r.port_unit = provided or r.cond_unit
            if provided and r.cond_unit != "1":
                try:
                    Quantity(1.0, provided).to(r.cond_unit)
                except UnitError as exc:
                    problems.append(
                        f"rule {r.as_text()!r}: port {r.port!r} is provided in "
                        f"{provided!r} but the rule compares it to "
                        f"{r.cond_unit!r} ({exc})")
        if problems:
            raise RuleError("; ".join(problems))
        self._compiled = {r.target: r for r in self.rules}
        return self

    def fires(self, state):
        return [r for r in self.rules if r.test(state)]

    def evaluate(self, state):
        """Return {target: (effect, value, unit)} for every satisfied rule."""
        fired = {}
        for r in self.rules:
            if r.test(state):
                fired[r.target] = (r.effect, r.value_number, r.value_unit, r)
        return fired

    def apply(self, state, target_values):
        """Apply fired rules to a dict of target names -> current numbers."""
        out = dict(target_values)
        for target, (effect, num, unit, rule) in self.evaluate(state).items():
            cur = out.get(target, 0.0)
            if effect == "set":
                out[target] = num
            elif effect == "scale":
                out[target] = cur * num
            elif effect == "add":
                out[target] = cur + num
        return out

    # ------------------------------------------------------------------
    def provenance_summary(self):
        s = {}
        for r in self.rules:
            s[r.basis] = s.get(r.basis, 0) + 1
        s["_total"] = len(self.rules)
        s["_fraction_not_measured"] = (
            (s.get("illustrative", 0) + s.get("assumed", 0)) / len(self.rules)
            if self.rules else 0.0)
        return s

    def annotate(self):
        """Auto-generated documentation (the PhysiCell 'annotation' lesson)."""
        rows = [f"### 规则集 `{self.name}`（自动生成，请勿手改）", "",
                f"共 {len(self.rules)} 条规则；出处统计：{self.provenance_summary()}",
                "",
                "| 条件 | 作用对象 | 作用 | 值 | 出处类别 | 来源 | 备注 |",
                "|---|---|---|---|---|---|---|"]
        for r in self.rules:
            rows.append(f"| `{r.condition}` | `{r.target}` | {r.effect} | "
                        f"`{r.value}` | {r.basis} | {r.source or '—'} | "
                        f"{r.note or '—'} |")
        return "\n".join(rows)

    def to_json(self):
        import json
        return json.dumps({"name": self.name,
                           "provenance": self.provenance_summary(),
                           "rules": [r.to_dict() for r in self.rules]},
                          indent=2, ensure_ascii=False)
