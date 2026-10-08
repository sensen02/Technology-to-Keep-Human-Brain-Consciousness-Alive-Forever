"""engine.params -- parameter registry with ENFORCED provenance.

WHY THIS EXISTS
---------------
The single hardest limit on this project is not compute, it is that most of its
parameters are not identifiable from any available measurement.  That is only
survivable if every parameter's origin is recorded and the engine refuses to
let an unlabelled number enter a result.

This module turns the project's hand-kept discipline into a mechanism:

  * ``measured``   -- requires a source string.  No source, no entry.
  * ``derived``    -- requires the names of the parameters it is computed from,
                      and those must already be registered (so a derivation
                      cannot silently depend on a number that was never
                      declared).
  * ``illustrative`` -- chosen by us.  Allowed, but counted and reported.
  * ``assumed``    -- a modelling choice that has no measurement and no
                      plausible range (e.g. "chemotaxis sensitivity").

Two failure modes the project has already hit are checked explicitly:

  * **species mixing** -- a parameter measured in mouse must not be used
    silently in a Drosophila model.
  * **stage mixing** -- ``MEASURED_ANCHORS.md`` carries an explicit warning that
    its anchors come from different developmental stages (notum 12-13.5 hAPF
    vs 18-26 hAPF vs embryo stage 5) and that every comparison must declare its
    stage.  ``check_stage`` enforces that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

__all__ = ["Provenance", "Param", "ParamRegistry", "ProvenanceError"]

MEASURED = "measured"
DERIVED = "derived"
ILLUSTRATIVE = "illustrative"
ASSUMED = "assumed"
_VALID = (MEASURED, DERIVED, ILLUSTRATIVE, ASSUMED)


class ProvenanceError(ValueError):
    pass


@dataclass
class Param:
    name: str
    value: float
    unit: str = "1"
    provenance: str = ILLUSTRATIVE
    source: str | None = None          # citation, required for `measured`
    derived_from: list = field(default_factory=list)   # required for `derived`
    species: str | None = None
    stage: str | None = None
    notes: str = ""
    range: tuple | None = None         # (lo, hi) plausible range if known

    def __post_init__(self):
        if self.provenance not in _VALID:
            raise ProvenanceError(
                f"{self.name}: provenance must be one of {_VALID}, "
                f"got {self.provenance!r}")
        if self.provenance == MEASURED and not self.source:
            raise ProvenanceError(
                f"{self.name}: provenance 'measured' requires a source citation. "
                f"If you do not have one, it is not measured.")
        if self.provenance == DERIVED and not self.derived_from:
            raise ProvenanceError(
                f"{self.name}: provenance 'derived' requires derived_from=[...]")
        if self.range is not None:
            lo, hi = self.range
            if not (lo <= self.value <= hi):
                raise ProvenanceError(
                    f"{self.name}: value {self.value} outside stated range "
                    f"{self.range}")

    def to_dict(self):
        d = asdict(self)
        d["range"] = list(self.range) if self.range else None
        return d


class ParamRegistry:
    """A named, validated collection of parameters."""

    def __init__(self, name="registry"):
        self.name = name
        self._p: dict[str, Param] = {}

    # ------------------------------------------------------------------
    def add(self, param):
        if param.name in self._p:
            raise ProvenanceError(f"duplicate parameter {param.name!r}")
        self._p[param.name] = param
        return param

    def define(self, name, value, unit="1", provenance=ILLUSTRATIVE, **kw):
        return self.add(Param(name, value, unit, provenance, **kw))

    def __getitem__(self, name):
        if name not in self._p:
            raise KeyError(
                f"parameter {name!r} is not registered; refusing to use an "
                f"undeclared number")
        return self._p[name].value

    def unit(self, name):
        return self._p[name].unit

    def quantity(self, name):
        from .units import Quantity
        p = self._p[name]
        return Quantity(p.value, p.unit)

    def get(self, name, default=None):
        return self._p[name].value if name in self._p else default

    def names(self):
        return sorted(self._p)

    def __len__(self):
        return len(self._p)

    def __contains__(self, name):
        return name in self._p

    # ------------------------------------------------------------------
    def validate(self):
        """Every `derived` parameter must depend only on registered names."""
        problems = []
        for p in self._p.values():
            for dep in p.derived_from:
                if dep not in self._p:
                    problems.append(
                        f"{p.name} derives from unregistered parameter {dep!r}")
        if problems:
            raise ProvenanceError("; ".join(problems))
        return True

    # ------------------------------------------------------------------
    def check_species(self, model_species, allow_cross=False):
        """Flag measured parameters from a different species.

        Returns a list of (name, param_species).  Raises if not allow_cross and
        the model declares itself strict.
        """
        offenders = []
        for p in self._p.values():
            if p.provenance not in (MEASURED, DERIVED):
                continue
            if p.species and model_species and p.species != model_species:
                offenders.append((p.name, p.species))
        if offenders and not allow_cross:
            raise ProvenanceError(
                f"model species is {model_species!r} but these measured "
                f"parameters come from another species: {offenders}. "
                f"Pass allow_cross=True and say so explicitly in the report.")
        return offenders

    def check_stage(self, model_stage):
        """Flag measured parameters from a different developmental stage."""
        offenders = []
        for p in self._p.values():
            if p.provenance != MEASURED:
                continue
            if p.stage and model_stage and p.stage != model_stage:
                offenders.append((p.name, p.stage))
        return offenders

    # ------------------------------------------------------------------
    def summary(self):
        out = {k: 0 for k in _VALID}
        by_unit = {}
        for p in self._p.values():
            out[p.provenance] += 1
            by_unit[p.unit] = by_unit.get(p.unit, 0) + 1
        out["_total"] = len(self._p)
        out["_fraction_illustrative_or_assumed"] = (
            (out[ILLUSTRATIVE] + out[ASSUMED]) / len(self._p) if self._p else 0.0)
        return out

    def measured_names(self):
        return sorted(n for n, p in self._p.items() if p.provenance == MEASURED)

    def unmeasured_names(self):
        return sorted(n for n, p in self._p.items()
                      if p.provenance in (ILLUSTRATIVE, ASSUMED))

    def require_measured(self, names):
        """Refuse to make a validated claim about parameters that are not measured."""
        bad = [n for n in names
               if n not in self._p or self._p[n].provenance != MEASURED]
        if bad:
            raise ProvenanceError(
                f"cannot claim these as measured: {bad}")
        return True

    # ------------------------------------------------------------------
    def snapshot(self):
        return {n: p.to_dict() for n, p in sorted(self._p.items())}

    def to_json(self, indent=2):
        return json.dumps(self.snapshot(), indent=indent)

    def markdown_table(self):
        rows = ["| 参数 | 值 | 单位 | 出处类别 | 来源 | 物种 | 阶段 |",
                "|---|---|---|---|---|---|---|"]
        for n in self.names():
            p = self._p[n]
            rows.append(
                f"| `{n}` | {p.value:g} | {p.unit} | {p.provenance} | "
                f"{p.source or '—'} | {p.species or '—'} | {p.stage or '—'} |")
        return "\n".join(rows)
