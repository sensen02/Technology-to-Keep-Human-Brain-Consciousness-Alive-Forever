"""engine.units -- explicit physical units with dimensional arithmetic.

WHY THIS EXISTS
---------------
This project has already produced several real unit bugs:

  * m^2 -> um^2 converted with 1e6 instead of 1e12, which silently made a
    diffusion length 1000x too small and made chemotaxis do nothing;
  * Chebyshev/engine coordinates treated as um, giving a "cell radius" off by
    orders of magnitude;
  * PhysiCell's own oxygen block declares "dimensionless" units while holding
    mmHg-like numbers (recorded in ORGANISM_SCALE_PLAN.md).

A silent unit error does not crash; it produces a plausible wrong number, which
is the worst failure mode for a simulation.  So units are explicit here and
mismatches raise.

DESIGN
------
Base dimensions: length m, time s, mass kg, amount mol, temperature K.
A unit string is parsed into {base_dimension: exponent} plus a scale factor to
SI.  Supported syntax: "um", "um^2", "um/s", "mol/m^3", "mmHg", "1/min",
"mmol/L", and products/quotients with ``*``, ``/`` and integer powers.

This is deliberately a small system: it covers what this project uses.  It is
not a general units library and does not attempt offset units (degC) -- use K.
"""

from __future__ import annotations

import math

__all__ = ["UnitError", "Dims", "Quantity", "Q", "unit_of", "convert"]

# base symbol -> (SI factor, dimension name)
_BASE = {
    "m": (1.0, "length"),
    "s": (1.0, "time"),
    "kg": (1.0, "mass"),
    "mol": (1.0, "amount"),
    "K": (1.0, "temperature"),
}

# prefixed / derived units, expressed in base units
_UNITS = {
    # length
    "nm": (1e-9, {"m": 1}), "um": (1e-6, {"m": 1}), "mm": (1e-3, {"m": 1}),
    "cm": (1e-2, {"m": 1}), "dm": (1e-1, {"m": 1}), "km": (1e3, {"m": 1}),
    "m": (1.0, {"m": 1}),
    # time
    "ns": (1e-9, {"s": 1}), "us": (1e-6, {"s": 1}), "ms": (1e-3, {"s": 1}),
    "s": (1.0, {"s": 1}), "min": (60.0, {"s": 1}), "h": (3600.0, {"s": 1}),
    "day": (86400.0, {"s": 1}), "MCS": (1.0, {"s": 1}),
    # mass
    "ug": (1e-9, {"kg": 1}), "mg": (1e-6, {"kg": 1}), "g": (1e-3, {"kg": 1}),
    "kg": (1.0, {"kg": 1}),
    # amount
    "mol": (1.0, {"mol": 1}), "mmol": (1e-3, {"mol": 1}),
    "umol": (1e-6, {"mol": 1}), "nmol": (1e-9, {"mol": 1}),
    "pmol": (1e-12, {"mol": 1}), "fmol": (1e-15, {"mol": 1}),
    # volume
    "L": (1e-3, {"m": 3}), "mL": (1e-6, {"m": 3}), "uL": (1e-9, {"m": 3}),
    "nL": (1e-12, {"m": 3}), "pL": (1e-15, {"m": 3}),
    # pressure
    "Pa": (1.0, {"kg": 1, "m": -1, "s": -2}),
    "kPa": (1e3, {"kg": 1, "m": -1, "s": -2}),
    "mmHg": (133.322, {"kg": 1, "m": -1, "s": -2}),
    "torr": (133.322, {"kg": 1, "m": -1, "s": -2}),
    "bar": (1e5, {"kg": 1, "m": -1, "s": -2}),
    # voltage: kg m^2 / (A s^3) expressed in SI as kg m^2 s^-3 A^-1; we carry
    # the base dimensions explicitly so conversions stay exact.
    "V": (1.0, {"kg": 1, "m": 2, "s": -3, "A": -1}),
    "mV": (1e-3, {"kg": 1, "m": 2, "s": -3, "A": -1}),
    "uV": (1e-6, {"kg": 1, "m": 2, "s": -3, "A": -1}),
    # concentration: amount / volume.  Physiology uses molarity everywhere, so
    # these are first-class rather than requiring "umol/L" each time.
    "M": (1e3, {"mol": 1, "m": -3}),      # 1 mol/L = 1000 mol/m^3
    "mM": (1.0, {"mol": 1, "m": -3}),
    "uM": (1e-3, {"mol": 1, "m": -3}),
    "nM": (1e-6, {"mol": 1, "m": -3}),
    "pM": (1e-9, {"mol": 1, "m": -3}),
    # dimensionless
    "1": (1.0, {}), "": (1.0, {}), "%": (0.01, {}),
}


class UnitError(ValueError):
    """Raised for unknown units, dimension mismatch, or bad unit syntax."""


class Dims:
    """A dimension signature: mapping base dimension -> exponent."""

    __slots__ = ("e",)

    def __init__(self, e=None):
        self.e = {k: v for k, v in (e or {}).items() if v != 0}

    def __mul__(self, other):
        d = dict(self.e)
        for k, v in other.e.items():
            d[k] = d.get(k, 0) + v
        return Dims(d)

    def __truediv__(self, other):
        d = dict(self.e)
        for k, v in other.e.items():
            d[k] = d.get(k, 0) - v
        return Dims(d)

    def __pow__(self, n):
        return Dims({k: v * n for k, v in self.e.items()})

    def __eq__(self, other):
        return isinstance(other, Dims) and self.e == other.e

    def __hash__(self):
        return hash(tuple(sorted(self.e.items())))

    def __repr__(self):
        if not self.e:
            return "dimensionless"
        num = " ".join(f"{k}^{v}" for k, v in sorted(self.e.items()) if v > 0)
        den = " ".join(f"{k}^{-v}" for k, v in sorted(self.e.items()) if v < 0)
        return (num or "1") + (f" / {den}" if den else "")

    def is_dimensionless(self):
        return not self.e


def _split_factor(token):
    """'um' -> ('um', 1); 'um^2' -> ('um', 2); '1' -> ('1', 1)."""
    if "^" in token:
        name, _, exp = token.partition("^")
        try:
            return name, int(exp)
        except ValueError:
            raise UnitError(f"bad exponent in unit {token!r}")
    return token, 1


def unit_of(text):
    """Parse a unit string -> (scale_to_SI, Dims).  Raises UnitError."""
    if text is None:
        return 1.0, Dims()
    text = text.strip()
    if text in _UNITS:
        f, d = _UNITS[text]
        return f, Dims(d)
    scale, dims = 1.0, Dims()
    # normalise 'a/b/c' and 'a*b'
    expr = text.replace(" ", "")
    num, _, den = expr.partition("/")
    for part, sign in ((num, 1), (den, -1)):
        if not part:
            continue
        for token in part.split("*"):
            if not token:
                continue
            name, exp = _split_factor(token)
            if name not in _UNITS:
                raise UnitError(
                    f"unknown unit {name!r} in {text!r}; "
                    f"known: {', '.join(sorted(_UNITS))}")
            f, d = _UNITS[name]
            scale *= f ** (sign * exp)
            dims = dims * (Dims(d) ** (sign * exp))
    return scale, dims


class Quantity:
    """A value with units.  Arithmetic keeps dimension bookkeeping exact."""

    __slots__ = ("value", "text", "_scale", "_dims")

    def __init__(self, value, unit="1"):
        self.value = float(value)
        self.text = unit
        self._scale, self._dims = unit_of(unit)

    # ---- introspection ----
    @property
    def dims(self):
        return self._dims

    @property
    def si(self):
        """Value in SI base units."""
        return self.value * self._scale

    def is_dimensionless(self):
        return self._dims.is_dimensionless()

    def __repr__(self):
        return f"{self.value:g} {self.text}"

    # ---- construction from SI ----
    @classmethod
    def from_si(cls, si_value, unit):
        s, _ = unit_of(unit)
        return cls(si_value / s, unit)

    # ---- conversion ----
    def to(self, unit):
        s, d = unit_of(unit)
        if d != self._dims:
            raise UnitError(
                f"cannot convert {self.text!r} ({self._dims}) to {unit!r} ({d})")
        return Quantity(self.si / s, unit)

    def in_unit(self, unit):
        return self.to(unit).value

    # ---- arithmetic ----
    def _coerce(self, other):
        if isinstance(other, Quantity):
            return other
        if isinstance(other, (int, float)):
            return Quantity(other, "1")
        raise TypeError(f"cannot combine Quantity with {type(other).__name__}")

    def __mul__(self, other):
        o = self._coerce(other)
        q = Quantity(self.value * o.value, "1")
        q._scale = self._scale * o._scale
        q._dims = self._dims * o._dims
        q.text = f"({self.text}*{o.text})"
        return q

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._coerce(other)
        q = Quantity(self.value / o.value, "1")
        q._scale = self._scale / o._scale
        q._dims = self._dims / o._dims
        q.text = f"({self.text}/{o.text})"
        return q

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, n):
        q = Quantity(self.value ** n, "1")
        q._scale = self._scale ** n
        q._dims = self._dims ** n
        q.text = f"({self.text})^{n}"
        return q

    def _add_sub(self, other, sign):
        o = self._coerce(other)
        if o._dims != self._dims:
            raise UnitError(
                f"cannot {'add' if sign > 0 else 'subtract'} {self!r} and {o!r}: "
                f"{self._dims} vs {o._dims}")
        si = self.si + sign * o.si
        return Quantity.from_si(si, self.text)

    def __add__(self, other):
        return self._add_sub(other, +1)

    def __radd__(self, other):
        return self._add_sub(other, +1)

    def __sub__(self, other):
        return self._add_sub(other, -1)

    def __neg__(self):
        return Quantity(-self.value, self.text)

    def __eq__(self, other):
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return self._dims == o._dims and math.isclose(self.si, o.si, rel_tol=1e-12)

    def __hash__(self):
        return hash((round(self.si, 15), self._dims))


def Q(value, unit="1"):
    """Shorthand constructor."""
    return Quantity(value, unit)


def convert(value, frm, to):
    """Convert a bare number between compatible units.  Raises on mismatch."""
    return Quantity(value, frm).in_unit(to)
