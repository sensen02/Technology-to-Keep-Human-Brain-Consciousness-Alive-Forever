"""Declarative, parameterised scene layer for the embodied fly body.

WHAT THIS MODULE IS
-------------------
``body_backend.py`` pins the fly.  This module pins *the world the fly stands
in*: contact friction per geom class, micro-relief of the ground, illumination,
scene objects for the visual pathway, and a capillary/secretion adhesion term.
Everything is described by one frozen dataclass (:class:`SceneConfig`) so that a
scene can be named, swept, reproduced and reported rather than assembled
ad hoc at each call site.

WHO TALKS TO MUJOCO
-------------------
MuJoCo is imported *inside* functions, never at module import time.  The neural
environment (``venv``) has no MuJoCo, so ``engine.embodied.scene`` must be
importable there: only the ledger, the dataclasses, the provenance rules and
:func:`sensitivity_sweep` (with an injected evaluator) are usable without it.

VERIFIED FACTS ABOUT THE SCENE (measured in this session, not assumed)
---------------------------------------------------------------------
Measured on the installed stack (FlyGym 2.1.0 / MuJoCo 3.9.0, model units mm):

* Today's degenerate scene is ``FlatGroundWorld`` + ``make_locomotion_fly``:
  ``ngeom == 70`` (1 ground plane + 69 fly mesh geoms), ``nlight == 0``,
  ``nhfield == 0``, ``npair == 55``, ``nu == 48``, ``nsensor == 6``,
  ``ncam == 0``.
* Every geom (plane and fly alike) reports ``model.geom_friction``
  ``[1.0, 0.005, 0.0001]`` and ``priority == 0``: one uniform row, no spatial
  or class variation.  The ground plane is at ``z = 0``, ``size = [1000, 1000, 1]``.
* **The ground plane and all 69 fly geoms have ``contype == 0`` and
  ``conaffinity == 0``.**  MuJoCo's automatic broadphase therefore generates no
  fly-ground contacts whatsoever.  Every fly-ground contact in this model goes
  through the 55 explicit MJCF ``<pair>`` elements that
  ``_GroundContactMixin._set_ground_contact`` writes, whose friction FlyGym
  takes from ``ContactParams`` (default ``[1, 1, 0.02, 1e-4, 1e-4]`` as a
  5-tuple).  **This is why the friction that actually governs the physics is the
  pair friction, not ``geom_friction``.**
* Which mechanism MuJoCo 3.9 honours for those pairs was tested directly:
  with all 55 pairs' friction set to zero the *measured* contact friction
  (``data.contact[i].friction``) became ``1e-5`` — the ``mjMINMU`` clamp of the
  value it was given — and **not** the row implied by the geom
  ``priority`` rule (tarsus priority 2 with slide 1.6 vs ground priority 0).
  Conclusion, and the mechanism this module uses: **an explicit pair's friction
  is used verbatim with no fallback to the geom-priority / elementwise-max rule.**
  Per-geom ``friction`` and ``priority`` are therefore also set (they make the
  classes distinguishable in ``geom_friction`` and govern any contact that is
  *not* pair-governed), but they are *not* what the fly's feet feel here.
  This is stated in every report's ``warnings``.
* Heightfields work: ``spec.add_hfield(size=[rx, ry, elevation_z, base_z], ...)``
  plus a ``mjGEOM_HFIELD`` geom that references it, with ``base_z`` **strictly
  positive** (MuJoCo 3.9 rejects ``base_z <= 0`` with "size parameter is not
  positive in hfield").  Fly-ground pairs can be deleted and re-added against the
  heightfield geom name, and the 6 ground-contact sensors' ``refname`` can be
  retargeted to it, so ``Simulation.get_ground_contact_info`` keeps working.
  Verified: after retargeting, contacts are reported between
  ``roughness_hfield_surface`` and ``nmf/*_tarsus5``.
* ``MjsBody`` exposes ``add_geom``/``add_light``/``add_site``/``add_camera`` but
  NOT a generic ``add()``; ``MjSpec`` has NO ``add_light`` (lights are added to a
  body).  ``MjSpec.add_actuator`` called by hand needs ``trntype=mjTRN_BODY`` for
  body-targeted actuators or compilation fails with "invalid transmission type";
  FlyGym's ``flygym.utils.mjcf.add_actuator`` sets this correctly.  MuJoCo names
  containing ``/`` are legal (FlyGym itself uses ``nmf/...``), but the elements
  added here avoid ``/`` so their names are unambiguous.
* The compiled model's total ``body_mass`` sums to ``1.0243e-3 kg`` and gravity
  is ``-9810 mm/s^2``, so the model's weight is ``10.05 mN``.  A real
  *Drosophila* is about three orders of magnitude lighter; this model is not
  mass-accurate to a real fly, which is recorded here because the capillary
  magnitude below is quoted as a fraction of the *model's* weight.

HONESTY BOUNDARY ON NUMBERS
---------------------------
No physical constant in this module is marked ``MEASURED_CITED``.  That label is
machine-enforced (:class:`ParameterRecord`) to require a non-empty source string
carrying an author/year/venue or a URL; **no source was verified in this
session, so no record carries the label.**  The friction, roughness and
capillary values are ``ASSUMED`` or ``ENGINEERING_DEFAULT``, each with an
explicit ``sweep_range``, and they encode *directions of change* (tarsal pads
grip more than cuticle; rough ground perturbs more than a plane) rather than
measured material properties.  Do not quote them as literature values.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    # configuration
    "SceneConfig", "RoughnessConfig", "CapillaryConfig", "LightSpec", "ObjectSpec",
    "SCENE_PRESETS",
    # applying
    "apply_scene", "SceneApplicationReport", "measured_after_compile",
    "drive_capillary_actuators", "capillary_actuator_names",
    "measure_ground_contact_forces",
    # provenance
    "ParameterRecord", "ParameterLedger", "PARAMETER_LEDGER",
    "scene_provenance_report", "UncitedMeasuredClaimError", "ProvenanceError",
    "PROVENANCE_CLASSES", "MEASURED_CITED", "MEASURED_LOCAL", "ASSUMED",
    "ENGINEERING_DEFAULT",
    # sweeps
    "sensitivity_sweep", "set_scene_parameter", "get_scene_parameter",
    # notes
    "FRICTION_MECHANISM_NOTE", "HFIELD_BASE_OFFSET_MIN_MM",
    "ROUGHNESS_GEOM_PREFIX", "OBJECT_GEOM_PREFIX", "CAPILLARY_ACTUATOR_PREFIX",
    "MODEL_UNITS_NOTE",
]

# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

MEASURED_CITED = "MEASURED_CITED"
MEASURED_LOCAL = "MEASURED_LOCAL"
ASSUMED = "ASSUMED"
ENGINEERING_DEFAULT = "ENGINEERING_DEFAULT"

PROVENANCE_CLASSES: tuple[str, ...] = (
    MEASURED_CITED, MEASURED_LOCAL, ASSUMED, ENGINEERING_DEFAULT)

#: Provenance classes that are the author's choice rather than a measurement,
#: and which therefore MUST carry a sweep_range (see module docstring).
_SWEEP_REQUIRED = (ASSUMED, ENGINEERING_DEFAULT)

MODEL_UNITS_NOTE = (
    "model units are millimetres, seconds, radians and kilograms "
    "(mujoco_globals.yaml: gravity [0,0,-9810] mm/s^2); the force unit is "
    "therefore kg*mm/s^2 = 1e-3 N = mN")

_YEAR_RE = re.compile(r"(?:18|19|20)\d{2}")


class ProvenanceError(ValueError):
    """A parameter record violates the provenance contract."""


class UncitedMeasuredClaimError(ProvenanceError):
    """A record claims MEASURED_CITED without a usable citation.

    This is the dedicated exception demanded by the project's provenance rule:
    a value may only be labelled as measured-and-cited if the record carries a
    non-empty ``source`` string containing an author/year/venue or a URL.
    """


def _has_usable_citation(source: str) -> bool:
    """Mechanical gate for a 'citation'.  It cannot detect fabrication.

    Accepts a URL, or any string of reasonable length containing a 4-digit year.
    It is deliberately documented as a *structural* check: it stops an unlabelled
    claim, it does not make a wrong citation right.
    """
    text = (source or "").strip()
    if len(text) < 12:
        return False
    if "http://" in text or "https://" in text:
        return True
    return bool(_YEAR_RE.search(text))


@dataclass(frozen=True)
class ParameterRecord:
    """One number used by the scene layer, with where it came from.

    ``sweep_range`` is a 2-tuple ``(low, high)`` for a continuous parameter or a
    tuple of candidate values for a discrete one.  It is REQUIRED for
    ``ASSUMED`` and ``ENGINEERING_DEFAULT`` records, because those are the
    classes where the value is a choice that the experiment must be able to move.
    """
    name: str
    value: Any
    unit: str
    provenance: str
    source: str = ""
    sweep_range: tuple | None = None

    def __post_init__(self):
        if self.provenance not in PROVENANCE_CLASSES:
            raise ProvenanceError(
                f"{self.name!r}: provenance {self.provenance!r} is not one of "
                f"{PROVENANCE_CLASSES}")
        if not str(self.name).strip():
            raise ProvenanceError("parameter name must be non-empty")
        if self.provenance == MEASURED_CITED and not _has_usable_citation(self.source):
            raise UncitedMeasuredClaimError(
                f"{self.name!r} claims {MEASURED_CITED} but its source is not a "
                f"usable citation (need author/year/venue or a URL): {self.source!r}")
        if self.provenance in _SWEEP_REQUIRED:
            rng = self.sweep_range
            if rng is None or len(tuple(rng)) < 2:
                raise ProvenanceError(
                    f"{self.name!r}: provenance {self.provenance} requires a "
                    f"sweep_range with at least two entries, got {rng!r}")
        if self.provenance in (MEASURED_CITED, MEASURED_LOCAL) and not str(self.source).strip():
            raise ProvenanceError(
                f"{self.name!r}: a {self.provenance} record must say where it was measured")

    def as_dict(self) -> dict:
        return {"name": self.name, "value": _jsonable(self.value), "unit": self.unit,
                "provenance": self.provenance, "source": self.source,
                "sweep_range": _jsonable(self.sweep_range)}


class ParameterLedger:
    """An append-only bag of :class:`ParameterRecord` objects, keyed by name."""

    def __init__(self, records: Iterable[ParameterRecord] = ()):
        self._records: list[ParameterRecord] = []
        self._by_name: dict[str, ParameterRecord] = {}
        for r in records:
            self.add(r)

    # -- writing ---------------------------------------------------------
    def add(self, record: ParameterRecord) -> ParameterRecord:
        if not isinstance(record, ParameterRecord):
            raise TypeError(f"expected ParameterRecord, got {type(record).__name__}")
        if record.name in self._by_name:
            raise ProvenanceError(f"duplicate ledger entry {record.name!r}")
        self._records.append(record)
        self._by_name[record.name] = record
        return record

    def record(self, name: str, value: Any, unit: str, provenance: str,
               source: str = "", sweep_range: tuple | None = None) -> ParameterRecord:
        return self.add(ParameterRecord(name=name, value=value, unit=unit,
                                        provenance=provenance, source=source,
                                        sweep_range=sweep_range))

    def record_vector(self, name: str, values: Sequence[float], unit: str,
                      provenance: str, source: str = "",
                      sweep_range: tuple | None = None) -> list[ParameterRecord]:
        out = []
        for i, v in enumerate(values):
            rng = sweep_range
            if rng is not None and provenance in _SWEEP_REQUIRED:
                lo, hi = float(rng[0]), float(rng[1])
                half = max(abs(float(v)), 1e-12) * 0.5
                rng = (min(lo, float(v) - half), max(hi, float(v) + half))
            out.append(self.record(f"{name}.{i}", float(v), unit, provenance,
                                   source=source, sweep_range=rng))
        return out

    # -- reading ---------------------------------------------------------
    def get(self, name: str) -> ParameterRecord:
        return self._by_name[name]

    def names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def counts(self) -> dict[str, int]:
        out = {p: 0 for p in PROVENANCE_CLASSES}
        for r in self._records:
            out[r.provenance] += 1
        return out

    def as_dicts(self) -> list[dict]:
        return [r.as_dict() for r in self._records]


# --------------------------------------------------------------------------- #
# element naming (kept slash-free so the names are unambiguous)
# --------------------------------------------------------------------------- #

ROUGHNESS_GEOM_PREFIX = "roughness_"
OBJECT_GEOM_PREFIX = "sceneobj_"
CAPILLARY_ACTUATOR_PREFIX = "capillary_"
HFIELD_ASSET_NAME = "roughness_hfield"
HFIELD_GEOM_NAME = ROUGHNESS_GEOM_PREFIX + "hfield_surface"
#: MuJoCo 3.9 rejects an hfield whose base_z is <= 0 ("size parameter is not
#: positive in hfield").  Verified in this session.
HFIELD_BASE_OFFSET_MIN_MM = 0.001

FRICTION_MECHANISM_NOTE = (
    "FlyGym routes every fly-ground contact through explicit MJCF <pair> elements: "
    "FlatGroundWorld sets the ground plane contype=conaffinity=0 and all 69 fly geoms "
    "are contype=conaffinity=0 as well, so MuJoCo's automatic broadphase produces no "
    "fly-ground contact at all (measured: ngeom 70, npair 55, all contacts come from "
    "pairs).  For a pair-governed contact MuJoCo 3.9 uses the pair's 5-tuple friction "
    "VERBATIM: verified by zeroing all 55 pair frictions, which produced a measured "
    "contact friction of 1e-5 (the mjMINMU clamp) rather than the geom-priority row "
    "(tarsus priority 2, slide 1.6, vs ground priority 0).  So the per-class friction "
    "that the feet actually feel is the per-class PAIR friction written here; per-geom "
    "friction + priority are set as well (tarsus 2 / body 1 / ground 0) so the classes "
    "are distinguishable in model.geom_friction and so any non-pair contact path "
    "resolves per class, but they do not govern the fly-ground contact in this model.")


# --------------------------------------------------------------------------- #
# scene description
# --------------------------------------------------------------------------- #

LEG_ORDER: tuple[str, ...] = ("lf", "lm", "lh", "rf", "rm", "rh")


@dataclass(frozen=True)
class RoughnessConfig:
    """Micro-relief of the ground.

    ``kind``:
      * ``"none"``       -- perfect plane (the control).
      * ``"heightfield"``-- one MuJoCo ``hfield`` asset + ``mjGEOM_HFIELD`` geom,
                            which then *becomes* the ground geoms used for the
                            fly-ground contact pairs and the contact sensors.
      * ``"scattered"``  -- fallback: ``n_scatter`` small spheres placed as geoms
                            and paired against the six tarsal-tip geoms.  Used if
                            a heightfield ever proves impossible; it is a
                            different contact topology and says so in the report.
    """
    kind: str = "none"
    amplitude_mm: float = 0.0
    wavelength_mm: float = 1.0
    seed: int = 0
    n_rows: int = 1
    n_cols: int = 1
    # -- extras (all optional) -------------------------------------------
    half_extent_mm: float = 30.0
    base_offset_mm: float = HFIELD_BASE_OFFSET_MIN_MM
    n_scatter: int = 48
    scatter_radius_mm: float = 0.03
    scatter_half_extent_mm: float = 15.0
    #: Cleared disc patches for the grass terrain, as ``(x_mm, y_mm, radius_mm)`` or
    #: ``(x_mm, y_mm, radius_mm, rim_mm)``.  Flattened to exactly zero so objects on them
    #: have an unobstructed line of sight to every camera.  Meaningless for other kinds.
    grass_clearings_mm: tuple = ()
    #: Material to paint the heightfield with.  FlyGym's FlatGroundWorld already
    #: defines a checker material called "grid" for its plane; reusing it keeps the
    #: rendered ground comparable between the control and the rough scene.  If the
    #: material is absent from the spec the heightfield is added without one (and
    #: the report says so) rather than failing.
    hfield_material: str | None = "grid"

    def __post_init__(self):
        # "grass" is a heightfield whose GRID comes from a different generator (a stand
        # of blades, see grass_scene.py).  It is a separate kind rather than an option on
        # "heightfield" because everything downstream that asks "is this rough?" has to
        # treat it identically anyway, while the two look very different in a report.
        if self.kind not in ("none", "heightfield", "grass", "scattered"):
            raise ValueError(f"roughness kind {self.kind!r} not in "
                             f"('none','heightfield','grass','scattered')")
        for f in ("amplitude_mm", "wavelength_mm", "half_extent_mm", "base_offset_mm",
                  "scatter_radius_mm", "scatter_half_extent_mm"):
            v = float(getattr(self, f))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"roughness.{f} must be finite and >= 0, got {v}")
        if self.kind != "none":
            if self.amplitude_mm <= 0:
                raise ValueError("roughness.amplitude_mm must be > 0 when roughness is on")
            if self.wavelength_mm <= 0:
                raise ValueError("roughness.wavelength_mm must be > 0")
            if self.half_extent_mm <= 0:
                raise ValueError("roughness.half_extent_mm must be > 0")
        if self.kind in ("heightfield", "grass"):
            if int(self.n_rows) < 2 or int(self.n_cols) < 2:
                raise ValueError("heightfield needs n_rows >= 2 and n_cols >= 2")
            if self.base_offset_mm <= 0:
                raise ValueError(
                    "MuJoCo 3.9 requires the hfield base_z to be strictly positive "
                    "(measured: base_z=0 raises 'size parameter is not positive in hfield')")
            # Resolution check: the grid must be able to represent the wavelength.
            # For the grass kind the "wavelength" IS the blade spacing, so this check
            # is exactly the statement that the grid resolves a blade.
            cell = 2.0 * self.half_extent_mm / max(int(self.n_rows), int(self.n_cols))
            if self.wavelength_mm < 4.0 * cell:
                raise ValueError(
                    f"heightfield grid cell {cell:.4f} mm cannot resolve wavelength "
                    f"{self.wavelength_mm} mm (need >= 4 cells per wavelength)")

    @property
    def grid_cell_mm(self) -> float:
        if self.kind not in ("heightfield", "grass"):
            return 0.0
        return 2.0 * self.half_extent_mm / max(int(self.n_rows), int(self.n_cols))


@dataclass(frozen=True)
class CapillaryConfig:
    """Tarsal secretion / capillary adhesion term.

    ``humidity`` is recorded as METADATA ONLY and does not enter the physics: no
    verified humidity->force law was found, and inventing a linear one would be a
    fabricated physical model.  The report carries a warning to that effect.
    """
    enabled: bool = False
    secretion_force_mN: float = 0.0
    humidity: float = 0.0
    applies_to: tuple[str, ...] = LEG_ORDER
    provenance: str = ASSUMED

    def __post_init__(self):
        if self.provenance not in PROVENANCE_CLASSES:
            raise ValueError(f"capillary.provenance {self.provenance!r} is not a "
                             f"provenance class")
        if not math.isfinite(float(self.secretion_force_mN)) or self.secretion_force_mN < 0:
            raise ValueError("capillary.secretion_force_mN must be finite and >= 0")
        if not 0.0 <= float(self.humidity) <= 1.0:
            raise ValueError("capillary.humidity must be in [0, 1] (relative humidity)")
        unknown = [leg for leg in self.applies_to if leg not in LEG_ORDER]
        if unknown:
            raise ValueError(f"capillary.applies_to contains unknown legs {unknown}")
        if self.enabled and self.secretion_force_mN <= 0:
            raise ValueError("capillary is enabled but secretion_force_mN is 0")


@dataclass(frozen=True)
class LightSpec:
    """One MuJoCo light.  ``kind`` is directional/point/spot; ``mode`` is a
    mjtCamLight name (fixed/track/trackcom/targetbody/targetbodycom)."""
    kind: str = "directional"
    pos_mm: tuple[float, float, float] = (0.0, 0.0, 200.0)
    dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    diffuse: tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular: tuple[float, float, float] = (0.2, 0.2, 0.2)
    ambient: tuple[float, float, float] = (0.1, 0.1, 0.1)
    castshadow: bool = True
    name: str = "light"
    mode: str = "fixed"

    def __post_init__(self):
        if self.kind not in ("directional", "point", "spot"):
            raise ValueError(f"light kind {self.kind!r} not in "
                             f"('directional','point','spot')")
        if self.mode not in ("fixed", "track", "trackcom", "targetbody",
                             "targetbodycom"):
            raise ValueError(f"light mode {self.mode!r} is not a MuJoCo camera/light mode")
        for f in ("pos_mm", "dir", "diffuse", "specular", "ambient"):
            v = tuple(float(x) for x in getattr(self, f))
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError(f"light.{f} must be three finite numbers, got {v!r}")
        if self.kind == "directional" and all(v == 0.0 for v in self.dir):
            raise ValueError("a directional light needs a non-zero dir")


@dataclass(frozen=True)
class ObjectSpec:
    """One scene object for the visual pathway.

    ``size_mm`` follows MuJoCo's own geom-size semantics for the chosen
    ``geom_type``: half-extents for a box, (radius, 0, 0) for a sphere,
    (radius, half-length) for a capsule/cylinder.  ``collidable`` defaults to
    False, which sets contype=conaffinity=0: the object is then purely visual and
    the ``natural`` vs ``natural_lit_no_objects`` comparison isolates the visual
    contribution exactly.
    """
    name: str = "object"
    kind: str = "target"
    pos_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    geom_type: str = "box"
    collidable: bool = False

    def __post_init__(self):
        if self.geom_type not in ("box", "sphere", "cylinder", "capsule", "ellipsoid"):
            raise ValueError(f"object geom_type {self.geom_type!r} is not supported")
        for f in ("pos_mm", "size_mm"):
            v = tuple(float(x) for x in getattr(self, f))
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError(f"object.{f} must be three finite numbers, got {v!r}")
        if len(tuple(self.rgba)) != 4:
            raise ValueError("object.rgba must have four entries")
        if not all(0.0 <= float(c) <= 1.0 for c in self.rgba):
            raise ValueError("object.rgba entries must be in [0, 1]")
        if float(self.size_mm[0]) <= 0:
            raise ValueError("object.size_mm[0] must be > 0")


@dataclass(frozen=True)
class SceneConfig:
    """A complete, reproducible description of the world the fly stands in."""
    name: str
    friction_by_class: Mapping[str, tuple[float, float, float]]
    roughness: "RoughnessConfig | None" = None
    capillary: "CapillaryConfig | None" = None
    adhesion_gain: float = 40.0
    adhesion_added: bool = True
    lights: tuple[LightSpec, ...] = ()
    objects: tuple[ObjectSpec, ...] = ()
    # -- extras ----------------------------------------------------------
    description: str = ""
    world_half_size_mm: float = 1000.0
    seed: int = 0
    geom_priority_by_class: Mapping[str, int] = field(
        default_factory=lambda: {"ground": 0, "body": 1, "tarsus": 2})
    objects_collidable: bool = False
    #: When True, ``apply_scene`` writes the per-geom friction/priority rows but
    #: leaves the explicit fly-ground pairs exactly as FlyGym wrote them.  This is
    #: what makes the ``blank`` control a faithful reproduction of the existing
    #: scene: today's pairs carry (1, 1, 0.02, 1e-4, 1e-4), whose torsional term
    #: (0.02) is *not* the geom row's (0.005), and it is the pair row MuJoCo uses.
    #: Only legal when every class asks for the same row.
    leave_pairs_untouched: bool = False

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("SceneConfig.name must be non-empty")
        missing = {"ground", "tarsus", "body"} - set(self.friction_by_class)
        if missing:
            raise ValueError(f"friction_by_class is missing {sorted(missing)}")
        for cls, row in self.friction_by_class.items():
            if len(tuple(row)) != 3:
                raise ValueError(f"friction_by_class[{cls!r}] must be "
                                 f"(sliding, torsional, rolling)")
            for v in row:
                if not math.isfinite(float(v)) or float(v) < 0:
                    raise ValueError(f"friction_by_class[{cls!r}] has a bad entry: {row!r}")
        if float(self.friction_by_class["tarsus"][0]) <= 0:
            raise ValueError("tarsus sliding friction must be > 0")
        if not math.isfinite(float(self.adhesion_gain)) or self.adhesion_gain < 0:
            raise ValueError("adhesion_gain must be finite and >= 0")
        if not math.isfinite(float(self.world_half_size_mm)) or \
                self.world_half_size_mm <= 0:
            raise ValueError("world_half_size_mm must be finite and > 0")
        for cls, prio in self.geom_priority_by_class.items():
            if cls not in ("ground", "tarsus", "body"):
                raise ValueError(f"geom_priority_by_class has unknown class {cls!r}")
            if int(prio) < 0:
                raise ValueError("geom priorities must be >= 0")
        if self.leave_pairs_untouched:
            rows = {cls: tuple(float(v) for v in self.friction_by_class[cls])
                    for cls in ("ground", "tarsus", "body")}
            if len(set(rows.values())) != 1:
                raise ValueError(
                    "leave_pairs_untouched=True is only legal when every class asks "
                    "for the SAME friction row; otherwise it would silently suppress "
                    f"the per-class pair friction this module exists to provide: {rows}")
            if set(self.geom_priority_by_class.values()) != {0}:
                raise ValueError("leave_pairs_untouched=True must come with priority 0 "
                                 "for every class, so the control scene is unchanged")

    def tarsus_friction(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.friction_by_class["tarsus"])  # type: ignore

    def body_friction(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.friction_by_class["body"])  # type: ignore

    def ground_friction(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.friction_by_class["ground"])  # type: ignore


# --------------------------------------------------------------------------- #
# THE NUMBERS, AND WHERE THEY CAME FROM
# --------------------------------------------------------------------------- #

#: The friction row every geom in today's scene carries.  MEASURED_LOCAL: read
#: back from ``model.geom_friction`` of the compiled control scene.
BASELINE_GEOM_FRICTION = (1.0, 0.005, 0.0001)
#: The 5-tuple today's 55 fly-ground pairs carry.  MEASURED_LOCAL: read back from
#: ``model.pair_friction``; it is FlyGym's own ``ContactParams`` default
#: (sliding 1.0, torsional 0.02, rolling 1e-4).
BASELINE_PAIR_FRICTION = (1.0, 1.0, 0.02, 1e-4, 1e-4)

#: ASSUMED (NOT cited).  Direction of the assumption: a tarsal pad with claws and
#: adhesive secretion grips a rough surface appreciably better than the model's
#: default, and fly body cuticle slides far more easily.
FRICTION_TARSUS = (1.6, 0.03, 0.0002)
FRICTION_BODY = (0.35, 0.008, 0.00005)

#: ENGINEERING_DEFAULT: the ground keeps MuJoCo's own default row (as today).
FRICTION_GROUND = (1.0, 0.005, 0.0001)

ROUGHNESS_AMPLITUDE_MM = 0.05
ROUGHNESS_WAVELENGTH_MM = 2.0
ROUGHNESS_N_ROWS = 129
ROUGHNESS_N_COLS = 129
ROUGHNESS_HALF_EXTENT_MM = 30.0
ROUGHNESS_SEED = 0
ROUGHNESS_SCATTER_N = 48
ROUGHNESS_SCATTER_RADIUS_MM = 0.03
ROUGHNESS_SCATTER_HALF_EXTENT_MM = 15.0

#: ASSUMED (NOT cited).  Magnitude as a fraction of the MODEL's own weight
#: (measured 10.05 mN): 0.05 mN per leg x 6 legs = 3% of body weight.  A real
#: Drosophila is ~1000x lighter than this model, so this is a placeholder whose
#: only defensible content is "small compared with body weight"; hence ASSUMED
#: plus a sweep range rather than a citation.
CAPILLARY_FORCE_MN = 0.05
CAPILLARY_HUMIDITY = 0.6

ADHESION_GAIN = 40.0

SUN_LIGHT = LightSpec(
    name="sun", kind="directional", pos_mm=(0.0, 0.0, 200.0),
    dir=(-0.35, -0.25, -1.0), diffuse=(0.85, 0.85, 0.80),
    specular=(0.30, 0.30, 0.30), ambient=(0.22, 0.22, 0.22),
    castshadow=True, mode="fixed")
FILL_LIGHT = LightSpec(
    name="fill", kind="directional", pos_mm=(0.0, 0.0, 150.0),
    dir=(0.45, 0.30, -1.0), diffuse=(0.22, 0.25, 0.32),
    specular=(0.0, 0.0, 0.0), ambient=(0.06, 0.06, 0.08),
    castshadow=False, mode="fixed")
AMBIENT_LIGHT = LightSpec(
    name="sky_ambient", kind="directional", pos_mm=(0.0, 0.0, 100.0),
    dir=(0.0, 0.0, -1.0), diffuse=(0.0, 0.0, 0.0),
    specular=(0.0, 0.0, 0.0), ambient=(0.12, 0.12, 0.14),
    castshadow=False, mode="fixed")

OBJECT_LANDMARK_DARK = ObjectSpec(
    name="landmark_dark", kind="target", pos_mm=(6.0, 0.0, 0.75),
    size_mm=(0.75, 0.75, 0.75), rgba=(0.04, 0.04, 0.05, 1.0), geom_type="box")
OBJECT_LANDMARK_LIGHT = ObjectSpec(
    name="landmark_light", kind="target", pos_mm=(-6.0, 2.5, 0.75),
    size_mm=(0.75, 0.75, 0.75), rgba=(0.95, 0.95, 0.90, 1.0), geom_type="box")
OBJECT_PEBBLE = ObjectSpec(
    name="pebble_contrast", kind="texture", pos_mm=(3.0, 3.0, 0.20),
    size_mm=(0.20, 0.0, 0.0), rgba=(0.75, 0.25, 0.05, 1.0), geom_type="sphere")


def _build_parameter_ledger() -> ParameterLedger:
    led = ParameterLedger()
    measured = "read back from the compiled FlyGym 2.1.0 / MuJoCo 3.9.0 control scene in this session"

    # -- MEASURED_LOCAL: facts about the machine, not choices ------------
    led.record_vector("baseline.geom_friction_row", BASELINE_GEOM_FRICTION,
                      "dimensionless (sliding, torsional, rolling)",
                      MEASURED_LOCAL, source="model.geom_friction, " + measured)
    led.record_vector("baseline.pair_friction_row", BASELINE_PAIR_FRICTION,
                      "dimensionless (2x sliding, torsional, 2x rolling)",
                      MEASURED_LOCAL, source="model.pair_friction, " + measured)
    led.record("baseline.n_geom", 70, "count", MEASURED_LOCAL,
               source="model.ngeom, " + measured)
    led.record("baseline.n_fly_geom", 69, "count", MEASURED_LOCAL,
               source="model.ngeom - 1 ground plane, " + measured)
    led.record("baseline.n_pair", 55, "count", MEASURED_LOCAL,
               source="model.npair, " + measured)
    led.record("baseline.n_light", 0, "count", MEASURED_LOCAL,
               source="model.nlight, " + measured)
    led.record("baseline.n_hfield", 0, "count", MEASURED_LOCAL,
               source="model.nhfield, " + measured)
    led.record("baseline.n_camera", 0, "count", MEASURED_LOCAL,
               source="model.ncam for make_locomotion_fly, " + measured)
    led.record("baseline.n_ground_geom", 1, "count", MEASURED_LOCAL,
               source="len(world.ground_geoms), " + measured)
    led.record("baseline.geom_contype", 0, "bitmask", MEASURED_LOCAL,
               source="model.geom_contype (ground plane and all fly geoms), " + measured)
    led.record("model.gravity", -9810.0, "mm/s^2", MEASURED_LOCAL,
               source="model.opt.gravity and mujoco_globals.yaml, " + measured)
    led.record("model.body_mass_total", 1.0243e-3, "kg", MEASURED_LOCAL,
               source="sum(model.body_mass), " + measured)
    led.record("model.weight_total", 10.048, "mN", MEASURED_LOCAL,
               source="measured body_mass_total * |model.opt.gravity|, " + measured)
    led.record("capillary.force_fraction_of_model_weight", 0.0299, "dimensionless",
               MEASURED_LOCAL,
               source="6 * secretion_force_mN / model.weight_total, this session")

    # -- ASSUMED: the author's choices where no number was verified ------
    led.record_vector("natural.friction.tarsus", FRICTION_TARSUS,
                      "dimensionless (sliding, torsional, rolling)", ASSUMED,
                      source="hand-chosen direction-of-change, no source verified",
                      sweep_range=(0.8, 2.4))
    led.record_vector("natural.friction.body", FRICTION_BODY,
                      "dimensionless (sliding, torsional, rolling)", ASSUMED,
                      source="hand-chosen direction-of-change, no source verified",
                      sweep_range=(0.1, 0.8))
    led.record("natural.capillary.secretion_force_mN", CAPILLARY_FORCE_MN, "mN per leg",
               ASSUMED,
               source="hand-chosen placeholder, no source verified; see module docstring",
               sweep_range=(0.0, 0.5))
    led.record("natural.capillary.humidity", CAPILLARY_HUMIDITY,
               "relative humidity (metadata only)", ASSUMED,
               source="recorded for provenance only; it does NOT enter the physics",
               sweep_range=(0.2, 0.95))
    led.record("natural.capillary.applies_to", 6, "leg count", ASSUMED,
               source="all six legs; no source verified",
               sweep_range=(1, 6))

    # -- ENGINEERING_DEFAULT: modelling/visual choices ------------------
    for cls in ("ground", "tarsus", "body"):
        led.record_vector(f"blank.friction.{cls}", BASELINE_GEOM_FRICTION,
                          "dimensionless (sliding, torsional, rolling)",
                          ENGINEERING_DEFAULT,
                          source="identical to MuJoCo's own geom default, i.e. to "
                                 "today's scene; the pairs are left untouched",
                          sweep_range=(0.3, 2.0))
        led.record(f"blank.priority.{cls}", 0, "priority (integer)",
                   ENGINEERING_DEFAULT,
                   source="0 everywhere, so the control scene is byte-for-byte the "
                          "same friction as today (leave_pairs_untouched=True)",
                   sweep_range=(0, 4))
    led.record("blank.adhesion_gain", ADHESION_GAIN, "dimensionless actuator gain",
               ENGINEERING_DEFAULT,
               source="make_locomotion_fly default, i.e. today's scene",
               sweep_range=(0.0, 80.0))
    led.record("blank.world_half_size_mm", 1000.0, "mm", ENGINEERING_DEFAULT,
               source="FlatGroundWorld default half_size, i.e. today's scene",
               sweep_range=(20.0, 1000.0))
    led.record_vector("natural.friction.ground", FRICTION_GROUND,
                      "dimensionless (sliding, torsional, rolling)",
                      ENGINEERING_DEFAULT,
                      source="MuJoCo 3.9 geom default, kept as today",
                      sweep_range=(0.3, 2.0))
    for cls, prio in (("ground", 0), ("body", 1), ("tarsus", 2)):
        led.record(f"natural.priority.{cls}", prio, "priority (integer)",
                   ENGINEERING_DEFAULT,
                   source="ordering chosen so the classes are distinguishable in "
                          "model.geom_friction; pairs override it",
                   sweep_range=(0, 4))
    led.record("natural.roughness.amplitude_mm", ROUGHNESS_AMPLITUDE_MM, "mm",
               ENGINEERING_DEFAULT, source="hand-chosen micro-relief scale",
               sweep_range=(0.01, 0.5))
    led.record("natural.roughness.wavelength_mm", ROUGHNESS_WAVELENGTH_MM, "mm",
               ENGINEERING_DEFAULT, source="hand-chosen micro-relief scale",
               sweep_range=(0.5, 8.0))
    led.record("natural.roughness.seed", ROUGHNESS_SEED, "integer",
               ENGINEERING_DEFAULT, source="arbitrary fixed seed for reproducibility",
               sweep_range=(0, 999))
    led.record("natural.roughness.n_rows", ROUGHNESS_N_ROWS, "grid samples",
               ENGINEERING_DEFAULT,
               source="grid resolution chosen to keep >= 4 cells per wavelength",
               sweep_range=(33, 257))
    led.record("natural.roughness.n_cols", ROUGHNESS_N_COLS, "grid samples",
               ENGINEERING_DEFAULT,
               source="grid resolution chosen to keep >= 4 cells per wavelength",
               sweep_range=(33, 257))
    led.record("natural.roughness.half_extent_mm", ROUGHNESS_HALF_EXTENT_MM, "mm",
               ENGINEERING_DEFAULT,
               source="must cover the intended travel distance of a short rollout",
               sweep_range=(10.0, 100.0))
    led.record("natural.roughness.base_offset_mm", HFIELD_BASE_OFFSET_MIN_MM, "mm",
               ENGINEERING_DEFAULT,
               source="smallest base_z MuJoCo 3.9 accepts (it rejects base_z <= 0)",
               sweep_range=(0.001, 0.05))
    led.record("scattered.n_scatter", ROUGHNESS_SCATTER_N, "count",
               ENGINEERING_DEFAULT, source="fallback roughness count",
               sweep_range=(8, 200))
    led.record("scattered.scatter_radius_mm", ROUGHNESS_SCATTER_RADIUS_MM, "mm",
               ENGINEERING_DEFAULT, source="fallback peak size",
               sweep_range=(0.01, 0.2))
    led.record("scattered.scatter_half_extent_mm", ROUGHNESS_SCATTER_HALF_EXTENT_MM,
               "mm", ENGINEERING_DEFAULT, source="fallback scatter extent",
               sweep_range=(2.0, 40.0))
    led.record("natural.adhesion_gain", ADHESION_GAIN, "dimensionless actuator gain",
               ENGINEERING_DEFAULT,
               source="flygym_demo.complex_terrain.common.make_locomotion_fly default, "
                      "kept identical to the control so adhesion is not a confound",
               sweep_range=(0.0, 80.0))
    led.record("natural.world_half_size_mm", 1000.0, "mm", ENGINEERING_DEFAULT,
               source="matches BodyConfig.world_half_size_mm (visual half-extent; "
                      "the plane collides infinitely)",
               sweep_range=(20.0, 1000.0))
    for light in (SUN_LIGHT, FILL_LIGHT, AMBIENT_LIGHT):
        led.record_vector(f"natural.light.{light.name}.pos_mm", light.pos_mm, "mm",
                          ENGINEERING_DEFAULT, source="visual choice, not measured",
                          sweep_range=(-500.0, 500.0))
        led.record_vector(f"natural.light.{light.name}.dir", light.dir,
                          "unit-less direction", ENGINEERING_DEFAULT,
                          source="visual choice, not measured",
                          sweep_range=(-1.0, 1.0))
        led.record_vector(f"natural.light.{light.name}.diffuse", light.diffuse,
                          "rgb in [0,1]", ENGINEERING_DEFAULT,
                          source="visual choice, not measured", sweep_range=(0.0, 1.0))
        led.record_vector(f"natural.light.{light.name}.specular", light.specular,
                          "rgb in [0,1]", ENGINEERING_DEFAULT,
                          source="visual choice, not measured", sweep_range=(0.0, 1.0))
        led.record_vector(f"natural.light.{light.name}.ambient", light.ambient,
                          "rgb in [0,1]", ENGINEERING_DEFAULT,
                          source="visual choice, not measured", sweep_range=(0.0, 0.5))
    for obj in (OBJECT_LANDMARK_DARK, OBJECT_LANDMARK_LIGHT, OBJECT_PEBBLE):
        led.record_vector(f"natural.object.{obj.name}.pos_mm", obj.pos_mm, "mm",
                          ENGINEERING_DEFAULT,
                          source="placed by hand near the spawn point; not measured",
                          sweep_range=(-10.0, 10.0))
        led.record_vector(f"natural.object.{obj.name}.size_mm", obj.size_mm, "mm",
                          ENGINEERING_DEFAULT,
                          source="MuJoCo geom-size semantics for the given geom_type",
                          sweep_range=(0.0, 2.0))
    return led


#: Module-level ledger, actually populated with every number the presets use.
PARAMETER_LEDGER = _build_parameter_ledger()


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #

_BLANK = SceneConfig(
    name="blank",
    friction_by_class={"ground": BASELINE_GEOM_FRICTION,
                       "tarsus": BASELINE_GEOM_FRICTION,
                       "body": BASELINE_GEOM_FRICTION},
    roughness=None,
    capillary=None,
    adhesion_gain=ADHESION_GAIN,
    adhesion_added=True,
    lights=(),
    objects=(),
    description=("Control: reproduces today's degenerate scene as closely as the spec "
                 "layer can -- one uniform friction row on every geom, no roughness, no "
                 "lights, no objects, no capillary term.  FlyGym's own pair friction "
                 "(1, 1, 0.02, 1e-4, 1e-4) is left untouched via leave_pairs_untouched, "
                 "which is exactly what the unmodified backend produces."),
    geom_priority_by_class={"ground": 0, "tarsus": 0, "body": 0},
    leave_pairs_untouched=True,
)

_NATURAL_ROUGHNESS = RoughnessConfig(
    kind="heightfield", amplitude_mm=ROUGHNESS_AMPLITUDE_MM,
    wavelength_mm=ROUGHNESS_WAVELENGTH_MM, seed=ROUGHNESS_SEED,
    n_rows=ROUGHNESS_N_ROWS, n_cols=ROUGHNESS_N_COLS,
    half_extent_mm=ROUGHNESS_HALF_EXTENT_MM,
    base_offset_mm=HFIELD_BASE_OFFSET_MIN_MM,
    n_scatter=ROUGHNESS_SCATTER_N, scatter_radius_mm=ROUGHNESS_SCATTER_RADIUS_MM,
    scatter_half_extent_mm=ROUGHNESS_SCATTER_HALF_EXTENT_MM)

_NATURAL_CAPILLARY = CapillaryConfig(
    enabled=True, secretion_force_mN=CAPILLARY_FORCE_MN,
    humidity=CAPILLARY_HUMIDITY, applies_to=LEG_ORDER, provenance=ASSUMED)

_NATURAL = SceneConfig(
    name="natural",
    friction_by_class={"ground": FRICTION_GROUND, "tarsus": FRICTION_TARSUS,
                       "body": FRICTION_BODY},
    roughness=_NATURAL_ROUGHNESS,
    capillary=_NATURAL_CAPILLARY,
    adhesion_gain=ADHESION_GAIN,
    adhesion_added=True,
    lights=(SUN_LIGHT, FILL_LIGHT, AMBIENT_LIGHT),
    objects=(OBJECT_LANDMARK_DARK, OBJECT_LANDMARK_LIGHT, OBJECT_PEBBLE),
    description=("Realistic-ish scene: per-class friction (tarsal pads grip, cuticle "
                 "slides), a heightfield micro-relief, three lights, three "
                 "high-contrast scene objects, and a capillary/secretion term.  The "
                 "magnitudes are ASSUMED or ENGINEERING_DEFAULT, not cited."),
    geom_priority_by_class={"ground": 0, "tarsus": 2, "body": 1},
)
SCENE_PRESETS: dict[str, SceneConfig] = {
    "blank": _BLANK,
    "natural": _NATURAL,
    # Same as natural but with no scene objects: a second control that isolates
    # the contribution of the objects.  Nothing else differs.
    "natural_lit_no_objects": dataclasses.replace(
        _NATURAL, name="natural_lit_no_objects", objects=(),
        description=("Second control: identical to 'natural' in every field except "
                     "`objects`, which is empty.  Because objects are visual-only "
                     "(contype=conaffinity=0), this isolates the objects' contribution "
                     "to what the fly could see, and changes nothing in the physics.")),
}
assert set(SCENE_PRESETS) == {"blank", "natural", "natural_lit_no_objects"}


def scene_provenance_report() -> dict:
    """Machine-readable dump of the ledger plus a count per provenance class."""
    counts = PARAMETER_LEDGER.counts()
    cited = [r.name for r in PARAMETER_LEDGER if r.provenance == MEASURED_CITED]
    return {
        "ledger": PARAMETER_LEDGER.as_dicts(),
        "counts_by_provenance": counts,
        "n_records": len(PARAMETER_LEDGER),
        "n_measured_cited": counts[MEASURED_CITED],
        "measured_cited_names": cited,
        "n_measured_local": counts[MEASURED_LOCAL],
        "n_assumed": counts[ASSUMED],
        "n_engineering_default": counts[ENGINEERING_DEFAULT],
        "citation_policy": ("a MEASURED_CITED record is rejected with "
                            "UncitedMeasuredClaimError unless `source` carries an "
                            "author/year/venue or a URL; ASSUMED and "
                            "ENGINEERING_DEFAULT records must carry a sweep_range"),
        "citation_honesty_note": (
            "This module contains NO MEASURED_CITED record: no external source was "
            "verified in this session, so no number is presented as a literature "
            "value.  The friction, roughness and capillary magnitudes are the "
            "author's choices, and the module says so."),
        "presets": {k: v.name for k, v in SCENE_PRESETS.items()},
        "model_units_note": MODEL_UNITS_NOTE,
    }


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return repr(obj)
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy arrays / scalars
        return _jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    return repr(obj)


def _import_mujoco():
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise RuntimeError(
            "this function needs MuJoCo: run it with the physics environment "
            "(venv_body/bin/python).  The neural environment (venv) has no MuJoCo "
            "on purpose.") from exc
    return mujoco


# --------------------------------------------------------------------------- #
# spec mutation helpers
# --------------------------------------------------------------------------- #

def _classify_fly_geoms(fly) -> dict[str, list[str]]:
    """Split the fly's geoms into the 'tarsus' and 'body' classes.

    Uses FlyGym's own ``bodyseg_to_mjcfgeom`` mapping rather than guessing from
    names: a geom is a tarsus geom iff its owning body segment's link name
    contains ``tarsus``.
    """
    out: dict[str, list[str]] = {"tarsus": [], "body": []}
    mapping = getattr(fly, "bodyseg_to_mjcfgeom", None)
    if not mapping:
        raise RuntimeError("fly object has no bodyseg_to_mjcfgeom mapping; cannot "
                           "classify geoms into tarsus/body")
    for seg, geoms in mapping.items():
        link = str(getattr(seg, "link", seg)).lower()
        cls = "tarsus" if "tarsus" in link else "body"
        for g in geoms:
            name = str(getattr(g, "name", g))
            if name not in out[cls]:
                out[cls].append(name)
    return out


def _geom_class_of_name(geom_name: str, tarsus_pattern: str = "_tarsus") -> str:
    base = geom_name.split("/")[-1]
    if base.startswith(ROUGHNESS_GEOM_PREFIX):
        return "ground"
    if geom_name == "ground_plane" or base.startswith("ground"):
        return "ground"
    return "tarsus" if tarsus_pattern in base else "body"


def _row3_to_pair5(row: Sequence[float]) -> list[float]:
    """MuJoCo wants 5 friction coefficients on a pair: 2 sliding, 1 torsional,
    2 rolling (documented in FlyGym's ``ContactParams.get_friction_tuple``)."""
    s, t, r = (float(v) for v in row)
    return [s, s, t, r, r]


def _heightfield_data(mj, cfg: "RoughnessConfig", scene_cfg=None):
    """Deterministic normalised elevation grid in [0, 1] with p2p exactly 1.0.

    For ``kind == "grass"`` the grid is the GRASS generator's own field, normalised the
    same way, so that MuJoCo's ``hfield_size[2]`` -- which is the physical peak-to-peak --
    stays the single source of truth for the terrain's amplitude whatever produced the
    grid.  The grass field is fetched through the ``natural_scene`` module's cache so the
    SAME array is used for the geometry and for the report's statistics; generating it
    twice would let the reported statistics describe a field that is not the one the fly
    walks on.
    """
    import numpy as np
    if getattr(cfg, "kind", None) == "grass":
        raw = _grass_heightfield_for(cfg, scene_cfg)
        lo, hi = float(raw.min()), float(raw.max())
        if hi - lo <= 0:
            raise ValueError("grass heightfield is flat; refusing to build terrain")
        return (raw - lo) / (hi - lo)
    n, m = int(cfg.n_rows), int(cfg.n_cols)
    ext = float(cfg.half_extent_mm)
    rng = np.random.default_rng(int(cfg.seed))
    xs = np.linspace(-ext, ext, n)
    ys = np.linspace(-ext, ext, m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    k = 2.0 * math.pi / float(cfg.wavelength_mm)
    h = 0.5 + 0.35 * np.sin(k * X) + 0.15 * np.cos(k * Y)
    h = h + 0.10 * rng.random((n, m))
    h = (h - h.min()) / (h.max() - h.min())
    return h


def _grass_heightfield_for(cfg: "RoughnessConfig", scene_cfg=None):
    """The grass elevation array.

    The GENERATOR lives in ``grass_scene.py`` and is called here directly.  It is NOT
    fetched through ``natural_scene``: importing that module from here would close an
    import cycle (natural_scene imports this module), and a second code path that built
    the grid independently would let the statistics in the report describe a field the
    fly never walked on.  ``natural_scene`` records this function's output in its own
    report, so there is still exactly one grid.
    """
    try:
        from grass_scene import generate_grass_heightfield
    except ImportError as exc:      # pragma: no cover
        raise RuntimeError(
            "roughness.kind='grass' needs grass_scene.py, which is not importable: "
            f"{type(exc).__name__}: {exc}") from exc
    # ``grass_clearings_mm`` on the roughness config: disc patches flattened to zero so
    # fiducial markers on them are visible from every camera.  MEASURED need -- with the
    # markers on uncleared grass, EVERY projection landed on a dark patch and no marker was
    # visible from any camera, because blades occlude a 0.15 mm sphere at a 29 degree
    # elevation.  Empty by default, so a preset that does not need clearings is unchanged.
    clr = tuple(getattr(cfg, "grass_clearings_mm", ()) or ())
    elev, _meta = generate_grass_heightfield(n_samples=int(cfg.n_rows),
                                             extent_mm=float(cfg.half_extent_mm),
                                             seed=int(cfg.seed),
                                             clearings_mm=clr)
    return elev


def _roughness_scatter_positions(cfg: "RoughnessConfig"):
    import numpy as np
    rng = np.random.default_rng(int(cfg.seed) + 7919)
    n = int(cfg.n_scatter)
    ext = float(cfg.scatter_half_extent_mm)
    xy = rng.uniform(-ext, ext, size=(n, 2))
    r = float(cfg.scatter_radius_mm)
    return [(float(x), float(y), r) for x, y in xy]


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

@dataclass
class SceneApplicationReport:
    """What :func:`apply_scene` did, and (when it could compile) what MuJoCo says.

    The configured fields are what was REQUESTED.  ``measured`` is what the
    compiled ``mujoco.MjModel`` actually contains; when the two disagree the
    disagreement is listed in ``warnings`` rather than hidden.
    """
    name: str
    friction_rows_applied: dict[str, tuple[float, float, float]]
    n_geoms: int
    n_lights: int
    n_hfield: int
    n_objects: int
    roughness_peak_to_peak_mm: float
    capillary_force_mN: float
    warnings: tuple[str, ...] = ()
    # -- extras that make the report usable instead of merely present ---
    pair_rows_applied: dict[str, tuple[float, float, float, float, float]] = \
        field(default_factory=dict)
    geom_priority_applied: dict[str, int] = field(default_factory=dict)
    n_pairs: int = 0
    n_roughness_geoms: int = 0
    n_capillary_actuators: int = 0
    n_geom_by_class: dict[str, int] = field(default_factory=dict)
    ground_geom_names: tuple[str, ...] = ()
    roughness_kind: str = "none"
    roughness_fallback_used: bool = False
    adhesion_gain_applied: float | None = None
    objects_collidable: bool = False
    friction_mechanism: str = FRICTION_MECHANISM_NOTE
    measured: dict | None = None
    measured_note: str = ""

    def as_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))

    # -- the ground-truth check -----------------------------------------
    @classmethod
    def measured_after_compile(cls, model, *, fly_name: str = "nmf",
                               ground_geom_names: Sequence[str] = (),
                               tarsus_pattern: str = "_tarsus",
                               n_roughness_geoms: int | None = None) -> dict:
        """RE-READ a compiled ``mujoco.MjModel`` and report what is actually there.

        This is the ground-truth check: it never consults the config.  Everything
        returned is read out of the compiled model arrays.
        """
        import numpy as np

        def _i(arr) -> int:
            """MuJoCo 3.9 returns 1-element arrays for many MjModel fields; NumPy 2
            refuses ``int()`` on them, so ravel first (measured, not guessed)."""
            return int(np.ravel(np.asarray(arr))[0])

        ground = set(str(g) for g in ground_geom_names)
        rows_by_class: dict[str, dict[str, list[float]]] = {
            "ground": {}, "tarsus": {}, "body": {}}
        counts_by_type: dict[str, int] = {}
        n_fly = n_rough = n_obj = 0
        for i in range(int(model.ngeom)):
            name = str(model.geom(i).name)
            tname = mujoco_geom_type_name(_i(model.geom_type[i]))
            counts_by_type[tname] = counts_by_type.get(tname, 0) + 1
            if name.startswith(f"{fly_name}/"):
                n_fly += 1
                cls_ = ("tarsus" if tarsus_pattern in name.split("/")[-1] else "body")
            elif name in ground or name == "ground_plane" or \
                    name.startswith(ROUGHNESS_GEOM_PREFIX):
                cls_ = "ground"
            else:
                cls_ = None
            if name.startswith(ROUGHNESS_GEOM_PREFIX):
                n_rough += 1
            if name.startswith(OBJECT_GEOM_PREFIX):
                n_obj += 1
            if cls_ is not None:
                rows_by_class[cls_][name] = [
                    float(v) for v in np.asarray(model.geom_friction[i])]
        pair_rows_by_class: dict[str, dict[str, list[float]]] = {
            "ground": {}, "tarsus": {}, "body": {}}
        for i in range(int(model.npair)):
            n1 = str(model.geom(_i(model.pair_geom1[i])).name)
            n2 = str(model.geom(_i(model.pair_geom2[i])).name)
            fly_geom = n1 if n1.startswith(f"{fly_name}/") else (
                n2 if n2.startswith(f"{fly_name}/") else None)
            cls_ = (_geom_class_of_name(fly_geom, tarsus_pattern) if fly_geom
                    else "ground")
            pair_rows_by_class[cls_][f"{n1}|{n2}"] = [
                float(v) for v in np.asarray(model.pair_friction[i])]
        hfield_p2p_mm = None
        if int(model.nhfield) > 0:
            hf = model.hfield(0)
            size = np.asarray(model.hfield_size[0], dtype=float)
            data = np.asarray(model.hfield_data, dtype=float)
            adr = _i(model.hfield_adr[0])
            n = _i(model.hfield_nrow[0]) * _i(model.hfield_ncol[0])
            chunk = data[adr:adr + n]
            hfield_p2p_mm = float((chunk.max() - chunk.min()) * size[2]) if chunk.size else None
            hfield_size = [float(v) for v in size]
            hfield_nrow = _i(model.hfield_nrow[0])
            hfield_ncol = _i(model.hfield_ncol[0])
            hfield_name = str(hf.name)
        else:
            hfield_size, hfield_nrow, hfield_ncol, hfield_name = None, None, None, None
        lights = []
        for i in range(int(model.nlight)):
            li = model.light(i)
            lights.append({
                "name": str(li.name), "type": _i(model.light_type[i]),
                "dir": [float(v) for v in np.asarray(model.light_dir[i])],
                "castshadow": bool(_i(model.light_castshadow[i])),
                "active": bool(_i(model.light_active[i])),
                "diffuse": [float(v) for v in np.asarray(model.light_diffuse[i])],
                "ambient": [float(v) for v in np.asarray(model.light_ambient[i])],
            })
        distinct_geom_rows = sorted({
            tuple(round(v, 9) for v in vals)
            for per_class in rows_by_class.values() for vals in per_class.values()})
        distinct_pair_rows = sorted({
            tuple(round(v, 9) for v in vals)
            for per_class in pair_rows_by_class.values() for vals in per_class.values()})
        return {
            "ngeom": int(model.ngeom), "nlight": int(model.nlight),
            "nhfield": int(model.nhfield), "npair": int(model.npair),
            "nu": int(model.nu), "nsensor": int(model.nsensor),
            "ncam": int(model.ncam), "nbody": int(model.nbody),
            "n_fly_geoms": n_fly,
            "n_roughness_geoms": (int(n_roughness_geoms) if n_roughness_geoms
                                  is not None else n_rough),
            "n_object_geoms": n_obj,
            "geom_counts_by_type": counts_by_type,
            "geom_friction_rows_by_class": {
                cls_: sorted({tuple(round(v, 9) for v in vals)
                              for vals in per_class.values()})
                for cls_, per_class in rows_by_class.items()},
            "distinct_geom_friction_rows": distinct_geom_rows,
            "geom_friction_row_is_uniform": len(distinct_geom_rows) == 1,
            "pair_friction_rows_by_class": {
                cls_: sorted({tuple(round(v, 9) for v in vals)
                              for vals in per_class.values()})
                for cls_, per_class in pair_rows_by_class.items()},
            "distinct_pair_friction_rows": distinct_pair_rows,
            "hfield_peak_to_peak_mm": hfield_p2p_mm,
            "hfield_size": hfield_size,
            "hfield_nrow": hfield_nrow, "hfield_ncol": hfield_ncol,
            "hfield_name": hfield_name,
            "lights": lights,
            "gravity_mm_s2": [float(v) for v in np.asarray(model.opt.gravity)],
            "timestep_s": float(model.opt.timestep),
        }


def mujoco_geom_type_name(gtype: int) -> str:
    """Human-readable MuJoCo geom type (``7`` -> ``'mesh'``)."""
    mj = _import_mujoco()
    out = {int(mj.mjtGeom.mjGEOM_PLANE): "plane",
           int(mj.mjtGeom.mjGEOM_HFIELD): "hfield",
           int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
           int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
           int(mj.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
           int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
           int(mj.mjtGeom.mjGEOM_BOX): "box",
           int(mj.mjtGeom.mjGEOM_MESH): "mesh"}
    return out.get(int(gtype), f"type_{int(gtype)}")


def measured_after_compile(model, **kwargs) -> dict:
    """Functional alias for :meth:`SceneApplicationReport.measured_after_compile`."""
    return SceneApplicationReport.measured_after_compile(model, **kwargs)


# --------------------------------------------------------------------------- #
# apply_scene
# --------------------------------------------------------------------------- #

def _find_geom_element(spec, name):
    for g in spec.geoms:
        if str(g.name) == name:
            return g
    return None


def _replace_ground_refs(spec, old_names: set[str], new_name: str) -> int:
    """Re-point every explicit pair from the old ground geoms to ``new_name``.

    Returns the number of pairs rewritten.  MuJoCo resolves pair geom references
    by name at compile time, so the pair has to be deleted and re-added.
    """
    rewritten = 0
    for p in list(spec.pairs):
        n1, n2 = str(p.geomname1), str(p.geomname2)
        if n1 in old_names:
            other, swap = n2, True
        elif n2 in old_names:
            other, swap = n1, False
        else:
            continue
        fric = [float(v) for v in p.friction]
        condim = int(p.condim)
        solref = [float(v) for v in p.solref]
        solimp = [float(v) for v in p.solimp]
        margin = float(p.margin)
        name = str(p.name)
        spec.delete(p)
        args = dict(name=name, condim=condim, solref=solref, solimp=solimp,
                    margin=margin, friction=fric)
        if swap:
            args["geomname1"], args["geomname2"] = new_name, other
        else:
            args["geomname1"], args["geomname2"] = other, new_name
        spec.add_pair(**args)
        rewritten += 1
    return rewritten


def _retarget_contact_sensors(spec, old_names: set[str], new_name: str) -> int:
    mj = _import_mujoco()
    changed = 0
    for s in spec.sensors:
        if int(s.type) == int(mj.mjtSensor.mjSENS_CONTACT) and \
                int(s.reftype) == int(mj.mjtObj.mjOBJ_GEOM) and \
                str(s.refname) in old_names:
            s.refname = new_name
            changed += 1
    return changed


def apply_scene(world, fly, cfg: SceneConfig,
                ledger: "ParameterLedger | None" = None, *,
                measure: bool = True) -> SceneApplicationReport:
    """Mutate ``world``'s MJCF spec (and ``fly``) in place to realise ``cfg``.

    Must be called BEFORE ``world.compile()``.  Adding or rewriting MJCF elements
    after construction is the only way to get per-class friction, micro-relief,
    lights and objects into a FlyGym world: the spec, not the compiled model, is
    the thing that can still be edited.

    Steps, in order:
      1. classify the fly's geoms into tarsus / body from FlyGym's own
         ``bodyseg_to_mjcfgeom`` mapping;
      2. write per-class ``geom.friction`` and ``geom.priority``;
      3. (heightfield only) build the hfield asset + geom and re-point the 55
         fly-ground pairs and the 6 contact sensors at it;
      4. write per-class pair friction -- the mechanism that actually governs
         these contacts (see :data:`FRICTION_MECHANISM_NOTE`);
      5. add scattered roughness geoms (fallback kind only);
      6. add lights;
      7. add scene-object geoms;
      8. add the capillary adhesion actuators and set the fly's own adhesion gain;
      9. optionally compile once to read back what MuJoCo actually built.

    ``ledger`` is accepted for API compatibility and used to look up (never to
    invent) provenance; a record that is missing is a warning, not a silent
    default.
    """
    mj = _import_mujoco()
    from flygym.utils.mjcf import GEOM_TYPES, add_actuator as fg_add_actuator

    if not isinstance(cfg, SceneConfig):
        raise TypeError(f"cfg must be a SceneConfig, got {type(cfg).__name__}")
    warnings: list[str] = []

    spec = world.mjcf_root
    fly_name = str(getattr(fly, "name", None) or "fly")
    classes = _classify_fly_geoms(fly)
    if not classes["tarsus"]:
        warnings.append("no tarsus geoms were found on this fly: the 'tarsus' friction "
                        "class is then inert and tarsus==body in practice")
    if ledger is not None and not isinstance(ledger, ParameterLedger):
        raise TypeError("ledger must be a ParameterLedger or None")

    ground_geoms = list(getattr(world, "ground_geoms", []) or [])
    if not ground_geoms:
        warnings.append("world.ground_geoms is empty: no ground pairs could be "
                        "classified or re-pointed")
    old_ground_names = {str(g.name) for g in ground_geoms}

    # -- 1/2. per-class geom friction and priority -----------------------
    prio = dict(cfg.geom_priority_by_class)
    wanted = {"tarsus": cfg.tarsus_friction(), "body": cfg.body_friction()}
    applied_geom_rows: dict[str, int] = {"tarsus": 0, "body": 0, "ground": 0}
    for cls in ("tarsus", "body"):
        for gname in classes[cls]:
            g = _find_geom_element(spec, gname)
            if g is None:
                warnings.append(f"fly geom {gname!r} was not found in the spec")
                continue
            g.friction = list(wanted[cls])
            g.priority = int(prio.get(cls, 0))
            applied_geom_rows[cls] += 1
    for gname in old_ground_names:
        g = _find_geom_element(spec, gname)
        if g is None:
            continue
        g.friction = list(cfg.ground_friction())
        g.priority = int(prio.get("ground", 0))
        applied_geom_rows["ground"] += 1

    # -- 3. heightfield --------------------------------------------------
    n_hfield = 0
    n_roughness_geoms = 0
    roughness_p2p = 0.0
    roughness_kind = cfg.roughness.kind if cfg.roughness else "none"
    fallback_used = False
    ground_after = list(old_ground_names)
    if cfg.roughness is not None and cfg.roughness.kind in ("heightfield", "grass"):
        rc = cfg.roughness
        # The data comes from the SCENE layer's own generator for "heightfield" and from
        # the grass generator (injected by natural_scene before compile) for "grass".
        # pass the SCENE config as scene_cfg, NOT as cfg: the second parameter of
        # _heightfield_data is already named cfg and means the roughness config,
        # so calling it with cfg= raised "got multiple values for argument cfg".
        data = _heightfield_data(mj, rc, scene_cfg=cfg)
        spec.add_hfield(name=HFIELD_ASSET_NAME,
                        size=[float(rc.half_extent_mm), float(rc.half_extent_mm),
                              float(rc.amplitude_mm), float(rc.base_offset_mm)],
                        nrow=int(rc.n_rows), ncol=int(rc.n_cols),
                        userdata=[float(v) for v in data.reshape(-1)])
        hgeom_kwargs = dict(
            name=HFIELD_GEOM_NAME, type=GEOM_TYPES["hfield"],
            hfieldname=HFIELD_ASSET_NAME, pos=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0],
            contype=0, conaffinity=0, group=0,
            friction=list(cfg.ground_friction()),
            priority=int(prio.get("ground", 0)))
        material_name = getattr(rc, "hfield_material", None)
        if material_name:
            have = {str(m.name) for m in spec.materials}
            if material_name in have:
                hgeom_kwargs["material"] = material_name
            else:
                warnings.append(
                    f"roughness.hfield_material={material_name!r} is not a material in "
                    f"this spec (have {sorted(have)}); the heightfield was added without "
                    f"one, so it will render with MuJoCo's default grey")
        hgeom = spec.worldbody.add_geom(**hgeom_kwargs)
        n_hfield = 1
        n_roughness_geoms = 1
        roughness_p2p = float(rc.amplitude_mm) * float(data.max() - data.min())
        n_pairs_rewritten = _replace_ground_refs(spec, old_ground_names, HFIELD_GEOM_NAME)
        n_sensors = _retarget_contact_sensors(spec, old_ground_names, HFIELD_GEOM_NAME)
        if n_pairs_rewritten == 0:
            warnings.append(
                "a heightfield was requested but no ground pair referenced the old "
                "ground geom, so no pair was re-pointed: the heightfield will not be "
                "touchable by the fly (the pairs are the only contact path)")
        if n_sensors != 6:
            warnings.append(
                f"expected 6 ground contact sensors to retarget to the heightfield, "
                f"retargeted {n_sensors}; leg-in-contact sensing may be degraded")
        # The original plane is kept in the model purely as an inert visual floor:
        # its contype/conaffinity are 0 and nothing pairs with it any more, so it
        # cannot generate a contact.  Reported, not hidden.
        warnings.append(
            "the original ground plane geom was KEPT as an inert visual floor: it has "
            "contype=conaffinity=0 and no pair references it any more, so MuJoCo "
            "generates no contact with it.  world.ground_geoms was set to the "
            "heightfield geom, and the fly's 6 contact sensors were re-pointed to it.")
        warnings.append(
            f"the heightfield is a FINITE patch (half extent {rc.half_extent_mm} mm, so "
            f"it spans {2 * rc.half_extent_mm:.0f} mm).  Outside it there is NO "
            f"collision at all -- the original plane has no pairs and no contype -- so a "
            f"fly that walks past the edge falls through the ground.  Raise "
            f"roughness.half_extent_mm for rollouts longer than ~{rc.half_extent_mm:.0f} mm "
            f"of travel.")
        warnings.append(
            "reusing the plane's material on the heightfield makes the texture scale "
            "differ inside and outside the patch (MuJoCo maps a texture per geom), so "
            "the rendered checker is finer within the heightfield.  Cosmetic only.")
        ground_after = [HFIELD_GEOM_NAME]
        try:
            world.ground_geoms = [hgeom]
        except Exception as exc:  # pragma: no cover
            warnings.append(f"could not replace world.ground_geoms: {exc!r}")

    elif cfg.roughness is not None and cfg.roughness.kind == "scattered":
        rc = cfg.roughness
        fallback_used = True
        positions = _roughness_scatter_positions(rc)
        made = []
        for i, (x, y, z) in enumerate(positions):
            name = f"{ROUGHNESS_GEOM_PREFIX}{i:03d}"
            g = spec.worldbody.add_geom(
                name=name, type=GEOM_TYPES["sphere"], pos=[x, y, z],
                size=[float(rc.scatter_radius_mm), 0.0, 0.0],
                contype=0, conaffinity=0, group=0,
                friction=list(cfg.ground_friction()),
                priority=int(prio.get("ground", 0)))
            made.append(name)
        n_roughness_geoms = len(made)
        roughness_p2p = 2.0 * float(rc.scatter_radius_mm)
        # Only the tarsal tips can touch the scattered pebbles: pairing every fly
        # geom with every pebble would add ~1400 candidate pairs for no benefit.
        tip_names = [n for n in classes["tarsus"] if "tarsus5" in n.split("/")[-1]]
        if not tip_names:
            tip_names = list(classes["tarsus"])
        n_new_pairs = 0
        for name in made:
            for tip in tip_names:
                spec.add_pair(name=f"{tip}-{name}-scatter", geomname1=tip,
                              geomname2=name, condim=3,
                              solref=[0.0002, 1.0], solimp=[0.98, 0.99, 1e-5, 0.5, 3.0],
                              margin=1e-3,
                              friction=_row3_to_pair5(cfg.tarsus_friction()))
                n_new_pairs += 1
        warnings.append(
            f"roughness kind 'scattered' is an APPROXIMATION of micro-relief: "
            f"{len(made)} sphere geoms were placed and paired ONLY with the six "
            f"tarsal-tip geoms ({n_new_pairs} extra pairs).  The flat ground plane is "
            f"still the main contact surface, so the contact topology differs from the "
            f"heightfield variant and results from the two are not interchangeable.")
        ground_after = list(old_ground_names)
    elif cfg.roughness is not None and cfg.roughness.kind == "none":
        pass

    # -- 4. per-class pair friction (the mechanism that governs contact) --
    pair_rows: dict[str, list[float]] = {}
    n_pairs_touched = 0
    if cfg.leave_pairs_untouched:
        warnings.append(
            "this preset leaves the explicit fly-ground pairs EXACTLY as FlyGym wrote "
            "them, so it reproduces the existing scene: the pair row is "
            "(1, 1, 0.02, 1e-4, 1e-4) and its torsional term 0.02 is NOT the geom "
            "row's 0.005 -- and it is the pair row MuJoCo actually uses.")
    for p in ([] if cfg.leave_pairs_untouched else list(spec.pairs)):
        n1, n2 = str(p.geomname1), str(p.geomname2)
        in1, in2 = n1.startswith(f"{fly_name}/"), n2.startswith(f"{fly_name}/")
        if not (in1 or in2):
            continue          # not a fly-ground pair
        partner = n2 if in1 else n1
        if partner not in set(ground_after) and not partner.startswith(ROUGHNESS_GEOM_PREFIX):
            continue
        fly_geom = n1 if in1 else n2
        cls = _geom_class_of_name(fly_geom)
        row = cfg.tarsus_friction() if cls == "tarsus" else cfg.body_friction()
        p.friction = _row3_to_pair5(row)
        pair_rows.setdefault(cls, _row3_to_pair5(row))
        n_pairs_touched += 1
    if pair_rows:
        warnings.append(
            "the fly's contact friction is set on the PAIRS, which is what MuJoCo 3.9 "
            "actually honours for these contacts; the per-geom friction/priority rows "
            "are also written but do not govern fly-ground contact in this model.")
    warnings.append(FRICTION_MECHANISM_NOTE)

    # -- 5. lights -------------------------------------------------------
    for light in cfg.lights:
        kind = {"directional": mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                "point": mj.mjtLightType.mjLIGHT_POINT,
                "spot": mj.mjtLightType.mjLIGHT_SPOT}[light.kind]
        mode = {"fixed": mj.mjtCamLight.mjCAMLIGHT_FIXED,
                "track": mj.mjtCamLight.mjCAMLIGHT_TRACK,
                "trackcom": mj.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                "targetbody": mj.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                "targetbodycom": mj.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM}[light.mode]
        spec.worldbody.add_light(
            name=light.name, type=kind, mode=mode,
            pos=[float(v) for v in light.pos_mm],
            dir=[float(v) for v in light.dir],
            diffuse=[float(v) for v in light.diffuse],
            specular=[float(v) for v in light.specular],
            ambient=[float(v) for v in light.ambient],
            castshadow=1 if light.castshadow else 0, active=1)

    # -- 6. objects ------------------------------------------------------
    for obj in cfg.objects:
        collidable = bool(obj.collidable or cfg.objects_collidable)
        spec.worldbody.add_geom(
            name=OBJECT_GEOM_PREFIX + obj.name,
            type=GEOM_TYPES[obj.geom_type],
            pos=[float(v) for v in obj.pos_mm],
            size=[float(v) for v in obj.size_mm],
            rgba=[float(v) for v in obj.rgba],
            contype=1 if collidable else 0,
            conaffinity=1 if collidable else 0,
            group=0)
    if cfg.objects and not (cfg.objects_collidable or any(
            o.collidable for o in cfg.objects)):
        warnings.append(
            "scene objects are VISUAL ONLY (contype=conaffinity=0).  Physics therefore "
            "cannot differ between presets that differ only in `objects`, so "
            "'natural' vs 'natural_lit_no_objects' isolates the objects' contribution "
            "to what could be seen -- it is not a physics control.")

    # -- 7. capillary ----------------------------------------------------
    n_cap = 0
    cap_force = 0.0
    if cfg.capillary is not None and cfg.capillary.enabled:
        cap = cfg.capillary
        cap_force = float(cap.secretion_force_mN) * len(cap.applies_to)
        for leg in cap.applies_to:
            body_name = f"{fly_name}/{leg}_tarsus5"
            fg_add_actuator(spec, "adhesion",
                            name=f"{CAPILLARY_ACTUATOR_PREFIX}{leg}_tarsus5",
                            body=body_name, gain=float(cap.secretion_force_mN),
                            ctrlrange=(0.0, 1.0))
            n_cap += 1
        warnings.append(
            "the capillary term is an ADDITIONAL mjACT_ADHESION actuator per leg "
            "(gain = secretion_force_mN, ctrl driven to 1) applying a normal "
            "attraction force.  Its magnitude is ASSUMED, not measured or cited: "
            f"{cap.secretion_force_mN} mN/leg x {len(cap.applies_to)} legs = "
            f"{cap_force:.4f} mN against a measured model weight of 10.048 mN "
            "(a real Drosophila is ~1000x lighter than this model).")
        warnings.append(
            f"capillary.humidity={cap.humidity} is recorded as METADATA ONLY and does "
            "NOT enter the physics: no humidity-to-force law was verified, and "
            "inventing one would be a fabricated physical model.")
    elif cfg.capillary is not None and not cfg.capillary.enabled:
        warnings.append("cfg.capillary exists but is disabled: no capillary actuator "
                        "was added")

    # -- 8. the fly's own adhesion actuators ----------------------------
    adhesion_gain_applied = None
    existing_adhesion = list(getattr(fly, "leg_to_adhesionactuator", {}) or {})
    if cfg.adhesion_added:
        if existing_adhesion:
            for leg, act in fly.leg_to_adhesionactuator.items():
                try:
                    act.gainprm[0] = float(cfg.adhesion_gain)
                except Exception as exc:  # pragma: no cover
                    warnings.append(f"could not set adhesion gain for leg {leg}: {exc!r}")
            adhesion_gain_applied = float(cfg.adhesion_gain)
        else:
            fly.add_leg_adhesion(gain=float(cfg.adhesion_gain))
            adhesion_gain_applied = float(cfg.adhesion_gain)
            warnings.append("the fly had no adhesion actuators; they were added by "
                            "apply_scene with cfg.adhesion_gain")
    else:
        if existing_adhesion:
            warnings.append("cfg.adhesion_added is False but the fly already carries "
                            "leg adhesion actuators; apply_scene does NOT remove them "
                            "(their control can be held at 0 instead)")
        else:
            adhesion_gain_applied = 0.0
    warnings.append(
        "adhesion_gain is kept IDENTICAL to the control preset in every preset here, "
        "so that tarsal adhesion is not a confound in the blank-vs-natural comparison.")

    # -- 9. prepare the report -------------------------------------------
    report = SceneApplicationReport(
        name=cfg.name,
        friction_rows_applied={k: tuple(float(v) for v in cfg.friction_by_class[k])
                               for k in ("ground", "tarsus", "body")},
        n_geoms=0, n_lights=len(cfg.lights), n_hfield=n_hfield,
        n_objects=len(cfg.objects),
        roughness_peak_to_peak_mm=float(roughness_p2p),
        capillary_force_mN=float(cap_force),
        warnings=tuple(warnings),
        pair_rows_applied={k: tuple(v) for k, v in pair_rows.items()},
        geom_priority_applied={k: int(v) for k, v in cfg.geom_priority_by_class.items()},
        n_pairs=n_pairs_touched,
        n_roughness_geoms=n_roughness_geoms,
        n_capillary_actuators=n_cap,
        n_geom_by_class={"tarsus": applied_geom_rows["tarsus"],
                         "body": applied_geom_rows["body"],
                         "ground": applied_geom_rows["ground"]},
        ground_geom_names=tuple(ground_after),
        roughness_kind=roughness_kind,
        roughness_fallback_used=fallback_used,
        adhesion_gain_applied=adhesion_gain_applied,
        objects_collidable=bool(cfg.objects_collidable or any(
            o.collidable for o in cfg.objects)))

    if measure:
        try:
            model, _ = world.compile()
            measured = SceneApplicationReport.measured_after_compile(
                model, fly_name=fly_name, ground_geom_names=ground_after,
                n_roughness_geoms=n_roughness_geoms)
            report.measured = measured
            report.n_geoms = int(measured["ngeom"])
            report.n_lights = int(measured["nlight"])
            report.n_hfield = int(measured["nhfield"])
            report.n_objects = int(measured["n_object_geoms"])
            report.measured_note = ("read back from a compiled copy of the spec; the "
                                    "caller's own world.compile() produces an "
                                    "equivalent model")
            # honest cross-checks, recorded rather than assumed
            if report.n_lights < len(cfg.lights):
                report.warnings = report.warnings + (
                    f"only {report.n_lights} of {len(cfg.lights)} configured lights "
                    f"reached the compiled model",)
            if report.n_hfield != n_hfield:
                report.warnings = report.warnings + (
                    f"configured {n_hfield} heightfield(s), compiled model has "
                    f"{report.n_hfield}",)
            if report.n_objects != len(cfg.objects):
                report.warnings = report.warnings + (
                    f"configured {len(cfg.objects)} objects, compiled model has "
                    f"{report.n_objects} object geoms",)
            if measured["hfield_peak_to_peak_mm"] is not None:
                report.roughness_peak_to_peak_mm = float(
                    measured["hfield_peak_to_peak_mm"])
                if cfg.roughness is not None and abs(
                        report.roughness_peak_to_peak_mm
                        - float(cfg.roughness.amplitude_mm)) > 1e-9:
                    report.warnings = report.warnings + (
                        f"configured roughness amplitude {cfg.roughness.amplitude_mm} mm "
                        f"but the compiled heightfield measures "
                        f"{report.roughness_peak_to_peak_mm} mm peak-to-peak",)
            if measured["geom_friction_row_is_uniform"]:
                report.warnings = report.warnings + (
                    "the compiled model reports ONE uniform geom_friction row for every "
                    "geom, i.e. the per-class geom friction did not survive "
                    "compilation -- check this before trusting any per-class claim",)
        except Exception as exc:
            report.measured_note = f"could not compile to measure: {exc!r}"
            report.warnings = report.warnings + (
                f"apply_scene could not compile the spec to verify itself: {exc!r}",)
    return report


# --------------------------------------------------------------------------- #
# capillary driving and contact-force measurement
# --------------------------------------------------------------------------- #

def capillary_actuator_names(cfg: SceneConfig) -> tuple[str, ...]:
    if cfg.capillary is None or not cfg.capillary.enabled:
        return ()
    return tuple(f"{CAPILLARY_ACTUATOR_PREFIX}{leg}_tarsus5"
                 for leg in cfg.capillary.applies_to)


def drive_capillary_actuators(model, data, cfg: SceneConfig) -> dict:
    """Set ``data.ctrl`` for the capillary actuators and report what was written.

    MuJoCo does NOT re-apply ``ctrl`` by itself across ``mj_step`` calls, so this
    must be called *every* step of a rollout for the term to be present, exactly
    like any other actuator.
    """
    mj = _import_mujoco()
    names = capillary_actuator_names(cfg)
    idx, missing = [], []
    for name in names:
        i = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            missing.append(name)
        else:
            idx.append(int(i))
    for i in idx:
        data.ctrl[i] = 1.0
    gains = [float(model.actuator_gainprm[i][0]) for i in idx]
    return {"names": list(names), "missing": missing, "indices": idx,
            "gains_mN": gains, "n_driven": len(idx),
            "nominal_force_mN": float(sum(gains))}


def measure_ground_contact_forces(model, data,
                                  ground_geom_names: Sequence[str]) -> dict:
    """Read the ACTUAL contact friction and forces out of the live ``MjData``.

    For each active contact between the ground and a fly geom, MuJoCo's contact
    frame has the normal as its first axis, so the normal force is ``f[0]`` and
    the sliding/tangential force is ``|(f[1], f[2])|``.  ``friction[0]`` is the
    sliding coefficient MuJoCo actually resolved for that contact -- the ground
    truth for every friction claim in this module.
    """
    mj = _import_mujoco()
    import numpy as np
    ground = set(str(g) for g in ground_geom_names)
    res = np.zeros(6, dtype=float)
    rows = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        n1 = str(model.geom(int(c.geom1)).name)
        n2 = str(model.geom(int(c.geom2)).name)
        if not (n1 in ground or n2 in ground or
                n1.startswith(ROUGHNESS_GEOM_PREFIX) or n2.startswith(ROUGHNESS_GEOM_PREFIX)):
            continue
        mj.mj_contactForce(model, data, i, res)
        f = np.array(res, dtype=float)
        fr = np.asarray(c.friction, dtype=float)
        rows.append({
            "geom1": n1, "geom2": n2,
            "normal_force_mN": float(f[0]),
            "tangential_force_mN": float(np.linalg.norm(f[1:3])),
            "force_norm_mN": float(np.linalg.norm(f[:3])),
            "friction_row": [float(v) for v in fr],
            "sliding_mu": float(fr[0]),
            "dist_mm": float(c.dist),
        })
    if not rows:
        return {"n_ground_contacts": 0, "contacts": [],
                "mean_normal_force_mN": None, "mean_tangential_force_mN": None,
                "mean_sliding_mu": None, "distinct_friction_rows": []}
    tang = [r["tangential_force_mN"] for r in rows]
    nrm = [r["normal_force_mN"] for r in rows]
    mu = [r["sliding_mu"] for r in rows]
    return {
        "n_ground_contacts": len(rows),
        "contacts": rows,
        "mean_normal_force_mN": float(np.mean(nrm)),
        "mean_tangential_force_mN": float(np.mean(tang)),
        "mean_sliding_mu": float(np.mean(mu)),
        "distinct_friction_rows": sorted({tuple(round(v, 9) for v in r["friction_row"])
                                          for r in rows}),
        "sum_tangential_force_mN": float(np.sum(tang)),
        "sum_normal_force_mN": float(np.sum(nrm)),
    }


# --------------------------------------------------------------------------- #
# parameter sweeps
# --------------------------------------------------------------------------- #

def _fields_of(obj) -> set[str]:
    try:
        return {f.name for f in dataclasses.fields(obj)}
    except TypeError:
        return set()


def set_scene_parameter(cfg: SceneConfig, path: str, value: Any) -> SceneConfig:
    """Return a copy of ``cfg`` with a nested field replaced.

    ``path`` is dotted and may index into tuples or mappings, e.g.
    ``"roughness.amplitude_mm"``, ``"friction_by_class.tarsus.0"``,
    ``"lights.0.diffuse.1"``, ``"capillary.enabled"``.  Assignment goes through
    the dataclass constructor, so every validator in the module runs on the
    result: a sweep cannot silently produce an invalid scene.
    """
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise ValueError("path must not be empty")
    return _with_path(cfg, parts, value)


def _with_path(obj: Any, parts: list[str], value: Any) -> Any:
    if not parts:
        return value
    head, rest = parts[0], parts[1:]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if head not in _fields_of(obj):
            raise KeyError(f"{type(obj).__name__} has no field {head!r}")
        cur = getattr(obj, head)
        return dataclasses.replace(obj, **{head: _with_path(cur, rest, value)})
    if isinstance(obj, Mapping):
        if head not in obj:
            raise KeyError(f"mapping has no key {head!r}")
        new = dict(obj)
        new[head] = _with_path(obj[head], rest, value)
        return new
    if isinstance(obj, (list, tuple)):
        idx = int(head)
        new = list(obj)
        new[idx] = _with_path(new[idx], rest, value)
        return tuple(new)
    raise KeyError(f"cannot descend into {type(obj).__name__} with {head!r}")


def get_scene_parameter(cfg: SceneConfig, path: str) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur: Any = cfg
    for head in parts:
        if isinstance(cur, Mapping):
            cur = cur[head]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(head)]
        else:
            cur = getattr(cur, head)
    return cur


def sensitivity_sweep(cfg: SceneConfig, parameter_name: str,
                      values: Sequence[Any], *,
                      evaluator: Callable[[SceneConfig], Any],
                      ledger: "ParameterLedger | None" = None,
                      rollout_seconds: float | None = None) -> dict:
    """Sweep one parameter of ``cfg`` and return a MEASURED outcome per value.

    ``evaluator`` is injected so this function needs no MuJoCo: the self-test can
    pass a pure-Python function, and a rollout script can pass one that builds a
    world, runs ``rollout_seconds`` of simulation and returns measurements.  The
    value actually read back from each variant is recorded alongside the requested
    one, so a silently-ignored assignment is visible in the output rather than
    inferred from a flat curve.
    """
    if evaluator is None or not callable(evaluator):
        raise TypeError("sensitivity_sweep needs an injectable `evaluator` callable")
    outcomes = []
    for v in values:
        variant = set_scene_parameter(cfg, parameter_name, v)
        readback = get_scene_parameter(variant, parameter_name)
        entry: dict[str, Any] = {
            "value_requested": _jsonable(v), "value_read_back": _jsonable(readback),
            "variant_scene": variant.name, "outcome": None, "error": None}
        try:
            entry["outcome"] = _jsonable(evaluator(variant))
        except Exception as exc:            # keep the sweep going, report the failure
            entry["error"] = f"{type(exc).__name__}: {exc}"
        outcomes.append(entry)
    return {
        "scene": cfg.name,
        "parameter": parameter_name,
        "base_value": _jsonable(get_scene_parameter(cfg, parameter_name)),
        "n_values": len(list(values)),
        "rollout_seconds": rollout_seconds,
        "values_applied": [o["value_read_back"] for o in outcomes],
        "n_errors": sum(1 for o in outcomes if o["error"] is not None),
        "outcomes": outcomes,
        "evaluator_note": ("the evaluator is injected; this function itself touches no "
                           "MuJoCo, so a synthetic evaluator keeps it testable in the "
                           "neural environment"),
    }
