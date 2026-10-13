"""CANONICAL PHYSICS / UNITS / GATE MODULE  (round 39 full-fix)

WHY THIS FILE EXISTS
--------------------
Two defects were found by measurement, both of which made earlier rounds' *numbers*
mean something other than what they said:

  D1  UNIT LABEL DEFECT.  This module's FIRST version (round 39) declared the model length
      unit to be the CENTIMETRE and set ``MM_PER_MODEL_LENGTH = 10.0``, and on that basis
      "corrected" the project's historical reports by a factor of ten - claiming for example
      that a standing thorax height of 2.04535 really means 20.45 mm.

      THAT DECLARATION WAS WRONG AND IS RETRACTED (independent audit, 2026-10-11).  The model
      length unit is the MILLIMETRE.  Four independent lines of evidence:

        1. the canonical asset scales every mesh by 1000: 71 ``<mesh ... scale="1000 1000 1000"/>``
           (lines 46-116), and the cached STL coordinates are in METRES (Thorax.stl spans
           1.069e-3 m), so 1.0692 model units = 1.0692 mm;
        2. geometry cross-check: the ratio mesh-span / body-offset over 70 pairs has median
           0.9408, which is only consistent with shared units (a cm/mm mismatch gives 9.41
           or 0.094);
        3. flygym states it: ``compose/fly/base_fly.py:166-168`` ("we simulate length in mm,
           not m ... SCALE = 1000"), ``simulation.py:188`` ("Body positions in mm"),
           ``compose/fly/flybody.py:106``;
        4. this project's OWN generator says mm: the canonical XML header comment reads
           "the keyframe places the thorax at z=2.0453 mm", and gravity=9801 mm/s^2 =
           9.801 m/s^2 (the sibling upstream asset uses -981 for its centimetre family).

      The historical label was therefore RIGHT and round 39's "fix" was backwards by 10x.
      The numbers fed to MuJoCo never changed (model-space values are untouched); only the
      reported millimetre values were wrong.  Correct readings: thorax height 2.04535 mm,
      body length 2.6 mm, mesh extent 5.78 mm - all right for Drosophila, whereas the
      centimetre reading made the fly 8-10x too large.

      This module now converts 1:1 and keeps ``SEATING_PENETRATION_MODEL`` at the value the
      historical code actually applied, so fixing labels does NOT move the physics.

  D2  MIXED-UNIT CONSUMERS.  Measured before the unit question was settled:
      ``qpos[2] = pose['root_z_mm'] / 10.0`` puts the fly 10x too low - measured 179 contacts
      and the thorax mesh below the floor plane, so the floor ejected the body instead of the
      body collapsing.  The gate here rejects such a pose.  That divisor is a genuine
      mismatched-convention bug REGARDLESS of which unit wins, because the saved and applied
      values were produced under different conventions.

THE RULE THIS MODULE ENFORCES
-----------------------------
  * Every physical quantity that crosses a function boundary carries its unit IN THE
    NAME.  No bare `_mm` field may hold model units.
  * All model-space values are named `*_model` and are in MILLIMETRES; all report values are
    named `*_mm` / `*_s` and are converted exactly here (a 1:1 conversion, kept as a function
    so the unit lives in one place).
  * A stance only counts when a GATE passes: only tarsus geoms touch the floor, no
    body geom penetrates the floor, and the root is solved so the lowest foot point
    sits at the declared penetration.  A failed gate raises; it never silently
    produces a number.

This is a hand-built research prototype, not a finished product.  It reports what it
measures and refuses to report what it cannot verify.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# ---------------------------------------------------------------------------
# Units.  The ONLY place where the model unit is turned into SI-ish report units.
# ---------------------------------------------------------------------------
MODEL_LENGTH_UNIT_NAME = "mm"
MM_PER_MODEL_LENGTH = 1.0           # 1 model length unit = 1 mm (see the D1 evidence above)
#: MASS IS NOT ESTABLISHED.  The asset carries 71 ``<inertial mass=...>`` entries summing to
#: 0.002494271478 and NO ``density`` attribute, so the unit is not declared anywhere.  Three
#: mutually exclusive readings exist in this project's own files: (a) kg
#: (this module's round-39 text), (b) gram (``tools_stand_and_ground.py:704``,
#: ``diag_leg_load_test.py:38-45``, ``LEG_TUNING.zh-CN.md:117``), (c) 1 unit = 1 mg
#: (round 39's constant).  The only geometric constraint available - the summed mesh volume
#: 1.4679 mm^3 is an UPPER bound - excludes kg (1.7e6 kg/m^3) and mg (1.70 kg/m^3, barely
#: heavier than air) and leaves GRAM as the only non-absurd reading (1699 kg/m^3).  That is
#: an inference from geometry, NOT a declaration, so it is marked as inferred and every
#: artifact that reports an absolute force carries the caveat.
MODEL_MASS_UNIT_KG = 1e-3           # INFERRED (gram), not established - see the note above
MODEL_MASS_UNIT_STATUS = "INFERRED_FROM_MESH_VOLUME_DENSITY_ARGUMENT_NOT_ESTABLISHED"
MODEL_TIME_UNIT_S = 1.0             # seconds
GRAVITY_MODEL = 9801.0              # mm/s^2 = 9.801 m/s^2, matches <option gravity="0 0 -9801">


def model_to_mm(x):
    """Model length units (mm) -> millimetres (1:1, kept as a function so the unit lives once)."""
    return np.asarray(x, dtype=float) * MM_PER_MODEL_LENGTH


def mm_to_model(x):
    """Millimetres -> model length units (mm).  See the D1 retraction above: 1:1."""
    return np.asarray(x, dtype=float) / MM_PER_MODEL_LENGTH


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The DECLARED seating penetration: ONE number, ONE unit, ONE place.
#
# RETRACTED NARRATIVE (round 39 was wrong here; corrected 2026-10-11).
#
# Round 39 claimed that rounds 14-38 passed the labelled `-0.05` as MODEL UNITS while the
# model unit was a CENTIMETRE, so the applied penetration was "really 0.5 mm" and the label
# was 10x off.  Both halves of that are wrong, because the model unit is the MILLIMETRE
# (see D1): `-0.05` model units IS -0.05 mm, exactly what the historical constant said.
# The old label was right; round 39's relabelling to "-0.5 mm" was the error.
#
# What this module does now:
#   * DECLARES the penetration in millimetres as the value the historical code ACTUALLY
#     applied: -0.05 mm;
#   * keeps SEATING_PENETRATION_MODEL equal to the same number, so `seat_root` seats the
#     body exactly where rounds 14-38 seated it and NO physics moves;
#   * records the two values side by side, because the only thing that was ever wrong here
#     was the label, and a label error must not be repaired by moving the physics.
# ---------------------------------------------------------------------------
SEATING_PENETRATION_MM = -0.05            # the value ACTUALLY APPLIED, now correctly labelled
SEATING_PENETRATION_LABEL_MM = -0.5       # round 39's mistaken relabelling, kept for audit
SEATING_PENETRATION_MODEL = float(mm_to_model(SEATING_PENETRATION_MM))   # = -0.05, unchanged


# ---------------------------------------------------------------------------
# The canonical assets (never modified in place by any tool in this round)
# ---------------------------------------------------------------------------
CANONICAL_XML = PROJECT / "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml"
CANONICAL_XML_SHA256 = "65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6"

# The upstream calibrated asset shipped with flygym 2.1.0 (read-only reference).
UPSTREAM_FLYBODY_XML = (
    PROJECT / "venv_body/lib/python3.12/site-packages/flygym/assets/model/flybody/fruitfly.xml"
)


# ---------------------------------------------------------------------------
# Contact parameter sets.  A set is a NAMED, documented bundle; changing one
# field without the others is what produced round 38's ambiguous result.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactSet:
    name: str
    provenance: str
    solref: tuple | None = None          # (timeconst_s, dampratio)
    solimp: tuple | None = None          # (dmin, dmax, width, midpoint, power)
    margin: float | None = None          # geometric contact margin, model units
    friction: tuple | None = None        # (slide, torsional, roll) applied to geoms
    cone: str | None = None
    noslip_iterations: int | None = None
    viscosity: float | None = None
    density: float | None = None
    integrator: str | None = None
    timestep: float | None = None
    iterations: int | None = None
    tolerance: float | None = None
    note: str = ""


CONTACT_SETS = {
    # What the canonical asset compiles to today: no solref/solimp/margin anywhere,
    # so MuJoCo's own defaults apply.  This is a MODEL DEFAULT, not a measurement,
    # and it is 100x softer on the constraint time constant than the upstream asset.
    "mujoco_default": ContactSet(
        name="mujoco_default",
        provenance="Measured on the compiled canonical asset: solref/solimp are absent "
                   "from the XML (0 occurrences of 'solref'), so MuJoCo's own defaults "
                   "apply - solref=0.02 1, solimp=0.9 0.95 0.001 0.5 2.0, friction=1.0 "
                   "0.005 0.0001.  The geometric margin is NOT a default: the asset sets "
                   "geom margin=0.001 (0.01 mm) explicitly at line 247.",
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        margin=1e-3,
        friction=(1.0, 0.005, 0.0001),
        note="what rounds 34-38 measured on, without saying so",
    ),
    # The upstream NeuroMechFly-v2 / flybody asset's own numbers.  Copied verbatim;
    # NOT a measurement of a real fly foot either.
    "upstream_flybody": ContactSet(
        name="upstream_flybody",
        provenance="flygym 2.1.0 assets/model/flybody/fruitfly.xml, verbatim: "
                   "geom solref=0.0002 1, solimp=0.95 0.99 0.01, friction=0.5, "
                   "option cone=elliptic, noslip_iterations=3, viscosity=0.000185, "
                   "density=0.00128, timestep=0.0001",
        solref=(2e-4, 1.0),
        solimp=(0.95, 0.99, 0.01, 0.5, 3.0),
        margin=1e-3,
        friction=(0.5, 0.005, 0.0001),
        cone="elliptic",
        noslip_iterations=3,
        viscosity=0.000185,
        density=0.00128,
        timestep=1e-4,
        note="a documented bundle, not a single number.  Round 38 changed solref alone and "
         "got a WORSE escape time (25 ms) than the default; this round shows the field "
         "that actually helps is cone=elliptic + noslip_iterations, not stiffness.  Note "
         "the upstream asset's friction is 0.5, i.e. SOFTER grip than this project's 1.0.",
    ),
}


# ---------------------------------------------------------------------------
# XML variant construction.  The canonical file is never touched: variants are
# written next to it (so its ./meshes/stl/... paths resolve) and deleted after use.
# ---------------------------------------------------------------------------
def build_variant_xml(src_text: str, cs: ContactSet, *, timestep=None,
                      integrator=None, iterations=None, tolerance=None) -> str:
    """Return the canonical XML text with ONLY the named contact set applied."""
    import re

    src = src_text

    # 1) <option> attributes
    opt = re.search(r"<option[^>]*/>", src)
    assert opt, "canonical asset has no <option/> tag"
    tag = opt.group(0)
    want = {
        "timestep": repr(timestep if timestep is not None else cs.timestep)
        if (timestep or cs.timestep) else None,
        "cone": cs.cone,
        "noslip_iterations": str(cs.noslip_iterations) if cs.noslip_iterations else None,
        "viscosity": repr(cs.viscosity) if cs.viscosity else None,
        "density": repr(cs.density) if cs.density else None,
        "integrator": integrator or cs.integrator,
        "iterations": str(iterations or cs.iterations) if (iterations or cs.iterations) else None,
        "tolerance": repr(tolerance or cs.tolerance) if (tolerance or cs.tolerance) else None,
    }
    for k, v in want.items():
        if v is None:
            continue
        if re.search(rf'{k}="[^"]*"', tag):
            tag = re.sub(rf'{k}="[^"]*"', f'{k}="{v}"', tag)
        else:
            tag = tag[:-2] + f' {k}="{v}"/>'
    src = src[: opt.start()] + tag + src[opt.end():]

    # 2) a root-level <default><geom .../></default> BEFORE class="main" so that
    #    every geom inherits it unless it sets its own value
    bits = []
    if cs.solref is not None:
        bits.append(f'solref="{cs.solref[0]:g} {cs.solref[1]:g}"')
    if cs.solimp is not None:
        bits.append('solimp="{}"'.format(" ".join(f"{x:g}" for x in cs.solimp)))
    if cs.margin is not None:
        bits.append(f'margin="{cs.margin:g}"')
    if cs.friction is not None:
        bits.append(f'friction="{cs.friction[0]:g} {cs.friction[1]:g} {cs.friction[2]:g}"')
    if bits:
        m = re.search(r"<mujoco[^>]*>", src)
        assert m, "no <mujoco> root"
        block = "  <default>\n    <geom " + " ".join(bits) + "/>\n  </default>\n"
        src = src[: m.end()] + "\n" + block + src[m.end():]
    return src


def load_model(xml_text: str | None = None, cs: ContactSet | None = None, **kw):
    """Compile the canonical asset, optionally with a contact set applied."""
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    if xml_text is None:
        return _load_mjcf(str(CANONICAL_XML)).compile()
    tmp = CANONICAL_XML.parent / ".round39_variant_tmp.xml"
    try:
        tmp.write_text(xml_text)
        return _load_mjcf(str(tmp)).compile()
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------------------
# The health gate.  Every stance in this round must pass it or the run stops.
# ---------------------------------------------------------------------------
@dataclass
class StanceFacts:
    root_z_model: float
    thorax_centre_model: float
    n_contact: int
    n_floor_contact: int
    n_body_floor_contact: int
    min_floor_distance_model: float
    tarsus_only: bool
    worst_floor_penetration_model: float
    contacting_nonfoot_geoms: list = field(default_factory=list)

    # explicitly-named report units, converted in exactly one place
    @property
    def root_z_mm(self) -> float:
        return float(model_to_mm(self.root_z_model))

    @property
    def thorax_centre_mm(self) -> float:
        return float(model_to_mm(self.thorax_centre_model))

    @property
    def worst_floor_penetration_mm(self) -> float:
        return float(model_to_mm(self.worst_floor_penetration_model))

    def as_dict(self) -> dict:
        return {
            "root_z_mm": self.root_z_mm,
            "thorax_centre_mm": self.thorax_centre_mm,
            "n_contact": self.n_contact,
            "n_floor_contact": self.n_floor_contact,
            "n_body_floor_contact": self.n_body_floor_contact,
            "tarsus_only": bool(self.tarsus_only),
            "worst_floor_penetration_mm": self.worst_floor_penetration_mm,
            "contacting_nonfoot_geoms": list(self.contacting_nonfoot_geoms),
        }


class StanceGateError(RuntimeError):
    """Raised when a pose is not a usable stance.  Never swallowed."""


def tarsus_geom_ids(model) -> list:
    import mujoco

    return [g for g in range(model.ngeom)
            if "Tarsus" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]


def floor_geom_id(model) -> int:
    import mujoco

    for g in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "floor":
            return g
    raise StanceGateError("no geom named 'floor' in the model")


def seat_root(model, data, base_qpos, target_penetration_model=None, iters=80) -> float:
    """Raise/lower the root (qpos[2], MODEL UNITS) until the lowest tarsus point sits
    at `target_penetration_model` from the floor plane.  Monotone in z -> bisection.

    `target_penetration_model` defaults to SEATING_PENETRATION_MODEL, which is the
    project's historical convention expressed once, in mm, and converted HERE instead
    of being written as a bare number at every call site.
    """
    import mujoco

    if target_penetration_model is None:
        target_penetration_model = SEATING_PENETRATION_MODEL
    tarsus = tarsus_geom_ids(model)
    floor = floor_geom_id(model)

    def lowest(z):
        data.qpos[:] = base_qpos
        data.qpos[2] = z
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return min(float(mujoco.mj_geomDistance(model, data, g, floor, 1e3, None))
                   for g in tarsus)

    lo, hi = 1e-3, 1e3
    if lowest(lo) > target_penetration_model or lowest(hi) < target_penetration_model:
        raise StanceGateError("cannot bracket a root height for the requested penetration")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if lowest(mid) < target_penetration_model:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    data.qpos[:] = base_qpos
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return float(z)


def stance_facts(model, data, *, raise_on_fail=True) -> StanceFacts:
    """Measure a stance and GATE it.  `tarsus_only` is the project's own long-standing
    criterion; `min_floor_distance` additionally catches a body buried in the floor,
    which is the D2 defect and which the contact *count* alone did not catch."""
    import mujoco

    tarsus = set(tarsus_geom_ids(model))
    floor = floor_geom_id(model)
    name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"

    n_floor = 0
    nonfoot = []
    for c in range(data.ncon):
        g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
        if floor in (g1, g2):
            n_floor += 1
            other = g2 if g1 == floor else g1
            if other not in tarsus:
                nonfoot.append(name(other))

    dists = []
    for g in range(model.ngeom):
        if g == floor:
            continue
        d = float(mujoco.mj_geomDistance(model, data, g, floor, 1e3, None))
        dists.append((d, g))
    worst_d, worst_g = min(dists)

    facts = StanceFacts(
        root_z_model=float(data.qpos[2]),
        thorax_centre_model=float(data.xpos[model.body("Thorax").id][2]),
        n_contact=int(data.ncon),
        n_floor_contact=n_floor,
        n_body_floor_contact=len(nonfoot),
        min_floor_distance_model=worst_d,
        tarsus_only=(n_floor > 0 and not nonfoot),
        worst_floor_penetration_model=worst_d,
        contacting_nonfoot_geoms=sorted(set(nonfoot)),
    )
    if raise_on_fail:
        if not facts.tarsus_only:
            raise StanceGateError(
                f"stance gate FAILED (tarsus_only): {facts.n_body_floor_contact} "
                f"non-foot geoms touch the floor: {facts.contacting_nonfoot_geoms[:6]}"
            )
        # a foot may sit at the declared 0.05 mm penetration; BODY geoms may not
        body_pen = min((d for d, g in dists if g not in tarsus), default=0.0)
        if body_pen < -0.05e-1:
            raise StanceGateError(
                f"stance gate FAILED (buried body): worst non-foot geom is at "
                f"{model_to_mm(body_pen):+.3f} mm from the floor plane - the body is "
                f"inside the floor, so any 'collapse' measured from here is the floor "
                f"ejecting it, not the fly falling"
            )
    return facts


# ---------------------------------------------------------------------------
# Result contract.  Every artifact written by this round carries these fields.
# ---------------------------------------------------------------------------
def write_result(path, *, question, status, measured, provenance,
                 what_this_does_not_support, **extra) -> Path:
    assert status in ("PASS", "FAIL", "NOT_RUN"), f"bad status {status!r}"
    doc = {
        "status": status,
        "question": question,
        "measured": measured,
        "provenance": provenance,
        "what_this_does_not_support": what_this_does_not_support,
        "units": {
            "length_model": MODEL_LENGTH_UNIT_NAME,
            "length_reported": "mm",
            "mm_per_model_length": MM_PER_MODEL_LENGTH,
            "mass_model": "INFERRED gram (NOT established): the asset declares no mass unit "
                          "and no density; see MODEL_MASS_UNIT_STATUS in canonical_physics",
            "mass_model_status": MODEL_MASS_UNIT_STATUS,
            "mass_unit_kg_inferred": MODEL_MASS_UNIT_KG,
            "time": "s",
            "gravity_model": GRAVITY_MODEL,
        },
        "illustrative_vs_measured": "measured simulator output; TRANSLATION PARAMETERS, "
                                    "ALIASES and the mass-unit inference are DECLARED "
                                    "assumptions, not measurements",
    }
    doc.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return path
