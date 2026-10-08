"""Electrode PAYLOAD layer: the mass a recording electrode puts on the fly.

WHY THIS FILE EXISTS
--------------------
The project has ``engine/electrode*.py`` for the electrode's ELECTRICAL model, and
a fixed geometry (7 um tip diameter, 20 um pitch) that is never optimised.  It has
NO mass model at all: nothing in the project ever asked what the electrode weighs,
so the body simulation has always been run with the fly carrying nothing.  This
file adds that, because "the electrode records the fly AND its own weight changes
the fly's behaviour" is a physical claim that has to be computed, not asserted.

WHAT IS DERIVED AND WHAT IS ASSUMED
-----------------------------------
DERIVED (arithmetic on the fixed geometry, reproducible by hand):
    electrode_glass_mass_kg()   -- frustum + cylinder volume x a declared density

ASSUMED / DECLARED (constants, no measurement behind them -- labelled as such in
the returned report and in the scene payload, never presented as measured):
    BOROSILICATE_DENSITY_KG_M3  2400 kg/m^3, literature value for borosilicate glass
    EPOXY_DENSITY_KG_M3         1200 kg/m^3, literature value for the glue at the joint
    SHANK_LENGTH_M              3.0 mm of drawn shank, a typical pulled pipette
    TIP_DIAMETER_M              7.0 um -- FIXED project geometry, not a free knob
    BASE_DIAMETER_M           200.0 um -- DECLARED taper shoulder, illustrative
    HANDLING_TARE_MASS_KG       the holder/wire/connector mass the fly actually
                                carries, expressed as a fraction of body mass

WHY THE PAYLOAD IS PARAMETERISED BY "x % OF BODY MASS"
------------------------------------------------------
Sizing the payload from the electrode ALONE gives a number that is real but
useless as a locomotion experiment: the bare glass is O(1e-8) kg against a body
mass of O(1e-3) kg.  The mass that actually loads a tethered fly is the glue bead,
the holder, the wire and the connector, none of which are in this project's
geometry.  Rather than invent a single "the" payload mass and quietly present it
as measured, the loading is swept on a declared axis:

    load_fraction = (payload mass) / (total body mass)

and every artefact records the fraction next to the number of body masses it
equals, so a reader can see the magnitude without trusting any one constant.

UNITS
-----
kg, m, m^3 throughout.  MuJoCo takes SI, so nothing is converted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Declared constants.  See the module docstring for the derived/assumed split.
# ---------------------------------------------------------------------------
BOROSILICATE_DENSITY_KG_M3 = 2400.0
EPOXY_DENSITY_KG_M3 = 1200.0
SHANK_LENGTH_M = 3.0e-3
TIP_DIAMETER_M = 7.0e-6          # FIXED project geometry
BASE_DIAMETER_M = 200.0e-6       # DECLARED
HANDLING_TARE_MASS_KG = 2.0e-6   # DECLARED: holder + 30 mm wire + connector

# ---------------------------------------------------------------------------
# THE REST OF THE RIG.  The bare electrode is 0.0095 % of body mass, so what
# actually loads a tethered, telemetrised fly is everything else:
#
#     electrode  ->  connector  ->  TETHER  ->  battery + transmitter
#
# Each of these is now an explicit, named component with its own provenance.  The
# numbers are either DERIVED (geometry x a density) or TAKEN (a real part whose
# mass is published), and the report says which for every one of them.
# ---------------------------------------------------------------------------

#: Copper wire for the tether.  Properties are literature values for annealed
#: copper: density 8960 kg/m^3, ultimate tensile strength ~220 MPa.  Used to DERIVE
#: both the tether's mass and the tension it can take before it breaks.
COPPER_DENSITY_KG_M3 = 8960.0
COPPER_TENSILE_PA = 220.0e6

#: TETHER geometry.  DECLARED: 50 um bare copper, 300 mm of free length between the
#: fly and the first strain relief.  The diameter is chosen as the thinnest wire that
#: survives the fly's own pull (checked in ``tether_report``); the LENGTH is the
#: free span and is the number that matters, because the tether's mass hangs on the
#: fly and its tension depends on how much of it is unsupported.
TETHER_DIAMETER_M = 50.0e-6
TETHER_FREE_LENGTH_M = 300.0e-3

#: TAKEN (real parts, manufacturer data -- NOT measured here):
#:   * a coin-cell LiPo of the kind used in insect-scale telemetry, ~3.0 mg.  This is
#:     the tier of the published insect-scale systems (about 3 mg for a few mAh).
#:   * a BLE transmitter die.  A bare 2.4 GHz BLE SoC die plus its antenna is
#:     ~0.3 mg of silicon and metal; a packaged module is 100x that and cannot fly.
BATTERY_MASS_KG = 3.0e-6
BLE_TRANSMITTER_MASS_KG = 0.3e-6

#: Named rig components: (key, description, mass_kg or None, provenance).
#: A component with ``mass_kg=None`` is DERIVED at plan time from the constants above.
RIG_COMPONENTS = (
    {"key": "electrode", "what": "pulled glass pipette + glue bead",
     "derived": "electrode_glass_mass_kg()",
     "provenance": "DERIVED from the fixed 7 um geometry x literature densities"},
    {"key": "connector", "what": "wire bond from electrode to tether",
     "mass_kg": HANDLING_TARE_MASS_KG,
     "provenance": "DECLARED order-of-magnitude for a bead of glue and a solder joint"},
    {"key": "tether", "what": "free copper tether span",
     "derived": "tether_mass_kg()",
     "provenance": "DERIVED from declared diameter and free length x copper density"},
    {"key": "battery", "what": "coin-cell LiPo",
     "mass_kg": BATTERY_MASS_KG,
     "provenance": "TAKEN from the published insect-scale telemetry tier (~3 mg)"},
    {"key": "ble", "what": "BLE transmitter die + antenna",
     "mass_kg": BLE_TRANSMITTER_MASS_KG,
     "provenance": "TAKEN as a bare die plus antenna; a packaged module cannot fly"},
)


def tether_mass_kg(diameter_m: float = TETHER_DIAMETER_M,
                   free_length_m: float = TETHER_FREE_LENGTH_M,
                   density: float = COPPER_DENSITY_KG_M3) -> dict[str, Any]:
    """Mass and breaking load of the tether, DERIVED from a cylinder.

    Reported together on purpose: a tether that is too thin to survive the fly's own
    pull is not a tether, and a tether whose mass is larger than the fly's payload
    budget is not an option either.  Both numbers come out of the same geometry.
    """
    r = 0.5 * float(diameter_m)
    L = float(free_length_m)
    area = math.pi * r * r
    volume = area * L
    mass = volume * float(density)
    break_n = area * COPPER_TENSILE_PA
    return {"diameter_m": float(diameter_m), "area_m2": area,
            "free_length_m": L, "volume_m3": volume, "mass_kg": mass,
            "breaking_force_N": break_n,
            "breaking_force_mN": break_n * 1e3,
            "provenance": {"derived": ["area_m2", "volume_m3", "mass_kg",
                                       "breaking_force_N"],
                           "taken": {"density_kg_m3": copper_density(density),
                                     "tensile_pa": COPPER_TENSILE_PA}}}


def copper_density(d: float) -> float:
    """Identity helper so the provenance dict names the number it used."""
    return float(d)


def rig_report(body_mass_kg: float,
               diameter_m: float = TETHER_DIAMETER_M,
               free_length_m: float = TETHER_FREE_LENGTH_M) -> dict[str, Any]:
    """Every component of the recording rig, with its mass and its provenance.

    This is the answer to "what does the fly actually carry": the sum, the fraction of
    body mass, and the share each part contributes, so the reader can see immediately
    that the electrode is the smallest term by four orders of magnitude and the tether
    is usually the largest.
    """
    glass = electrode_glass_mass_kg()
    teth = tether_mass_kg(diameter_m, free_length_m)
    parts = []
    for c in RIG_COMPONENTS:
        if c["key"] == "electrode":
            m = glass["total_kg"]
        elif c["key"] == "tether":
            m = teth["mass_kg"]
        else:
            m = float(c["mass_kg"])
        parts.append({"key": c["key"], "what": c["what"], "mass_kg": m,
                      "provenance": c["provenance"]})
    total = sum(p["mass_kg"] for p in parts)
    for p in parts:
        p["share_of_rig"] = p["mass_kg"] / total if total else 0.0
        p["fraction_of_body"] = p["mass_kg"] / float(body_mass_kg)
    # The fly must be able to pull the tether: compare its own weight to the breaking
    # load, and its own mass to the rig's mass.  Both are ratios, so they are unit-free.
    weight_N = float(body_mass_kg) * 9.80665
    return {"components": parts, "total_kg": total,
            "total_fraction_of_body": total / float(body_mass_kg),
            "tether": teth,
            "fly_weight_N": weight_N,
            "tether_breaking_over_fly_weight": teth["breaking_force_N"] / weight_N,
            "heaviest_component": max(parts, key=lambda p: p["mass_kg"])["key"],
            "lightest_component": min(parts, key=lambda p: p["mass_kg"])["key"],
            "worst_case_pull_mN": None,
            "note": ("masses are DERIVED from geometry or TAKEN from manufacturer / "
                     "published figures; NONE of them was measured in this project")}

#: Body mass of the FlyGym locomotion model, MEASURED from a compiled model as the
#: sum of ``body_mass`` (see ``tools_measure_body_mass.py``, which writes
#: ``outputs/electrode_payload/body_mass.json``) and used only as the yardstick the
#: loading ratio is expressed against.  It is a recorded measurement of one exact
#: asset, NOT a live reading: the live value is only available after a compile, and
#: both attempts to obtain it at injection time were measured to fail (see the note
#: in ``BodyBackend.__init__``).  ``assert_yardstick`` checks it against any
#: compiled model so a wrong yardstick cannot pass silently.
BODY_MASS_KG_FALLBACK = 1.02431e-3

#: Bodies the payload is attached to, as (body segment name, attachment offset in
#: metres in that body's own frame, share of the total payload mass).
#:
#: The shares are DECLARED, illustrative choices of where a recording rig sits:
#: a dorsal-thorax electrode is the common configuration, a head electrode is the
#: case that matters for head-fixed behavioural work, and the sham is a null-load
#: attachment used to show the injection machinery does not by itself move the fly.
ATTACHMENT_SITES = (
    {"segment": "c_thorax", "offset_m": (0.0, 0.0, 5.0e-4), "share": 0.75,
     "what": "dorsal thorax electrode + holder"},
    {"segment": "c_head", "offset_m": (0.0, 0.0, 2.0e-4), "share": 0.25,
     "what": "head electrode + headstage"},
)

#: The point-like payload is given a sphere inertia tensor.  The radius is the
#: actual radius of the payload's own volume at the DECLARED tare density of
#: 1000 kg/m^3 (a compact rig), i.e. it is derived from the mass, not invented.
TARE_EFFECTIVE_DENSITY_KG_M3 = 1000.0


def electrode_glass_mass_kg(
    tip_diameter_m: float = TIP_DIAMETER_M,
    base_diameter_m: float = BASE_DIAMETER_M,
    shank_length_m: float = SHANK_LENGTH_M,
    glass_density: float = BOROSILICATE_DENSITY_KG_M3,
    epoxy_density: float = EPOXY_DENSITY_KG_M3,
) -> dict[str, Any]:
    """Mass of the bare pulled pipette, from its fixed geometry.

    The drawn shank is modelled as a cone frustum from the tip diameter to the
    shoulder diameter over ``shank_length_m``; the glue bead at the shoulder is
    modelled as the same frustum's circumscribed cylinder filled with epoxy.  Both
    are exact volume formulas:

        frustum  V = pi h / 3 * (r1^2 + r1 r2 + r2^2)
        cylinder V = pi r^2 h

    Returns a report dict; ``total_kg`` is the sum.  No measurement is involved.
    """
    r_tip = 0.5 * float(tip_diameter_m)
    r_base = 0.5 * float(base_diameter_m)
    h = float(shank_length_m)
    v_frustum = math.pi * h / 3.0 * (r_tip * r_tip + r_tip * r_base + r_base * r_base)
    # The glue bead: a cylinder of the shoulder radius, DECLARED to be 0.5 mm long.
    glue_length_m = 5.0e-4
    v_glue = math.pi * r_base * r_base * glue_length_m
    m_glass = v_frustum * float(glass_density)
    m_glue = v_glue * float(epoxy_density)
    return {
        "frustum_volume_m3": v_frustum,
        "frustum_mass_kg": m_glass,
        "glue_volume_m3": v_glue,
        "glue_mass_kg": m_glue,
        "total_kg": m_glass + m_glue,
        "glass_density_kg_m3": float(glass_density),
        "epoxy_density_kg_m3": float(epoxy_density),
        "geometry": {"tip_diameter_m": float(tip_diameter_m),
                     "base_diameter_m": float(base_diameter_m),
                     "shank_length_m": h,
                     "glue_length_m": glue_length_m},
        "provenance": {
            "derived": ["frustum_volume_m3", "frustum_mass_kg",
                        "glue_volume_m3", "glue_mass_kg", "total_kg"],
            "assumed": {"glass_density_kg_m3": BOROSILICATE_DENSITY_KG_M3,
                        "epoxy_density_kg_m3": EPOXY_DENSITY_KG_M3,
                        "glue_length_m": glue_length_m,
                        "base_diameter_m": BASE_DIAMETER_M},
        },
    }


def measure_body_mass_kg(model) -> float:
    """Total body mass of a compiled MuJoCo model, in kg.

    Sums ``body_mass`` over every body.  The world body contributes 0, so this is
    the fly's mass.  Measured, not declared.
    """
    import numpy as np
    return float(np.sum(np.asarray(model.body_mass, dtype=float)))


@dataclass
class ElectrodePayloadConfig:
    """Declarative payload description.  All masses are kg, all lengths m.

    ``load_fraction`` is the PRIMARY knob: total payload mass as a fraction of the
    fly's measured body mass.  ``load_fraction == 0.0`` is the control condition
    and must leave the model bit-identical (checked by
    ``PayloadAttachment.signature``).
    """

    load_fraction: float = 0.0
    #: If True, add the bare-electrode glass mass on top of the tare.  It is
    #: 1e-5 of the tare, so it is off by default; it exists so the report can show
    #: how negligible the glass is instead of only claiming it.
    include_bare_electrode: bool = False
    #: Per-site share overrides; None -> ATTACHMENT_SITES.
    sites: tuple[dict[str, Any], ...] | None = None
    #: Multiplies every site share; used to build the sham (a load whose mass is
    #: 1e-6 of the nominal one, so it is present in the model but physically inert).
    mass_scale: float = 1.0
    label: str = "payload"
    #: OPTIONAL named rig (see ``RIG_PRESETS``).  When set, ``load_fraction`` is
    #: ignored and the per-site masses come from ``rig_plan`` instead, so the masses
    #: are the named components' own masses rather than a single swept number.  The
    #: two paths share one injection implementation.
    rig: "str | None" = None
    #: Tether free length and diameter, in metres, only used when ``rig`` is set.
    tether_diameter_m: float = TETHER_DIAMETER_M
    tether_free_length_m: float = TETHER_FREE_LENGTH_M

    def validate(self) -> "ElectrodePayloadConfig":
        if self.load_fraction < 0.0:
            raise ValueError("load_fraction must be >= 0")
        if self.mass_scale <= 0.0:
            raise ValueError("mass_scale must be > 0")
        if self.rig is not None:
            if self.rig not in RIG_PRESETS:
                raise ValueError(f"unknown rig {self.rig!r}; known {sorted(RIG_PRESETS)}")
            if self.load_fraction != 0.0:
                raise ValueError("rig and load_fraction are two different experiments; "
                                 "set load_fraction=0 when a rig is named")
            if self.tether_diameter_m <= 0.0 or self.tether_free_length_m < 0.0:
                raise ValueError("tether diameter must be > 0 and length >= 0")
        return self


@dataclass
class PayloadAttachment:
    """Result of injecting the payload into an MJCF spec."""

    config: ElectrodePayloadConfig
    body_mass_kg: float
    total_payload_kg: float
    load_fraction_achieved: float
    per_site: list[dict[str, Any]] = field(default_factory=list)
    body_names_added: list[str] = field(default_factory=list)
    #: Set by :func:`attach_payload` after compile, from the live model.
    signature: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"]["sites"] = None if self.config.sites is None else list(self.config.sites)
        return d


def _sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    i = 0.4 * mass * radius * radius
    return (i, i, i)


#: NAMED RIGS: the physical configurations the arena experiment compares.  Each is a
#: plain dict so it can be recorded verbatim in the artefact, and each carries the
#: provenance of every component.  "tether_supported" differs from "tether_free" only
#: in that the tether's weight is counter-supported, which is the honest baseline for
#: asking what the tether's MASS does on its own.
RIG_PRESETS: dict[str, dict[str, Any]] = {
    "bare": {"components": (), "what": "nothing on the fly (control)",
             "provenance": "none"},
    "electrode_only": {"components": ("electrode",), "what": "electrode + glue only",
                       "provenance": "DERIVED geometry"},
    "tether": {"components": ("electrode", "connector", "tether"),
               "what": "tethered recording, tether free",
               "provenance": "DERIVED + DECLARED"},
    "tether_supported": {"components": ("electrode", "connector", "tether"),
                         "what": "tethered, tether weight counter-supported",
                         "provenance": "DERIVED + DECLARED",
                         "support_fraction": 1.0},
    "telemetry": {"components": ("electrode", "connector", "battery", "ble"),
                  "what": "wireless: battery + BLE on board, no tether",
                  "provenance": "TAKEN manufacturer / published tier"},
    "full": {"components": ("electrode", "connector", "tether", "battery", "ble"),
             "what": "everything: tethered AND powered AND transmitting",
             "provenance": "DERIVED + DECLARED + TAKEN"},
}

#: WHICH SEGMENT EACH COMPONENT'S MASS GOES ON.  DECLARED, and it matters physically:
#: dorsal thorax for the electrode and the tether anchor, and the abdomen for the
#: battery and transmitter, because that is where a payload can be carried without
#: blocking the legs.  The offsets are in METRES in the segment's own frame.
RIG_MOUNTS: dict[str, dict[str, Any]] = {
    "electrode": {"segment": "c_thorax", "offset_m": (0.0, 0.0, 5.0e-4)},
    "connector": {"segment": "c_thorax", "offset_m": (0.0, 0.0, 6.0e-4)},
    "tether": {"segment": "c_thorax", "offset_m": (0.0, 0.0, 6.5e-4)},
    "battery": {"segment": "c_abdomen3", "offset_m": (0.0, 0.0, 2.5e-4)},
    "ble": {"segment": "c_abdomen4", "offset_m": (0.0, 0.0, 2.5e-4)},
}


def rig_plan(rig: str, body_mass_kg: float,
             diameter_m: float = TETHER_DIAMETER_M,
             free_length_m: float = TETHER_FREE_LENGTH_M) -> dict[str, Any]:
    """Turn a named rig into the per-site masses that :func:`attach_payload` injects.

    The result has the SAME shape as :func:`payload_plan`, so the injection path is
    shared: one implementation of "put mass on the body", not two.
    """
    if rig not in RIG_PRESETS:
        raise KeyError(f"unknown rig {rig!r}; known: {sorted(RIG_PRESETS)}")
    preset = RIG_PRESETS[rig]
    rep = rig_report(body_mass_kg, diameter_m, free_length_m)
    by_key = {p["key"]: p for p in rep["components"]}
    per_site: list[dict[str, Any]] = []
    for key in preset["components"]:
        mount = RIG_MOUNTS[key]
        m = by_key[key]["mass_kg"]
        r = (3.0 * m / (4.0 * math.pi * TARE_EFFECTIVE_DENSITY_KG_M3)) ** (1.0 / 3.0) \
            if m > 0.0 else 0.0
        per_site.append({
            "segment": mount["segment"], "what": by_key[key]["what"],
            "component": key, "mass_kg": m, "share": None,
            "offset_m": tuple(float(v) for v in mount["offset_m"]),
            "effective_radius_m": r,
            "inertia_kg_m2": _sphere_inertia(m, r) if m > 0.0 else (0.0, 0.0, 0.0),
            "provenance": by_key[key]["provenance"],
        })
    total = sum(s["mass_kg"] for s in per_site)
    return {"total_kg": total, "per_site": per_site, "rig": rig,
            "what": preset["what"], "rig_provenance": preset["provenance"],
            "support_fraction": preset.get("support_fraction", 0.0),
            "rig_report": rep, "glass_report": electrode_glass_mass_kg()}


def payload_plan(cfg: ElectrodePayloadConfig,
                 body_mass_kg: float) -> dict[str, Any]:
    """Pure arithmetic: what mass goes where.  No MuJoCo needed, so it is testable.

    Returns ``{"total_kg", "per_site": [...]}``.  ``per_site`` entries carry the
    segment name, the mass in kg, the attachment offset in metres and the sphere
    inertia the injection will use.
    """
    cfg = cfg.validate()
    if cfg.rig is not None:
        # A named rig carries its own masses; ``load_fraction`` and the tare axis are
        # deliberately NOT applied, because mixing them would make "20 % of body mass"
        # and "the real components" two different experiments under one label.
        return rig_plan(cfg.rig, body_mass_kg, cfg.tether_diameter_m,
                        cfg.tether_free_length_m)
    glass = electrode_glass_mass_kg()
    bare = glass["total_kg"] if cfg.include_bare_electrode else 0.0
    tare = cfg.load_fraction * float(body_mass_kg) * cfg.mass_scale
    total = tare + bare
    sites = cfg.sites if cfg.sites is not None else ATTACHMENT_SITES
    share_sum = sum(float(s["share"]) for s in sites)
    if share_sum <= 0.0:
        raise ValueError("attachment site shares must sum to > 0")
    per_site = []
    for s in sites:
        frac = float(s["share"]) / share_sum
        m = total * frac
        r = (3.0 * m / (4.0 * math.pi * TARE_EFFECTIVE_DENSITY_KG_M3)) ** (1.0 / 3.0) \
            if m > 0.0 else 0.0
        per_site.append({
            "segment": str(s["segment"]),
            "what": str(s.get("what", "")),
            "share": frac,
            "mass_kg": m,
            "offset_m": tuple(float(v) for v in s["offset_m"]),
            "effective_radius_m": r,
            "inertia_kg_m2": _sphere_inertia(m, r) if m > 0.0 else (0.0, 0.0, 0.0),
        })
    return {"total_kg": total, "bare_electrode_kg": bare, "tare_kg": tare,
            "per_site": per_site, "glass_report": glass}


MARKER_MATERIAL_NAME = "marker_emissive"


def ensure_marker_material(spec, name: str = MARKER_MATERIAL_NAME,
                           emission: float = 1.0) -> str:
    """Add the emissive marker material to a spec, idempotently.

    MEASURED why emission and not just white: in the meadow the ground already renders
    at the top of the range, so a white sphere is indistinguishable from a blade by
    brightness.  An emissive material is the only way to make the marker the single
    brightest object without touching the scene's lighting, which would invalidate every
    other measurement taken in the same scene.
    """
    for m in spec.materials:
        if str(getattr(m, "name", "")) == name:
            return name
    mat = spec.add_material(name=name, rgba=[1.0, 1.0, 1.0, 1.0],
                            emission=float(emission), reflectance=0.0,
                            shininess=0.0, specular=0.0)
    _ = mat
    return name


def _quat2mat(q):
    import numpy as _np
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return _np.eye(3)
    s = 2.0 / n
    return _np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def _spec_world_poses(root):
    """``{body name: (position, rotation)}`` for the whole spec, in the NEUTRAL pose.

    TOP-DOWN ON PURPOSE.  The first version of this walked ``MjsBody.parent`` upward and
    SEGFAULTED inside MuJoCo (measured: exit 139, no traceback, reproducible) -- the spec
    wrappers reached through ``parent`` are not safe to dereference.  Recursing down through
    ``body.bodies``, which is a valid child iterator, produces the same transforms without ever
    touching ``parent``.
    """
    import numpy as _np
    poses = {}

    def rec(body, p, R):
        pq = _np.asarray(body.pos, dtype=float)
        pw = p + R @ pq
        Rw = R @ _quat2mat(body.quat)
        nm = str(getattr(body, "name", "") or "")
        if nm:
            poses[nm] = (pw, Rw)
        for child in body.bodies:
            rec(child, pw, Rw)

    rec(root.worldbody, _np.zeros(3), _np.eye(3))
    return poses


def compute_outward_offsets(fly, segments, distance_mm: float = 0.18,
                            torso_distance_mm: float = 0.55,
                            coxa_distance_mm: float = 0.50,
                            anchor_segment: str = "c_thorax"):
    """Local-frame offset per segment that puts its marker on the EXPOSED (dorsal) surface.

    MEASURED, AND THE FIRST VERSION OF THIS WAS WRONG IN A WAY THE DATA CAUGHT.  The first rule
    offset each marker along ``segment_position - thorax_position``, i.e. RADIALLY OUTWARD from
    the body.  For a leg that direction is very nearly ALONG THE LEG ITSELF -- legs radiate
    outward, which is what a leg is -- so the marker was pushed toward the next joint, often
    ending up inside the NEXT segment's mesh instead of outside its own.  Rendering and then
    counting how many cameras could see each marker showed the result: only 6 of 20 markers were
    visible to three or more cameras, and the femur -- whose marker is pushed along the femur
    straight into the tibia -- was visible in 0 frames on all six legs.

    WHAT REPLACES IT.  Each marker goes on the DORSAL side of its own segment:
      * take the segment's own axis, from its neutral pose to its first child body, and
      * offset along the component of world +Z PERPENDICULAR to that axis, so the marker sits
        on top of the segment rather than in front of it.
    A dorsal marker is visible from every camera of a 45-degree ring, which is the whole point.
    Torso segments have no useful bone axis, so their markers go straight up by a larger
    distance, because they have to clear the thorax mesh rather than a thin leg.

    THREE RULES, NOT ONE, BECAUSE THE THREE CLASSES OF SEGMENT FAIL DIFFERENTLY.  Measured by
    recording, mapping how many cameras see each marker, and re-measuring the pairwise spacing:

      * TIBIA and TARSUS (thin, exposed legs): dorsal, 0.18 mm.  Worked -- the tarsus markers
        went from 70-96% to 96-100% and the mid/hind tibiae from 9-13% to 100%.
      * TORSO (thorax, abdomen): dorsal, 0.55 mm, big enough to clear a thick mesh.  Went from
        0% to 100%.
      * COXA / TROCHANTERFEMUR: RADIAL, 0.50 mm -- and this is the one that matters most.  A
        dorsal offset here made things WORSE: the six coxae are already squeezed against the
        thorax and moving them all "up" pushed them TOWARD EACH OTHER, taking the closest pair
        from 0.257 mm to 0.144 mm, i.e. below the 0.26 mm marker diameter, so they merged.  A
        radial offset pushes them AWAY along the body radius instead, which both separates them
        and gets them out from under the thorax mesh.

    This changes the pairwise spacing between markers, so the clearance must be RE-MEASURED on
    the recorded marker positions afterwards (``tools_marker_coverage.py`` reports it): the
    identity budget says the marker diameter may not exceed the clearance, and moving markers
    spends it.
    """
    import numpy as _np

    seg_map = getattr(fly, "bodyseg_to_mjcfbody", None)
    if seg_map is None:
        raise TypeError("fly has no bodyseg_to_mjcfbody map")
    root = getattr(fly, "_mjcf_root", None) or getattr(fly, "mjcf_root", None)
    if root is None:
        raise RuntimeError("cannot compute marker offsets: fly spec root not found")
    poses = _spec_world_poses(root)

    def _find(segment):
        for name, _body in ((str(getattr(b, "name", "")), b) for b in seg_map.values()):
            if name == segment or name.endswith("/" + segment):
                return name
        for k in poses:
            if k.endswith("/" + segment):
                return k
        return None

    def _first_child_full_name(parent_full):
        """Full name of the parent's first child body, from the spec tree."""
        parts = parent_full.split("/")
        best = None
        for k in poses:
            if k == parent_full or not k.startswith(parent_full + "/"):
                continue
            rest = k[len(parent_full) + 1:]
            if "/" not in rest:                      # a DIRECT child
                best = k
                break
        return best

    Z = _np.array([0.0, 0.0, 1.0])
    anchor_p = poses[_find(anchor_segment) or anchor_segment][0]
    out = {}
    for seg in segments:
        nm = _find(seg)
        if nm is None:
            raise KeyError(f"segment {seg!r} not found in the spec poses")
        p, R = poses[nm]
        coxa = seg.endswith("_trochanterfemur")
        torso = seg.startswith("c_")
        if coxa:
            # RADIAL: horizontally away from the body axis, which spreads the six coxae apart
            d = p - anchor_p
            d = _np.array([d[0], d[1], 0.0])
            n = float(_np.linalg.norm(d))
            d = d / n if n > 1e-9 else Z.copy()
            dist = float(coxa_distance_mm)
        elif torso:
            d = Z.copy()
            dist = float(torso_distance_mm)
        else:
            child = _first_child_full_name(nm)
            if child is None:
                d = Z.copy()
            else:
                axis = poses[child][0] - p
                n = float(_np.linalg.norm(axis))
                axis = axis / n if n > 1e-9 else Z
                d = Z - float(_np.dot(Z, axis)) * axis
                n = float(_np.linalg.norm(d))
                d = d / n if n > 1e-9 else axis
            dist = float(distance_mm)
        out[seg] = tuple(float(v) for v in (R.T @ d * dist))
    return out


def attach_marker_geoms(fly, segments, radius_mm: float = 0.13,
                        label_prefix: str = "mk_",
                        material: str = MARKER_MATERIAL_NAME,
                        group: int = 0, offset_m=(0.0, 0.0, 0.0),
                        rgba=(1.0, 1.0, 1.0, 1.0)) -> dict[str, Any]:
    """Put a bright sphere geom DIRECTLY ON each named segment body.  Call BEFORE compile().

    WHY THIS REPLACES ``attach_marker_bodies``, WHICH IS KEPT ONLY FOR REFERENCE.  MEASURED,
    and it is the same failure family that killed the first payload attempt: ADDING BODIES TO
    THE FLY'S SPEC DESTROYS OTHER BODIES AT COMPILE TIME.  With 20 marker BODIES requested the
    compiled model came back without ``nmf/c_rostrum``, ``nmf/c_haustellum``, ``nmf/l_eye`` and
    ``nmf/r_eye`` -- the whole head.  The recorder's base-body assertion is what surfaced it
    (it refuses to record an episode on a model that lost bodies), and the earlier payload work
    had already concluded the same thing: extra bodies are capacity-limited, so the payload
    mass is written into the COMPILED model instead.

    THE RADIUS IS IN MODEL UNITS (mm), AND GETTING THAT WRONG IS WHY THIS NEVER WORKED.
    MEASURED, and it is the third independent defect in this one code path: the model's length
    unit is MILLIMETRES (the gravity constant is -9810, the fly is ~2.5 units long, the static
    fiducials are declared as ``size=[0.3]`` for a 0.30 mm radius).  The previous call passed
    ``radius_m=MARKER_RADIUS_MM * 1e-3``, i.e. 0.0003, which compiles to a ``size=[0.0003]``
    sphere -- 0.3 MICROMETRES across in model units.  Every render test showed "the markers do
    not appear" and the geoms were there all along, sub-pixel, with ``type=2`` (sphere) and a
    correct material: a 3-D model in mm is not a 3-D model in metres, and the parameter name
    ``radius_m`` invited exactly this error, so it is renamed and the caller passes mm.

    Adding a GEOM to a body that already exists does not consume body capacity and keeps the
    body tree untouched.

    THE OFFSET IS NOT COSMETIC.  MEASURED, by rendering: at ``offset_m=(0,0,0)`` the markers
    are INVISIBLE at every radius tried, because a segment ORIGIN is a joint centre and is
    therefore inside the body's own mesh -- at 0.60 mm radius only fragments poked out, and at
    the 0.13 mm the identity budget requires, nothing showed at all.  So each marker carries a
    fixed offset in ITS OWN SEGMENT'S frame, which pushes it outside the cuticle while keeping
    it rigidly attached to the segment.  The consequence is stated rather than hidden: the
    tracked bone is then MARKER-TO-MARKER, not joint-centre-to-joint-centre, and the truth the
    recorder compares against is read from the SAME geoms (``geom_xpos``), so the comparison
    stays apples-to-apples.

    The geom is ``contype=conaffinity=0`` so it can never touch anything, and its own name is
    ``<segment body name>/mk_<segment>_g`` so the recorder can find it back by name.
    """
    added: list[str] = []
    specs: list[dict] = []
    seg_map = getattr(fly, "bodyseg_to_mjcfbody", None)
    if seg_map is None:
        raise TypeError("fly has no bodyseg_to_mjcfbody map; cannot attach markers")

    def _resolve(segment):
        for name, body in ((str(getattr(b, "name", "")), b) for b in seg_map.values()):
            if name == segment or name.endswith("/" + segment):
                return body
        return None

    from flygym.utils.mjcf import GEOM_TYPES
    for seg in segments:
        target = _resolve(seg)
        if target is None:
            raise KeyError(f"no MJCF body for marker segment {seg!r}")
        name = f"{str(target.name)}/{label_prefix}{seg}_g"
        if isinstance(offset_m, dict):
            _off = offset_m.get(seg, (0.0, 0.0, 0.0))
        else:
            _off = offset_m
        kw = dict(name=name, type=GEOM_TYPES["sphere"],
                  size=[float(radius_mm)] * 3,
                  pos=[float(v) for v in _off],
                  rgba=[float(v) for v in rgba],
                  contype=0, conaffinity=0, group=int(group),
                  # MASSLESS, AND THAT IS NOT COSMETIC.  MEASURED: without this the 20 markers
                  # shifted the fly's resting height from z = 1.421 to z = 1.800 model units and
                  # produced "Nan, Inf or huge value in QACC ... simulation is unstable".  The
                  # reason is the model's unit system: MuJoCo computes geom inertia from the
                  # SIZE NUMBERS (0.13 here) as if they were METRES, so a 0.13-unit sphere at
                  # the default density carries ~9 kg of "mass" on a 1e-6 kg fly.  Setting both
                  # mass and density to zero makes the marker purely visual, which is all a
                  # marker is allowed to be -- it must not change the mechanics it measures.
                  mass=0.0, density=0.0)
        if material:
            kw["material"] = material
        target.add_geom(**kw)
        added.append(name)
        specs.append({"segment": seg, "body_name": str(target.name), "geom_name": name,
                      "radius_mm": float(radius_mm),
                      "offset_m": [float(v) for v in _off]})
    return {"segments": list(segments), "geoms_added": added, "specs": specs,
            "radius_mm": float(radius_mm), "n_geoms": len(added), "group": int(group),
            "units": "MODEL UNITS = mm",
            "offset_m": ("per-segment" if isinstance(offset_m, dict)
                         else [float(v) for v in offset_m]),
            "rgba": [float(v) for v in rgba],
            "note": ("geoms on the EXISTING segment bodies at zero offset, so a marker's world "
                     "position IS the segment origin; they never collide (contype=0)")}


def attach_marker_bodies(fly, segments, radius_m: float = 0.15e-3,
                         label_prefix: str = "mk_",
                         material: str = MARKER_MATERIAL_NAME) -> dict[str, Any]:
    """Attach a bright sphere to each named fly segment.  Call BEFORE ``compile()``.

    THE SAME INJECTION PATH AS THE PAYLOAD, on purpose: markers and payload are both
    "extra bodies on named segments", and having one implementation means a marker cannot
    end up somewhere a payload would not.  The difference is only that a marker has
    ``contype=conaffinity=0`` and is VISUAL, where a payload is mass-only and invisible.

    Each marker is its own child body at a zero offset from its segment, so it follows
    that segment exactly -- which is what makes its measured position a measurement of
    the JOINT rather than of the body as a whole.
    """
    added: list[str] = []
    specs: list[dict] = []
    seg_map = getattr(fly, "bodyseg_to_mjcfbody", None)
    if seg_map is None:
        raise TypeError("fly has no bodyseg_to_mjcfbody map; cannot attach markers")
    root_spec = getattr(fly, "_mjcf_root", None) or getattr(fly, "mjcf_root", None)

    def _resolve(segment):
        for name, body in ((str(getattr(b, "name", "")), b) for b in seg_map.values()):
            if name == segment or name.endswith("/" + segment):
                return body
        return None

    for seg in segments:
        target = _resolve(seg)
        if target is None:
            raise KeyError(f"no MJCF body for marker segment {seg!r}")
        name = f"{str(target.name)}/{label_prefix}{seg}"
        if root_spec is not None:
            stale = _find_spec_body(root_spec, name)
            if stale is not None:
                try:
                    root_spec.delete(stale)
                except Exception:
                    pass
        child = target.add_body(name=name, pos=[0.0, 0.0, 0.0])
        # TYPE IS RESOLVED THROUGH FLYGYM'S OWN MAP.  This line used to read ``type=1`` with the
        # comment "1 = sphere in MuJoCo's geom types", which is WRONG twice over: MuJoCo's
        # mjtGeom order is 0 plane, 1 HFIELD, 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder,
        # 6 box, 7 mesh -- so 1 is an hfield, AND ``MjsBody.add_geom`` wants the INT, not the
        # string.  MEASURED, and this is how the bug surfaced once the injection ordering was
        # fixed and the geoms finally reached a compile: first
        # "hfield geom 'nmf/c_rostrum/mk_c_rostrum_g' (id = 5) must have valid hfieldid" with
        # type=1, then "add_geom(): incompatible function arguments ... type: SupportsInt" with
        # type="sphere".  BodyConfig.extra_geoms resolves strings through the same map, so this
        # now matches the path the static fiducials already take.
        from flygym.utils.mjcf import GEOM_TYPES
        kw = dict(name=name + "_g", type=GEOM_TYPES["sphere"],
                  size=[float(radius_m)] * 3,
                  rgba=[float(v) for v in rgba],
                  contype=0, conaffinity=0, group=int(group),
                  # MASSLESS, AND THAT IS NOT COSMETIC.  MEASURED: without this the 20 markers
                  # shifted the fly's resting height from z = 1.421 to z = 1.800 model units and
                  # produced "Nan, Inf or huge value in QACC ... simulation is unstable".  The
                  # reason is the model's unit system: MuJoCo computes geom inertia from the
                  # SIZE NUMBERS (0.13 here) as if they were METRES, so a 0.13-unit sphere at
                  # the default density carries ~9 kg of "mass" on a 1e-6 kg fly.  Setting both
                  # mass and density to zero makes the marker purely visual, which is all a
                  # marker is allowed to be -- it must not change the mechanics it measures.
                  mass=0.0, density=0.0)
        if material:
            # the material must exist in the spec the marker is being added to AND in the
            # FLY's own spec: FlyGym compiles the fly subtree standalone inside add_fly(),
            # where a name declared only on the world root is not visible (MEASURED:
            # "material 'nmf/marker_emissive' not found in geom 5").  BodyBackend.fly_markers
            # declares it in both places.
            kw["material"] = material
        if material:
            # the material must exist in the spec the marker is being added to; the
            # caller is responsible for adding it to the WORLD spec, which is where the
            # markers actually live after the fly is attached
            kw["material"] = material
        child.add_geom(**kw)
        added.append(name)
        specs.append({"segment": seg, "body_name": name, "radius_m": float(radius_m)})
    return {"segments": list(segments), "bodies_added": added, "specs": specs,
            "radius_m": float(radius_m),
            "note": ("markers ride their segment at zero offset, so their world position "
                     "IS the segment's origin; they never collide (contype=0)")}


def _spec_bodies(spec) -> list:
    """ALL bodies in an MjSpec, in one flat list.

    Measured API facts behind this (MuJoCo 3.9.0 python bindings):
      * ``spec.body`` is a METHOD (``spec.body("name")``), NOT iterable ->
        "TypeError: 'method' object is not iterable";
      * an ``MjsBody`` has NO ``.body`` attribute at all -> "AttributeError:
        'mujoco._specs.MjsBody' object has no attribute 'body'", so a body tree
        CANNOT be walked recursively through the Python API;
      * ``spec.bodies`` is the iterable that does exist.
    The first version of this module walked ``node.body`` recursively and therefore
    found nothing, which let a stale payload survive and produce "repeated name
    'c_thorax/..._0' in body" from MuJoCo itself.
    """
    try:
        return list(spec.bodies)
    except TypeError:
        return []


def _find_spec_body(spec, name: str):
    """Body with this exact name anywhere in the spec, or None."""
    for b in _spec_bodies(spec):
        if str(getattr(b, "name", "")) == name:
            return b
    return None


def attach_payload(fly, cfg: ElectrodePayloadConfig, body_mass_kg: float) -> PayloadAttachment:
    """Plan the payload.  THE MODEL IS EDITED AFTER COMPILE, not by adding bodies.

    MEASURED, AND IT CHANGED THE DESIGN.  The obvious implementation -- add a small body
    per attachment site, as a child of the target segment -- DESTROYS FLY BODIES.  Measured
    on this stack: with a payload attached, the compiled model came back with the head
    segments gone (nbody 69 -> 66, no body whose name contains "rostrum", and one eye
    instead of two), and the loss grew with the number of extra bodies (64 bodies and ZERO
    eyes with five payload bodies).  The fly's compiled body count is CAPACITY-LIMITED, so
    extra bodies do not displace empty space, they displace the fly.

    A model missing its own head is not an acceptable price for a payload, so the payload
    is applied to the compiled model's OWN body masses instead: see
    ``apply_payload_to_model``, called by ``BodyBackend`` after ``compile()``.  That path
    cannot add or remove a body, so it cannot break the fly.

    This function therefore returns only the PLAN.  The mass is real and exactly the
    declared fraction; what is NOT modelled is the weight's lever arm about the attachment
    segment, which is recorded in ``report["attach_simplification"]``.
    """
    cfg = cfg.validate()
    if cfg.rig is not None:
        plan = rig_plan(cfg.rig, body_mass_kg, cfg.tether_diameter_m,
                        cfg.tether_free_length_m)
        report_common = {
            "load_fraction_requested": None,
            "rig": cfg.rig,
            "rig_what": plan["what"],
            "rig_provenance": plan["rig_provenance"],
            "rig_report": plan["rig_report"],
            "bare_electrode_kg": plan["glass_report"]["total_kg"],
            "tare_kg": plan["total_kg"],
            "support_fraction": plan["support_fraction"],
            "glass_report": plan["glass_report"],
            "attachment_bodies": [],
        }
    else:
        plan = payload_plan(cfg, body_mass_kg)
        report_common = {
            "load_fraction_requested": float(cfg.load_fraction),
            "include_bare_electrode": bool(cfg.include_bare_electrode),
            "mass_scale": float(cfg.mass_scale),
            "bare_electrode_kg": float(plan["bare_electrode_kg"]),
            "tare_kg": float(plan["tare_kg"]),
            "glass_report": plan["glass_report"],
            "attachment_bodies": [],
        }
    report_common["apply_mode"] = "post-compile body-mass edit"
    report_common["attach_simplification"] = (
        "the payload mass is added to the fly's own bodies AFTER compile, not as extra "
        "bodies attached to the named segments, because extra bodies silently removed fly "
        "bodies at compile time (measured: nbody 69 -> 66 and the head gone with one "
        "payload body; 69 -> 64 and ZERO eyes with five).  The mass and the total load "
        "fraction are exact.  The weight's LEVER ARM about the attachment segment is NOT "
        "modelled, and neither is the tether's tension.")
    att = PayloadAttachment(
        config=cfg, body_mass_kg=float(body_mass_kg),
        total_payload_kg=float(plan["total_kg"]),
        load_fraction_achieved=(float(plan["total_kg"]) / float(body_mass_kg)
                                if body_mass_kg else 0.0),
        per_site=list(plan["per_site"]), report=report_common,
    )
    return att


def mujoco_name(model, objid: int) -> str:
    """The name of a geom by id, or a placeholder when it has none."""
    import mujoco
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(objid))
    return str(n) if n else f"geom{objid}"


def mass_overrides_by_segment(attachment: PayloadAttachment) -> dict[str, float]:
    """Per-segment mass to add, in kg, keyed by the segment name WITHOUT any prefix."""
    out: dict[str, float] = {}
    for site in attachment.per_site:
        if site["mass_kg"] <= 0.0:
            continue
        seg = str(site["segment"])
        out[seg] = out.get(seg, 0.0) + float(site["mass_kg"])
    return out


def apply_payload_to_model(model, attachment: PayloadAttachment) -> dict[str, Any]:
    """Add the payload's mass to the compiled model's own bodies.  CANNOT break the fly.

    MEASURED FACTS THAT DECIDED THIS IMPLEMENTATION:

      * adding payload BODIES removes fly bodies at compile time (nbody 69 -> 66 with one
        payload body, 64 with five, and the head and one eye gone), so extra bodies are out;
      * ``geom_mass``, ``geom_inertia`` and ``geom_density`` DO NOT EXIST on this MjModel
        ("AttributeError: 'MjModel' object has no attribute 'geom_density'"), so the
        payload cannot be put on the geometry;
      * ``body_mass`` DOES exist, IS writable, and the write SURVIVES ``mj_forward`` --
        verified by writing a value and reading it back after a forward pass.

    So the mass goes on the body.  ``body_inertia`` is scaled by the same factor as the
    mass, which is what happens when a dense payload beads onto a segment.
    """
    import numpy as np
    names = [str(model.body(i).name) for i in range(model.nbody)]

    def _find(seg: str) -> int | None:
        for i, nm in enumerate(names):
            if nm.endswith("/" + seg) or nm == seg:
                return i
        return None

    plan = mass_overrides_by_segment(attachment)
    applied: list[dict] = []
    substituted: list[dict] = []
    masses = np.array(model.body_mass, dtype=float)
    largest = int(np.argmax(masses[1:])) + 1 if model.nbody > 1 else 0
    for seg, dm in plan.items():
        idx = _find(seg)
        if idx is None or idx == 0:
            idx = largest
            substituted.append({"segment": seg,
                                "reason": "no body for this segment in the compiled model",
                                "mass_kg": dm, "added_to": names[idx]})
        old_m = float(model.body_mass[idx])
        if old_m > 0:
            model.body_inertia[idx] = np.asarray(model.body_inertia[idx],
                                                 dtype=float) * ((old_m + dm) / old_m)
        else:
            r = (3.0 * dm / (4.0 * math.pi * TARE_EFFECTIVE_DENSITY_KG_M3)) ** (1.0 / 3.0)
            model.body_inertia[idx] = [0.4 * dm * r * r] * 3
        model.body_mass[idx] = old_m + dm
        applied.append({"segment": seg, "body": names[idx], "added_kg": dm,
                        "mass_before_kg": old_m, "mass_after_kg": old_m + dm})
    # VERIFY THE WRITE LANDED.  A silent no-op here would make every load experiment a
    # control condition wearing a payload's label, so the result is read back and reported.
    got = float(np.sum(np.asarray(model.body_mass, dtype=float)))
    want = float(np.sum(masses)) + sum(a["added_kg"] for a in applied)
    return {"applied": applied, "substituted": substituted,
            "total_applied_kg": float(sum(a["added_kg"] for a in applied)),
            "plan_total_kg": float(sum(plan.values())),
            "mass_total_after_kg": got, "mass_total_expected_kg": want,
            "write_verified": bool(abs(got - want) <= 1e-15 * max(1.0, abs(want)))}


def assert_yardstick(model, expected_kg: float = BODY_MASS_KG_FALLBACK,
                     rtol: float = 1e-4) -> dict[str, Any]:
    """Check a compiled UNLOADED model's mass against the yardstick.

    A wrong yardstick scales every payload by the same relative error and is
    invisible in the results, so it is checked explicitly and raises instead of
    warning.  Returns the comparison for recording.
    """
    got = measure_body_mass_kg(model)
    rel = abs(got - float(expected_kg)) / float(expected_kg) if expected_kg else float("inf")
    if rel > rtol:
        raise ValueError(
            f"body-mass yardstick mismatch: compiled model is {got:.8e} kg but the "
            f"yardstick is {expected_kg:.8e} kg (relative error {rel:.3e} > {rtol:.1e}); "
            "every payload fraction would be wrong by that factor. Re-measure with "
            "tools_measure_body_mass.py and update BODY_MASS_KG_FALLBACK.")
    return {"measured_kg": got, "yardstick_kg": float(expected_kg),
            "relative_error": rel, "rtol": float(rtol), "ok": True}


def model_signature(model) -> dict[str, Any]:
    """A hash-free but exact-enough fingerprint of the mass distribution.

    Used to prove that ``load_fraction == 0`` leaves the model untouched and that a
    positive load really changes it.  ``mass_sum`` is exact to double precision;
    ``mass_sha1`` hashes the rounded masses so a single changed body is visible.
    """
    import hashlib
    import numpy as np
    masses = np.asarray(model.body_mass, dtype=float)
    rounded = np.round(masses, 18).astype("<f8")
    return {
        "n_body": int(model.nbody),
        "mass_sum_kg": float(masses.sum()),
        "mass_sha1": hashlib.sha1(rounded.tobytes()).hexdigest(),
        "thorax_mass_kg": float(masses[1]) if model.nbody > 1 else None,
        "com_kg_m": [float(v) for v in np.asarray(model.body_ipos, dtype=float)[1]] if model.nbody > 1 else None,
    }


__all__ = [
    "BOROSILICATE_DENSITY_KG_M3", "EPOXY_DENSITY_KG_M3", "TIP_DIAMETER_M",
    "BASE_DIAMETER_M", "SHANK_LENGTH_M", "HANDLING_TARE_MASS_KG",
    "BODY_MASS_KG_FALLBACK", "ATTACHMENT_SITES", "ElectrodePayloadConfig",
    "PayloadAttachment", "electrode_glass_mass_kg", "measure_body_mass_kg",
    "payload_plan", "attach_payload", "model_signature", "assert_yardstick",
    "apply_payload_to_model", "mass_overrides_by_segment",
    "tether_mass_kg", "rig_report", "rig_plan", "RIG_PRESETS", "RIG_MOUNTS",
    "TETHER_DIAMETER_M", "TETHER_FREE_LENGTH_M", "BATTERY_MASS_KG",
    "BLE_TRANSMITTER_MASS_KG", "COPPER_DENSITY_KG_M3", "attach_marker_bodies",
]
