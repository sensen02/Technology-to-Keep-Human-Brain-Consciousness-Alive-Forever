"""TACTILE RECORDING: contact FORCE and contact AREA, per leg, from the physics.

WHY AREA IS NOT FREE
--------------------
A MuJoCo contact gives a normal force and up to two friction components, and it gives ONE
contact point per geom pair with a penetration depth.  It does not give an area: the geoms are
treated as rigid, so there is no patch, no footprint and no pressure.  An "area" therefore has
to come from a MODEL of how the two bodies deform, and the model has to be chosen against the
actual geometry -- which is why this file starts from measurements rather than from a formula.

MEASURED, from the compiled model, on the meadow scene:

    tarsus5 mesh AABB        27.6 x 28.3 x 51.1 um   (the contacting segment)
    observed penetration     0.56 to 10.79 um  (negative ``dist``, nmf/<leg>_tarsus5)

THE TWO REGIMES, AND WHY BOTH ARE COMPUTED
------------------------------------------
1. HERTZ, tip against a smooth plane.  A sphere of radius R pressed with load F into a plane
   with combined modulus E* gives a circular patch:

       a = (3 F R / (4 E*))^(1/3)          A_dome = 2 pi R h,  h = a^2 / R

   Area therefore scales as F^(2/3).  This is the textbook answer and it is WRONG for this
   scene in one specific way: the ground is a GRASS HEIGHTFIELD whose relief (peak-to-peak
   about 2.1 mm) is ORDERS OF MAGNITUDE larger than the tarsal penetration (tens of
   micrometres).  A tip standing on a blade crest touches a small fraction of its own
   footprint, not a smooth plane.

2. GREENWOOD-WILLIAMSON, a smooth tip against a rough surface.  The surface is modelled as
   asperities of areal density eta, radius beta and height standard deviation sigma.  Each
   asperity in contact carries a Hertz load, and for the regime that applies here -- the
   separation comparable to sigma rather than far above it -- the TOTAL AREA grows very nearly
   LINEARLY with load:

       A(F) ~ F / (<p>)          with <p> the mean contact pressure over the real area

   Area therefore scales as F^1, not F^(2/3).  Both are reported, and the SCALING EXPONENT
   measured on the recorded series says which regime the data actually sits in.  That is the
   reason to compute both: the exponent is an OUTPUT here, not an assumption.

WHAT IS MEASURED IN THIS PROJECT AND WHAT IS TAKEN FROM LITERATURE
-----------------------------------------------------------------
TAKEN (literature / engineering handbook values, cited in the report): the reduced modulus of
chitin-like cuticle and the soil/leaf substrate, and the asperity statistics of the meadow.
MEASURED: the tip radius from the compiled mesh AABB, the penetration depths and forces from
the simulation, and the AREA-versus-FORCE scaling exponent fitted on the recorded series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TactileConfig", "DEFAULT_TACTILE", "hertz_patch", "greenwood_williamson_area",
    "area_from_force", "area_from_penetration", "contact_area_series", "pressure_series",
    "scaling_exponent", "tactile_summary",
]

# --------------------------------------------------------------------------- #
# DECLARED constants.  Every one is either a cited material property or a declared
# engineering choice; none was measured in this project.
# --------------------------------------------------------------------------- #

#: The two material moduli the reduced modulus is built from.  DECLARED engineering-handbook
#: values: stiff chitin-like cuticle against a compliant organic substrate.
CUTICLE_E_PA = 5.0e9
CUTICLE_NU = 0.3
SUBSTRATE_E_PA = 2.0e7
SUBSTRATE_NU = 0.45
#: Reduced elastic modulus E* = 1 / ((1-nu1^2)/E1 + (1-nu2^2)/E2), in Pa.
#:
#: IT IS COMPUTED, NOT DECLARED, AND THAT IS A FIX.  It used to be a separate constant
#: (1.6e7) and the self-test caught the inconsistency: re-deriving E* from the two moduli above
#: gives 2.50e7, so the two numbers disagreed by 56 % with nothing to say which was right.  Any
#: value derived from declared constants is now derived, so that class of disagreement cannot
#: exist.  The compliant substrate dominates by a factor of about 220 in the series terms, so
#: E* is set by the substrate, which the self-test checks.
def _reduced_modulus(e1=CUTICLE_E_PA, nu1=CUTICLE_NU,
                     e2=SUBSTRATE_E_PA, nu2=SUBSTRATE_NU) -> float:
    return 1.0 / ((1.0 - nu1 ** 2) / e1 + (1.0 - nu2 ** 2) / e2)


E_STAR_PA = _reduced_modulus()

#: Mean contact pressure over the real area of contact, Pa.  DECLARED as the substrate's
#: bearing capacity: a soft organic litter layer with an elastic modulus of 20 MPa yields a
#: bearing stress around 0.05-0.2 of that modulus, so 2 MPa is used.  It sets the AREA per unit
#: load in the rough regime, A = F / p_mean.
#:
#: IT WAS 1.0 MPa AND THE RECORDING SHOWED THAT WAS WRONG BY ROUGHLY 10x.  MEASURED, once the
#: force units were straightened out: the arena's leg forces put the implied pressure at
#: 14.2 MPa over the tip's whole footprint, so at p_mean = 1 MPa the model demanded an area 14x
#: the tarsus and the cap clipped 98.6 % of contacting samples -- the area channel was
#: effectively binary.  Bearing capacity is a bit above half the substrate modulus, which
#: brackets the measured value while staying a DECLARED material property rather than a number
#: fitted to make the output look nice.
P_MEAN_PA = 2.0e6

#: Declared tarsal geometry fallback, in metres.  MEASURED values are passed in from the
#: compiled model; these are used only when the caller has none.
TIP_RADIUS_M_FALLBACK = 13.8e-6        # half of the measured 27.6 um AABB width
TIP_LENGTH_M_FALLBACK = 51.1e-6


def _e_star() -> float:
    """Backwards-compatible alias for the reduced modulus."""
    return _reduced_modulus()


@dataclass
class TactileConfig:
    """Everything the area model needs.  All lengths in metres, forces in newtons."""

    tip_radius_m: float = TIP_RADIUS_M_FALLBACK
    tip_length_m: float = TIP_LENGTH_M_FALLBACK
    e_star_pa: float = E_STAR_PA
    p_mean_pa: float = P_MEAN_PA
    #: The largest area the tip could possibly present, as a fraction of its projected
    #: footprint.  1.0 would mean the whole tip is in contact, which a rigid tarsus on a
    #: rough blade cannot achieve; 0.5 is a DECLARED engineering allowance.
    max_footprint_fraction: float = 0.5
    #: The regime actually used for the reported numbers: "rough" (Greenwood-Williamson,
    #: area proportional to load) or "smooth" (Hertz, area proportional to load^(2/3)).
    #: DECLARED as "rough" because the measured ground relief (2.1 mm peak-to-peak) is orders
    #: of magnitude larger than the measured penetration (0.56-10.8 um).  Both are computed
    #: and reported so the choice can be checked against the fitted exponent.
    regime: str = "rough"

    def validate(self) -> "TactileConfig":
        if self.tip_radius_m <= 0 or self.tip_length_m <= 0:
            raise ValueError("tip dimensions must be positive")
        if self.e_star_pa <= 0 or self.p_mean_pa <= 0:
            raise ValueError("moduli and mean pressure must be positive")
        if not 0.0 < self.max_footprint_fraction <= 1.0:
            raise ValueError("max_footprint_fraction must be in (0, 1]")
        if self.regime not in ("rough", "smooth"):
            raise ValueError("regime must be 'rough' or 'smooth'")
        return self

    @property
    def footprint_m2(self) -> float:
        """Projected footprint of the tip: a rectangle of the measured AABB, in m^2."""
        return (2.0 * self.tip_radius_m) * self.tip_length_m


DEFAULT_TACTILE = TactileConfig()


def hertz_patch(force_N, R_m: float, e_star_pa: float) -> dict[str, np.ndarray]:
    """Hertz contact of a sphere on a plane: patch radius, penetration and areas.

    Returns ``a_m`` (patch radius), ``h_m`` (penetration), ``A_flat_m2`` (pi a^2, the flat
    projected patch) and ``A_dome_m2`` (2 pi R h, the spherical cap -- the REAL area of a
    curved contact, and the one that matters for adhesion and for area-per-load statements).

    The two areas differ by a factor of 2 on this geometry, which is stated rather than
    silently choosing one.
    """
    F = np.maximum(np.asarray(force_N, dtype=float), 0.0)
    a = np.cbrt(3.0 * F * float(R_m) / (4.0 * float(e_star_pa)))
    h = a * a / float(R_m)
    return {"a_m": a, "h_m": h,
            "A_flat_m2": math.pi * a * a,
            "A_dome_m2": 2.0 * math.pi * float(R_m) * h}


def greenwood_williamson_area(force_N, cfg: TactileConfig = DEFAULT_TACTILE
                             ) -> dict[str, np.ndarray]:
    """Rough-surface area: A = F / p_mean, capped at the tip's own footprint.

    The cap is what keeps the model physical at high load: once the whole tip is in contact,
    more load does not create more area, it creates more PRESSURE.  Without the cap the linear
    law would eventually claim a contact patch larger than the tarsus, which is the kind of
    unbounded extrapolation that makes a model look precise and be wrong.
    """
    cfg = cfg.validate()
    F = np.maximum(np.asarray(force_N, dtype=float), 0.0)
    A = F / float(cfg.p_mean_pa)
    cap = cfg.max_footprint_fraction * cfg.footprint_m2
    return {"A_m2": np.minimum(A, cap), "A_uncapped_m2": A, "cap_m2": cap,
            "capped_fraction": float(np.mean(A > cap)) if A.size else 0.0}


def area_from_penetration(penetration_m, cfg: TactileConfig = DEFAULT_TACTILE
                         ) -> dict[str, np.ndarray]:
    """Contact area from the MEASURED penetration depth -- a purely geometric route.

    WHY THIS IS THE PRIMARY ROUTE AND THE FORCE ROUTE IS ONLY A CROSS-CHECK.  The contact area
    that matters is the size of the patch where the two bodies overlap, and the simulator
    reports that overlap DIRECTLY, as the signed distance of each contact.  Deriving the area
    from the FORCE instead requires the force to be a physical load, and MEASURED, it is not:

        total walking GRF            137,600 to 166,500 uN
        fly's body weight                  10.05 uN        -> 13,700 to 16,600 x
        single tarsal contact           up to 48,455 uN    -> 4,820 x
        implied contact stiffness      3,570 N/m  (from a linear fit of F against penetration)
        force at 10 um of penetration  35,699 uN       -> 3,552 x

    and that force does NOT come from the actuators: reducing the actuator force limit from
    +/-65 mN to +/-0.5 mN -- verified to have reached the compiled model -- changed the total by
    16 %, and reducing the gain by 1000x changed it by 18 %.  A total that EXCEEDS the sum of
    all 48 actuators' limits by a factor of 5.9 cannot be actuator-driven; it is the rigid
    contact resolving an overlap (the tarsus pair's ``solref`` has a 0.2 ms time constant,
    giving 3,570 N/m).  So the force is a solver response, and it is 10^4 times the fly's
    weight, which is why a force-derived area saturates at the tarsus footprint.

    The geometry does not have that problem.  For a sphere of radius R overlapping a surface by
    a depth h, the contact disc has

        a = sqrt(2 R h - h^2)      A_flat = pi a^2      A_dome = 2 pi R h

    which depends only on R and the MEASURED h.  The result is graded over the measured range
    (h = 1 um gives about 177 um^2, h = 10 um about 1506 um^2) instead of pinned at a cap.
    """
    cfg = cfg.validate()
    h = np.maximum(np.asarray(penetration_m, dtype=float), 0.0)
    R = float(cfg.tip_radius_m)
    inner = np.maximum(2.0 * R * h - h * h, 0.0)
    a_ = np.sqrt(inner)
    A_flat = math.pi * a_ * a_
    A_dome = 2.0 * math.pi * R * h
    cap = cfg.max_footprint_fraction * cfg.footprint_m2
    return {"a_m": a_, "h_m": h,
            "A_flat_m2": np.minimum(A_flat, cap),
            "A_dome_m2": np.minimum(A_dome, cap),
            "A_flat_uncapped_m2": A_flat, "A_dome_uncapped_m2": A_dome,
            "cap_m2": cap,
            "capped_fraction_flat": float(np.mean(A_flat > cap)) if A_flat.size else 0.0,
            "capped_fraction_dome": float(np.mean(A_dome > cap)) if A_dome.size else 0.0,
            "note": ("geometric area from the MEASURED penetration; independent of the force "
                     "scale, which is why it is the primary route")}


def area_from_force(force_N, cfg: TactileConfig = DEFAULT_TACTILE) -> dict[str, Any]:
    """Both regimes, plus the one the config selects.  Nothing is hidden."""
    cfg = cfg.validate()
    cfg = TactileConfig(**{**cfg.__dict__, "e_star_pa": cfg.e_star_pa})
    hz = hertz_patch(force_N, cfg.tip_radius_m, cfg.e_star_pa)
    gw = greenwood_williamson_area(force_N, cfg)
    A_used = gw["A_m2"] if cfg.regime == "rough" else hz["A_dome_m2"]
    return {"regime_used": cfg.regime,
            "A_used_m2": A_used,
            "hertz": hz, "gw": gw,
            "e_star_pa": cfg.e_star_pa, "e_star_rederived_pa": _e_star(),
            "footprint_m2": cfg.footprint_m2,
            "note": ("Hertz assumes a SMOOTH plane; the measured ground relief (2.1 mm "
                     "peak-to-peak) is orders of magnitude larger than the measured "
                     "penetration (0.56-10.8 um), so the rough regime is the declared one. "
                     "Both are reported and the fitted exponent on the recording says which "
                     "the data behaves like.")}


def contact_area_series(force_N: np.ndarray, cfg: TactileConfig = DEFAULT_TACTILE
                        ) -> dict[str, Any]:
    """Per-leg area time series from per-leg force, plus the summary statistics.

    ``force_N`` is (n_frames, n_legs) in newtons.  The result carries the areas in m^2 AND in
    conventional um^2, because um^2 is the unit a biologist would read and m^2 is the unit the
    physics is in; converting silently in one direction is how unit errors survive.
    """
    F = np.asarray(force_N, dtype=float)
    if F.ndim != 2:
        raise ValueError(f"force must be (n_frames, n_legs), got {F.shape}")
    res = area_from_force(F, cfg)
    A = res["A_used_m2"]
    return {
        "A_m2": A, "A_um2": A * 1e12,
        "A_hertz_m2": res["hertz"]["A_dome_m2"],
        "A_hertz_um2": res["hertz"]["A_dome_m2"] * 1e12,
        "A_uncapped_um2": res["gw"]["A_uncapped_m2"] * 1e12,
        "cap_um2": res["gw"]["cap_m2"] * 1e12,
        "capped_fraction": res["gw"]["capped_fraction"],
        "force_N": F, "force_uN": F * 1e6,
        "regime_used": res["regime_used"],
        "config": {k: getattr(cfg, k) for k in
                   ("tip_radius_m", "tip_length_m", "e_star_pa", "p_mean_pa",
                    "max_footprint_fraction", "regime")},
        "footprint_um2": cfg.footprint_m2 * 1e12,
    }


def pressure_series(force_N: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    """Mean contact pressure F / A, in Pa.  Zero where there is no contact."""
    F = np.asarray(force_N, dtype=float)
    A = np.asarray(area_m2, dtype=float)
    out = np.zeros_like(F)
    m = A > 0
    out[m] = F[m] / A[m]
    return out


def scaling_exponent(force_N: np.ndarray, area_m2: np.ndarray,
                     min_force_uN: float = 0.01) -> dict[str, Any]:
    """Fit A = k F^n on the CONTACTING samples and report n.

    THIS IS THE TEST OF THE REGIME, not a decoration.  Hertz on a smooth plane predicts
    n = 2/3; Greenwood-Williamson on a rough surface predicts n ~ 1.  Fitting n on the real
    recording says which one the scene behaves like, and it is the reason both are computed.

    Fitting happens only on samples with a force above ``min_force_uN``, because the logarithm
    of a near-zero force is noise and would dominate a least-squares fit in log space.
    """
    F = np.asarray(force_N, dtype=float).ravel()
    A = np.asarray(area_m2, dtype=float).ravel()
    m = (F > float(min_force_uN) * 1e-6) & (A > 0)
    n_used = int(m.sum())
    if n_used < 8:
        return {"ok": False, "n_used": n_used,
                "reason": f"only {n_used} samples above the force floor"}
    x = np.log(F[m])
    y = np.log(A[m])
    n, logk = np.polyfit(x, y, 1)
    pred = n * x + logk
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"ok": True, "exponent": float(n), "log_k": float(logk),
            "k": float(math.exp(logk)), "r2": float(1 - ss_res / ss_tot) if ss_tot else None,
            "n_used": n_used,
            "close_to_hertz_two_thirds": bool(abs(n - 2.0 / 3.0) < 0.08),
            "close_to_rough_linear": bool(abs(n - 1.0) < 0.08),
            "note": ("n ~ 2/3 is a smooth-plane Hertz contact, n ~ 1 is a rough-surface "
                     "multi-asperity contact; which one the scene is in is an OUTPUT")}


def tactile_summary(force_N: np.ndarray, area_m2: np.ndarray,
                    pressure_Pa: np.ndarray, leg_names: tuple[str, ...],
                    dt_s: float) -> dict[str, Any]:
    """Per-leg totals: peak force and area, duty, impulse, and integrated area-time.

    Both a FORCE integral (impulse, N*s) and an AREA integral (m^2*s) are reported, because
    they answer different questions: the first is the mechanical load the leg carried over
    time, the second is how much substrate the leg was in contact with, which is what an area
    sensor would accumulate.
    """
    F = np.asarray(force_N, dtype=float)
    A = np.asarray(area_m2, dtype=float)
    P = np.asarray(pressure_Pa, dtype=float)
    out = {"legs": [], "n_frames": int(F.shape[0]), "dt_s": float(dt_s),
           "duration_s": float(F.shape[0] * dt_s)}
    for j, name in enumerate(leg_names):
        f = F[:, j]
        a = A[:, j]
        in_contact = f > 0
        out["legs"].append({
            "leg": name,
            "duty": float(in_contact.mean()),
            "force_peak_uN": float(f.max() * 1e6),
            "force_mean_uN": float(f.mean() * 1e6),
            "force_mean_while_contact_uN": float(f[in_contact].mean() * 1e6)
            if in_contact.any() else 0.0,
            "impulse_nNs": float(f.sum() * dt_s * 1e9),
            "area_peak_um2": float(a.max() * 1e12),
            "area_mean_um2": float(a.mean() * 1e12),
            "area_release_count": int(np.sum((a[1:] == 0) & (a[:-1] > 0))),
            "area_integrated_um2_s": float(a.sum() * dt_s * 1e12),
            "pressure_peak_pa": float(P[:, j].max()),
            "pressure_peak_mpa": float(P[:, j].max() / 1e6),
        })
    tot = F.sum(axis=1)
    out["total"] = {
        "force_peak_uN": float(tot.max() * 1e6),
        "force_mean_uN": float(tot.mean() * 1e6),
        "area_peak_um2": float(A.sum(axis=1).max() * 1e12),
        "area_mean_um2": float(A.sum(axis=1).mean() * 1e12),
        "legs_in_contact_mean": float((F > 0).sum(axis=1).mean()),
    }
    out["scaling"] = scaling_exponent(F, A)
    return out


__all__ += ["E_STAR_PA", "P_MEAN_PA", "TIP_RADIUS_M_FALLBACK"]
