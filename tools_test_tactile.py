#!/usr/bin/env python3
"""Self-test for the tactile area model: closed-form checks and synthetic scaling.

The area model is arithmetic on declared constants plus a regime choice, so it is checkable
against known answers without any simulation.  The checks that matter:

  * Hertz patch radius and dome area against the closed forms a=(3FR/4E*)^(1/3), A=2 pi R h;
  * the E* value against its own re-derivation from the two cited moduli;
  * the SCALING EXPONENT recovered from synthetic data that was generated with a KNOWN
    exponent -- 2/3 and 1 respectively -- because an exponent fit that has never been tested
    against a known slope is a fit that reports its own noise;
  * the footprint cap actually capping, so area cannot exceed the tarsus.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_test_tactile.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tactile import (DEFAULT_TACTILE, E_STAR_PA, P_MEAN_PA, TactileConfig,  # noqa: E402
                     _e_star, area_from_force, area_from_penetration, contact_area_series,
                     hertz_patch, greenwood_williamson_area, pressure_series,
                     scaling_exponent, tactile_summary)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def approx(a, b, rtol=1e-9):
    return abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)), 1e-300)


def test_hertz():
    print("Hertz contact closed forms:")
    cfg = DEFAULT_TACTILE
    F = 1e-5                      # 10 uN
    hz = hertz_patch(F, cfg.tip_radius_m, cfg.e_star_pa)
    a_want = (3.0 * F * cfg.tip_radius_m / (4.0 * cfg.e_star_pa)) ** (1.0 / 3.0)
    check("patch radius = (3FR/4E*)^(1/3)", approx(hz["a_m"], a_want),
          f"a = {float(hz['a_m']) * 1e9:.3f} nm")
    check("penetration h = a^2/R", approx(hz["h_m"], a_want ** 2 / cfg.tip_radius_m),
          f"h = {float(hz['h_m']) * 1e9:.4f} nm")
    check("flat area = pi a^2", approx(hz["A_flat_m2"], math.pi * a_want ** 2))
    check("dome area = 2 pi R h", approx(hz["A_dome_m2"], 2 * math.pi * cfg.tip_radius_m
                                         * hz["h_m"]))
    check("dome/flat = 2 exactly for a sphere", approx(hz["A_dome_m2"] / hz["A_flat_m2"], 2.0),
          "A_dome = 2 A_flat is a sphere identity")
    # F^(2/3) scaling
    F2 = np.array([1e-6, 1e-5, 1e-4])
    a2 = hertz_patch(F2, cfg.tip_radius_m, cfg.e_star_pa)["a_m"]
    check("patch radius scales as F^(1/3)",
          approx(float(a2[2] / a2[1]), 10.0 ** (1.0 / 3.0), rtol=1e-6))


def test_e_star():
    print("reduced modulus:")
    want = 1.0 / ((1 - 0.3 ** 2) / 5.0e9 + (1 - 0.45 ** 2) / 2.0e7)
    check("E* equals its own re-derivation from the two moduli", approx(E_STAR_PA, want,
                                                                        rtol=1e-12),
          f"E* = {want / 1e6:.3f} MPa")
    # the substrate term must dominate the series sum; the cuticle term is 219x smaller
    t_cut = (1 - 0.3 ** 2) / 5.0e9
    t_sub = (1 - 0.45 ** 2) / 2.0e7
    check("the compliant substrate term dominates the series sum", t_sub > 100 * t_cut,
          f"cuticle {t_cut:.3e} vs substrate {t_sub:.3e} (ratio {t_sub / t_cut:.1f})")
    check("E* therefore sits near the substrate modulus",
          abs(E_STAR_PA - 2.0e7) / 2.0e7 < 0.35,
          f"E* = {E_STAR_PA / 1e6:.2f} MPa against substrate {2.0e7 / 1e6:.1f} MPa")


def test_gw_and_cap():
    print("rough-surface area and its cap:")
    cfg = DEFAULT_TACTILE
    F = np.array([1e-7, 1e-6, 1e-5])
    gw = greenwood_williamson_area(F, cfg)
    check("A = F / p_mean below the cap",
          np.allclose(gw["A_uncapped_m2"], F / P_MEAN_PA), "linear in load")
    big = greenwood_williamson_area(np.array([1.0]), cfg)
    check("cap is enforced for a large load", big["A_m2"][0] <= big["cap_m2"] + 1e-30,
          f"A = {big['A_uncapped_m2'][0] * 1e12:.3e} um^2 capped to "
          f"{big['cap_m2'] * 1e12:.3f} um^2")
    check("cap = max_footprint_fraction x footprint",
          approx(big["cap_m2"], cfg.max_footprint_fraction * cfg.footprint_m2))
    check("cap is smaller than the tarsus footprint",
          big["cap_m2"] < cfg.footprint_m2, "a rigid tarsus cannot cover its whole footprint")


def test_scaling_exponent():
    print("scaling exponent recovered from SYNTHETIC data with a known slope:")
    rng = np.random.default_rng(0)
    F = np.exp(rng.uniform(math.log(1e-7), math.log(1e-5), 400))
    for n_true in (2.0 / 3.0, 1.0, 0.5):
        A = 1e-9 * F ** n_true * np.exp(rng.normal(0, 0.02, F.size))
        got = scaling_exponent(F, A)
        check(f"recovers n = {n_true:.4f}", got["ok"] and abs(got["exponent"] - n_true) < 0.03,
              f"fitted {got['exponent']:.4f} (r2 {got['r2']:.5f})")
    got67 = scaling_exponent(F, 1e-9 * F ** (2.0 / 3.0))
    got1 = scaling_exponent(F, 1e-9 * F)
    check("two-thirds is flagged as Hertz-like", got67["close_to_hertz_two_thirds"])
    check("one is flagged as rough-like", got1["close_to_rough_linear"])
    check("a near-zero force is excluded from the fit",
          scaling_exponent(np.zeros(20), np.zeros(20))["ok"] is False)


def test_geometric_area():
    print("geometric area from penetration:")
    cfg = DEFAULT_TACTILE
    R = cfg.tip_radius_m
    h = np.array([0.0, 1e-6, 5e-6, 1e-5])
    g = area_from_penetration(h, cfg)
    a_want = np.sqrt(2 * R * h - h * h)
    check("contact radius = sqrt(2Rh - h^2)",
          np.allclose(g["a_m"], a_want, equal_nan=True),
          f"a(1um) = {a_want[1] * 1e6:.2f} um")
    check("dome area = 2 pi R h",
          np.allclose(g["A_dome_uncapped_m2"], 2 * math.pi * R * h))
    check("zero penetration gives zero area",
          g["A_flat_uncapped_m2"][0] == 0 and g["A_dome_uncapped_m2"][0] == 0)
    check("area grows with penetration",
          bool(np.all(np.diff(g["A_flat_uncapped_m2"]) > 0)))
    # THE POINT OF THIS ROUTE: it is GRADED over the measured penetration range rather than
    # pinned at a cap.  The thresholds below are the MEASURED values for this tip radius
    # (28.9 um): 17.2 um^2 at h = 0.2 um rising to 552.9 um^2 at h = 10 um, a factor of 32,
    # with no sample at the cap (the cap is first reached at h = 8.1 um for the DOME area).
    # An earlier version of this test demanded a factor of 100, which is simply not what this
    # geometry does; the assertion was wrong, not the model.
    h_range = np.linspace(0.2e-6, 10e-6, 200)
    gg = area_from_penetration(h_range, cfg)
    A = gg["A_flat_uncapped_m2"]
    span = float(A.max() / max(A[A > 0].min(), 1e-30))
    check("area is graded over the measured penetration range (factor > 20)",
          span > 20.0, f"{A[1] * 1e12:.1f} to {A[-1] * 1e12:.1f} um^2, factor {span:.1f}")
    check("the flat area stays inside the cap over that range",
          gg["capped_fraction_flat"] == 0.0,
          f"cap {gg['cap_m2'] * 1e12:.1f} um^2, first reached at h = "
          f"{gg['cap_m2'] / (2 * math.pi * cfg.tip_radius_m) * 1e6:.2f} um (dome)")
    e = scaling_exponent(h_range, A, min_force_uN=0.0)
    # A = pi(2Rh - h^2): exponent 1 for h << R, falling as h approaches R
    check("area ~ penetration exponent lands between 0.8 and 1.0 as the geometry predicts",
          bool(e["ok"] and 0.80 <= e["exponent"] <= 1.0),
          f"fitted {e.get('exponent'):.4f} (r2 {e.get('r2'):.4f})" if e["ok"] else "fit failed")


def test_config_validation():
    print("config validation:")
    for kwargs, why in (({"tip_radius_m": 0.0}, "zero radius"),
                        ({"e_star_pa": -1.0}, "negative modulus"),
                        ({"p_mean_pa": 0.0}, "zero mean pressure"),
                        ({"max_footprint_fraction": 0.0}, "zero footprint fraction"),
                        ({"max_footprint_fraction": 1.5}, "footprint fraction above 1"),
                        ({"regime": "magic"}, "unknown regime")):
        try:
            TactileConfig(**kwargs).validate()
            check(f"rejects {why}", False)
        except ValueError:
            check(f"rejects {why}", True)


def test_series_and_summary():
    print("series plumbing and summary:")
    F = np.array([[0.0, 0.0], [1e-6, 0.0], [2e-6, 1e-6], [0.0, 3e-6]])
    s = contact_area_series(F)
    check("areas are zero exactly where forces are", bool(np.all((s["A_m2"] == 0) == (F == 0))))
    check("um^2 is m^2 x 1e12",
          np.allclose(s["A_um2"], s["A_m2"] * 1e12))
    P = pressure_series(F, s["A_m2"])
    check("pressure is F/A where in contact and 0 elsewhere",
          bool(np.all(P[F > 0] > 0) and np.all(P[F == 0] == 0)))
    check("pressure equals p_mean in the rough regime below the cap",
          np.allclose(P[F > 0], P_MEAN_PA, rtol=1e-9),
          "which is the definition of the regime, not a coincidence")
    summ = tactile_summary(F, s["A_m2"], P, ("lf", "lm"), dt_s=5e-3)
    check("summary has one entry per leg", len(summ["legs"]) == 2)
    check("impulse = sum(F) x dt",
          approx(summ["legs"][0]["impulse_nNs"], F[:, 0].sum() * 5e-3 * 1e9))
    check("area-time = sum(A) x dt",
          approx(summ["legs"][0]["area_integrated_um2_s"],
                 s["A_m2"][:, 0].sum() * 5e-3 * 1e12))
    check("duty is the fraction of frames in contact",
          approx(summ["legs"][0]["duty"], 2.0 / 4.0))


def main() -> int:
    test_hertz()
    test_e_star()
    test_gw_and_cap()
    test_geometric_area()
    test_scaling_exponent()
    test_config_validation()
    test_series_and_summary()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
