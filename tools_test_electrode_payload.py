#!/usr/bin/env python3
"""Self-test for the electrode payload layer -- arithmetic that can be checked by hand.

The payload's mass, inertia and site assignment are pure arithmetic on declared
constants, so they are checkable to machine precision without MuJoCo.  The MuJoCo half
(the injection itself) is checked by the assertions in this file too, but only if the
body environment is available; without it those checks are SKIPPED and reported as
skipped, never as passed.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_test_electrode_payload.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from electrode_payload import (  # noqa: E402
    ATTACHMENT_SITES, BODY_MASS_KG_FALLBACK, EPOXY_DENSITY_KG_M3,
    TARE_EFFECTIVE_DENSITY_KG_M3, TIP_DIAMETER_M, BASE_DIAMETER_M,
    ElectrodePayloadConfig, electrode_glass_mass_kg, payload_plan,
)

FAILS: list[str] = []
SKIPS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def approx(a: float, b: float, rtol: float = 1e-12) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def test_frustum_volume() -> None:
    print("frustum volume (closed form, hand-checkable):")
    rep = electrode_glass_mass_kg()
    r1 = 0.5 * TIP_DIAMETER_M
    r2 = 0.5 * BASE_DIAMETER_M
    h = rep["geometry"]["shank_length_m"]
    v = math.pi * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2)
    check("frustum volume matches pi*h/3*(r1^2+r1r2+r2^2)", approx(v, rep["frustum_volume_m3"]),
          f"{v:.6e} m^3")
    # degenerate cases: a cone (r1=0) and a cylinder (r1=r2)
    v_cone = electrode_glass_mass_kg(tip_diameter_m=0.0)["frustum_volume_m3"]
    check("r1 -> 0 gives a cone: pi r2^2 h / 3",
          approx(v_cone, math.pi * r2 * r2 * h / 3.0), f"{v_cone:.6e} m^3")
    v_cyl = electrode_glass_mass_kg(tip_diameter_m=2 * r2)["frustum_volume_m3"]
    check("r1 -> r2 gives a cylinder: pi r2^2 h",
          approx(v_cyl, math.pi * r2 * r2 * h), f"{v_cyl:.6e} m^3")
    # mass = volume x density, exactly
    m = rep["frustum_volume_m3"] * rep["glass_density_kg_m3"]
    check("glass mass = volume x declared density", approx(m, rep["frustum_mass_kg"]))
    check("total = glass + glue",
          approx(rep["total_kg"], rep["frustum_mass_kg"] + rep["glue_mass_kg"]))
    check("glue mass = cylinder volume x epoxy density",
          approx(rep["glue_mass_kg"],
                 rep["glue_volume_m3"] * EPOXY_DENSITY_KG_M3))
    check("bare electrode is a negligible fraction of body mass (<0.1 %)",
          rep["total_kg"] / BODY_MASS_KG_FALLBACK < 1e-3,
          f"{100 * rep['total_kg'] / BODY_MASS_KG_FALLBACK:.4f} % of body")


def test_sphere_inertia() -> None:
    print("payload inertia (solid sphere, 2/5 m r^2):")
    cfg = ElectrodePayloadConfig(load_fraction=0.20, label="t")
    plan = payload_plan(cfg, BODY_MASS_KG_FALLBACK)
    for s in plan["per_site"]:
        m, r = s["mass_kg"], s["effective_radius_m"]
        want = 0.4 * m * r * r
        check(f"{s['segment']}: I = 0.4 m r^2", approx(want, s["inertia_kg_m2"][0]),
              f"I={want:.4e} kg m^2, r={r * 1e6:.2f} um")
        check(f"{s['segment']}: radius from volume at declared tare density",
              approx(r ** 3, 3.0 * m / (4.0 * math.pi * TARE_EFFECTIVE_DENSITY_KG_M3),
                     rtol=1e-9))


def test_plan_sums() -> None:
    print("plan conservation and the sweep axis:")
    for frac in (0.0, 0.01, 0.05, 0.20, 1.0):
        cfg = ElectrodePayloadConfig(load_fraction=frac, label="t")
        plan = payload_plan(cfg, BODY_MASS_KG_FALLBACK)
        total = sum(s["mass_kg"] for s in plan["per_site"])
        check(f"site masses sum to the plan total at load={frac}",
              approx(total, plan["total_kg"], rtol=1e-12),
              f"{total:.6e} kg")
        check(f"load_fraction is a fraction of body mass at load={frac}",
              approx(plan["tare_kg"], frac * BODY_MASS_KG_FALLBACK, rtol=1e-12))
    shares = sum(float(s["share"]) for s in ATTACHMENT_SITES)
    check("declared attachment shares are normalised by the plan",
          approx(sum(s["share"] for s in payload_plan(
              ElectrodePayloadConfig(0.1), BODY_MASS_KG_FALLBACK)["per_site"]), 1.0),
          f"declared raw sum {shares}")


def test_config_validation() -> None:
    print("config validation:")
    for bad, why in ((-1.0, "negative load"),):
        try:
            ElectrodePayloadConfig(load_fraction=bad).validate()
            check(f"rejects {why}", False)
        except ValueError:
            check(f"rejects {why}", True)
    try:
        ElectrodePayloadConfig(load_fraction=0.1, mass_scale=0.0).validate()
        check("rejects zero mass_scale", False)
    except ValueError:
        check("rejects zero mass_scale", True)


def test_mujoco_injection() -> None:
    print("MuJoCo injection (needs the body environment):")
    try:
        from engine.embodied.body_backend import BodyBackend, BodyConfig
        from electrode_payload import model_signature
    except Exception as exc:            # noqa: BLE001 - reported, not swallowed
        SKIPS.append(f"MuJoCo injection checks: {type(exc).__name__}: {exc}")
        print(f"  [SKIP] cannot import the body backend: {type(exc).__name__}: {exc}")
        return
    try:
        be = BodyBackend(BodyConfig(), gl_backend=None)
        ref = model_signature(be.model)
        be.close()
    except Exception as exc:            # noqa: BLE001
        SKIPS.append(f"MuJoCo not runnable: {type(exc).__name__}: {exc}")
        print(f"  [SKIP] cannot build an unloaded model: {type(exc).__name__}: {exc}")
        return

    # zero load must NOT touch the model
    be = BodyBackend(BodyConfig(electrode_payload=ElectrodePayloadConfig(
        load_fraction=0.0, label="zero")), gl_backend=None)
    sig0 = model_signature(be.model)
    be.close()
    check("load_fraction=0 leaves the model bit-identical",
          sig0["mass_sha1"] == ref["mass_sha1"] and
          approx(sig0["mass_sum_kg"], ref["mass_sum_kg"], rtol=0),
          f"sha {sig0['mass_sha1'][:12]}")

    for frac in (0.05, 0.20):
        be = BodyBackend(BodyConfig(electrode_payload=ElectrodePayloadConfig(
            load_fraction=frac, label="t")), gl_backend=None)
        sig = model_signature(be.model)
        rep = be.payload_report
        be.close()
        want = frac * ref["mass_sum_kg"]
        got = sig["mass_sum_kg"] - ref["mass_sum_kg"]
        check(f"load {frac}: mass added = {frac} x body mass",
              approx(got, want, rtol=1e-6), f"added {got:.6e} kg, wanted {want:.6e}")
        check(f"load {frac}: signature changed",
              sig["mass_sha1"] != ref["mass_sha1"])
        check(f"load {frac}: both attachment bodies present",
              len(rep["body_names_added"]) == len(
                  [s for s in rep["per_site"] if s["mass_kg"] > 0]))

    # sham: same machinery, inert mass
    be = BodyBackend(BodyConfig(electrode_payload=ElectrodePayloadConfig(
        load_fraction=0.20, mass_scale=1e-6, label="sham")), gl_backend=None)
    sig_s = model_signature(be.model)
    be.close()
    d = sig_s["mass_sum_kg"] - ref["mass_sum_kg"]
    check("sham load changes the model by < 1 % of body mass",
          abs(d) / ref["mass_sum_kg"] < 0.01, f"{100 * abs(d) / ref['mass_sum_kg']:.4f} %")

    # the yardstick must reproduce on this asset
    check("declared yardstick matches a freshly compiled unloaded model",
          approx(ref["mass_sum_kg"], BODY_MASS_KG_FALLBACK, rtol=1e-4),
          f"compiled {ref['mass_sum_kg']:.8e} vs declared {BODY_MASS_KG_FALLBACK:.8e}")

    # repeated builds must not accumulate payloads (the bug this guards)
    sums = []
    for _ in range(3):
        be = BodyBackend(BodyConfig(electrode_payload=ElectrodePayloadConfig(
            load_fraction=0.20, label="rep")), gl_backend=None)
        sums.append(model_signature(be.model)["mass_sum_kg"])
        be.close()
    check("repeated same-label builds give identical masses (no accumulation)",
          max(sums) - min(sums) < 1e-15, f"spread {max(sums) - min(sums):.3e} kg")


def main() -> int:
    test_frustum_volume()
    test_sphere_inertia()
    test_plan_sums()
    test_config_validation()
    test_mujoco_injection()
    print()
    if SKIPS:
        print(f"SKIPPED ({len(SKIPS)}):")
        for s in SKIPS:
            print(f"  - {s}")
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s): {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
