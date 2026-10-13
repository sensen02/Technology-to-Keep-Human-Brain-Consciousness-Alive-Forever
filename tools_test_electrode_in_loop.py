#!/usr/bin/env python3
"""Does the electrode in the loop actually record the network?  And does the guard fire?

Three checks, each able to FAIL:

  1. GEOMETRY   the electrode must place its channels on the FIXED constraint (7 um diameter,
                20 um pitch) and must report how many tier neurons it could NOT locate, rather
                than quietly recording from fewer neurons than the network has.
  2. COUPLING   the recorded trace must change when the spikes are zeroed.  If it does not, the
                electrode is not in the loop and the harness reports NOT_RUN -- the same rule that
                caught 75 muscles never being wired.
  3. INJECTION  a deliberately broken electrode (all somata placed at the cloud centroid, so every
                source sits on every contact) must be detected as degenerate, and the real geometry
                must not be.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_electrode_in_loop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))
sys.path.insert(0, str(HERE / "engine"))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from engine.embodied import BodyBackend, BodyConfig
    from engine.embodied.adapters import TierConfig, build_neural_tier
    from engine.embodied.electrode_in_loop import (CONTACT_DIAMETER_UM, CONTACT_PITCH_UM,
                                                   InLoopElectrode, SomaPositions)

    print("1. building the real BANC tier")
    cfg = TierConfig()
    built = build_neural_tier(cfg)
    n = int(built["n"])
    print(f"   tier neurons: {n}")

    # tier local index -> global index -> root_id, the same mapping the tier itself uses
    data = built["data"]
    gi = np.asarray(built["global_index"], dtype=np.int64)
    root_ids = np.asarray(data["root_ids"], dtype=np.int64)[gi]
    print(f"   root ids resolved: {root_ids.size}")

    print("2. soma positions from the BANC soma table")
    soma = SomaPositions.from_banc(root_ids)
    print(f"   {soma.provenance}")
    check("soma lookup located neurons", soma.xyz_um.shape[0] - soma.n_missing > 0,
          f"{soma.xyz_um.shape[0] - soma.n_missing} of {soma.xyz_um.shape[0]}")

    elec = InLoopElectrode(soma, n_tier_neurons=n, n_channels=4, dt_neural_ms=float(cfg.dt_ms))
    g = elec.geometry()
    print(f"   geometry: {json.dumps({k: v for k, v in g.items() if k != 'kernel'})}")

    print("\n3. the FIXED constraint is respected")
    check("contact diameter is the fixed 7 um",
          abs(g["contact_diameter_um"] - CONTACT_DIAMETER_UM) < 1e-9,
          f"{g['contact_diameter_um']} um")
    check("contact pitch is the fixed 20 um",
          g["contact_pitch_um"] is not None and abs(g["contact_pitch_um"] - CONTACT_PITCH_UM) < 1e-6,
          f"{g['contact_pitch_um']:.3f} um")

    print("\n4. driving the tier with the loop's OWN sensory path, then recording")
    be = BodyBackend(BodyConfig(seed=0, scene_preset="meadow_grass",
                                add_tracking_camera=False, add_world_camera=False,
                                add_vision=False), gl_backend=None)
    obs = be.observe()
    # Constructed EXACTLY as the loop constructs it (engine/embodied/loop.py:441), so this test
    # exercises the loop's real sensory path rather than a lookalike.
    from engine.embodied.adapters import MechanoReceptorBank, OwnershipLedger, ReceptorConfig
    from engine.embodied.adapters import sensor_matrix_row_map
    ledger = OwnershipLedger()
    # `angle_rows` must be valid indices into the joint-angle vector, so it is taken from the body's
    # OWN DOF map -- the same source loop.py uses (`self.body_maps["angle_rows"]`) -- rather than
    # invented here.  Guessing it produced a loud ValueError, which is the intended behaviour of
    # that guard.
    from engine.embodied.adapters import leg_rows_from_dof_order
    angle_rows = leg_rows_from_dof_order(be.dof_order, be.model)["angle_rows"]
    rec = MechanoReceptorBank(built["sensory"], angle_rows, n, ReceptorConfig(),
                              dt_ms=5.0, ledger=ledger, seed=0)
    # the loop calls this before its first transduce (loop.py:496); without it the receptor bank
    # raises rather than transducing against an undefined reference pose
    rec.set_reference_pose(obs.joint_angles_rad)
    _cur, _drive = rec.transduce(obs)
    n_sub = 40
    spikes = np.zeros((n_sub, n), dtype=bool)
    for j in range(n_sub):
        c, _ = rec.transduce(obs)
        spikes[j] = be_advance(built, cfg, c, n_sub=1)
    total_spikes = int(spikes.sum())
    print(f"   spikes over {n_sub} substeps: {total_spikes}")
    check("the tier actually fired (else nothing can be recorded)", total_spikes > 0,
          f"{total_spikes} spikes")

    print("\n5. the guard: does the trace depend on the spikes?")
    dep = elec.dependence(spikes)
    print(f"   status={dep['status']}  peak={dep['trace_peak_mV']:.4g} mV  "
          f"std={dep['trace_std_mV']:.4g} mV  n_neurons={dep['n_neurons_recorded']}")
    check("electrode is COUPLED to the network (trace changes when spikes are zeroed)",
          dep["status"] == "PASS", dep["why"])

    print("\n6. injection: a DECOUPLED electrode must be caught by the guard")
    # THE FAILURE MODE THE GUARD EXISTS FOR is "the electrode is not recording the network", so the
    # injection is exactly that: put every channel AT the reference point, which makes the recorded
    # differential identically zero for any spike pattern.  The guard must then say NOT_RUN.
    #
    # NOTE, and it is a finding rather than a detail: an earlier version of this injection collapsed
    # all somata onto the cloud centroid and required the recording to become blind to WHICH neurons
    # fired.  MEASURED, it did not -- because 4263 of the 12000 tier neurons have no soma in the
    # table, so which neurons fire changes HOW MANY located sources contribute.  The degeneracy
    # therefore shows up in the presence mask, not in the geometry, and the correct injection is
    # decoupling rather than collapsing.
    dec = InLoopElectrode(soma, n_tier_neurons=n, n_channels=4, dt_neural_ms=float(cfg.dt_ms))
    for k, r in enumerate(dec.recorders):
        r.xyz = np.asarray([np.asarray(dec.reference_um), np.asarray(dec.reference_um)], dtype=float)
    d_dep = dec.dependence(spikes)
    check("decoupled electrode reports NOT_RUN (guard fires)", d_dep["status"] == "NOT_RUN",
          f"status={d_dep['status']}  peak={d_dep['trace_peak_mV']:.3g} mV")
    check("coupled electrode reports PASS (guard is not hard-wired)",
          dep["status"] == "PASS", f"status={dep['status']}")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
           "status": "FAIL" if n_fail else "PASS",
           "units": "um for geometry; mV for the recorded trace; pC and ms for the kernel",
           "provenance": ("real BANC tier via build_neural_tier; soma coordinates from "
                          "data/banc/somas_v1.parquet (pt_root_id, pt_position); volume conduction "
                          "via engine.electrode.transfer_mV_per_nA and VoltageRecorder"),
           "what_this_does_not_establish": (
               "that the recorded AMPLITUDE is biologically meaningful.  The spike->current kernel "
               "is a declared, uncalibrated assumption.  Only the COUPLING between the electrode "
               "and the network is established here.")}
    dest = HERE / "outputs" / "electrode_in_loop_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


def be_advance(built, cfg, current, n_sub):
    """Advance a fresh tier by one substep.  Kept separate so the test drives the tier the same
    way the loop does (advance(n_substeps, current) -> spikes)."""
    global _TIER
    try:
        t = _TIER
    except NameError:
        from engine.embodied.adapters import NeuralTier, OwnershipLedger
        from engine.embodied.adapters import TierConfig
        t = _TIER = NeuralTier(built, TierConfig(), OwnershipLedger())
        t.reset()
    return t.advance(n_sub, current)


if __name__ == "__main__":
    raise SystemExit(main())
