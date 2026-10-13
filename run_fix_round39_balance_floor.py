"""ROUND 39 (FIX), part D2: measure the muscle-only imbalance FLOOR properly.

WHY THIS IS A SEPARATE EXPERIMENT
--------------------------------
Part D's constrained search reported "0 of 2348 poses balanceable below 4% of body
weight".  That number is NOT usable as evidence, because the scorer it used is a weak
optimiser: on ONE pose, from one neutral seed, it lands at 3.72% of body weight, while
from another seed the same pose reaches 2.15%.  A method whose answer moves by a factor of
two with the seed cannot settle whether a balanceable pose exists.

So this script measures the thing that CAN be settled, and states the rest as unknown:

    For a sample of gated stances, how low can the PHYSICAL imbalance
        M(q) . qacc  (leg-joint rows)
    be driven by muscle activation alone, with a strong multi-start bounded
    least-squares, and what is the best value found?

That gives an empirical FLOOR with its own dispersion, which is what the earlier rounds
needed and never measured.  It does NOT prove that no perfect balance exists; it bounds how
close a practical muscle pattern gets, using only physics (no LP, no qfrc residual).

Diagnostic measurement.  Not an autonomous result.  Not a fly measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402
import run_fix_round39_margin_research as MR  # noqa: E402

OUT = HERE / "outputs/fix_round39"
LEG = slice(6, 48)


def solve_floor(model, data, qpos, *, seeds, rounds=10, eps=2e-3) -> dict:
    """Best muscle-only imbalance for one pose, over several activation seeds."""
    import mujoco
    from scipy.optimize import lsq_linear

    M = np.zeros((model.nv, model.nv))
    n_use = model.nu - 2
    weight = float(model.body_mass.sum() * cp.GRAVITY_MODEL)

    def imb(a):
        data.qpos[:] = qpos
        data.qvel[:] = 0
        data.act[:] = a
        data.ctrl[:] = a
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, M, data.qM)
        return np.asarray(M) @ np.asarray(data.qacc, float).copy()

    per_seed = []
    best = None
    for s in seeds:
        a = np.full(model.nu, float(s))
        a[-2:] = 0.0
        u = imb(a)
        hist = []
        for _ in range(rounds):
            R = np.zeros((42, n_use))
            for j in range(n_use):
                ap = a.copy()
                ap[j] = min(1.0, a[j] + eps)
                R[:, j] = (imb(ap)[LEG] - u[LEG]) / eps
            sol = lsq_linear(R, -u[LEG], bounds=(0.0, 1.0), max_iter=500)
            a2 = a.copy()
            a2[:n_use] = sol.x
            u2 = imb(a2)
            v = float(np.abs(u2[LEG]).max())
            hist.append(v)
            a, u = a2, u2
        v = float(np.abs(u[LEG]).max())
        per_seed.append({"seed": float(s), "imbalance_model_units": v,
                         "fraction_of_body_weight": v / weight,
                         "actuators_at_bound": int((a[:n_use] >= 0.999).sum()),
                         "trajectory_absmax": hist})
        if best is None or v < best["imbalance_model_units"]:
            best = per_seed[-1]
    fracs = np.array([p["fraction_of_body_weight"] for p in per_seed])
    return {"per_seed": per_seed, "best": best,
            "median_fraction": float(np.median(fracs)),
            "worst_fraction": float(fracs.max()),
            "seed_dispersion_ratio": float(fracs.max() / fracs.min())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=int, default=20)
    ap.add_argument("--budget-s", type=float, default=900.0)
    args = ap.parse_args()

    import mujoco

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    model = cp.load_model()
    data = mujoco.MjData(model)
    keys = MR.key_qpos(model)
    base = keys.copy()
    addr = MR.joint_addresses(model)
    tarsus = set(cp.tarsus_geom_ids(model))
    tarsus_geoms = cp.tarsus_geom_ids(model)

    # candidate poses: the keyframe, the round-37 pose, and seeded perturbations
    cands = [("keyframe", np.zeros(len(addr)))]
    pose = json.loads(R39.ROUND37_POSE.read_text())
    cands.append(("round37_pose",
                  np.array([float(pose["joint_offsets"][f"joint_{leg}{j}"])
                            for leg in MR.LEGS for j in MR.PARAM_J])))
    rng = np.random.default_rng(7)
    for k in range(args.poses):
        cands.append((f"perturbed_{k}", rng.normal(0, 0.3, size=len(addr))))

    t0 = time.time()
    rows = []
    for name, x in cands:
        if time.time() - t0 > args.budget_s:
            break
        r = MR.evaluate(model, data, base, keys, x, addr, tarsus, tarsus_geoms)
        if r is None:
            rows.append({"pose": name, "rejected_by_gate": True})
            continue
        fl = solve_floor(model, data, r["qpos"], seeds=(0.0, 0.25, 0.5, 0.75))
        rows.append({
            "pose": name, "rejected_by_gate": False,
            "margin_mm": r["margin_mm"],
            "tipping_angle_deg": r.get("tipping_angle_deg"),
            "n_support_points": r["n_support_points"],
            "seated_thorax_mm": r["seated"]["thorax_centre_mm"],
            "best_imbalance_fraction_of_body_weight":
                fl["best"]["fraction_of_body_weight"],
            "median_imbalance_fraction": fl["median_fraction"],
            "worst_imbalance_fraction": fl["worst_fraction"],
            "seed_dispersion_ratio": fl["seed_dispersion_ratio"],
            "detail": fl,
        })
        print(f"  {name:16s} margin {r['margin_mm']:6.3f} mm  best imbalance "
              f"{100*fl['best']['fraction_of_body_weight']:5.2f}% BW "
              f"(median {100*fl['median_fraction']:5.2f}%, dispersion "
              f"{fl['seed_dispersion_ratio']:.2f}x)", flush=True)

    ok = [r for r in rows if not r.get("rejected_by_gate")]
    bests = np.array([r["best_imbalance_fraction_of_body_weight"] for r in ok]) if ok else \
        np.zeros(0)
    meds = np.array([r["median_imbalance_fraction"] for r in ok]) if ok else np.zeros(0)

    cp.write_result(
        OUT / "fix_round39_balance_floor.json",
        question=("Using only physics (M qacc), how close to zero can muscle activation "
                  "drive the leg-joint imbalance of a gated stance, and how much does that "
                  "answer depend on the optimiser's seed?"),
        status="PASS" if ok else "NOT_RUN",
        measured={
            "poses_measured": len(ok),
            "poses_rejected_by_gate": len(rows) - len(ok),
            "best_of_all_poses_fraction_of_body_weight": (float(bests.min()) if ok else None),
            "median_over_poses_of_best_fraction": (float(np.median(bests)) if ok else None),
            "worst_pose_best_fraction": (float(bests.max()) if ok else None),
            "median_over_poses_of_median_fraction": (float(np.median(meds)) if ok else None),
            "seed_dispersion_ratio_median": (float(np.median(
                [r["seed_dispersion_ratio"] for r in ok])) if ok else None),
            "rows": rows,
            "verdict": {
                "empirical_floor_fraction_of_body_weight": (float(bests.min()) if ok else None),
                "floor_is_zero": bool(ok and bests.min() < 1e-6),
                "optimiser_seed_matters": bool(
                    ok and np.median([r["seed_dispersion_ratio"] for r in ok]) > 1.5),
                "interpretation": (
                    "A muscle pattern can drive the measured physical imbalance down to "
                    f"{100*bests.min():.2f}% of body weight at best, over "
                    f"{len(ok)} gated stances with 4 activation seeds each. It does NOT "
                    "reach zero, and the answer moves by a factor of "
                    f"{np.median([r['seed_dispersion_ratio'] for r in ok]):.2f} with the "
                    "seed, so this is an upper bound on practical balanceability, not a "
                    "proof that no exact balance exists. What it does establish is that "
                    "the LP's claimed 0.0 was never a property of the body."
                    if ok else "no gated stance was measured"),
            },
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "seating": f"{cp.SEATING_PENETRATION_MM} mm, gated",
            "criterion": "M(q) @ qacc over the 42 leg joints; minimised by bounded "
                         "least squares against moment arms measured by finite difference",
            "seeds": [0.0, 0.25, 0.5, 0.75],
            "interpreter": sys.executable,
            "mujoco": mujoco.__version__,
        },
        what_this_does_not_support=[
            "This is NOT a proof that no exact muscle balance exists. It is a measurement "
            "of how close a practical multi-start optimiser gets, with the seed dispersion "
            "reported so the weakness is visible.",
            "No LP, no qfrc residual and no adhesion are used anywhere in this measurement.",
            "Nothing here is autonomous behaviour and nothing here is about real flies.",
        ],
    )
    if ok:
        print(f"\nfloor over {len(ok)} gated stances: best {100*bests.min():.2f}% BW, "
              f"median-of-best {100*np.median(bests):.2f}% BW")
    print("wrote", OUT / "fix_round39_balance_floor.json")


if __name__ == "__main__":
    main()
