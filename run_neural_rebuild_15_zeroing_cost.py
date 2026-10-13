"""NEURAL INTERFACE REBUILD, step 15: what does honest zeroing cost a feasible stance?

THE QUESTION
------------
Step 11 resolved 74 of 90 actuators to a motor-neuron pool on their OWN leg and side, and
refused to substitute another leg's pool.  Step 12 measured that in the round-37 feasible
stance, 43.2% of the activation sits on the 16 actuators that then have no own-leg source
(36.8% cross-leg substitution + 6.4% no pool anywhere).

So: if the runtime honestly zeroes those 16, can the OTHER 74 still balance the pose?  The
answer decides whether the corrected wiring leaves anything drivable at all, and it must be
measured before any stance re-search is worth running.

METHOD (no LP residual, no qfrc bookkeeping - the round-39 lesson)
------------------------------------------------------------------
Balance is scored by the PHYSICAL criterion used since step 8:

        imbalance = M(q) . qacc        over the 42 leg joints, in model force units

and the achievable floor is found by non-negative bounded least squares against the moment-arm
matrix MEASURED by finite differences of that same imbalance.  Three conditions are compared,
all at the SAME pose and with the same solver settings:

    ALL90     all 90 actuators free                       (the historical setting)
    COVERED74 the 16 unresolved actuators pinned to 0     (the corrected wiring)
    UNCOVERED only the 16 unresolved actuators free       (a control: how much do THEY carry)

A cost is then reported as the ratio of the imbalance floor, and the pose search is only
justified if that ratio is materially larger than 1.

Read-only on the canonical asset.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
POOLS = OUT / "neural_rebuild_11_mn_pools.json"
LEG = slice(6, 48)
SEEDS = (0.0, 0.25, 0.5, 0.75)
ROUNDS = 8


def main() -> None:
    import mujoco

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    pools = json.loads(POOLS.read_text())["measured"]
    resolved = [r["actuator"] for r in pools["resolved"]]
    unresolved = [u["actuator"] for u in pools["unresolved"]]
    assert len(resolved) == 74 and len(unresolved) == 16, (len(resolved), len(unresolved))

    model = cp.load_model()
    data = mujoco.MjData(model)
    pose = json.loads(R39.ROUND37_POSE.read_text())
    R39.apply_round37_pose(model, data, pose)
    base = data.qpos.copy()
    z = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data)
    seated = data.qpos.copy()

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
             for i in range(model.nu)]
    idx_res = np.array([names.index(n) for n in resolved if n in names])
    idx_unres = np.array([names.index(n) for n in unresolved if n in names])
    missing_from_model = [n for n in resolved + unresolved if n not in names]
    weight = float(model.body_mass.sum() * cp.GRAVITY_MODEL)

    M = np.zeros((model.nv, model.nv))

    def imb(act_vec):
        data.qpos[:] = seated
        data.qvel[:] = 0
        data.act[:] = act_vec
        data.ctrl[:] = act_vec
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, M, data.qM)
        return np.asarray(M) @ np.asarray(data.qacc, float).copy()

    def floor(free_idx, label):
        """Best leg-joint imbalance reachable using only `free_idx` actuators."""
        t0 = time.time()
        eps = 2e-3
        best = None
        used_seeds = []
        for s in SEEDS:
            a = np.zeros(model.nu)
            a[free_idx] = float(s)
            u = imb(a)
            for _ in range(ROUNDS):
                R = np.zeros((42, free_idx.size))
                for j, ai in enumerate(free_idx):
                    ap = a.copy()
                    ap[ai] = min(1.0, ap[ai] + eps)
                    R[:, j] = (imb(ap)[LEG] - u[LEG]) / eps
                sol = lsq_linear(R, -u[LEG], bounds=(0.0, 1.0), max_iter=500)
                a2 = a.copy()
                a2[free_idx] = sol.x
                u2 = imb(a2)
                a, u = a2, u2
            v = float(np.abs(u[LEG]).max())
            used_seeds.append(v)
            if best is None or v < best[0]:
                best = (v, a.copy())
        return {"label": label, "n_free_actuators": int(free_idx.size),
                "imbalance_model_units": best[0],
                "fraction_of_body_weight": best[0] / weight,
                "per_seed": used_seeds,
                "seed_dispersion_ratio": max(used_seeds) / min(used_seeds) if min(used_seeds) > 0 else None,
                "actuators_at_bound": int((best[1][free_idx] >= 0.999).sum()),
                "wall_s": time.time() - t0}

    all90 = np.arange(model.nu - 2)          # 88 muscles (the last two are adhesion pads)
    conditions = [
        floor(all90, "ALL_MUSCLES_FREE"),
        floor(idx_res, "COVERED74_ONLY"),
        floor(idx_unres, "UNCOVERED16_ONLY"),
    ]
    ratio = conditions[1]["fraction_of_body_weight"] / conditions[0]["fraction_of_body_weight"]
    verdict = {
        "cost_of_honest_zeroing_ratio": float(ratio),
        "zeroing_makes_balance_materially_harder": bool(ratio > 1.5),
        "reading": ("zeroing the 16 uncovered actuators raises the achievable imbalance floor "
                    "by this factor at the SAME pose; a factor near 1 means the uncovered "
                    "actuators were not load-bearing and no stance re-search is needed for "
                    "this reason, while a large factor means the pose must be re-searched"),
        "uncovered16_carry": conditions[2]["fraction_of_body_weight"],
    }

    cp.write_result(
        OUT / "neural_rebuild_15_zeroing_cost.json",
        question=("If the runtime honestly zeroes the 16 actuators that have no own-leg pool, "
                  "can the remaining 74 still balance a known-feasible stance?"),
        status="PASS",
        measured={
            "seated": gate.as_dict(),
            "body_weight_model_units": weight,
            "n_resolved": len(resolved), "n_unresolved": len(unresolved),
            "actuators_not_in_model": missing_from_model,
            "conditions": conditions,
            "verdict": verdict,
            "solver": {"seeds": list(SEEDS), "rounds": ROUNDS,
                       "criterion": "M(q) @ qacc over the 42 leg joints, minimised by bounded "
                                    "least squares against finite-difference moment arms"},
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_15_zeroing_cost.py"),
            "read_only": True,
            "pool_source": str(POOLS),
            "pose": str(R39.ROUND37_POSE),
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "The floor is the best value a bounded multi-start least-squares descent found, not a "
            "proof of the true minimum: the seed spread is reported so that weakness is visible.",
            "This measures ONE pose.  A pose that is feasible with all 90 actuators need not "
            "remain feasible without them, and a DIFFERENT pose might be feasible; only the "
            "measured cost at this pose is claimed.",
            "No controller, no body integration and no closed loop are involved.",
        ],
    )
    for c in conditions:
        print(f"  {c['label']:22s} free={c['n_free_actuators']:2d} "
              f"imbalance {c['imbalance_model_units']:.4f} "
              f"({100*c['fraction_of_body_weight']:.2f}% BW) "
              f"spread {c['seed_dispersion_ratio']}", flush=True)
    print(f"  cost of honest zeroing: {ratio:.2f}x")
    print("wrote", OUT / "neural_rebuild_15_zeroing_cost.json")


if __name__ == "__main__":
    main()
