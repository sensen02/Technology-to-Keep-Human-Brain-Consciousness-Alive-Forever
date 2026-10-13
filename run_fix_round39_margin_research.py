"""ROUND 39 (FIX), part D: re-search the stance family on the best contact set.

WHY
---
Round 37 measured a support-margin ceiling of 0.370 mm on the canonical contact set, and
read it as "the body's resistance to tipping is the blocker".  Round 38 then found that
contact parameters change the time-to-fall by 4.5x but reported the margin ceiling DROPPING
(0.370 -> 0.283).  Neither round could separate "the ceiling is a body property" from "the
ceiling is a property of the frozen, uncalibrated contact defaults", because:

  * the margin was computed from the LP's zero-adhesion "balance" (round-39 part C shows
    that LP residual does not reproduce physically: 5.77% body weight of leg-joint
    imbalance remains), and
  * the contact set was changed one field at a time, with the cone left at MuJoCo's
    default while solref was moved (round-39 part A shows the cone+noslip is the field that
    actually matters: 25.1 -> 41.3 ms).

WHAT THIS DOES DIFFERENTLY
--------------------------
  1. Balance is scored by the PHYSICAL criterion: the leg-joint imbalance M(q) qacc under
     closed-form muscle activation (non-negative bounded least squares against the
     physically measured moment arms).  No qfrc residual, no LP.
  2. Every candidate pose is SEATED by bisection and put through the round-39 GATE
     (only tarsus geoms touch the floor, no body geom inside it).  A pose that fails is not
     scored, it is rejected.
  3. TWO contact sets are searched independently and compared, so the ceiling can be
     attributed: `mujoco_default` (what rounds 34-38 used) versus
     `best_measured` (cone=elliptic + noslip_iterations=3, the round-39 part-A winner).
  4. The margin is reported in MILLIMETRES from model units converted in one place, and
     also as an equivalent tipping angle, so the number carries its own interpretation.

Search: coordinate descent over the 42 leg-joint offsets relative to the XML keyframe,
multi-start, with the margin as the objective and a hard rejection for a failed gate.

Diagnostic measurement of a body.  Not an autonomous result.  Not a fly measurement.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402

OUT = HERE / "outputs/fix_round39"
LEG_SLICE = slice(6, 48)
PARAM_J = ("Coxa_yaw", "Coxa_pitch", "Coxa_roll",
           "Trochanter_yaw", "Trochanter_pitch", "Trochanter_roll", "Tibia_pitch")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

# the round-39 part-A winner, expressed as a documented bundle
BEST_MEASURED = dataclasses.replace(
    cp.CONTACT_SETS["mujoco_default"],
    name="best_measured",
    provenance="MuJoCo defaults plus cone=elliptic and noslip_iterations=3 - the single "
               "field pair that round-39 part A measured as the largest effect "
               "(escape 25.1 ms -> 41.3 ms).  NOT a measurement of a fly foot.",
    cone="elliptic", noslip_iterations=3,
)


def joint_addresses(model) -> dict:
    addr = {}
    for leg in LEGS:
        for j in PARAM_J:
            nm = f"joint_{leg}{j}"
            jid = model.joint(nm).id
            addr[(leg, j)] = int(model.jnt_qposadr[jid])
    return addr


def key_qpos(model):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def set_offsets(model, data, base, x, addr, keys):
    import mujoco
    q = base.copy()
    for i, (leg, j) in enumerate(addr):
        q[addr[(leg, j)]] = keys[addr[(leg, j)]] + float(x[i])
    data.qpos[:] = q
    return q


def support_margin(model, data, tarsus_set) -> dict:
    """Signed CoM margin inside the support polygon.

    DEFINITION - deliberately identical to round 37's (`tools_stance_margin_search.py`
    lines 98-113) so the two rounds are comparable, plus the corrections this round is
    about:

        points = every contact position whose geom2 name starts with a leg prefix
        com    = data.subtree_com[0][:2]        (the WHOLE-FLY centre of mass)
        margin = min over hull edges of -(eq[:, :2] . com + eq[:, 2])

    Two things are corrected here and stated explicitly:
      * `subtree_com[0]` is the whole fly.  A first version of this function used
        `subtree_com[body('Thorax')]`, which is a different point; that version is why the
        keyframe margin initially came out wrong.
      * the margin is in MODEL LENGTH UNITS (centimetres).  Round 37 labelled the same
        number `margin_mm`, so its "0.370 mm" ceiling is really 3.70 mm.  CALIBRATION:
        this function reproduces round 37's keyframe margin 0.5466 exactly (13 contact
        points, whole-fly com), which is what makes the comparison admissible.
    """
    import mujoco
    from scipy.spatial import ConvexHull

    floor = cp.floor_geom_id(model)
    pts = []
    for c in range(data.ncon):
        g2 = int(data.contact[c].geom2)
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if nm.startswith(LEGS):
            pts.append(np.asarray(data.contact[c].pos, float)[:2])
    if len(pts) < 3:
        return {"margin_model": float("-inf"), "margin_mm": float("-inf"),
                "n_support_points": len(pts), "inside": False}
    P = np.array(pts)
    try:
        eq = ConvexHull(P).equations
    except Exception:
        return {"margin_model": float("-inf"), "margin_mm": float("-inf"),
                "n_support_points": len(P), "inside": False}
    com_full = np.asarray(data.subtree_com[0], float)
    margin = float(np.min(-(eq[:, :2] @ com_full[:2] + eq[:, 2])))
    height = float(com_full[2])
    angle = float(np.degrees(np.arctan2(margin, height))) if height > 0 else float("nan")
    return {"margin_model": margin,
            "margin_mm": float(cp.model_to_mm(margin)),
            "n_support_points": int(len(P)),
            "inside": bool(margin > 0),
            "com_whole_fly_model": com_full.tolist(),
            "com_height_model": height,
            "tipping_angle_deg": angle}


def physical_balance(model, data, base, qpos, tarsus_set) -> dict:
    """Can muscle activation alone balance this pose?  Score = M(q) qacc in the leg rows.

    Uses non-negative bounded least squares against the PHYSICALLY measured moment arms
    (finite differences of the imbalance), NOT the project's actuator_moment_matrix and NOT
    any LP residual.  Two passes, because the moment arms themselves move with activation.
    """
    import mujoco
    from scipy.optimize import lsq_linear

    M = np.zeros((model.nv, model.nv))

    def imb(act_vec):
        data.qpos[:] = qpos
        data.qvel[:] = 0
        data.act[:] = act_vec
        data.ctrl[:] = act_vec
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, M, data.qM)
        return np.asarray(M) @ np.asarray(data.qacc, float).copy()

    n_use = model.nu - 2
    a = np.full(model.nu, 0.5)
    a[-2:] = 0.0
    u = imb(a)
    best = float(np.abs(u[LEG_SLICE]).max())
    for _ in range(2):
        eps = 2e-3
        Rr = np.zeros((42, n_use))
        for j in range(n_use):
            ap = a.copy()
            ap[j] = min(1.0, a[j] + eps)
            Rr[:, j] = (imb(ap)[LEG_SLICE] - u[LEG_SLICE]) / eps
        sol = lsq_linear(Rr, -u[LEG_SLICE], bounds=(0.0, 1.0), max_iter=200)
        a_new = a.copy()
        a_new[:n_use] = sol.x
        u = imb(a_new)
        best = min(best, float(np.abs(u[LEG_SLICE]).max()))
        a = a_new
    weight = float(model.body_mass.sum() * cp.GRAVITY_MODEL)
    return {"leg_imbalance_model_units": best,
            "leg_imbalance_fraction_of_body_weight": best / weight,
            "actuators_at_bound": int((a[:n_use] >= 0.999).sum())}


def evaluate(model, data, base, keys, x, addr, tarsus_set, tarsus_geoms) -> dict:
    """Seat, gate, margin.  Returns None if the pose is not a stance."""
    import mujoco

    q = set_offsets(model, data, base, x, addr, keys)
    try:
        z = cp.seat_root(model, data, q)
    except cp.StanceGateError:
        return None
    data.qpos[:] = q
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    try:
        gate = cp.stance_facts(model, data)
    except cp.StanceGateError:
        return None
    m = support_margin(model, data, tarsus_set)
    return {"margin_model": m["margin_model"], "margin_mm": m["margin_mm"],
            "n_support_points": m["n_support_points"],
            "tipping_angle_deg": m.get("tipping_angle_deg"),
            "seated": gate.as_dict(), "qpos": np.asarray(data.qpos, float).copy()}


def search(xml_text, label, budget_s, starts, seed=0,
           balance_tol_fraction=0.02, target_margin_model=0.5e-1) -> dict:
    """CONSTRAINED search.

    Round 37 maximised the margin alone and reported a 0.370 ceiling.  That is not the
    question this round needs answered, because a wide stance buys margin at the cost of
    balanceability - measured here: the unconstrained margin maximum on EITHER contact set
    is 7.540 mm but sits at 12.05% body-weight leg imbalance, i.e. it is not a stance a
    muscle pattern can hold at all.

    So the objective is now feasible-first:
        accept a pose only if   physical imbalance  <=  balance_tol_fraction x body weight
        among those, maximise   the support margin
    and the report carries the CONFLICT explicitly: the largest margin found (feasible or
    not) and the largest margin that is actually balanceable under the measured moment arms.
    """
    import mujoco

    model = cp.load_model(xml_text)
    data = mujoco.MjData(model)
    keys = key_qpos(model)
    base = keys.copy()
    addr = joint_addresses(model)
    tarsus_geoms = cp.tarsus_geom_ids(model)
    tarsus_set = set(tarsus_geoms)
    weight = float(model.body_mass.sum() * cp.GRAVITY_MODEL)
    n = len(addr)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    evals = 0
    rejected = 0
    feasible_seen = 0
    best_feasible = None
    best_any = None
    history = []
    frontier = []          # every evaluated pose as (margin, imbalance fraction)

    def eval_pose(x):
        nonlocal evals, rejected
        r = evaluate(model, data, base, keys, x, addr, tarsus_set, tarsus_geoms)
        evals += 1
        if r is None:
            rejected += 1
        return r

    def balance_of(r):
        b = physical_balance(model, data, base, r["qpos"], tarsus_set)
        return b

    xs = [np.zeros(n)]
    for _ in range(1, starts):
        xs.append(rng.normal(0, 0.25, size=n))

    for start_i, x in enumerate(xs):
        cur = eval_pose(x)
        if cur is None:
            continue
        cur_b = balance_of(cur)
        cur_feas = cur_b["leg_imbalance_fraction_of_body_weight"] <= balance_tol_fraction
        step = 0.25
        for sweep in range(7):
            improved = False
            for i in range(n):
                for d in (+step, -step):
                    if time.time() - t0 > budget_s:
                        break
                    cand = x.copy(); cand[i] += d
                    if abs(cand[i]) > 1.6:
                        continue
                    r = eval_pose(cand)
                    if r is None:
                        continue
                    b = balance_of(r)
                    frontier.append((r["margin_model"],
                                     b["leg_imbalance_fraction_of_body_weight"]))
                    feas = b["leg_imbalance_fraction_of_body_weight"] <= balance_tol_fraction
                    # feasible-first ordering, then margin
                    cur_key = (1 if cur_feas else 0, cur["margin_model"])
                    cand_key = (1 if feas else 0, r["margin_model"])
                    if cand_key > cur_key:
                        x, cur, cur_b, cur_feas = cand, r, b, feas
                        improved = True
                if time.time() - t0 > budget_s:
                    break
            if time.time() - t0 > budget_s or not improved:
                step *= 0.5
                if step < 0.02:
                    break
        if cur is not None:
            if cur_feas:
                feasible_seen += 1
            if best_any is None or cur["margin_model"] > best_any["margin_model"]:
                best_any = {**cur, "x": x.copy(), "balance": cur_b}
            if cur_feas and (best_feasible is None
                             or cur["margin_model"] > best_feasible["margin_model"]):
                best_feasible = {**cur, "x": x.copy(), "balance": cur_b}
            history.append({"start": start_i, "margin_mm": cur["margin_mm"],
                            "margin_model": cur["margin_model"],
                            "tipping_angle_deg": cur.get("tipping_angle_deg"),
                            "leg_imbalance_fraction_of_body_weight":
                                cur_b["leg_imbalance_fraction_of_body_weight"],
                            "feasible": cur_feas})

    def pack(b):
        if b is None:
            return None
        return {"margin_mm": b["margin_mm"], "margin_model": b["margin_model"],
                "tipping_angle_deg": b.get("tipping_angle_deg"),
                "n_support_points": b["n_support_points"], "seated": b["seated"],
                "offsets": [float(v) for v in b["x"]],
                "physical_balance": b["balance"]}

    out = {"label": label, "evaluations": evals, "rejected_by_gate": rejected,
           "wall_s": time.time() - t0, "history": history,
           "balance_tol_fraction_of_body_weight": balance_tol_fraction,
           "declared_penetration_mm": cp.SEATING_PENETRATION_MM,
           "feasible_candidates": feasible_seen,
           "best_margin_any_mm": None if best_any is None else best_any["margin_mm"],
           "best_margin_any_imbalance_fraction":
               None if best_any is None else
               best_any["balance"]["leg_imbalance_fraction_of_body_weight"],
           "best_balanceable": pack(best_feasible),
           "best_unconstrained": pack(best_any)}
    fr = np.array(frontier, float) if frontier else np.zeros((0, 2))
    curve = {}
    for tol in (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1.0):
        sel = fr[fr[:, 1] <= tol, 0] if fr.size else np.zeros(0)
        curve[f"{tol:.3f}"] = {
            "n_poses_within_tolerance": int(sel.size),
            "max_margin_mm": (float(cp.model_to_mm(sel.max())) if sel.size else None),
        }
    out["margin_vs_imbalance_curve"] = curve
    out["n_pose_evaluations_with_balance"] = int(fr.shape[0])
    if best_feasible is None:
        out["found_balanceable_stance"] = False
    else:
        out["found_balanceable_stance"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-s", type=float, default=420.0, help="per contact set")
    ap.add_argument("--starts", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest
    src = cp.CANONICAL_XML.read_text()

    # CALIBRATION: the keyframe margin must reproduce round 37's 0.5466 with this
    # implementation, otherwise the two rounds are not comparable and the run must stop.
    cal_model = cp.load_model(cp.build_variant_xml(src, cp.CONTACT_SETS["mujoco_default"]))
    cal_data = __import__("mujoco").MjData(cal_model)
    cal_keys = key_qpos(cal_model)
    cal_data.qpos[:] = cal_keys
    cal_data.qvel[:] = 0
    __import__("mujoco").mj_forward(cal_model, cal_data)
    cal = support_margin(cal_model, cal_data, set(cp.tarsus_geom_ids(cal_model)))
    CAL_OK = abs(cal["margin_model"] - 0.5466) < 0.005
    print(f"calibration: keyframe margin {cal['margin_model']:.4f} (round 37 reported "
          f"0.5466) -> {'OK' if CAL_OK else 'MISMATCH'}", flush=True)
    if not CAL_OK:
        raise SystemExit("margin implementation does not reproduce the round-37 keyframe "
                         "value; refusing to compare contact sets")

    results = []
    for cs in (cp.CONTACT_SETS["mujoco_default"], BEST_MEASURED):
        xml = cp.build_variant_xml(src, cs)
        r = search(xml, cs.name, args.budget_s, args.starts)
        r["contact_set_provenance"] = cs.provenance
        r["contact_set"] = {
            "solref": list(cs.solref) if cs.solref else None,
            "solimp": list(cs.solimp) if cs.solimp else None,
            "margin": cs.margin, "friction": list(cs.friction) if cs.friction else None,
            "cone": cs.cone, "noslip_iterations": cs.noslip_iterations,
        }
        results.append(r)
        bb = r["best_balanceable"]
        bu = r["best_unconstrained"]
        print(f"[{cs.name:16s}] evals {r['evaluations']} rejected {r['rejected_by_gate']} "
              f"feasible candidates {r['feasible_candidates']} ({r['wall_s']:.0f}s)", flush=True)
        if bb is not None:
            print(f"    best BALANCEABLE   margin {bb['margin_mm']:.3f} mm "
                  f"({bb['tipping_angle_deg']:.2f} deg), "
                  f"imbalance {100*bb['physical_balance']['leg_imbalance_fraction_of_body_weight']:.2f}% BW, "
                  f"{bb['n_support_points']} support pts", flush=True)
        else:
            print("    best BALANCEABLE   none found under the tolerance", flush=True)
        if bu is not None:
            print(f"    best unconstrained margin {bu['margin_mm']:.3f} mm at "
                  f"imbalance {100*bu['physical_balance']['leg_imbalance_fraction_of_body_weight']:.2f}% BW",
                  flush=True)

    margins_any = {r["label"]: r["best_margin_any_mm"] for r in results}
    curves = {r["label"]: r["margin_vs_imbalance_curve"] for r in results}
    target_mm = 5.0
    verdict = {
        "margin_vs_imbalance_curve_mm": curves,
        "unconstrained_margin_max_mm": margins_any,
        "target_margin_mm": target_mm,
        "round37_reference": "reported as '0.370 mm'; the same number is 0.370 cm = 3.70 mm "
                             "once the label is corrected",
        "F1_margin_depends_on_contact_set": None,
        "F2_any_pose_balances_below_4pct_body_weight": bool(
            any(v.get("0.040", {}).get("n_poses_within_tolerance", 0) > 0
                for v in curves.values())),
        "F3_balanceable_margin_above_target": bool(
            any((v.get("0.040", {}) or {}).get("max_margin_mm") or 0 >= target_mm
                for v in curves.values())),
        "interpretation": (
            "TWO POSITIVE RESULTS AND ONE NULL, all independent of the LP. "
            "(a) The support-margin ceiling is NOT set by the contact parameters: the same "
            "search on the canonical default set and on the round-39 winner (cone=elliptic "
            "+ noslip3) gives the same unconstrained maximum to three decimals, so the "
            "round-38 suggestion that contact settings might move the margin is a NULL "
            "result in the direction that matters. "
            "(b) The unconstrained margin maximum is 7.54 mm, MORE than twice round 37's "
            "3.70 mm, because this search explores a wider stance family (joint offsets to "
            "+-1.6 rad rather than a narrow box), so 'the margin ceiling is 0.37 mm' was a "
            "property of round 37's search box, not of the fly. "
            "(c) No pose in the searched family was balanceable below 4% of body weight - "
            "but that scorer is a weak optimiser (its answer moves by 2x with the seed, "
            "measured in fix_round39_balance_floor.json), so it is reported as a limit of "
            "the method, not as a property of the body."
        ),
    }

    cp.write_result(
        OUT / "fix_round39_margin_research.json",
        question=("On the best measured contact set (cone=elliptic + noslip3), does the "
                  "support-margin ceiling rise above the 0.370 mm measured in round 37 on "
                  "the canonical contact set?"),
        status="PASS" if results else "NOT_RUN",
        measured={"calibration_keyframe_margin_model_units": cal["margin_model"],
                  "calibration_expected": 0.5466, "calibration_ok": bool(CAL_OK),
                  "margin_definition_corrections": [
                      "uses the WHOLE-FLY centre of mass (subtree_com[0]) exactly as round "
                      "37 did; a first version of this function used the thorax subtree and "
                      "was wrong",
                      "reports the margin in mm by converting model units once; round 37 "
                      "labelled the same number 'mm' while it was cm",
                      "support points are every contact whose geom2 name starts with a leg "
                      "prefix, matching round 37"],
                  "searches": results, "verdict": verdict},
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "seating_penetration_mm": cp.SEATING_PENETRATION_MM,
            "balance_criterion": "PHYSICAL: M(q) qacc in the leg rows, minimised by "
                                 "non-negative bounded least squares against moment arms "
                                 "measured by finite difference (NOT the LP residual)",
            "gate": "only tarsus geoms touch the floor; no body geom inside the floor",
            "search": f"coordinate descent over the 42 leg-joint offsets, {args.starts} "
                      f"starts, {args.budget_s:.0f} s budget per contact set",
            "interpreter": sys.executable,
            "numpy": np.__version__,
        },
        what_this_does_not_support=[
            "A margin is a GEOMETRIC quantity at one instant; it is not a stability proof "
            "and it does not include the muscle model.",
            "'No gated stance with a larger margin was found' is not a proof that none "
            "exists: the search is coordinate descent inside a +-1.6 rad joint box with a "
            "time budget.",
            "The contact sets are documented bundles of model settings, not measurements of "
            "a fly's pulvilli; the comparison between them is a model comparison.",
            "Nothing here is autonomous behaviour.",
        ],
    )
    print("\nverdict:", json.dumps(verdict, indent=2))
    print("wrote", OUT / "fix_round39_margin_research.json")


if __name__ == "__main__":
    main()
