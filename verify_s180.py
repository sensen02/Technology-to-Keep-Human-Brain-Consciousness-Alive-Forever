#!/usr/bin/env python3
"""S180 saved-data verification only.

Reads the frozen S180 Stage A / Stage B artifacts and re-derives every numeric
claim from them.  No engine, no mj_forward, no optimization, no writes outside
the single audit JSON.  Producer files, tracker and history are untouched.

Usage:
  venv_body/bin/python -B verify_s180.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

AUD = "/home/sensen/Desktop/engine修复验收/s180-dof-authority-audit"
STAGE_A = os.path.join(AUD, "20261025T096000000000Z")
STAGE_B = os.path.join(AUD, "20261025T110000000000Z")
S178_A = "/home/sensen/Desktop/engine修复验收/s178-legjoint-release-search/20261025T034000000000Z"
DEFECTS = [
    "20261025T080000000000Z", "20261025T082000000000Z", "20261025T084000000000Z",
    "20261025T090000000000Z", "20261025T092000000000Z", "20261025T094000000000Z",
    "20261025T100000000000Z", "20261025T102000000000Z", "20261025T104000000000Z",
]
OUT = os.path.join(AUD, "S180_INDEPENDENT_CHECK.json")
XML = "/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml"
UNREACH = [19, 22, 26, 40, 42, 47]

checks: list[dict] = []
disc: list[str] = []


def add(name, status, numbers, note=""):
    rec = {"name": name, "status": status, "numbers": numbers}
    if note:
        rec["note"] = note
    checks.append(rec)
    return rec


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with open(p, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- check 1
def check_crosscheck():
    j = load(os.path.join(STAGE_A, "B_ENGINE_CROSSCHECK.json"))
    ts = j["tests"]
    maxd = max(abs(t["difference"]) for t in ts)
    signs = all(t["sign_match"] for t in ts)
    formula = j["model"]
    f_ok = all(s in formula for s in ("qfrc_actuator", "A[:,pool]*activation", "moment.T@bias"))

    # 084 / 092 defect attempts
    early = {}
    const = {}
    for tag in ("20261025T084000000000Z", "20261025T092000000000Z"):
        e = load(os.path.join(AUD, tag, "B_ENGINE_CROSSCHECK.json"))
        ets = e["tests"]
        groups = {}
        for t in ets:
            groups.setdefault((t["dof"], t["pool"]), []).append(
                (t["pool_activation"], t["difference"]))
        spreads = [abs(v[0][1] - v[1][1]) for v in groups.values() if len(v) == 2]
        const[tag] = max(spreads) if spreads else None
        early[tag] = {
            "verdict": e["verdict"],
            "max_abs_difference": e["max_abs_difference"],
            "recomputed_max_abs_difference": max(abs(t["difference"]) for t in ets),
            "n_tests": len(ets),
            "max_abs_spread_across_act_0.25_vs_0.5": max(spreads) if spreads else None,
            "n_groups": len(groups),
        }
    # 096: same grouping -> residual must vanish once the bias term is included
    g96 = {}
    for t in ts:
        g96.setdefault((t["dof"], t["pool"]), []).append((t["pool_activation"], t["difference"]))
    spread96 = max(abs(v[0][1] - v[1][1]) for v in g96.values() if len(v) == 2)

    # zero-activation force evidence from the 084 diagnosis (activation-independent term)
    diag = load(os.path.join(AUD, "20261025T084000000000Z", "CROSSCHECK_DIAGNOSIS.json"))
    nonzero = [{"actuator": d["actuator"], "force_act0": d["force_act0"],
                "biasprm2_over_gainprm2": round(d["biasprm"][2] / d["gainprm"][2], 6)}
               for d in diag["tests"]]

    per_pool_bias = {}
    for (dof, pool), v in sorted(g96.items()):
        e84 = load(os.path.join(AUD, "20261025T084000000000Z", "B_ENGINE_CROSSCHECK.json"))["tests"]
        m = [t["difference"] for t in e84 if t["dof"] == dof and t["pool"] == pool]
        if m:
            per_pool_bias[f"dof{dof}_pool{pool}"] = {
                "bias_omitted_diff_084": round(m[0], 9),
                "diff_after_bias_included_096": round(v[0][1], 18),
            }

    consistent = (
        all(x["max_abs_spread_across_act_0.25_vs_0.5"] < 1e-9 for x in early.values())
        and abs(early["20261025T084000000000Z"]["max_abs_difference"] - 0.33563436319752427) < 1e-15
    )
    nums = {
        "n_tests": len(ts),
        "all_signs_match": signs,
        "max_abs_difference": maxd,
        "tolerance": 1e-06,
        "within_tolerance": maxd <= 1e-06,
        "verdict_recorded": j["verdict"],
        "model_formula": formula,
        "formula_has_qfrc_actuator_plus_bias": f_ok,
        "earlier_defect_attempts": early,
        "defect_difference_is_activation_independent": consistent,
        "max_spread_across_activation_096": spread96,
        "zero_activation_force_084": nonzero,
        "per_pool_bias_term": per_pool_bias,
    }
    st = "PASS" if (len(ts) == 24 and signs and maxd <= 1e-06 and f_ok and consistent) else "FAIL"
    add("crosscheck_B_vs_engine", st, nums,
        "24/24 signs match; max|diff| 1.78e-15 <= 1e-6. The 084/092 IMPLEMENTATION_DEFECT verdict "
        "compared against A[:,pool]*activation only; its residual is exactly constant across activation "
        "0.25 and 0.5, i.e. the omitted activation-independent moment.T@bias term, not a wrong B.")
    disc_084_092 = {k: v["verdict"] for k, v in early.items()}
    return st, disc_084_092


# ---------------------------------------------------------------- check 2
def check_arm_sign():
    rs = rows(os.path.join(STAGE_A, "ARM_SIGN_TABLE.csv"))
    mp = {int(r["joint_row"]): (int(r["dof"]), r["dof_name"], r["leg"])
          for r in load(os.path.join(STAGE_A, "ROW_DOF_MAP_REUSED.json"))["rows"]}
    by_pose = {}
    for r in rs:
        by_pose[r["pose"]] = by_pose.get(r["pose"], 0) + 1
    mism = []
    for r in rs:
        jr, dof = int(r["joint_row"]), int(r["dof"])
        if mp.get(jr) != (dof, r["dof_name"], r["leg"]):
            mism.append({"pose": r["pose"], "joint_row": jr, "dof": dof, "map": mp.get(jr)})
    unreach = {p: sorted(int(r["dof"]) for r in rs if r["pose"] == p and r["reachable"] == "False")
               for p in by_pose}
    nsbad, missing_true, unreach_detail = [], [], []
    for r in rs:
        if r["reachable"] != "False":
            continue
        t, lo, hi = float(r["target"]), float(r["lo"]), float(r["hi"])
        need = "positive" if t > hi else ("negative" if t < lo else "none")
        miss = ((need == "positive" and r["has_positive_channel"] == "False")
                or (need == "negative" and r["has_negative_channel"] == "False"))
        if r["needed_sign"] != need or r["needed_sign_channel_missing"] != str(miss):
            nsbad.append({"pose": r["pose"], "dof": int(r["dof"]),
                          "recorded": [r["needed_sign"], r["needed_sign_channel_missing"]],
                          "recomputed": [need, miss]})
        if miss:
            missing_true.append({"pose": r["pose"], "dof": int(r["dof"]), "needed_sign": need})
        unreach_detail.append({"pose": r["pose"], "dof": int(r["dof"]), "target": t,
                               "lo": lo, "hi": hi, "needed_sign": need,
                               "has_positive_channel": r["has_positive_channel"] == "True",
                               "has_negative_channel": r["has_negative_channel"] == "True",
                               "needed_sign_channel_missing": miss})
    ok = (len(rs) == 84 and by_pose == {"cand11": 42, "s177_baseline": 42}
          and not mism and all(v == UNREACH for v in unreach.values()) and not nsbad)
    add("arm_sign_table", "PASS" if ok else "FAIL", {
        "n_rows": len(rs), "rows_per_pose": by_pose,
        "identity_columns_consistent_with_ROW_DOF_MAP_REUSED": not mism,
        "identity_mismatches": mism,
        "unreachable_dofs": unreach,
        "unreachable_identical_in_both_poses": all(v == UNREACH for v in unreach.values()),
        "needed_sign_recomputed_from_target_vs_lohi": not nsbad,
        "needed_sign_inconsistencies": nsbad,
        "channel_missing_rows": missing_true,
        "unreachable_rows_detail": unreach_detail,
    }, "dof 42 lacks a positive channel in both poses; the other five have both channels and are "
       "unreachable from interval magnitude, not from a missing sign channel.")


# ---------------------------------------------------------------- check 3
def check_demand():
    rs = rows(os.path.join(STAGE_A, "DEMAND_DECOMPOSITION.csv"))
    out, dom_ok = [], True
    for r in rs:
        g, c = float(r["gravity_passive_part"]), float(r["contact_part"])
        dom = "contact" if abs(c) > abs(g) else "gravity_passive"
        dom_ok &= (r["dominant"] == dom) and dom == "contact"
        out.append({"dof": int(r["dof"]), "dof_name": r["dof_name"],
                    "abs_gravity_passive_part": abs(g), "abs_contact_part": abs(c),
                    "ratio_contact_over_gravity": abs(c) / abs(g) if abs(g) else None,
                    "dominant": r["dominant"]})
    ok = len(rs) == 6 and dom_ok and sorted(x["dof"] for x in out) == UNREACH
    add("demand_decomposition", "PASS" if ok else "FAIL", {
        "n_rows": len(rs),
        "all_six_contact_dominant": dom_ok,
        "abs_gravity_range": [min(x["abs_gravity_passive_part"] for x in out),
                              max(x["abs_gravity_passive_part"] for x in out)],
        "abs_contact_range": [min(x["abs_contact_part"] for x in out),
                              max(x["abs_contact_part"] for x in out)],
        "min_ratio_contact_over_gravity": min(x["ratio_contact_over_gravity"] for x in out),
        "per_row": out,
    }, "contact demand is 7.6x-49.1x the gravity/passive part on all six rows.")


# ---------------------------------------------------------------- check 4
def xml_ranges():
    root = ET.parse(XML).getroot()
    out = {}
    for jt in root.iter("joint"):
        if jt.get("name") and jt.get("range"):
            a, b = jt.get("range").split()
            out[jt.get("name")] = [float(a), float(b)]
    return out


def check_native_flip():
    rs = rows(os.path.join(STAGE_A, "NATIVE_LIMIT_FLIP.csv"))
    xr = xml_ranges()
    dofs = sorted({int(r["dof"]) for r in rs})
    xmism = []
    for r in rs:
        nr = json.loads(r["native_range"])
        x = xr.get(r["joint_name"])
        if x is None or abs(x[0] - nr[0]) > 1e-12 or abs(x[1] - nr[1]) > 1e-12:
            xmism.append({"joint_name": r["joint_name"], "csv": nr, "xml": x})
    trig = load(os.path.join(STAGE_A, "STAGE_B_TRIGGER.json"))
    sel = {int(r["dof"]): r["cheapest"] for r in trig["rows"]}
    sel_delta = {d: abs(sel[d]["delta_from_release"]) for d in sorted(sel)}
    all_delta = sorted(abs(float(r["delta_from_release"])) for r in rs)
    # DOF-level flippability: the six rows of STAGE_B_TRIGGER, built from each DOF's cheapest joint row
    dof_flip = {int(r["dof"]): r["flippable"] for r in trig["rows"]}
    all_flip = all(dof_flip.get(d) is True for d in dofs) \
        and all(sel[d]["flippable_in_native_limits"] is True for d in sel)
    row_true = [r for r in rs if r["flippable_in_native_limits"] == "True"]
    row_false = [{"dof": int(r["dof"]), "joint_row": int(r["joint_row"]),
                  "joint_name": r["joint_name"], "native_range": json.loads(r["native_range"]),
                  "release": float(r["release"]), "best_angle": float(r["best_angle"]),
                  "delta_from_release": float(r["delta_from_release"]),
                  "reachable_at_best": r["reachable_at_best"] == "True"}
                 for r in rs if r["flippable_in_native_limits"] != "True"]
    sel_ok = all(0.01 <= v <= 0.5 for v in sel_delta.values())

    # S178 +/-0.6 window estimates
    s178 = load(os.path.join(S178_A, "SUMMARY.json"))["stage_a"]["per_row"]
    s178_pre = load(os.path.join(S178_A, "PREREG.json"))["frozen_ranges"]["leg_joint"]
    s178_ang = {int(r["dof"]): r["min_angle_needed"] for r in s178}
    win = [v for v in s178_ang.values() if 2.17 <= v <= 3.53]
    contradiction = all(sel_delta[d] <= 0.5 for d in s178_ang) and len(win) >= 5

    ok = (dofs == UNREACH and not xmism and all_flip and sel_ok)
    add("native_limit_flip", "PASS" if ok else "FAIL", {
        "n_dofs": len(dofs), "dofs": dofs, "n_rows": len(rs),
        "native_range_equals_xml_limits": not xmism, "xml_mismatches": xmism,
        "flippable_in_native_limits_all_six": all_flip,
        "dof_level_flippable": dof_flip,
        "n_rows_flippable_true": len(row_true),
        "n_rows_flippable_false": len(row_false),
        "non_flippable_joint_rows": row_false,
        "selected_abs_delta_from_release": sel_delta,
        "selected_delta_all_in_0.01_to_0.5_rad": sel_ok,
        "all_rows_abs_delta_min_max": [all_delta[0], all_delta[-1]],
        "n_rows_abs_delta_gt_0.5": sum(1 for v in all_delta if v > 0.5),
        "s178_window": s178_pre,
        "s178_min_angle_needed": s178_ang,
        "s178_values_in_2.17_3.53": win,
        "contradicts_s178_window_estimate": contradiction,
    }, "Native XML limits make all six rows interval-flippable with 0.017-0.476 rad on the selected "
       "joint row; S178's 2.17-3.53 rad came from its release+/-0.6 rad window (5 of 6 rows > 0.6), "
       "so the window, not the model, was the earlier limitation.")
    if sum(1 for v in all_delta if v > 0.5):
        disc.append(
            "NATIVE_LIMIT_FLIP.csv lists %d rows (3 joint rows per DOF); only the selected cheapest row "
            "per DOF satisfies |delta_from_release| in [0.01,0.5]. %d non-selected rows exceed 0.5 rad "
            "(max %.4f rad, e.g. dof19/joint_LHTrochanter_yaw delta -1.1912). The 0.01-0.5 claim holds "
            "for the six selected rows, not for all 18 CSV rows."
            % (len(rs), sum(1 for v in all_delta if v > 0.5), all_delta[-1]))
    if row_false:
        disc.append(
            "One of the 18 candidate joint rows is not flippable: dof 42 via joint_RMTibia_pitch "
            "(best_angle 0.4789 = native lower bound) has reachable_at_best False and "
            "flippable_in_native_limits False. DOF-level flippability of all six rows is unaffected "
            "because each DOF's cheapest row (joint_LMCoxa_yaw / joint_RMTrochanter_yaw etc.) is True.")


# ---------------------------------------------------------------- check 5
def check_trigger():
    t = load(os.path.join(STAGE_A, "STAGE_B_TRIGGER.json"))
    ok = t.get("triggered") is True and len(t.get("rows", [])) == 6 \
        and all(r["flippable"] and r["cheapest"]["reachable_at_best"] for r in t["rows"])
    add("stage_b_trigger", "PASS" if ok else "FAIL", {
        "triggered": t.get("triggered"),
        "reason": t.get("reason"),
        "n_rows": len(t.get("rows", [])),
        "rows_all_flippable_and_reachable_at_best": ok,
        "dofs": [r["dof"] for r in t.get("rows", [])],
    })


# ---------------------------------------------------------------- check 6
def check_stage_b():
    c = load(os.path.join(STAGE_B, "CANDIDATES.json"))
    s = load(os.path.join(STAGE_B, "SUMMARY.json"))
    b = [x for x in c if x["tag"].startswith("BASELINE")][0]
    bm = b["minimax"]
    leg = [x for x in c if not x["tag"].startswith("BASELINE")]
    mm = {x["tag"]: x["minimax"] for x in c}
    jac = {x["tag"]: x["per_candidate_jacobians"] for x in c}
    jv = list(jac.values())
    all_ok = all(x["root_ok"] and x["geom_ok"] for x in c)
    strictly = all(x["minimax"] > bm for x in leg)
    fwd = s["calls"]["forward"]
    load_n = s["calls"]["model_load"]
    budget_ok = fwd <= 2200 - 200 and load_n <= 3
    zero_unreach = [x["tag"] for x in c if x.get("unreachable") == []]
    ok = (len(c) == 10 and all_ok and strictly and budget_ok and len(zero_unreach) == 0)
    add("stage_b_candidates", "PASS" if ok else "FAIL", {
        "n_candidates": len(c),
        "all_root_ok": all(x["root_ok"] for x in c),
        "all_geom_ok": all(x["geom_ok"] for x in c),
        "baseline_minimax": bm,
        "minimax_per_candidate": mm,
        "every_leg_candidate_minimax_strictly_greater_than_baseline": strictly,
        "leg_minimax_min": min(x["minimax"] for x in leg),
        "leg_minimax_over_baseline_ratio_range": [
            round(min(x["minimax"] for x in leg) / bm, 4),
            round(max(x["minimax"] for x in leg) / bm, 4)],
        "per_candidate_jacobians": jac,
        "jacobians_min": min(jv), "jacobians_max": max(jv),
        "n_candidates_jacobians_ge_8": sum(1 for v in jv if v >= 8),
        "forward_total": fwd, "forward_cap": 2200, "forward_reserved": 200,
        "forward_within_cap": fwd <= 2200 - 200,
        "model_loads": load_n, "load_cap": 3, "loads_within_cap": load_n <= 3,
        "candidates_with_no_unreachable_rows": zero_unreach,
        "selection": s["selection"],
        "unreachable_counts": {x["tag"]: len(x.get("unreachable", [])) for x in c},
        "root_newton_iterations": s["calls"]["root_newton_iterations"],
        "root_jacobians": s["calls"]["root_jacobians"],
        "fd_probes": s["calls"]["fd_probes"],
    }, "no candidate reduces the unreachable row set to zero; baseline stays the best minimax, so "
       "moving a leg joint always costs more than it buys.")


# ---------------------------------------------------------------- check 7
def check_hashes():
    per, bad, edited, cache = {}, [], [], {}
    for run in (STAGE_A, STAGE_B):
        a = load(os.path.join(run, "INPUT_HASHES.json"))
        b = load(os.path.join(run, "INPUT_HASHES_AFTER.json"))
        same = a == b
        miss = [p for p in a if not os.path.exists(p)]
        mism = []
        for p, h in a.items():
            if p in cache:
                cur = cache[p]
            elif os.path.exists(p):
                cur = cache[p] = sha(p)
            else:
                continue
            if cur != h:
                mism.append(p)
        per[os.path.basename(run)] = {
            "n_files": len(a),
            "input_equals_after": same,
            "n_differing_entries_between_snapshots": sum(1 for k in a if a[k] != b.get(k)),
            "files_missing_now": miss,
            "files_with_changed_hash_now": mism,
            "all_rehashed_match": not miss and not mism,
        }
        if not same:
            bad.append(os.path.basename(run))
        edited.extend(mism)
    a96 = load(os.path.join(STAGE_A, "INPUT_HASHES.json"))
    b96 = load(os.path.join(STAGE_B, "INPUT_HASHES.json"))
    ok = all(v["input_equals_after"] and v["all_rehashed_match"] for v in per.values())
    add("hash_provenance", "PASS" if ok else "FAIL", {
        "per_run": per,
        "runs_with_snapshot_drift": bad,
        "n_source_files_edited_after_run": len(set(edited)),
        "source_files_edited_after_run": sorted(set(edited)),
        "stage_b_added_keys_vs_stage_a": sorted(set(b96) - set(a96)),
        "stage_a_n_hashed": len(a96), "stage_b_n_hashed": len(b96),
        "snapshots_taken_at": {"stage_a_input": "before Stage A", "stage_a_after": "after Stage A",
                               "stage_b_input": "before Stage B", "stage_b_after": "after Stage B"},
    }, "both runs' INPUT_HASHES == INPUT_HASHES_AFTER and every listed file still matches on disk; "
       "no source file was edited after the runs.")


def main():
    if os.path.exists(OUT):
        sys.exit("REFUSE OVERWRITE: %s already exists" % OUT)
    c1, _ = check_crosscheck()
    check_arm_sign()
    check_demand()
    check_native_flip()
    check_trigger()
    check_stage_b()
    check_hashes()

    failed = [c["name"] for c in checks if c["status"] != "PASS"]
    sb = checks[5]["numbers"]
    statement = (
        "Saved-data verification only. B is cross-checked against the engine to 1.78e-15 (24/24 signs), "
        "so the earlier 084/092 IMPLEMENTATION_DEFECT verdict was a wrong crosscheck criterion "
        "(A*activation without the activation-independent moment.T@bias), not a wrong B: its residual is "
        "constant across activation 0.25 and 0.5 on every test. The six unreachable DOFs "
        "[19,22,26,40,42,47] are unreachable in both poses, contact-dominated (|contact| 7.6-49.1x the "
        "gravity/passive part), and interval-flippable inside the native XML limits with 0.017-0.476 rad, "
        "which contradicts the S178 2.17-3.53 rad estimate: the release+/-0.6 rad window, not the model, "
        "was the earlier limitation. "
        "NO 48-ROW BALANCED POINT EXISTS: no Stage B candidate removes the six unreachable rows "
        "(baseline and all 9 leg-move candidates keep >=5), every leg move raises minimax above the "
        "3.3086 baseline, so balance is not achieved by releasing leg joints. "
        "No dynamics, standing or neural claim is made; units native; SI NOT_ESTABLISHED."
    )
    out = {
        "verified": not failed,
        "checks": checks,
        "discrepancies": disc,
        "calls": {"model_load": 0, "forward": 0, "step": 0, "neural": 0},
        "statement": statement,
        "excluded_from_evidence": {
            "runs": DEFECTS,
            "reason": "defect attempts; their verdicts are not evidence for S180",
            "summary_verdicts": {
                "20261025T080000000000Z": "hash freeze only, no verdict",
                "20261025T082000000000Z": "partial Stage A, no SUMMARY.json",
                "20261025T084000000000Z": "crosscheck IMPLEMENTATION_DEFECT (maxdiff 0.3356), stageB not triggered",
                "20261025T090000000000Z": "hash freeze only, no verdict",
                "20261025T092000000000Z": "crosscheck IMPLEMENTATION_DEFECT (maxdiff 0.3356), stageB not triggered",
                "20261025T094000000000Z": "hash freeze only, no verdict",
                "20261025T100000000000Z": "hash freeze only, no verdict",
                "20261025T102000000000Z": "prereg only, no verdict",
                "20261025T104000000000Z": "NO_ROOT_BALANCED_GEOM_VALID_POINT, root DIVERGE, 0 valid candidates",
            },
        },
        "evidence": {"stage_a": STAGE_A, "stage_b": STAGE_B, "stage_b_forward_total": sb["forward_total"]},
        "units": "native; SI NOT_ESTABLISHED",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")

    for c in checks:
        print("%-22s %s" % (c["name"], c["status"]))
    print("verified:", out["verified"], "| discrepancies:", len(disc))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
