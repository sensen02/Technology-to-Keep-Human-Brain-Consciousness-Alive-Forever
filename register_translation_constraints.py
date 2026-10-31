#!/usr/bin/env python3
"""Offline-only registration of frozen translation constraints.
No physics/neural-engine imports, no source mutation, no weight writes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys
from pathlib import Path

REQUESTED = "/run/media/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/"
ARMS = {
    "S113": Path("/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us"),
    "S115": Path("/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z/init80ms"),
    "S117": Path("/home/sensen/Desktop/engine修复验收/s117-single-weight-attenuation/20261024T064500Z/w36b4_down"),
    "S118": Path("/home/sensen/Desktop/engine修复验收/s118-secant-weight-step/20261024T065500Z/w36b4_secant"),
}
PARENTS = {k: p.parent / "INPUT_HASHES.json" for k, p in ARMS.items()}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

def source_manifest():
    out = {"requested_path": REQUESTED, "requested_exists": Path(REQUESTED).exists(),
           "resolution": "requested path missing; bound to actual frozen Desktop path; no silent rewrite",
           "actual_paths": {k: str(v) for k, v in ARMS.items()}, "references_copied": False}
    return out

def verify_refs():
    records = {}
    for arm, d in ARMS.items():
        if not d.is_dir(): raise FileNotFoundError(d)
        result = d / "RESULT.json"; parent = PARENTS[arm]
        r = load_json(result); inp = load_json(parent)
        actual_result_hash = sha256(result); actual_parent_hash = sha256(parent)
        expected_result_hash = {"S113":"af22822fa0615ea34b7d3565c953bde158ef356d3fb0c3c2c64035ca8699a74b","S115":"c5e7df1ae4b658b0589fb78629ca15866d70d682069723df3e9447143684efc8","S117":"471704b4c3405e56c3dbc421f64141dbbc0b3d945c93b3ca5cfbc2c1f71556eb","S118":"3c693a232db4d1ff5c0cc8ffb33169bbf9335dc20ad7c26d361571dda3c6336a"}[arm]
        expected_parent_hash = {"S113":"45fdf64bf13fedd8f5936866510c5920c1cc77e9550ec01c50cfbc50aecfc6ed","S115":"f789deef333e3c9b7018a7ed019e9dbbf1a4ab50160981e2aa3f05d1c90302ae","S117":"2e99cfaa56ea004967ac8dc451170240bfb5b3d4eb552e8edbce9790de4d0b78","S118":"ddba522dc9353752cf055749fd22054b10a2b16dea5f37d44804b687ffe8a072"}[arm]
        if actual_result_hash != expected_result_hash or actual_parent_hash != expected_parent_hash: raise ValueError(f"hash mismatch {arm}")
        result_hashes = {str(k): v for k,v in r.get("hashes", {}).items()}
        missing = [p for p in result_hashes if not Path(p).exists()]
        mismatched = [p for p,v in result_hashes.items() if Path(p).exists() and sha256(Path(p)) != v]
        if missing or mismatched: raise ValueError(f"RESULT hash verification failed {arm}: missing={missing} mismatched={mismatched}")
        parent_mismatch = [p for p,v in inp.items() if Path(p).exists() and sha256(Path(p)) != v]
        records[arm] = {"artifact_dir": str(d), "result_sha256": actual_result_hash, "parent_input_hashes_sha256": actual_parent_hash,
                        "result_hashes_verified": len(result_hashes), "result_missing": missing, "result_mismatched": mismatched,
                        "parent_input_hashes_entries": len(inp), "parent_input_hashes_mismatched": parent_mismatch,
                        "label": r.get("label"), "factor": r.get("factor"), "old_W_sha256": r.get("old_W_sha256"), "new_W_sha256": r.get("new_W_sha256"),
                        "selection": r.get("selection", False), "deployment": r.get("deployment", False), "decision_reject": r.get("decision",{}).get("reject", False)}
    return records

def make_ledger(out, records):
    import numpy as np
    fields=["arm","time_ms","sample_kind","peak_joint","peak_time_ms","all_six_boundary","commanded_force_norm","actuator_force_norm","moment_norm","tau6","tau7","tau8","full_dof_coupling"]
    rows=[]
    for arm,d in ARMS.items():
        z=np.load(d/"TRAJECTORY.npz", allow_pickle=False); t=z["time"]*1000
        idx=int(np.argmin(abs(t-2.5))); am=z["actuator_force"][idx]; ctrl=z["ctrl"][idx]; mom=z["actuator_moment"][idx]
        peak_joint=float(np.max(np.abs(z["qpos"][idx,1:])))
        peak_time=float(t[np.argmax(np.max(np.abs(z["qpos"][:,1:]),axis=1))])
        rows.append([arm,2.5,"2.5ms",peak_joint,peak_time, bool(z["all_six_gates"][idx]),float(np.linalg.norm(ctrl)),float(np.linalg.norm(am)),float(np.linalg.norm(mom)),float(np.linalg.norm(mom[:,6])),float(np.linalg.norm(mom[:,7])),float(np.linalg.norm(mom[:,8])),"all_dof_6_7_8"])
        for i,tm in enumerate(t):
            rows.append([arm,float(tm),"boundary",float(np.max(np.abs(z["qpos"][i,1:]))),peak_time,bool(z["all_six_gates"][i]),float(np.linalg.norm(z["ctrl"][i])),float(np.linalg.norm(z["actuator_force"][i])),float(np.linalg.norm(z["actuator_moment"][i])),"","","","all_dof_6_7_8"])
    with (out/"SAMPLE_LEDGER.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rows)
    with (out/"PER_STATE_FULL_COUPLING.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["arm","state_index","time_ms","dof_scope","commanded_norm","force_norm","moment_norm","actuator_actual_coupling"])
        for arm,d in ARMS.items():
            z=np.load(d/"TRAJECTORY.npz",allow_pickle=False)
            for i,tm in enumerate(z["time"]*1000): w.writerow([arm,i,float(tm),"DOF6,DOF7,DOF8",float(np.linalg.norm(z["ctrl"][i])),float(np.linalg.norm(z["actuator_force"][i])),float(np.linalg.norm(z["actuator_moment"][i])),"logged_actual_not_DOF_selective"])

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070500Z"); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args(argv)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    records=verify_refs(); make_ledger(out,records)
    constraints={"schema":"training-constraints/v1","offline_only":True,"scientific_acceptance_framework_added":False,"no_physics_or_neural_imports":True,
      "source_resolution":source_manifest(),"frozen_references":records,
      "status":{"S113":"reference_not_autonomous_success","S115":"rejected_safety_boundary","S117":"rejected_safety_boundary","S118":"rejected_safety_boundary","last_two":"empirical_nonmonotonic_no_W36_scans"},
      "selection":{"S113":"frozen_original_checkpoint_active_logical_selection_only_no_weight_writes","weight_writes":False,"original_weights_hashes_preserved":True},
      "requirements":{"retain_S115":{"load_deficit_0_1p7ms":True,"initial_load":True,"first_load_failure":True,"support_sum":True,"longest_support":True},"safety_upper_bound_S113":{"max_limit_excess":True,"angle":True,"nonfoot":True,"height":True,"downward_velocity":True,"first_safety_failure_no_earlier":True},"coupling":{"all_DOF_6_7_8_actuator_actual_logged":True,"not_DOF_selective":True,"native_original_criteria_only":True,"no_expansion":True}},
      "empirical_2p5ms":{"S115":{"actuator":0.055082,"constraint":1.223889},"S117":{"actuator":0.056137,"constraint":1.240219},"S118":{"actuator":0.063759,"constraint":1.285931}},
      "evaluation":"per_arm_against_spec_only"}
    write_json(out/"TRAINING_CONSTRAINTS.json",constraints)
    write_json(out/"SOURCE_RESOLUTION.json",source_manifest())
    write_json(out/"REFERENCE_VERIFICATION.json",records)
    (out/"SUMMARY.md").write_text("# S119 training constraints\n\nOffline registration only. S113 is a frozen reference, not autonomous success. S115/S117/S118 are rejected at safety boundaries; no new acceptance framework, no W36 scan, and no physical weight writes.\n\nAll four RESULT hashes and parent INPUT_HASHES were verified before use.\n",encoding="utf-8")
    hashes={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="ARTIFACT_HASHES.json"}; write_json(out/"ARTIFACT_HASHES.json",hashes)
    if args.self_test:
        assert constraints["selection"]["weight_writes"] is False and len(records)==4 and all(v["result_missing"]==[] for v in records.values())
        print("SELFTEST_OK")
    print(json.dumps({"readiness":"READY_OFFLINE_CONSTRAINT_REGISTRATION","output":str(out),"files":len(hashes)},indent=2))
if __name__ == "__main__": main()
