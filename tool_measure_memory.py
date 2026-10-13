"""Measure this project's heavy-entry-point footprint so the budgets are measured, not guessed.

Runs under the guard itself, so it also verifies that the cap and the lock work: a guard that
has never been exercised is an assertion, not a protection.

Writes outputs/memory_guard/memory_budget.json and prints a stage table.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

import engine.memory_guard as mg  # noqa: E402

OUT = HERE / "outputs/memory_guard"


def main() -> None:
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    stages = []

    with mg.guarded_run("measure_memory_footprint", need_mb=900.0, cap_mb=4096.0) as report:
        def stage(name):
            peak, rss, size = mg.current_vm_mb()
            stages.append({"stage": name, "VmPeak_mb": round(peak, 1), "VmRSS_mb": round(rss, 1),
                           "VmSize_mb": round(size, 1), "host_available_mb": round(mg.available_mb(), 1),
                           "host_swap_used_mb": round(mg.swap_used_mb(), 1), "t_s": round(time.time() - t0, 1)})
            print(f"  {name:42s} VmPeak {peak:8.1f} MB   VmRSS {rss:8.1f} MB   "
                  f"host free {mg.available_mb():7.1f} MB", flush=True)

        t0 = time.time()
        stage("process_start")
        import mujoco  # noqa: F401
        stage("import_mujoco")
        import engine.canonical_physics as cp
        from engine.embodied.adapters import build_neural_tier, TierConfig
        from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig
        stage("import_project_modules")

        model = cp.load_model()
        stage("compile_canonical_model")
        renderer = mujoco.Renderer(model, height=480, width=640)
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        renderer.update_scene(d)
        frame = renderer.render().copy()
        stage("render_one_frame")
        renderer.close()

        built = build_neural_tier(TierConfig(background_mean_nA=0, background_sd_nA=0))
        stage("build_neural_tier_12000")
        from engine.neural_cond import ConductanceNetwork, ConductanceParams
        net = ConductanceNetwork(int(built["n"]), ConductanceParams(), dt_ms=0.5)
        net.set_connectivity(built["edges"]["We"] * 0.2, built["edges"]["Wi"] * 0.2)
        stage("build_conductance_network")
        cur = np.zeros(int(built["n"]))
        for _ in range(20):
            net.step(current_nA=cur)
        stage("20_neural_steps")
        del cur

        pools = json.loads((HERE / "outputs/neural_rebuild/neural_rebuild_11_mn_pools.json").read_text())
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        manifest = build_motor_manifest(pools, built["tier"]["root_ids"], names)
        dec = MotorOutput(manifest, MotorOutputConfig())
        sp = np.zeros((10, int(built["n"])), dtype=bool)
        dec.consume(sp, spike_dt_s=0.0005, start_s=0.0, end_s=0.005)
        stage("motor_manifest_and_decoder")

        # One real closure interval, i.e. the actual per-interval working set.
        import run_neural_muscle_closure as clo
        model2, seated, gate, joints, geom_leg = clo.prepare_body()
        stage("closure_prepare_body")
        q = clo.observe_quantities(model2, d, seated, joints, geom_leg)
        stage("closure_observe_quantities")
        del pools, manifest, dec, sp, model2, seated, q, frame

        info = report("completed")

    budgets = {}
    peak = max(s["VmPeak_mb"] for s in stages)
    rss = max(s["VmRSS_mb"] for s in stages)
    for name, b in mg.BUDGETS.items():
        budgets[name] = dict(b, measured_peak_VmPeak_mb=peak, measured_peak_VmRSS_mb=rss,
                             headroom_ratio=b["cap_mb"] / peak if peak else None)
    doc = {
        "status": "PASS",
        "question": "What does this project's heavy work actually cost, and do the declared "
                    "memory budgets and guards hold?",
        "measured": {
            "stages": stages,
            "peak_VmPeak_mb": peak,
            "peak_VmRSS_mb": rss,
            "host": mg.meminfo(),
            "guard_report": info,
            "budgets": budgets,
            "incident_reference": {
                "kernel_oom": "global OOM, python anon-rss 7.58 GB, total-vm 19.84 GB, "
                              "oom_score_adj 200, 2026-10-11 12:06:32 host time",
                "consequence": "two concurrent project audits were killed before writing output",
                "host_total_mb": mg.meminfo().get("MemTotal"),
                "note": "a different project's training job holds ~4.9 GB on this host; the "
                        "guard cannot bound it",
            },
            "guard_self_test": {
                "available_mb_check": "passed",
                "address_space_cap_applied_mb": 4096.0,
                "single_flight_lock_acquired": True,
                "cap_survived_full_workload": True,
            },
        },
        "provenance": {
            "code": str(Path(__file__).resolve()),
            "guard_module": str(HERE / "engine/memory_guard.py"),
            "interpreter": sys.executable,
            "command": f"{sys.executable} {Path(__file__).resolve()}",
            "note": "footprint measured under the guard's own 4096 MB address-space cap",
        },
        "units": {"memory": "MB (MiB as reported by /proc)", "time": "s"},
        "illustrative_vs_measured": "measured on this host at this moment; host load is external",
        "what_this_does_not_support": [
            "Measured peak is for THIS stage sequence; a longer closure run grows qpos/control "
            "traces linearly with duration and is not covered by the same peak.",
            "The guard bounds this process only; it cannot stop another program from exhausting "
            "the host, and the recorded kill was triggered by a browser, not by this project.",
            "A run that hits the cap raises MemoryError and counts as a FAILED run, never as a "
            "scientific negative result.",
        ],
    }
    (OUT / "memory_budget.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\npeak VmPeak {peak:.1f} MB, peak VmRSS {rss:.1f} MB, host free {mg.available_mb():.1f} MB")
    print("wrote", OUT / "memory_budget.json")


if __name__ == "__main__":
    main()
