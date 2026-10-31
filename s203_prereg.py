"""S203 freeze the pre-registration BEFORE any run. Writes a unique timestamp directory."""
import json
import datetime
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit')
stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
B = ROOT / stamp
B.mkdir(parents=True, exist_ok=False)

prereg = {
    "round": "S203",
    "title": "attribution / chaos audit of the S202 tactile contrast",
    "directory": str(B),
    "designation": "P2 sensory->motor closed loop: causal attributability of the S202 differences",
    "written_before_runs": True,
    "written_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "question": (
        "S202 reported B(vs A) horizontal displacement +0.1866 m and C +0.0229 m, and D(negative force) bitwise equal to A. "
        "Before any of that can be called a tactile effect this round must decide (i) whether a single 5 s end-state "
        "displacement is even repeatable under a same-configuration infinitesimal perturbation, and (ii) whether the "
        "A/D difference is explained by the structure of the decode clamp rather than by a mechanical rectifier."
    ),
    "part_A_offline": {
        "A1_ctrl_divergence": {
            "source": "instrumented runs (0.02 s = 800 body steps, full ctrl/act/current dumped) for arms A,B,C,D",
            "windows": {"pre_collapse_s": [0.0, 0.020], "post_collapse_s": [0.020, 5.0]},
            "reports": ["number of command intervals that differ", "number of body steps that differ",
                        "fraction", "max|dctrl|", "first diverging command index and its time",
                        "actuator indices involved"],
            "cadence_note": (
                "ctrl is piecewise constant over a 200-body-step command interval, so the pre-collapse window contains only "
                "4 distinct ctrl samples; body-step counts are reported as 200x the differing command intervals and the "
                "cadence is stated explicitly"
            ),
        },
        "A2_saturation_by_actuator": {
            "reports": ["per-actuator ctrl==0 fraction", "per-actuator ctrl==1 fraction",
                        "per-window (pre/post collapse)",
                        "hard DOFs 19/22/26/40/42/47 with their agonist/antagonist actuators from ROW_DOF_MAP"],
        },
        "A3_clamp_audit": {
            "reports": ["source file:line for every clamp/rectification on the sensory->current->spike->ctrl path",
                        "explicit statement whether negative current is structurally rejected",
                        "instrumented per-leg current pre/post any clamp for arms A and D"],
        },
    },
    "part_B_arms": {
        "note": "same frozen start, same 5 s, same neural tier/decoder/physics; only the initial value changes",
        "arms": {
            "A": {"config": "no tactile", "delta": 0.0},
            "A_d1": {"config": "no tactile", "delta": 1e-9, "delta_joint": "joint_LFCoxa_pitch"},
            "A_d2": {"config": "no tactile", "delta": 1e-6, "delta_joint": "joint_LFCoxa_pitch"},
            "B": {"config": "raw contact force, divisor 1.0", "delta": 0.0},
            "B_d1": {"config": "raw contact force, divisor 1.0", "delta": 1e-9, "delta_joint": "joint_LFCoxa_pitch"},
        },
        "deferred": {
            "arm": "E (fixed permutation of the six-leg force vector)",
            "reason": (
                "E would be a sixth 200000-step arm: 6*200000 = 1200000 steps > the frozen 1100000 cap. The cap is not "
                "relaxed, so E is NOT run this round and is recorded here BEFORE the fact as deferred, not as a missing "
                "result. It is the first action after this round if the chaos gate does not fire."
            ),
        },
        "delta_note": (
            "the delta is added to the qpos entry of joint_LFCoxa_pitch AND the same delta enters set_reference_pose(q0), so "
            "the sensory deviation signals are exactly 0 at t=0 in every arm and the only difference is the physical "
            "initial condition"
        ),
    },
    "conclusion_gates": {
        "chaotic": {
            "rule": "A_d1 (delta=1e-9) end-state displacement spread vs A >= 0.05 m OR end-state rootz spread >= 0.005 m",
            "consequence": (
                "CHAOTIC_AMPLIFICATION: the S202 B/C differences must be re-written as 'inside the amplification range of a "
                "same-configuration infinitesimal perturbation, no attributable tactile effect', single-run 5 s end-state "
                "displacement is forbidden from now on as an A/B comparison statistic, and dynamics comparisons must use "
                "ensembles over perturbations/start points with a reported distribution"
            ),
        },
        "attributable": {
            "rule": "A_d1 spread < the limits above AND |B_d1 - B| far larger than the A-side spread at the same delta",
            "consequence": "tactile effect attributable; then and only then discuss direction and size",
        },
        "d_mechanism": {
            "values": ["RECTIFIED", "SATURATION_MASKED", "UNRESOLVED"],
            "rule": (
                "RECTIFIED only if a clamp/rectifier on the current path structurally rejects negative input; "
                "SATURATION_MASKED if the negative input reached the neurons (spikes changed) but every actuator it could "
                "have moved was already clipped at 0 or 1 in the decode step; otherwise UNRESOLVED. Leaving 'rectification' "
                "as an unproven phrase is forbidden."
            ),
        },
    },
    "forbidden": {"words": ["改善", "improved"],
                  "note": "never claim improvement; report only 'measurably different', 'no attributable effect', or 'unresolved'"},
    "budget": {
        "max_steps_per_arm": 200000,
        "total_max_steps": 1100000,
        "planned": {
            "partA_short_arms": ["A", "B", "C", "D"], "partA_short_steps_per_arm": 800,
            "partA_short_total": 3200,
            "partB_arms": ["A", "A_d1", "A_d2", "B", "B_d1"], "partB_steps_per_arm": 200000,
            "partB_total": 1000000, "total": 1003200, "remaining": 96800,
        },
    },
    "guards": {
        "xml_changes": 0, "training_steps": 0, "prewarming": False,
        "forbidden_controllers": "no CPG, no PD, no per-step pose correction, no self-made signals",
        "calls_reported": ["model loads", "forward", "mj_step", "LP", "neural ticks"],
        "pre_load_regressions": ["lowercase leg keys", "encoder set_reference_pose + assign",
                                 "geom mapping assertions", "noise-floor self-check"],
    },
    "deliverables": ["PREREG.json", "SOURCE_HASHES.json", "CTRL_DIVERGENCE.json",
                     "SATURATION_BY_ACTUATOR.csv", "CLAMP_PATH_AUDIT.json", "SENSITIVITY_ARMS.json",
                     "D_MECHANISM_VERDICT.json", "ATTEMPT_LEDGER.json", "STOP_AND_BUDGET.json",
                     "INDEPENDENT_VERIFICATION.json", "CONCLUSION.zh-CN.md",
                     "FIG1_CTRL_DIVERGENCE_TIMELINE.png", "FIG2_DELTAQPOS_GROWTH.png",
                     "FIG3_TACTILE_VS_SATURATION.png"],
    "history_preserved": "S169-S202 tracker entries and their hashes are not modified; only a new S203 entry is added",
}
(B / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))
print(B)
