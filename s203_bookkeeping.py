"""S203 bookkeeping: attempt ledger, stop reason + budget, source/artifact hashes."""
import json
import hashlib
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


ARMS = {'A': 200000, 'A_d1': 200000, 'A_d2': 200000, 'B': 200000, 'B_d1': 200000}
SHORT = {'short_A': 800, 'short_B': 800, 'short_C': 800, 'short_D': 800,
         'short2_A': 800, 'short2_D': 800,
         'short3_A': 800, 'short3_B': 800, 'short3_C': 800, 'short3_D': 800, 'short3_T': 0}
# short3_T was run twice: once with a patch that did not apply (a 800-step duplicate of B) and once correctly
SHORT_STEPS = dict(SHORT)
SHORT_STEPS['short3_T'] = 800
EXTRA_T_ATTEMPT = 800
total = sum(ARMS.values()) + sum(SHORT_STEPS.values()) + EXTRA_T_ATTEMPT

ledger = {
    "round": "S203",
    "failures": [
        {"id": "S203-A1", "what": "run_s203_arm.py joint scan",
         "defect": "compared the lowercase joint name against 'joint_' + LEG where LEG is uppercase, so every leg matched 0 joints and the assertion fired ('LF', 0, 0)",
         "caught_by": "the in-run assertion, before any dynamics", "status": "fixed (leg.lower()), no data produced"},
        {"id": "S203-A2", "what": "run_s203_arm.py leg_rows / probe",
         "defect": "channel_index_map keys are lowercase; three successive patches used uppercase ('LF') and '.size' on list values",
         "caught_by": "KeyError / AttributeError, then the nonzero-current assertion", "status": "fixed, no data produced"},
        {"id": "S203-A3", "what": "run_s203_arm.py probe keys",
         "defect": "the per-leg nonzero-current probe passed UPPERCASE leg keys; encode() silently ignores them, so the probe measured all-zero current",
         "caught_by": "the probe assertion itself fired ('a leg got zero current')",
         "status": "fixed (lowercase probe keys); this is the same defect class that cost two earlier rounds, caught here by construction"},
        {"id": "S203-A4", "what": "arm T (touch-only control), first attempt",
         "defect": "the edit that should have set touch=1.0 / force=0 did not apply to observe(); arm T therefore ran the arm-B code path and produced a BITWISE COPY of B",
         "caught_by": "the bitwise comparison B vs T (identical ctrl/qpos/pools together with identical force values) made the duplication visible; a summary-level comparison alone would have hidden it",
         "status": "the duplicate run is withdrawn, the patch was re-applied, and short3_T was re-run; 800 steps of the duplicate are counted against the budget"},
        {"id": "S203-A5", "what": "analysis print block", "defect": "referenced pool_changed / Z['...']['total_spikes'] (summary key, not a TRACE array)",
         "caught_by": "NameError / KeyError", "status": "fixed, no artifact written from the broken pass"},
        {"id": "S203-A6", "what": "FIG1 first draft",
         "defect": "annotated the A-vs-B divergence as starting 'before the floor contact', which is wrong: at t=0 every tarsus is already in contact (1.70-3.90 N)", "caught_by": "reading the figure and then checking force[0]",
         "status": "figure rewritten; the correct statement is 'first command, identical release pose'"},
    ],
    "defects_found_in_the_platform": [
        {"id": "S203-P1", "what": "ChannelSensoryEncoder.encode()",
         "defect": "encode() writes into the global group map, not into the per-leg rows, so it does NOT require assign(); assign() is only needed for quantities()/deviation(). An uppercase leg key is silently dropped, giving exactly zero current with no exception",
         "evidence": "REGRESSIONS_OFFLINE.json: encoder:encode_drives_via_global_group_map_not_assign, encoder:uppercase_keys_silently_ignored",
         "status": "documented; not changed (0 XML / 0 platform edits this round)"},
        {"id": "S203-P2", "what": "decoder readout coverage",
         "defect": "only ~7% of the tier spikes fall inside the 90 decoder pools (588 of 8240 in the 0.02 s arm); the large spike changes produced by the bristle touch channel land entirely outside every pool",
         "evidence": "CTRL_DIVERGENCE.json first_command_probe / FIG1a / D_MECHANISM_VERDICT.json",
         "status": "measured; this is a property of the frozen pipeline, not a change made here"}],
}

(RUN / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

budget = {
    "round": "S203",
    "stop_reason": (
        "All pre-registered Part A items (ctrl divergence, saturation by actuator, clamp audit) and Part B arms "
        "(A, A_d1, A_d2, B, B_d1) are complete. Arm E was pre-registered as DEFERRED before the runs because it would "
        "have broken the 1100000-step cap; it is not run and is not a missing result. One extra touch-only control (T, "
        "800 steps) was added because it separates the two tactile channels and fits inside the remaining budget. "
        "No new tactile configuration and no XML change was introduced, so the round stops here."),
    "budget": {
        "max_steps_per_arm": 200000,
        "total_max_steps": 1100000,
        "steps": {"partB": ARMS, "partA_short": SHORT_STEPS, "withdrawn_duplicate_T": EXTRA_T_ATTEMPT},
        "total_used": total,
        "remaining": 1100000 - total,
        "model_loads": {"partB": 5, "partA_short": len(SHORT_STEPS), "withdrawn_duplicate_T": 1, "analysis_actuator_map": 1},
        "forward_calls": 5 + len(SHORT_STEPS) + 1 + 1,
        "mj_step_calls": total,
        "lp_calls": 0,
        "neural_ticks": (5 * 1000 + (len(SHORT_STEPS) - 1) * 4 + 4) * 10,
        "training_steps": 0, "prewarming": False, "xml_changes": 0,
        "platform_source_edits": 0,
    },
    "environment": {
        "python": "venv_body/bin/python 3.12",
        "mujoco": "3.9.0",
        "env": "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -B",
        "xml": "MODEL_CALIB_H_VERIFIED.xml sha256 " + sha(XML),
    },
}
budget["artifacts"] = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
budget["sources"] = {str(p): sha(p) for p in [
    ROOT / 's203_prereg.py', ROOT / 'regress_s203_offline.py', ROOT / 'run_s203_arm.py',
    ROOT / 'run_s203_analyze.py', ROOT / 'run_s203_figs.py', ROOT / 's203_hard_dof.py',
    ROOT / 'verify_s203.py', ROOT / 's203_bookkeeping.py', XML] if p.exists()}
(RUN / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(RUN / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']),
                                                   indent=2))
print(json.dumps(dict(total_used=total, remaining=1100000 - total, files=len(budget['artifacts']),
                      sources=len(budget['sources'])), indent=1))
