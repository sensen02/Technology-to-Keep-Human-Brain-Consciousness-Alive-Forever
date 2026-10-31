"""S205 freeze the pre-registration BEFORE any run."""
import json
import datetime
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s205-unresolved-mn-and-silent-pools')
stamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
B = ROOT / stamp
B.mkdir(parents=True, exist_ok=False)

prereg = {
    "round": "S205",
    "title": "provenance of the 16 uncommandable coxa muscles + diagnosis of the zero-spike pools",
    "directory": str(B),
    "designation": "P3/P2 gap closure requested by the S204 conclusion (tracker next action 1 and 2)",
    "written_before_runs": True,
    "written_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "no model change, no mapping substitution, no training, no XML edit; diagnosis only",
    "part_A_provenance": {
        "source": "outputs/neural_rebuild/neural_rebuild_11_mn_pools.json (measured.unresolved / unresolved_by_reason / "
                  "cross_leg_substitution_the_old_runtime_performs) plus the BANC feather metadata it was built from",
        "questions": [
            "for each of the 16 unresolved actuators: is the muscle target absent from the dataset entirely, or present only on "
            "another leg/side, or present here but rejected for another reason",
            "what does the dataset actually contain for each target (which legs, how many motor neurons)",
            "which of the six hard DOFs (19/22/26/40/42/47) lose how much mechanical authority to these rows",
        ],
        "the_question_this_does_NOT_answer": "whether a cross-leg or mirror substitution would be legitimate. Substituting cells "
                                             "would be inventing synapses, which stays forbidden; this round records the gap only",
    },
    "part_B_zero_spike_pools": {
        "question": "the S204 calibration arms showed that 72 of 222 non-empty actuator-window values carry ZERO spikes and stay "
                    "at ctrl=0 whatever the full scale is. Is that because the pool cells receive no sensory current, or because "
                    "they are driven but stay below threshold (or are gated by the network)?",
        "arm": "one instrumented arm, 0.05 s = 2000 steps, rate_for_full_control_hz=100, no tactile, same frozen start",
        "instrumentation": "per command window: the full external current vector (12000), every cell's spike count, and every "
                           "pool cell's membrane potential; nothing else in the loop changes",
        "classes": ["NO_EXTERNAL_CURRENT", "DRIVEN_BUT_NO_SPIKE_BELOW_THRESHOLD",
                    "DRIVEN_NO_SPIKE_ABOVE_THRESHOLD_GATED", "FIRED"],
        "rule": "a pool is classified by its 0.05 s window set: NO_EXTERNAL_CURRENT only if every pool cell had |I| == 0 in every "
                "window; DRIVEN_BUT_NO_SPIKE_BELOW_THRESHOLD if some cell had |I| > 0 and no cell of the pool ever spiked while "
                "all its cells stayed below threshold; DRIVEN_NO_SPIKE_ABOVE_THRESHOLD_GATED if a cell reached threshold without "
                "spiking; FIRED if any pool cell spiked",
    },
    "gates": {
        "provenance": "the 16 rows must be classified into the dataset-coverage reasons with counts that sum to 16, each named",
        "silent_pools": "the zero-spike actuator-window count must be reproduced offline from the new arm, and each silent pool "
                        "must fall into exactly one of the four classes with the counts reported",
    },
    "budget": {"max_steps": 10000, "planned_steps": 2000, "planned_model_loads": 2, "training_steps": 0, "xml_changes": 0},
    "forbidden": {"words": ["改善", "improved"],
                  "note": "no movement claim; this round diagnoses a wiring gap and a silence mechanism only"},
    "deliverables": ["PREREG.json", "UNRESOLVED_PROVENANCE.json", "SILENT_POOL_DIAGNOSIS.json",
                     "ATTEMPT_LEDGER.json", "STOP_AND_BUDGET.json", "INDEPENDENT_VERIFICATION.json",
                     "CONCLUSION.zh-CN.md", "FIG1_UNRESOLVED_REASONS.png", "FIG2_SILENT_POOL_CLASSES.png"],
}
(B / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))
print(B)
