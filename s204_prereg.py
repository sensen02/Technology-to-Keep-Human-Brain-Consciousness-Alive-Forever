"""S204 freeze the pre-registration BEFORE any run."""
import json
import datetime
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit')
stamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
B = ROOT / stamp
B.mkdir(parents=True, exist_ok=False)

prereg = {
    "round": "S204",
    "title": "readout structure audit and decoder calibration verdict",
    "directory": str(B),
    "designation": "P2 sensory->motor closed loop: is the saturation a body property or a decoder artefact?",
    "written_before_runs": True,
    "written_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "question": (
        "S203 reported 84.7% of pre-collapse actuator values at a rail in the no-tactile arm. That number was computed "
        "without knowing the readout structure. If part of the 90 decoder rows can never receive a command at all, and if "
        "the full-scale rate is set far below one spike per sample, then the rail statistics measure the DECODER, not the "
        "body. This round decides that question before any further tactile or postural work."
    ),
    "five_labels": {
        "SATURATION_IS_CALIBRATION": "H1/H2 raise the off-rail fraction of non-empty actuators above 50% with bitwise identical input",
        "POOL_TOO_SMALL_FOR_GRADATION": "even at H2 >= 10*H0 the off-rail fraction stays <= 25%: 1-7 cell pools at 0.5 ms sampling cannot express gradation",
        "UNDETERMINED": "neither gate is met; report the measured points and extrapolate nothing",
        "TOPO_UNREACHABLE": "most non-pool cells cannot reach a pool cell in <= 2 hops of the frozen connectivity",
        "SUBTHRESHOLD_REACHABLE": "most non-pool cells CAN reach a pool cell in <= 2 hops, so the silence is a threshold issue",
        "rule": "exactly one of TOPO_UNREACHABLE / SUBTHRESHOLD_REACHABLE is chosen, with the connection evidence attached; 'rectification' may not be written",
    },
    "part_A_offline": {
        "A1_readout_census": "n from build_neural_tier (asserted 12000); per-row dump of actuator/actuator_index/leg/side/muscle_target/status/pool_size/local_indices; the 16 empty rows named with their legs; coverage, pool-size distribution, per-leg non-empty counts; the six hard DOFs 19/22/26/40/42/47 checked for empty-row members",
        "A2_transfer_function": "from motor_output.py (filter_tau_s, rate_for_full_control_hz, max_control), solve the per-cell firing rate needed for ctrl=1.0 and ctrl=0.5 for pool sizes k=1..7; then recompute every command-window ctrl exactly from the stored per-sample spike matrices and compare BITWISE with the stored ctrl (hard check; if it fails the round's evidence is void)",
        "A3_saturation_decomposition": "four classes: structural zero (empty row), pool zero spikes, pool spiked but ctrl<1, ctrl at 1.0; counts with and without the 16 structural-zero rows, compared against the S203 numbers 84.7/38.3/46.4",
        "A4_reachability": "BFS over the frozen We|Wi adjacency from the pool cells; report the count and share of non-pool cells reachable in <= 2 hops, using the stored connection structure only",
    },
    "part_B_arms": {
        "note": "same frozen start, same neural tier / physics / manifest / filter; only rate_for_full_control_hz changes",
        "duration_s": 0.05,
        "steps_per_arm": 2000,
        "arms": {
            "H0": {"rate_for_full_control_hz": 100.0, "role": "current value"},
            "H1": {"rate_for_full_control_hz": "2 * median(pool_rate_over_nonempty_pools, commands 1-3 of H0)", "role": "data-derived"},
            "H2": {"rate_for_full_control_hz": "max(10*H0, 2*P90(pool_rate))", "role": "data-derived and guaranteed >=10x so the frozen 10x gate is decidable"},
        },
        "derivation_note": (
            "H1 is the median-derived full-scale value. H2 is defined as max(10*H0, 2*P90) so that the pre-registered "
            "'>=10x' clause is always evaluable; if 2*P90 < 10*H0 then H2 = 10*H0 and the P90 value is still reported. "
            "No other parameter changes."
        ),
        "reporting": "commands 1-3 off-rail fraction over NON-EMPTY actuators, ctrl quantiles, per-actuator ctrl for the first 4 windows, spike tickets to prove the three arms had identical input",
        "forbidden": "no 5 s end-state displacement and no rootz comparison may be used (S203 methodological constraint)",
    },
    "gates": {
        "calibration": "off-rail fraction of non-empty actuators at H1 or H2 > 50% while the spike tickets are bitwise identical across arms -> SATURATION_IS_CALIBRATION",
        "pool_too_small": "off-rail fraction at H2 (>=10*H0) <= 25% -> POOL_TOO_SMALL_FOR_GRADATION",
        "otherwise": "UNDETERMINED",
    },
    "budget": {
        "max_steps": 20000,
        "planned": {"arms": ["H0", "H1", "H2"], "steps_per_arm": 2000, "total": 6000, "remaining": 14000},
    },
    "guards": {
        "xml_changes": 0, "training_steps": 0, "prewarming": False,
        "forbidden_controllers": "no CPG, no PD, no per-step pose correction, no self-made signals",
        "calls_reported": ["model loads", "forward", "mj_step", "LP", "neural ticks"],
        "parameter_change_ledger": "every changed parameter is recorded with old file / old sha256 / new file / new sha256 / reason; no frozen XML or historical artefact is overwritten",
        "param_note": "rate_for_full_control_hz is a decoder gain and changing it is calibration, which the owner explicitly allows; it replaces neither neurons nor physics",
    },
    "deliverables": ["PREREG.json", "SOURCE_HASHES.json", "READOUT_CENSUS.json", "DECODER_TRANSFER.json",
                     "SATURATION_DECOMPOSITION.csv", "REACHABILITY.json", "CALIBRATION_ARMS.json",
                     "ATTEMPT_LEDGER.json", "STOP_AND_BUDGET.json", "INDEPENDENT_VERIFICATION.json",
                     "CONCLUSION.zh-CN.md", "FIG1_POOLSIZE_VS_REQUIRED_RATE.png",
                     "FIG2_SATURATION_DECOMPOSITION.png", "FIG3_ARM_CTRL_TRACES.png"],
}
(B / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))
print(B)
