# Local physical sensory encoder: verified engineering interface

Implemented only the stateless sensory encoder for the approved 20,000-cell BANC local candidate. Numerical verification passes 189 checks. No neural network was instantiated or stepped, no training ran, and no MuJoCo model, forward call or physical step was used. This is an engineering interface result, not calibrated cell physiology, functional neural propagation or locomotion success. Task relation: S3/S4 sensory boundary for the approved platform.

Primary implementation: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py`.

Desktop report: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/encoder/REPORT.md`.

## Exact API

```python
import importlib.util
import numpy as np
path = '/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py'
spec = importlib.util.spec_from_file_location('banc_sensory_encoder', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
encoder = module.SensoryEncoder()
# q42_rad and qvel42_rad_s must follow encoder.joint_names exactly.
# foot_load6_native must follow ('LF', 'LM', 'LH', 'RF', 'RM', 'RH').
currents_nA = encoder.encode(q42_rad, qvel42_rad_s, foot_load6_native)
assert currents_nA.shape == (20000,)
current_allowed_ports = encoder.local_indices
```

Initializer uses only NumPy and the Python standard library. It verifies source artifact/cache SHA256 against the source manifest, checks candidate root identities, loads all 42 joint names and bounds from their specified sources, qualifies source records and exposes `feature_contract`, `source_contract`, `qualified_count`, `local_indices`, `channel_local_indices`, `allowed_mask`, `root_ids`, `cache_indices`, `joint_names`, `joint_ranges_rad`, `knee_indices`, `coxa_indices`, `foot_order`, `body_weight_native`, `normalize_angles(q42_rad)` and `contract()`.

Inputs have exact shapes `(42,)`, `(42,)`, `(6,)`; all entries must be real numeric and finite. Shape mismatch, NaN, infinity, complex and string inputs raise `ValueError`. Outputs are new finite nonnegative float64 vectors bounded to `[0, 0.06]` nA, with exact zeros outside the 556 qualified local indices. Negative loads clamp to zero. Angle inputs clamp before normalization, including extreme finite values.

## Identity and admission

The 3,903 channel source rows are joined by exact decimal `root_888`, never a float or legacy root fallback. Admitted rows must match the full cache root/index, local candidate root/index, `leg_sensory_nerve_union`, the leg group, the exact channel group, a unique matching leg nerve, the source subclass and the expected cache class/sensory superclass. Repeated source identities or reused indices are rejected. No cells or edges are synthesized.

All 3,347 excluded records and their individual reasons are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/EXCLUSIONS.json` and in the complete `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/CONTRACT.json`. The contract contains every qualified exact root888 ID and local index, the full local candidate root order, source hashes and runtime provenance.

| Leg | claw_position | hook_direction | hairplate_angle | campaniform_strain | Total |
|---|---:|---:|---:|---:|---:|
| LF | 27 | 25 | 21 | 27 | 100 |
| LM | 34 | 20 | 31 | 2 | 87 |
| LH | 38 | 21 | 35 | 2 | 96 |
| RF | 25 | 25 | 32 | 24 | 106 |
| RM | 13 | 22 | 42 | 2 | 79 |
| RH | 23 | 23 | 40 | 2 | 88 |
| Total | 160 | 136 | 201 | 59 | 556 |

Exclusion reasons overlap: 335 club rows lack acceleration; 36 multidendritic rows lack calibrated damage; 2,945 bristle rows lack individual hair collision; 30 rows lack cache identity; 70 are outside the local candidate; 74 are outside the leg sensory union; 76 have nerve/source-leg disagreement or ambiguity; 76 are outside the matching leg group; 7 have channel/cache-class contradictions; 3 have subclass/modality contradictions. All four reviewed ambiguous identities carry the explicit `review4_modality_ambiguity` reason. The seven class contradictions include those four plus hairplate/chordotonal, campaniform/hairplate and hook/hairplate conflicts. Front hairplate LF14/RF11 exclusions also expose leg-nerve qualification failures; they are not rescued by generic source channel labels.

## Physical feature contract and assumptions

The foot order is LF, LM, LH, RF, RM, RH. The 42-joint action order is LF, LH, LM, RF, RH, RM. RF coxa indices 21, 22, 23 are **roll, yaw, pitch**, respectively; the encoder resolves axes by exact names. Full bounds are preserved from `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/current42_candidates.csv` and cross-checked against `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/STUDENT_INTERFACE.json`.

For each joint, normalized angle is `(clip(q, lower, upper)-lower)/(upper-lower)`.

- `claw_position`, cache class `chordotonal_organ_neuron`: same-leg tibia pitch normalized by its actual range; current `0.01 + 0.04 * normalized_angle` nA.
- `hook_direction`, cache class `chordotonal_organ_neuron`: same-leg tibia pitch speed; current `0.05 * tanh(abs(speed_rad_s)/10)` nA. Source name is preserved, but individual directional tuning is unknown; the engineering proxy uses speed magnitude only.
- `hairplate_angle`, cache class `hair_plate_neuron`: mean of same-leg individually normalized coxa yaw, pitch and roll; current `0.01 + 0.04 * mean` nA. This common three-coordinate proxy is an explicit engineering choice, not inferred receptor axis or tuning direction.
- `campaniform_strain`, cache class `campaniform_sensillum_neuron`: current `0.05 * tanh(max(foot_load_native, 0)/(24.446354755878005/6))` nA. This uses the task-specified original body weight and a common load proxy; local strain calibration and absolute SI force are NOT_ESTABLISHED.
- `club_vibration`, `bristle_touch`, `multidendritic_noxious`: always zero, including baseline.

All cells within a leg/channel get identical declared transduction. Positive angle-coordinate slope, baseline, spans and speed scale are illustrative engineering parameters; no random cell-specific thresholds, directions, ID phases or oscillators are introduced. Only qualified angle cells have the declared 0.01 nA baseline. Zero q is a coordinate diagnostic and lies outside some actual joint ranges; it is not a body pose or zero-current condition. Zero velocity/load removes hook/campaniform input, while qualified angle currents remain.

No root position, root orientation, root velocity, time, phase, teacher, target, history or neural state enters the API. Trochanter coordinates/velocities and coxa velocities are validated but unused. `/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py` was read for its explicit units and illustrative-parameter caveat; no network class was imported or executed.

## Verification and execution record

Preregistration `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/PREREG.json` was saved before construction tests. Verification implementation `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/verify_encoder.py` checks all 20,000 root identities, exact source groups/classes/nerves, every 42 coordinate pose/velocity perturbation, six independent foot loads, RF ordering, source ranges and clamping, extreme finite inputs, 31 deterministic sweeps, identical within-class tuning, independent repeated initialization, 24 invalid input cases and nine derived-record identity/qualification mutation fixtures. The fixtures never enter the saved contract or runtime output and do not fabricate biological cells.

Final result `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/VERIFICATION.json`: `PASS_NUMERICAL_ENGINEERING_ONLY`, 189 checks. Run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/verify_encoder.py
```

Failures are retained in the execution record: initial environment probe `python` exited127 because this machine exposes `python3`; first numerical test exited1 on empty disabled-channel arrays introduced by defaultdict lookups, fixed by omitting empty runtime ports; second numerical test exited1 because a midpoint rounding check used tolerance `1e-16`, corrected to four float64 machine epsilons. Final verification exited0. No rerun involved neural or physical dynamics.

The parent separately owns `LocalCircuit(current_allowed_ports)`, MN-only linear readout training and physical diagnostics. The saved slow4 motions remain failed references, never success labels. No tracked files were modified by this encoder task; the preexisting AGENTS/ROADMAP/TRACKER modifications belong to the parent session. This report supplies the parent with the S3/S4 evidence for its tracker update.
