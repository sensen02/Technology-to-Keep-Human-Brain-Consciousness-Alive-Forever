# Prepared fixed neural command profiles — NOT PHYSICALLY RUN

Root: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/command_profiles/`.

Parent must select and qualify the candidate through constant-command tests before launch. No physical jobs have been launched by this implementation task.

## CLI after parent selection

Interpreter: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/command_profiles/evaluate_profile.py --checkpoint /ABSOLUTE/PARENT_SELECTED_CHECKPOINT.pt --sha256 EXACT_SELECTED_64_LOWERCASE_HEX --profile neutral --out /ABSOLUTE/NEW_PROFILE_OUTPUT_DIRECTORY
```

Run exactly the five preregistered profile names, each with a fresh absolute output directory and the same selected checkpoint/digest:

| Profile | First4.8s turn | Last4.8s turn | Direction criterion |
|---|---:|---:|---|
| neutral | 0 | 0 | None; original physical gates and cycles apply |
| halfleft | +.5 | +.5 | Whole signed yaw >=.005rad |
| halfright | -.5 | -.5 | Whole signed yaw >=.005rad |
| left_to_right | +.5 | -.5 | Each4.8s signed yaw >=.002rad |
| right_to_left | -.5 | +.5 | Each4.8s signed yaw >=.002rad |

All forward inputs are exactly0. Halfleft/halfright mean half-amplitude TURN, not half forward speed. Entire9.6s original all5 engineering gates and all6 sensed cycles are required for every profile. Both switching halves include their initial/transient responses. All5 profiles must qualify; there is no best-profile selection.

## Implementation review

- Strict selected checkpoint digest and training source/data checks use `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/checked_motor.py`.
- Legacy modules are pinned/rejected on conflicting imports. Actual original runtime/compiled model/actuator identity uses unchanged baseline sw.load_student before execution. Runtime capture includes evaluator, preregistration, strict wrapper, conditional module, geometry audit and checkpoint. File identities and embedded training identities are rechecked after execution.
- Actual unchanged Core performs1s zero-command settle, original4.8s teacher preheat with neural zero-command hidden priming, then9.6s direct neural action. Teacher called960 times before handover and never after. No pose writes, support forces or gait schedule are introduced. The only supplied schedule is the highlevel neural turn input.
- Handover and halfway snapshots check continuous body/actuator/previous-action/hidden state. Only the read-only cycle observer is initialized at handover; no hidden reset occurs at command switch.
- Actual trace must contain1000settle+4800teacher+9600autonomous1ms endpoints, with2880 real5ms decisions (960prime+1920autonomous). Actual applied autonomous actions are checked bitwise against actor outputs.
- Thorax yaw is continuously unwrapped. Boundaries are exact trace indices5799,10599,15399, corresponding to actual5.8/10.6/15.4s endpoints. No leading/final autonomous interval is removed.
- Head pose and bilateral virtual sensor positions are read directly from actual xpos/xmat every1ms. Their midpoint/path diagnostics do not replace gates or count as food arrival evidence.
- Each profile writes IDENTITY.json, trace_1ms.npz, neural_decisions.npz, geometry_1ms.npz and RESULT.json to a new directory. These are endpoint engineering checks, not full native D1–D8 certification.

## Offline validation performed

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/command_profiles/check_profiles.py` ran with -B using the stated interpreter, exit0, 47checks PASS. It verifies fixed schedules, zero forward input, exact decision switch, split yaw indices, sign/threshold criteria, angle unwrap crossing and syntax. Its imports do not load torch, MuJoCo or physical backend. Report: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/command_profiles/OFFLINE_CHECKS.json`.

Limit: evaluator physical execution is intentionally untested until parent selects a qualified candidate. Offline checks do not establish command response, switch robustness or food approach.
