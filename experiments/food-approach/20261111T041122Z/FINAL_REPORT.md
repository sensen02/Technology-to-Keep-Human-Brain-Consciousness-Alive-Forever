# Fixed ±0.5 neural command-domain tests: neither profile qualifies

Both preregistered actual tests completed. Managed physical pipeline `bash-22` returned exit0 and was collected; both evaluations and both independent saved-array audits returned0. No model tensor, optimizer, body, actuator, scoring gate, window or denominator changed. No further command magnitude was tested.

| Fixed final turn command | Maximum slip, original <=.5 | Whole yaw rad, signed magnitude >=.01 | Signed forward | Result |
|---|---:|---:|---:|---|
| +.5 | .5142855862877349 | +.009259551106626348 | .07171767948595652 | Slip FAIL and yaw magnitude FAIL |
| -.5 | .5276619022605319 | -.02696263893214417 | .07034485124161609 | Slip FAIL |

Both profiles passed support, nonfoot, COM and forward original gates, strict nonfoot zero and all six sensed cycles (LF1, other five legs2). The left yaw threshold remains .01; it was not reduced to .005. The left slip threshold remains .5 and is not rounded to a pass.

Each saved-array audit passed88/88 checks, including all2880 actual checkpoint means/actions/192-dimensional hidden histories, previous-action alignment, exact held-action/source timing, all15400 endpoints, zero extra forces, unchanged original prefix,71 opaque mesh compiled runtime, full9.6s summary/cycles/yaw and the new actual half-command formula. Integrity does not alter either failure. The full-right(-1) parent-trajectory comparator was deliberately not applied to half-right(-.5): those inputs differ. The new half-right was judged on its OWN actual checkpoint and full original gates; the earlier full-right PASS remains valid only for that earlier fixed stimulus.

## Unchanged model and declared input domain

Exact checkpoint: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt`, SHA eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df. All197608 ANN parameters and the original ReadoutMotor class/128+64 memories are untouched. The new test input map is only `(0,+.5)` or `(0,-.5)` multiplied by the same1s smoothstep after withdrawal. This is a declared high-level test stimulus; actual42 joint actions remain current-observation/recurrent ANN outputs. No clock/phase/food source coordinate was added to the actor.

Timing remains1s zero settle +4.8s actual physical teacher priming +9.6s pure ANN. Native12.5us, target1ms, policy5ms,42 finite PD, complete71 meshes, zero adhesion/applied extra forces and no root trajectory writes remain unchanged. Both recurrent states are continuous, with immediately saved HANDOVER hashes. Scoring includes the entire onset and full9.6s.

The exact same-checkpoint already passed neutral physical run and audit are reused with hash bindings in PREPARATION.json and FINAL_SUMMARY.json; no unnecessary neutral simulation was repeated. Preparation verified the strict unchanged checkpoint, old neutral artifact hashes and full2880 neutral production outputs/states. The original evaluator/auditor source digests are pinned to known frozen hashes before execution. The physical loop and scoring are unchanged; only explicit profile map, new source/output identity and checkpoint loader binding differ.

## Preparation issue retained

Initial preparation job `bash-20` exited1 BEFORE any physics because an algebraically equivalent independent smoothstep expansion was compared bitexactly against the factored production expression. Maximum discrepancy is2.220446049250313e-16. The failed PREPARATION is preserved as PREPHYSICS_ARITHMETIC_DIAGNOSTIC.json; PREPHYSICS_NOTE.json explains the correction. Production command computation was unchanged. Preparation now uses the same predeclared1e-14 algebraic comparison tolerance used in earlier profile checks; actual saved command rows are audited exactly against production command_at. No physical gate or yaw threshold changed. Corrected preparation `bash-21` passed14 checks, exit0 and collected, then and only then physical22 launched.

## Conclusion and outputs

`FINAL_SUMMARY.json` is COMPLETE with both_fixed_profiles_pass=false. These two fixed half inputs do not establish a usable restricted command domain, and no continuous [-.5,.5] range, switching, initial-state robustness, native formal certification or food behavior is claimed. No additional magnitude, candidate, optimizer or physical condition is registered by this task. Existing full-left failures and full-right passes remain immutable.

Public inference alias `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T041122Z/model_io.py` exposes `load_checked(path,sha,require_final=True)`, accepting only the exact original eeced checkpoint and delegating its strict source/data/architecture checks. Each profile directory contains complete actual records and VERIFY.json. All final captured source hashes are unchanged and sealed in ARTIFACT_SHA256.json.
