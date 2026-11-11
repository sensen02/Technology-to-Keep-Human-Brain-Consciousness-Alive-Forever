# Four fixed RH ANN weight calibrations: no passing left turn

All four registered actual left-turn conditions completed. Managed job `bash-10` returned exit0 and was collected; each evaluator and independent saved-array auditor returned0. There are no active jobs. No optimizer, additional alpha, neutral baseline rerun, or selected-candidate confirmation was performed.

| Alpha multiplying RH readout rows | Maximum slip (must be <=.5) | Signed forward speed | Yaw rad | Result |
|---|---:|---:|---:|---|
| 0 | .5243074851683935 | .07120734571333664 | +.06189826069199637 | Slip FAIL |
| .5 | .5169789923969749 | .07121556692016451 | +.06184914707945411 | Slip FAIL |
| 1.5 | .5206314506483894 | .07123549707436831 | +.06178078260041838 | Slip FAIL |
| 2 | .5312039097792911 | .07124580084273062 | +.06179799484143499 | Slip FAIL |

Each condition passed the other four original gates, strict nonfoot zero, all six sensed cycles (LF1, other legs2), and the registered yaw sign/magnitude criterion. Each independent actual audit passed92/92 checks, including all2880 candidate means/actions/192-state chains, full15400 endpoints, source and timing counts, previous-action alignment, original prefix, zero applied extra forces, runtime71 opaque meshes, complete-window metrics/cycles/yaw and source identities. These integrity passes do not change the slip failures.

The existing alpha1 parent left result remains .5148415850460619, better than the four tested calibrations but itself FAIL. Selection is null. Per preregistration no neutral/right confirmation is run when no left candidate passes. No qualified motor or food capability is claimed. Finite RH connection scaling alone did not solve the remaining coupled-body slip problem; this is not a body-impossibility conclusion.

## Exact intervention and preflight

Actual frozen joint order was inspected with Core construction and zero integration. Rows28:35 correspond to joint_RHCoxa_yaw, joint_RHCoxa_pitch, joint_RHCoxa_roll, joint_RHTrochanter_yaw, joint_RHTrochanter_pitch, joint_RHTrochanter_roll, joint_RHTibia_pitch. Only `readout.weight[28:35,:]` and `readout.bias[28:35]` change, multiplied by each registered alpha. All other tensors must match the exact parent checkpoint. The original ReadoutMotor class, .25 tanh correction bound, positive-command ReLU activation, physical observations, recurrent memory and final motor actions remain neural. No per-step rule, phase input, pose write or force support is added.

All four final candidate checkpoints and source hashes were saved before physical execution. Eighteen preflight checks passed, including all2880 neutral/right means and192 hidden states bitexact to the parent for every alpha. Parent identity and source/data hashes are checked by the loader; every candidate load recomputes the expected exact tensor intervention and rejects any other change.

An initial pre-physics auditor import hit the predecessor's namespace guard, exit1 before any integration. It was corrected in the NEW auditor binding and checked successfully. All initial pre-physics checkpoints/manifests are retained under PREPHYSICS_REV0; the SAME four final candidates were regenerated with final source hashes before any physical run. `PREPHYSICS_FIX.json` records this correction. Jobs8/9 preparation both exit0 and collected; no physics was repeated.

## API and outputs

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T024558Z/model_io.py` exposes `load_checked(candidate_path, expected_sha, require_final=True) -> original ReadoutMotor, payload`. Candidate payload retains original model identity and state_dict/config lineage and adds candidate_id, alpha, parent_sha256 and calibration_source_hashes. These four candidates are diagnostic failures and are not deployment-qualified.

`FINAL_SUMMARY.json` records all four outcomes and selected=null. Each candidate directory holds candidate.pt, EVAL_left/ complete records, and VERIFY_left.json. `PAD_SPEED_DIAGNOSIS.json` is a read-only whole-window numerator/denominator decomposition; it adds no new scoring gate or causal claim. `ARTIFACT_SHA256.json` seals outputs and explicitly verifies all final captured sources unchanged. No prior timestamp file was modified.
