# Four fixed RH output-neuron biases: all left-turn cases remain below qualification

Managed physical pipeline `bash-16` completed with exit0 and was collected. All four actual evaluations and all four independent audits returned0. There are no active jobs and no optimization was run.

| Exact changed neuron bias | Delta | Maximum slip (must be <=.5) | Signed forward | Yaw rad | Result |
|---|---:|---:|---:|---:|---|
| joint_RHTibia_pitch, index34 | -.05 | .5184743239053445 | .07115721971854014 | +.06139746199753331 | Slip FAIL |
| joint_RHTibia_pitch, index34 | +.05 | .5135699022944087 | .07128899171646293 | +.062115819881903685 | Slip FAIL |
| joint_RHTrochanter_pitch, index32 | -.05 | .5249652247799116 | .07100667691598193 | +.060835917230827465 | Slip FAIL |
| joint_RHTrochanter_pitch, index32 | +.05 | .5094544386920862 | .07142287847813633 | +.06266490548701936 | Slip FAIL |

All four passed the other four original gates, strict nonfoot force zero, all six sensed cycles (LF1, every other leg2), and the registered positive yaw magnitude. Each independent saved-array audit passed92/92 checks: all2880 candidate checkpoint means/actions/192-dimensional state chain, full15400 endpoints, sources/clock, actual previous-action alignment, unchanged original prefix, zero extra forces,71 opaque mesh compiled runtime, complete9.6-second summary/cycles/yaw, source and artifact identities.

Selection is null. Per preregistration no neutral/right confirmation was run because no left candidate passed. Parent delta0 left slip remains .5148415850460619. Trochanter+.05 produced the smallest measured slip of these four but still FAIL; neither that comparison nor local finite differences permit extrapolating a passing unseen bias or extending the registered grid. No extra joint, delta, model, optimizer or case was added.

## Implementation boundary

Each checkpoint copies the exact immutable head2000 parent SHA eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df. Only one `readout.bias` scalar changes by the registered delta. Every load reconstructs the expected tensor dictionary and verifies every other weight/bias bitexact. The original ReadoutMotor class, positive-command ReLU*.25*tanh branch, both128+64 recurrent memories, observations,42 bounded neural action outputs and physical actuators are unchanged. These are constant ANN neuron biases, not runtime pose or gait rules.

Joint names were checked against the sealed actual zero-integration joint mapping from the previous finite-gain experiment. Eighteen preflight checks passed with zero integration: exact parent identity, sealed index32/34 names and all four candidates' full2880 neutral/right means and192 hidden histories bitexact to the parent. All four final checkpoint/source hashes were frozen before any physical execution. All runtime source paths exist before Core construction. Frozen024558 evaluator/auditor/pipeline bodies are reused read-only in a new explicit namespace; no old file was modified.

Actual runs retain complete1s zero settle +4.8s physical teacher priming +9.6s pure ANN, one-second smooth turn command, continuous both memories with immediate HANDOVER records, native12.5us/target1ms/policy5ms,71 opaque meshes and42 finite PD actuators, zero adhesion/applied extra forces and no root trajectory writes. No threshold, slip denominator, scoring window or controller source was changed.

## Outputs and API

- `FINAL_SUMMARY.json`: all four outcomes, selected=null, no confirmation cases.
- Each named candidate directory: `candidate.pt`, complete `EVAL_left/` arrays/identity/result/HANDOVER and `VERIFY_left.json`.
- `BIAS_FINITE_DIFFERENCES.json`: descriptive slopes between the two fixed observed signs; no new physical case or pass prediction.
- `ARTIFACT_SHA256.json`: verified output and frozen source digests.

Public API `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T033228Z/model_io.py` exposes `load_checked(candidate_path,expected_sha,require_final=True) -> original ReadoutMotor,payload`. Payload retains original architecture and parent training state and adds candidate_id,bias_index,bias_delta,parent_sha256,calibration_source_hashes. These are failed diagnostic candidates; no one is deployment-qualified.

This bounded operating-point test establishes that these four particular single-neuron bias changes do not qualify left turn. It does not prove the body cannot turn, identify a unique cause, establish neural deceleration or food approach, or certify original connectome/biology/full nativeD1-D8. All original failures remain immutable.
