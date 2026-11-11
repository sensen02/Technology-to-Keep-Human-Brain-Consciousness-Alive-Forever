# Registered mixed-control data collection: complete but safety rejected

Exactly two actual trajectories were executed after the parent's `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/RUN_PLAN.json` registration. Managed jobs224left/225right completed exit0 and were collected. No optimization, new parameter search or repeated physical collection occurred. Both use the exact fixed3000 checkpoint SHA`17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`, unchanged original full body/42finitePD/native12.5µs and zero adhesion/external forces.

Each contains10600 actual1ms endpoints and1920 neural/teacher decisions:1s zero settle,4.8s physical teacher prime, then4.8s mixed24×(25ms student+175ms teacher). Continuous128 hidden/actuator/body history preserved, teacher queried every5ms on the actual state including student intervals. Teacher phase is provenance only. Seeds are recorded but no physical initial perturbation was added; different seeds do not establish independent initial-state coverage.

| Actual mixed condition | Signed forward | Maximum slip | COM minimum | Nonfoot | Admission |
|---|---:|---:|---:|---:|---|
| Left | .070818224 | .549528870 | 1.251916922 | 0 | Reject: slip>.5 |
| Right | .071879203 | .551908018 | 1.250467994 | 0 | Reject: slip>.5 |

Both complete without early stop and pass the other four original gates. The draft's whole-window slip<=.5 requirement remains unchanged. **Neither dataset is admitted for the proposed successful-data training curriculum.** Preservation and exact labels are not safety qualification. `autonomous_walking_pass` and `food_success` remain false.

## Independent saved actual inference and physical evidence

`VERIFICATION.json` and each `COLLECT_left/VERIFIED_SAMPLES.json`, `COLLECT_right/VERIFIED_SAMPLES.json` report integrity true and dataset_safety_admitted false. They bind absolute run, samples/trace/result/checkpoint SHA, original identity and source digests.

All1920 loaded-checkpoint means, actions and hidden states match bitexactly; the entire hidden chain, actual previous-action input, schedule/source counts, held actions, original5800physical/960neural prefixes, complete clocks, zero external forces, whole collection summaries and cycle events recompute exactly. The original window denominator and all five gates are retained. Disjoint source summaries are diagnostics only, not continuous autonomous windows.

## Independent ordered teacher label reconstruction

Jobs228left/229right replayed1920 ordered teacher queries each on a private Core restored from saved qpos/qvel/ctrl/target and `mj_forward` only. `mj_step`, `mj_step1`, `mj_step2` were blocked. The teacher was constructed at the saved actual1s settled state, then its private desired-state/IK history evolved through the entire original query order. This is not a stateless oracle inference from one pose.

Every teacher42 label matches bitexactly with max error0; all root-balance residuals also match. Maximum residuals are7.42578446e-5 and1.03217063e-4, with zero queries above.02. These residuals are algorithm diagnostics, not a new admission gate or proof of physical recoverability. `LABEL_REPLAY.json` in each run records hashes and zero physical integration. All228/229 replay jobs completed exit0 and were collected.

## Matched original teacher control

Job230 used only cached saved arrays. Original left/right teacher trajectories were restricted to the identical first4.8s post-prime window `[5800:10600]`, rather than comparing a different duration.

| Condition | Mixed slip | Matched pure teacher slip | Mixed XY denominator | Pure teacher XY denominator | Mixed RM absolute weighted foot speed | Pure teacher RM speed |
|---|---:|---:|---:|---:|---:|---:|
| Left | .549528870 | .479011726 | .070829997 | .071342638 | .038923128 | .030533076 |
| Right | .551908018 | .447020885 | .072326509 | .072750532 | .039917580 | .030154795 |

The matched pure-teacher windows pass the slip gate. Mixed RM foot motion increases while denominators change little. Left RM alone exceeds.5; right RM and LH(.508840325) exceed.5. `SEGMENT_DIAGNOSIS.json` contains all six weighted foot speeds, per-leg ratios, load>=.1W numerator shares and raw source-segment integrals. This rejects the explanation that the4.8s window alone caused the mixed admission failure; it does not isolate a single causal controller error.

## Descriptive recovery-state measurements

Against the original teacher at the same sample times, standardized full-observation RMS across23 complete boundary comparisons is:

| Condition | Student segment start | After25ms student | After175ms teacher |
|---|---:|---:|---:|
| Left | .00697159 | .21521136 | .00707340 |
| Right | .00709568 | .25521182 | .00732638 |

All23 compared teacher segments end with less deviation than their preceding student endpoints. The final burst lacks a subsequent recorded decision, so it is excluded from this specific endpoint comparison while its full1ms trajectory remains present. Different durations, changing gait phases and the lack of a matched no-recovery intervention prevent a causal recovery claim. Raw endpoints do not prove safe recovery, and the whole-window safety rejection stands.

## Next boundary

No new actual run or training is authorized by this report. The parent may use these exact failed trajectories to diagnose student-distribution error and recovery-dependent loaded foot motion. Extending collection duration or changing burst ratios would require a new preregistered experiment; it cannot retroactively admit these datasets. Completed sources, results and hashes remain unchanged.
