# Final native/neural data audit — S2/S5

This is a read-only preparation artifact for a future candidate. It does not run a physical simulation, change a checkpoint, change acceptance thresholds, or establish the full walking goal. Historical run files, frozen evaluator, recorder, measurement freeze, master state and tracker remain untouched.

The reusable entry point is `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152725Z/final_native_audit.py`.

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152725Z/final_native_audit.py \
  --run /ABS/COMPLETE/AUDITED_NATIVE_DIR \
  --baseline /ABS/SAME_CHECKPOINT/FAST_EVAL_DIR \
  --out /ABS/NEW_AUDIT_REPORT.json
```

All three arguments must be absolute paths. The output must be new and outside both immutable input directories; its parent must already exist. Exit code 0 means the checked data/metric conjunction passes, 1 means a measured gate failed or remains unknown, 2 means an evidence/input error. Every produced report explicitly leaves `full_goal_status=NOT_ESTABLISHED`. Same-run complete video and dependency/scope review remain separate requirements.

Complete native phase counts must be exactly80000 settle /384000 AUX /768000 ANN. Actual recorded source timestamps must align within1e-8s of nominal1/5.8/15.4s and match decision/trace boundaries exactly. The evaluator and diagnostics use these actual boundary timestamps so accumulated MuJoCo clock rounding cannot drop the final native interval. The earlier nominal-endpoint diagnostic is retained as `EXISTING_FULL_NATIVE_NOMINAL_ENDPOINT_DIAGNOSTIC.json`; its767999 ANN count exposed this completeness defect. `PREREG.json` records the clarification before candidate audit. A last-interval force/contact-spike regression proves the interval is included and strict D1 fails.

The native input must be the completed deterministic nominal 1s settle +4.8s teacher +9.6s ANN run from the immutable `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T131413Z/audited_eval.py` with `--native`. The fast baseline must use the same checkpoint. This wrapper intentionally does not accept perturbation runs with a different timeline or historical300 without a synchronous neural archive. It never launches the recorder. `RESULT.json` is the completion barrier.

Verification includes both identity digests and all source/asset/checkpoint hashes, imported deployment sources, compiled physical arrays and numerical packages, full49qpos/48qvel/42motor body, 71 opaque mesh geoms, mass/gravity/timestep, zero adhesion, exact independent logical trace/decision replay, prescribed source blocks, full hidden chain, all2880 saved neural forward outputs independently recomputed from checkpoint, actual1920 ANN actions and all14400 applied decision-held trace actions. Constructor/forward and neural inference are allowed; integration steps remain zero.

The existing frozen `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T054510Z/native_acceptance.py` executes unchanged. Its entire original report, including D7 `UNKNOWN` and historical `PROPOSED` labels, is retained. A separate `predeclared_measurement_criteria` reports conservative D7 evidence: median carrying count at >=.1W >=3 is sufficient for every reported lower positive threshold. Failure of this conservative proof is `UNKNOWN`, not a proof that the underspecified original D7 failed. Threshold histograms use 1e-6/.001/.01/.05/.1W. Original numeric criteria remain unchanged, including D1 exact zero, same-run AUX[1,5.8)/ANN[5.8,15.4) D3/D5, original D8 h0, and signed D6 p95 <=.15W.

A second streaming pass independently reports raw and continuous causal EMA .25/.5/1/2ms D6 signed and absolute p95, external-accounted variants, all actual external/generalized/adhesion force peaks, contact offsets, and every15400 native/trace endpoint (qpos/qvel/COM/action/target/commanded torque) exactly. The primary1ms D6 and .1W D7 must agree with the frozen evaluator. Every array is header-checked before wrapper allocation, at most2,000,000 scalar elements; native trajectories stay on disk in chunks. Selected scalar quantile buffers are individually capped at2M.

All five original engineering gates are reported on native records, with all per-leg slip/load integrals and original1ms legacy metrics separately. Strict six-leg cycle events, including aborts/timeouts, are replayed from the1ms sensed trace with the unchanged diagnostic CycleTracker and compared exactly to the committed events; this diagnostic does not generate commands.

Dependency evidence verifies the frozen zero-output control identity, archive hashes, and all common physical trace fields across the matched5800-row physical prefix. Its actual zero-control forward/cycle results and candidate results are reported without treating actuator holding as neural locomotion. Forty actor-only input-zero cuts across eight sensory groups at five actual saved hidden states report changes to commands. These are explicitly offline counterfactuals and make no claim about physical outcomes. No new requirement that every sensory cut must cause falling is introduced.

The recorder's RESULT omits teacher metadata. The wrapper reads reference/teacher metadata explicitly from the fast baseline RESULT only after exact trace/decision replay, verifies the reference/evidence hashes, and records that provenance instead of inventing missing recorder fields.

Validation evidence in this new directory:

- `TEST_RESULTS.json`: bounded synthetic regression, including strict D1 rejection, D7 legacy UNKNOWN, finite allocation cap, replay mismatches, hidden-chain tamper, explicit extra forces, and dynamic cross-chunk filter equivalence/sensitivity.
- `EXISTING_BOUNDED_CHECK.json`: first2048 actual native300 rows and matched existing2000/zero5800-row prefixes.
- `EXISTING_FULL_NATIVE_CHECK.json`: all1,232,000 historical native300 rows and15400 physical endpoints; primary D6 agrees with historical frozen signed/absolute p95 within6.1e-16; .1W D7 median4 AUX+NN. Historical300 slip remains FAIL.
- `RUNTIME_FORWARD_CHECK.json`: compiled body/runtime and all1920 existing2000 actual ANN actions reproduced bitexact with zero integration.
- `CYCLE_REPLAY_CHECK.json`: original9600-row sensed cycle diagnostic replay on existing2000.

No candidate2500/3000 final native/neural acceptance has been run by this wrapper preparation. The full native300 check validates measurement implementation only; it cannot retroactively supply a missing synchronous neural archive or make the failed300 a passed walking candidate.
