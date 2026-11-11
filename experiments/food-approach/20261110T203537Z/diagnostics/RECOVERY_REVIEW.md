# Independent read-only recovery review

Reviewed `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/recover_pipeline3000.py`, frozen `pipeline3000.py`, `eval3000.py`, `verify_actual3000.py`, and new `report_from_summary.py`. Read current summary/association manifests and inspected the original incomplete directory. No source/data modification, physics, optimization, duplicate audit, or new simulation was performed. This report is the only new output of this review.

## Finding

**No concrete recovery, checkpoint-binding or scoring defect found for the actual five completed recovery cases.** At the review snapshot, `RECOVERY_FINAL3000_SUMMARY.json` was already COMPLETE. It records neutral engineering PASS and four non-neutral engineering FAILs, while independent saved-trajectory integrity is true for all five. This distinction is preserved by the recovery runner and the report generator.

| Case | Engineering eligible | Saved integrity | Recorded audit checks |
|---|---|---|---:|
| neutral | true | true | 96 |
| left | false | true | 72 |
| right | false | true | 72 |
| speed085 | false | true | 75 |
| speed090 | false | true | 75 |

Actual summary criteria additionally show left fails directional magnitude and both turn cases fail six-cycle completion; speed085 fails strict nonfoot zero. The report's table does not enumerate every failed subcriterion, but does not claim these failures passed. Functional success must continue to come from `criteria.eligible`, not audit count or process exit code.

## Interrupted evidence and non-repetition

Original `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/EVAL_neutral/` still contains only `HANDOVER.json`, `IDENTITY.json` and `runtime_geometry.npz`. There is no original RESULT, saved neural trajectory or physical endpoint trace. Original `FINAL3000_SUMMARY.json` has an empty conditions dictionary. No original `EVAL_left`, `EVAL_right`, `EVAL_speed085` or `EVAL_speed090` directories exist.

The recovery checks exactly these preconditions before proceeding, refuses already completed/original non-neutral runs and refuses preexisting recovery output directories. It writes new `RECOVERY_EVAL_*` directories and new logs/audits; it does not overwrite or delete the interrupted directory. Consequently this execution restarted an unauditable incomplete neutral trajectory and then ran four previously unstarted conditions. It did not repeat a completed condition.

The script's `processes_surviving=False` and interruption reason are supplied orchestration facts rather than process-discovery assertions implemented in this script. This review checks the resulting preserved artifacts and manifests, not historical process liveness or the cause of the interruption. No numerical continuation from the interrupted in-memory body is claimed.

## Exact model and source association

The final checkpoint is fixed to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/TRAIN/checkpoint_003000.pt`, SHA-256 `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`.

Read-only hashing verified all 18 entries assembled from the neutral association's evaluator source freeze, original source freeze, recovery-runner hash, three interrupted artifacts, original summary and final checkpoint. All matched their recorded digests. The old partial files and old summary remained unchanged at this snapshot.

Each recovery association records the exact command, output directory, checkpoint/path/SHA, new recovery-runner hash, immutable evaluator/original source identities, and interrupted artifacts. Frozen `eval3000.py` retains its original source capture; the new orchestration source is bound in the separate association and final summary. The recovery runner rechecks evaluator readiness, checkpoint and its own source hash after completing all groups. This is explicit separate orchestration provenance, not relabeling the old evaluator source.

One scope qualification: the end-of-run code rechecks runner/evaluator/checkpoint hashes but does not itself re-hash the original partial files or old summary. Those are hashed at recovery start; this review independently re-hashed and confirmed them unchanged. No evidence of mutation was found.

## Scheduling and neutral reference

The condition list is fixed to exactly neutral, left, right, speed085 and speed090. Groups are neutral first, left/right second, speed085/speed090 last; each `ThreadPoolExecutor` is capped at two, each condition performs its evaluator and auditor sequentially, and all numerical-thread environment variables are set to one. The speed group runs only if the recovered neutral is eligible.

Both speed runs point to the new recovered neutral RESULT. Frozen evaluator checks same checkpoint path/SHA, neutral engineering PASS, matching compiled arrays and evaluation-source hashes, plus saved neutral trace/neural hashes. Independent audit checks the neutral reference digest and repeats same-model/source/geometry/speed binding. Scoring remains original all-five physical gates, all-six cycles, strict nonfoot zero, signed turn magnitude >=0.01, and speed <=0.95 times the qualified matched neutral. The 2880-decision teacher-prime/autonomous control loop, timestep and handover recording are unchanged.

## Integrity versus functional acceptance

Recovery `eligible` requires evaluator engineering pass AND audit engineering pass AND audit integrity AND both checkpoint SHAs matching the fixed final model. `saved_trajectory_verified` requires audit integrity only. Final `all_conditions_eligible` and `saved_trajectory_verification` are computed independently. A numerically consistent failure therefore remains an engineering failure even when evaluator/auditor exits are zero.

The frozen audit reconstructs full checkpoint inference/hidden/action chains, all15400 endpoints, original prefixes and source boundaries, zero extra forces, saved handover hashes, whole-window summary/cycle events, and exact criteria. It does not claim full native D1–D8 or biological certification. No audit was rerun by this reviewer.

## Report generator

`report_from_summary.py` reads the recovery summary, requires COMPLETE and all five condition keys, writes per-case engineering/integrity flags separately, prints the corresponding forward/slip/yaw measurements, and explicitly states that integrity is not behavioral success. It preserves the interruption/recovery history and makes no food-arrival, ingestion, satiety or full-native claim. Its training duration and checkpoint-history prose are fixed statements rather than recomputed validations; they correspond to the documented completed continuation, and this report generator is a summary, not a replacement for training/audit verification.

Generic limitation that does not affect the present completed execution: COMPLETE is based on the five-key set, and missing/unrun conditions can still occupy keys after a failed neutral or process exception. The report generator does not additionally require five actual execution/result records and would show missing numbers as `None` while retaining an 'all five actual' title. In the current summary, all five actual evaluations and audits are present, so this latent generic wording issue does not misstate the current outcome. Do not reuse this formatter as proof of actual execution when a future summary contains NOT_RUN/error placeholders.

The report does not independently verify summary/result hashes; it trusts the just-produced summary and records hashes of itself/report/summary afterward. Here the frozen runner, per-condition result/audit association and the read-only source/partial checks provide the relevant evidence. No source change is requested during the captured run.

## Execution and communication scope

Only bounded reads, directory inspection and a read-only 18-digest/current-summary check were run. No background job was started by this review. A requested `send_message` to the supplied parent id `session-80ccae55-a957-4374-b48e-7ba52c0ab34c` was rejected by the tool as belonging to another parent session. The review is therefore returned through this file and the final response; no cross-session delivery success is claimed.
