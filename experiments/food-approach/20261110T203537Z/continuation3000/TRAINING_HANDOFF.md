# Exact1000→3000 lower-lr continuation: training complete, physical evidence pending

Parent explicitly authorized this controlled continuation after the fixed1000 physical candidate failed. No original captured source/data file was edited.

Parent checkpoint SHA: `3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28`.

Final checkpoint `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/TRAIN/checkpoint_003000.pt` SHA: `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`.

## What changed

Exactly2000 additional updates, global1001–3000. Adam lr alone changed0.001→0.0001. Frozen base,53034-parameter instantaneous residual, command normalization/gate/bound1, exact four teacher episodes, equal postprime bounded-action MSE and clip1 unchanged. Saved exact loaded weights/Adam moments+all six counters1000/RNG/full parent0–1000 curve in `TRAIN/RESUME_STATE.pt`; `RESUME_CHECKS.json` verifies old/new state differs only in optimizer group lr. Actual resumed predictions exactly match parent final saved after-update metrics.

Managed training218 completed exit0 and was collected. Total elapsed203.682s. Checkpoints1500/2000/2500/3000 each passed22 independent fresh-process checks including weights, Adam states/counters, RNG, full curve prefix, immutable sources/data and neutral2880mean/action/hidden parity. No intermediate checkpoint was physically selected.

| Global step | Left fit MSE | Right fit MSE | Speed085 fit MSE | Speed090 fit MSE |
|---|---:|---:|---:|---:|
|1000 parent|0.000741510|0.000765372|0.001757067|0.001460654|
|1500|0.000251114|0.000232384|0.000950176|0.001205243|
|2000|0.000241885|0.000223292|0.000897891|0.001147754|
|2500|0.000229508|0.000210999|0.000838781|0.001078395|
|3000|0.000213642|0.000195378|0.000774388|0.000997906|

All values are post-update teacher-distribution fit, not heldout or physical success. Final raw gradient norm0.0001030; the observed continuation is substantially steadier than the old0.001 run. This supports the limited optimization-stability hypothesis; it does not prove the original closed-loop failure cause or imply actual success.

## Physical wrapper identity

Explicit new `eval3000.py`/`verify_actual3000.py` accept only the final path/step3000 and actual original FastMotor source. Same original2880decision loop, scoring, timing, neutral reference semantics, HANDOVER saved before autonomous step and full source/compiled model verification. `EVAL_PREPARATION_CHECKS.json`13 offlinechecks passed; actual physical paths require execution. Evaluators capture original plus new wrapper/prereg/pipeline identities separately. Training captured its own frozen trainer/verifier/prereg/preflight and all original parent sources/data; it did not require future evaluator files to be hashed before their preparation completed.

The original physical pipeline219 was launched then runtime/session failed. On parent recovery request, both parent and delegate confirmed no live pipeline/evaluation/audit process. Old `EVAL_neutral/` contains only IDENTITY, runtime_geometry and HANDOVER; no saved trajectory, neural arrays, RESULT or FAILURE. Original219 runtime status became unknown. These partial artifacts and stale original summary are retained. No complete physical condition existed to repeat.

A separately captured recovery runner will rerun the necessary incomplete neutral into fresh `RECOVERY_EVAL_neutral/`, then the four entirely unstarted conditions. Frozen evaluator/scoring/control sources remain unchanged; old interrupted evidence is never relabeled complete. Neutral first then left/right concurrency2 and speed085/speed090 concurrency2, CPUthreads1. Each completed condition must pass independent saved-actual inference/hidden/force/prefix/gate/cycle audit before any result is accepted. See `RECOVERY_FINAL3000_SUMMARY.json` for actual subsequent status; do not infer pass from this training report.
