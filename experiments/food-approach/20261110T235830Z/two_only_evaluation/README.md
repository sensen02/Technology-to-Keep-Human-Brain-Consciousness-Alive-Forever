# Original-two-trajectory control evaluation — CODE ONLY

This directory prepares the independent budget-matched two-teacher control registered in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/TWO_TRAJECTORY_CONTROL_AMENDMENT.json`. It does not start physical evaluation, training or optimization.

## Explicit derivation and identity

The primary frozen evaluator, auditor and pipeline under `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/` are SHA-pinned in this directory's `EVAL_PREREG.json`. They remain unchanged on disk. The own wrappers compile that exact source with their own `__file__`, so output paths and captured source identity refer to this control directory. The evaluator changes one descriptive `kind` string and explicitly binds `TRAIN`/`CHECKPOINT`/`source_paths` to the control. No physical integration, inference loop, command formula, scoring, handover or audit equation is changed.

Actual checkpoint path is ONLY:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`

The actual loader is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/model_io.py`, using `load_checked(checkpoint,sha,require_final=True)`. It validates the control storage kind, fixed final1500, original architecture and frozen base, all source/data identities and exactly `['left_full','right_full']` with loss episode count2. A primary checkpoint path is rejected before loader invocation; intermediate1000 is rejected. Same network identity does not imply same training/data/execution identity.

The actual source map includes the control amendment, own wrappers/prereg/readiness/checks, actual control training Python/JSON files, and explicitly frozen primary source identities. It does NOT enumerate primary training sources. Imported original physical modules are captured by original runtime identity machinery. Neutral is newly run at the control checkpoint and independently audited before left/right; a primary candidate's neutral is never reused as if it had the same model weights or identity.

## Unchanged experiment

Exactly neutral, left and right are prepared:1s actual zero settle +4.8s original teacher/neutral neural prime +9.6s pure ANN. Forward modulation is structural0; left/right turn use the same one-second smoothstep as qualified teacher observations. Both recurrent memories `(base128,residual64)` continue without reset and are logged as192 dimensions. Immediate HANDOVER hashes precede autonomous steps. All original five gates, six sensed cycles, strict nonfoot zero, signed turn magnitude≥.01rad and the full9.6s scoring window are unchanged.

The independent auditor constructs no Core and performs no physical integration. It loads each NPZ once, replays all2880 actual checkpoint means/actions/both hidden states, verifies192-dimensional chains, complete15400 endpoints/sources/clock/zero extra force, original5800 physical prefix/960 neural prime, whole9600 summary/cycle events, actual root yaw, diagnostic head geometry, compiled71-mesh opacity, and eight observable handover hashes. Neutral additionally compares complete original shared physical arrays and base128 neural output/state bitexactly. Optional individual `--native` only records chunks; original fullnative certification remains separate.

## Parent-controlled launch

Only after the parent confirms final control checkpoint integrity/SHA and releases physical resources:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/pipeline.py --training-complete --sha256 EXACT_CONTROL1500_SHA
```

Own outputs are `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/EVAL_neutral`, `EVAL_left`, `EVAL_right`, corresponding `VERIFY_*.json` and `FINAL_SUMMARY.json`. Neutral and audit run first; turns run only if both pass, at most2 CPU-single-thread processes. The parent coordinates the shared physical-process limit with other experiments. Existing outputs are never overwritten.

Individual evaluation and audit use own `evaluate_motor.py` and `verify_actual.py` with the same explicit checkpoint/SHA/condition/fresh-output arguments as the primary interface. Actual execution requires `--training-complete`; code import or `pipeline.py --prepare-only` never runs physics. Partial exceptions preserve available arrays and explicit failure records.

## Offline checks and limits

`READINESS.json` records source hashes and offline checks. AST comparison confirms original scoring, physical/inference loops and command formula remain unchanged; binding checks distinguish control versus primary paths and data identities. Both actual old immutable2880 teacher streams verify public control tuple API means and all192 states unchanged by flatten/split. No new control1500 checkpoint is claimed evaluated by these checks.

A fixed-condition engineering pass would authorize only further parent-registered command/holdout qualification, not food success, native D1–D8 or biological claims. The original-two-only comparison is predeclared; it does not prove every failure mechanism or silently relabel earlier failed candidates.
