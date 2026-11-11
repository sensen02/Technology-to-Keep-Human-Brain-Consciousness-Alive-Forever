# Fixed2000 positive-command neural readout evaluation

Code preparation only. No physical integration, optimization or new trajectory was executed by this delegated task.

The evaluator derives the previously frozen physics/inference/scoring implementation from `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/evaluate_motor.py`. Explicit changes are final checkpoint path/step2000, new source metadata and strict new loader, new root preregistration path, and an additional measured parent-right preservation comparison. Source identities from the previous evaluation and new one remain separately recorded. Old files are neither rewritten nor relabelled.

The new `train/model_io.py` must expose `load_checked(checkpoint,sha,require_final=True)`. The actual checkpoint is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt`, SHA `eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df`. The loaded payload's final curve step and `physical_eval_only_step` must both be2000; wrong paths fail before model loading.

Parent-authorized launch:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/evaluation/pipeline.py --training-complete --sha256 eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df
```

Neutral physical evaluation and independent audit run first; only after both pass do left/right run with maximum two single-thread CPU processes. Each uses1000ms zero settle,4800ms actual teacher priming and9600ms pure ANN. The turn command has the original1s smoothstep. Original five gates, six sensed cycles, strict nonfoot0 and signed yaw magnitude0.01rad apply over the entire autonomous interval. Both explicit128+64 states persist; HANDOVER hashes are saved before autonomous integration. Finally blocks preserve partial artifacts on errors. No source positions, phase or clock enter the actor.

The actual neutral trajectory is checked against the original verified walking baseline. Actual right additionally compares every shared saved physical, neural and head/sensor geometry array to the passed parent right trajectory in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/EVAL_right/`, including both hidden memories. The parent independent audit and all artifact digests are verified. This is a fixed-right-history comparison; it says nothing about switching after positive-command history.

`verify_actual.py` independently loads each archive once and replays the actual new checkpoint over all2880 observations/commands, validates192-state chains and actions,15400 timestamps/source counts/zero external force, compiled71-mesh geometry, derived handover hashes, whole9600-window metrics and cycle events. It does not construct `Core` or integrate physics.

Preparation validation: `check_prepared.py` passed36 checks using the exact provided interpreter. Actual final2000 production-sized inference on all2880 saved parent-right observations matched parent means, actions and both states bitexactly. The preservation comparator also passed the full saved parent arrays and rejected an intentionally changed residual hidden value. AST tests retain original scoring, command, pinning, flattening and parser functions and independent audit helper equations. This offline inference is not a new physical trajectory or final qualification.

`READINESS.json` freezes the five evaluator/check files. Training sources and top-level JSON files are captured at actual execution; they must remain immutable throughout all three physical runs. This Markdown file is documentation only and is not part of that source set. No speed, food, fullnative D1-D8, biological or connectome certification is claimed.
