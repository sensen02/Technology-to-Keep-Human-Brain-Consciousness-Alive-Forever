# S141/P1 neural-derived one-time activation initialization — SOURCE READY protocol

Primary new source: `/run/media/sensen/Data2/cell_wound_prototype/run_neural_activation_initialized_probe.py`.
Frozen original runner: `/run/media/sensen/Data2/cell_wound_prototype/run_extended_initialization_probe.py`.
Frozen passive recorder: `/run/media/sensen/Data2/cell_wound_prototype/run_constraint_row_recording.py`.
Fixed fresh output: `/home/sensen/Desktop/engine修复验收/s141-neural-activation-initialization/20261024T200305Z`.

This document precedes actual execution. The source-writing agent performs NO actual model load, forward, physical step, or neural tick. Parent alone executes after independent review. No historical source, tracker, XML, pose, weights or checkpoint is edited. This is a NEW initial condition, NOT an eligible candidate under the same-condition S119 seventeen-rule admission. All comparisons with S121 are descriptive; no standing/walking/biological claim. Native physical units remain NOT_ESTABLISHED.

## Source audit and sole intervention

Original source line225 performs the initial zero-activation forward used for actual initial sensory observation. Original lines230–233 perform PRE160 genuine 0.5ms neural steps (80ms), holding that observation, with NO body stepping or activation integration. `dec.consume` returns `u` each tick but does not write `d.ctrl`. Original lines234–237 reconstruct the final release command from the final actual FIR history and original W. Original lines238–241 serialize zero act/ctrl; line243 asserts zero activation/control/velocity/external loads and unchanged pose. Original sample0 is collected at line268 BEFORE tick0. Original lines275–283 consume the first physical neural tick (.080–.0805s) then install its command.

There is NO existing safe callback between the line243 zero-state guard and initial sample. Wrapping COM capture would be too late because `sample()` has already copied act/ctrl by that time. The new wrapper compiles a private string of the exact original source with ONE exact, uniquely asserted insertion immediately AFTER the retained line243 guard and BEFORE sample0. No original file is changed. Both insertion text/anchor and compiled SHA are preregistered; executed private source is archived in `PRIVATE_RUNNER_SOURCE.py`. All original zero-state guards are retained.

The inserted hook requires `m.nu == m.na == 90`, every compiled `actuator_actnum == 1`, and `actuator_actadr` a permutation of 0..89. Any mismatch aborts, never guesses activation indexing. It uses the actual final `warmu[-1]`, verifies exact equality with reconstructed release_decoded, assigns ctrl to that command and sets `d.act[actuator_actadr] = command` ONCE. This assignment is solely neural-history-derived; no external controller, posture target, servo, fixed pose, or synthetic action sequence. It adds ONE explicitly reserved forward so the initial physical sample reflects the initialized activation and recomputed real contact/COM/constraint force state. Subsequent original tick0 replaces ctrl normally and original free neural-body integration proceeds unchanged.

Original zero-state initialization is preserved as `PRE_RELEASE_ZERO_INITIALIZATION.npz`. The authoritative `INITIALIZATION.npz` is written after the one-time activation assignment and release forward, preserving all actual neural warm history while replacing body_act/body_ctrl with the real release values. It additionally records before-initialization act/ctrl/qpos/qvel, compiled address/count mapping, and the actual command. `ACT_INITIALIZATION.json` and all90-row `PHASE_ACT_INIT.csv` declare the event. Original INITIAL_OBSERVATION remains legitimately the zero-act observation used during PRE160, not falsely relabeled as release observation.

## Bound experiment and ledger

One fixed full-body trial only: original XML, original saved pose, original W, original real 17099-neuron network, PRE160 then 6 further neural ticks, RK4 dt25us, 20 body substeps/tick, physical 3ms, 121 physical samples. Total actual explicit calls: model_load=1, forward=122, step=120, neural_ticks=166. The forward count is initial sensory + one release recomputation +120 after-step; MuJoCo RK4 internal evaluations are not explicit forward calls. Original hardcoded 121/6/166 diagnostic lengths are retained; 10ms extension is intentionally NOT attempted.

Every original trajectory field is retained including native body COM Jacobian/Jdot/force/impulse data and raw actuator moment CSR. Passive S121 recorder captures all121 actual constraint states including raw sparse-or-dense J, row metadata and contact mapping without extra calls. `ACTUAL_VS_S121.csv` gives all121 sample time/load/limit/angle/nonfoot/all-six-gates paired values and deltas; `BASELINE_COMPARISON.json` gives descriptive extrema only. Parent independent verifier should recompute full gates, first failures, support duration, limit maxima, neural/FIR identity, RK4 activation evolution from NEW initial activation, raw J.T force closure, COM closure, original source/hash immutability and ledger. A diagnostic improvement is not candidate promotion or autonomous physical success.

All source-chain SHA values are checked before and after, including frozen S121 inputs/artifacts, current wrapper and this preregistration document. `PREREG.json`/`S141_PREREG.json` are written before model construction. Source freeze means do not modify these source/doc files after parent starts the real trial.

## Parent command (NOT run by source-writing agent)

`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/run_neural_activation_initialized_probe.py --output /home/sensen/Desktop/engine修复验收/s141-neural-activation-initialization/20261024T200305Z`
