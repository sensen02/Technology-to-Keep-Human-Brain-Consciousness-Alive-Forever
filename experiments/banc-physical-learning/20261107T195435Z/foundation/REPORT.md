# Round10 canonical neural foundation and fixed same-leg basis

This foundation is ready for the direct physical-learning child. It creates a canonical neural state at the verified one-second physical handover, a transactional restore API, and a fixed same-leg MN basis with a bounded engineering actor. It performs no physics steps, optimizer updates or body/model copying.

Primary workspace: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation/`.

Desktop mirror: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/foundation/`.

## Canonical state

The fixed conductance core is `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py`, SHA256 `c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250`. The old qualified encoder and complete body remain frozen. The canonical physics handover is the existing `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop/HANDOVER.npz`, with its physics state and actuator target referenced by `CANONICAL.json`; the body model is referenced by path/hash and was not copied.

`build_foundation.py` replayed the first 200 rows of the saved new-core POLICY sensory currents, each for 5 ms, with zero DN current. It used the saved current vectors only; policy actions and target labels were not passed to the neural circuit. The resulting 2,000-step state at 1,000 ms matched all available handover neural fields exactly: `v`, `ge`, `gi`, `g_adapt`, delay ring, ring index, filtered MN rates and MN voltages. `NEURAL_START.npz` also stores complete 20k voltage, conductance, adaptation, refractory, spike-count, delay, last-spike, filter and clock state plus every identity/port array and the typed parameter contract.

`runtime.restore_circuit(c, path)` validates every required shape, dtype, identity array, source hash, parameter contract, finite/nonnegative state, voltage domain, spike counts, clock and delay-ring consistency before mutating any state. Invalid state fixtures were rejected transactionally, including shape, NaN, negative conductance/rate/count, ring, clock, identity, motor order and contract corruption. Restoring a fresh core and advancing the next saved 5 ms matched the continuous prefix exactly; repeating restore and continuation was also exact. `load_circuit()` uses the canonical snapshot by default and works independently of cwd.

## Fixed basis and actor

`BASE.npz` is an uncentered per-leg SVD basis of the 192931Z `SOURCE_TRACE.npz` start-of-bin MN filtered rates divided by 100. The exact frozen old `TRAIN_TRACE.npz` training mask is used, with 1,160 rows. Its boundary is preserved exactly: index 40 has time `0.19999999999998003` and is included by the old mask; recomputing a decimal `.2 <= t` condition would incorrectly remove it. No action labels or targets are loaded for basis construction.

The six actual leg blocks are LF, LH, LM, RF, RH and RM, with four uncentered SVD directions each, giving `(323, 24)`. Each direction uses deterministic sign selection: its maximum-absolute loading is positive. The raw four directions are orthonormal before whitening. Each latent is divided by `max(training RMS, .05)`, with gain capped at 20; no centering is applied. Cross-leg loadings are zero. Root IDs, MN CSV order, MN labels, joint names and actual joint leg order are stored in `BASE.npz` and `BASIS.json`.

`Actor(theta)` accepts exactly 168 coefficients as flat `(168,)` or `(6,4,7)`, bounds each coefficient to `[-.25,.25]`, forms same-leg 4-latent to 7-joint blocks, clips effective weights to `[-1,1]`, and computes `30*tanh((MN/100) @ W)`. It has no bias, history, time, phase, root, teacher, target or body input. Zero coefficients and silent MN features produce exact zero action. The weights are an engineering translation layer, not biological synapses.

## Verification and accounting

Primary checks passed with 2,030 completed neural steps: 2,000 canonical prefix steps, 10 continuous next-input steps, 10 restored continuation steps and 10 repeated restored continuation steps. Basis construction performed six deterministic SVD factorizations. No physical steps or optimizer updates occurred.

Primary numerical checks reported zero prefix/handover error and actor manual maximum error `2.6645352591003757e-14`. Thirteen corrupted state fixtures and 12 actor/current guard inputs were rejected without mutation. The source graph, frozen core, encoder, handover, prior policy and body/model hashes remained unchanged. The first build attempt failed before stepping because it recomputed the old decimal boundary mask and detected the preserved floating-time row; `FAILED_FIRST_BUILD.json` records this exit-1 event and the repair. No model or source artifact was produced by that failed attempt.

The independent verifier `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation/independent_verify.py` passed with status `PASS_INDEPENDENT_VERIFICATION`. It independently restored the state without calling `runtime.restore_circuit`, reproduced the next 5 ms, reconstructed the per-leg SVD basis and actor math, and verified all identity/order, same-leg, zero/silent, clipping and guard properties. Its 2,030 completed neural steps were 2,000 saved-policy replay steps plus three 10-step continuation checks; physical steps and optimizer updates were zero, and no labels were loaded. The final delivery manifest includes this result and byte hashes. This foundation is engineering infrastructure for physical-learning experiments; it does not establish biological calibration, walking, or behavior success.
