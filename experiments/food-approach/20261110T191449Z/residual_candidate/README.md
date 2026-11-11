# Isolated neural residual motor candidate — offline only

All new files are confined to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/`. No active conditional motor, source identities, teacher files, original checkpoint or training files were edited. This candidate has NOT been selected and has no physical performance evidence.

## Architecture and rationale

Verified base actor128 runs its exact original step, with all 151636 parameters frozen. The residual receives 112 allowed physical channels selected from obs238 (excluded42:84,126:168,195:237), actual new base hidden128, and actual neural command2. Encoder242->64, slow GRU64, head64->42 have 43242 trainable scalar parameters. No dead weights are allocated for masked channels. The residual has an explicit64 state; base state remains128. Both advance on every actual sensory tick, including neutral priming, and both must be carried across teacher withdrawal without resets.

`mean = base_mean + tanh(sum(command**2)) * .5 * tanh(residual_head)`

Call `tanh(mean)` once to obtain deterministic42 actuator-rate commands, matching the existing mean/action contract. The base target-position inhibitory feedback is kept intact, frozen. The residual has no direct target-position/error/previous-action observation, and its direct current target Jacobian is exactly the base negative diagonal. This prevents direct latent cancellation via those channels; it does NOT prove stability or exclude indirect target effects via sensory history. The .5 latent bound is an engineering hypothesis requiring physical validation.

Neutral command[0,0] gives exactly zero residual regardless of residual state/weights. Thus the saved baseline neutral sequence retains bitexact mean, action and base hidden. If nonneutral control has changed the physical trajectory, returning to neutral means applying the verified base policy to that new trajectory; it does not undo the physical history or restore an old trajectory.

Nonzero head initialization scaled.01 permits gradients into every adapter layer immediately. Unlike the earlier zero-initial command encoder, nonneutral initialization is NOT exact baseline, only neutral is.

## API

```python
from residual_motor import ResidualMotor, load_baseline_actor
model = ResidualMotor(load_baseline_actor()).eval()
state = model.initial_state(batch_size=1)  # episode reset only
mean, state = model.step(obs_B238, command_B2, state)
action = mean.tanh()
# At handover: keep state, obs, physical body, target and actuator state continuous.
means, (base_states, residual_states) = model.sequence(obs_TB238, commands_TB2, state)
next_state = (base_states[-1], residual_states[-1])
optimizer = torch.optim.Adam(model.adapter_parameters(), lr=.001)
```

`save_candidate(new_path, model, optimizer, config, runtime_state)` writes only new paths inside this candidate directory. `load_candidate(path)` checks identity/source/checkpoint, restores frozen-base weights strictly, and returns `(model,payload)`; both recurrent states are `payload['runtime_state']`. Restore optimizer from `payload['optimizer']` when continuing. There is no training runner or physical evaluator here.

## Verification and restrictions

Final offline revision: 30/30 checks PASS, including nonzero gradients for all43242 adapter scalars; process exit0. All2880 neutral means/actions/base hidden bitexact before and after the temporary update.

Preregistered checks: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/PREREG.json`.
Results: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/CHECK_RESULTS.json`.
Run using `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/check_residual.py` with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.

Checks use all2880 saved verified neutral rows, including960 priming+1920 actual neural rows; true recorded teacher left_full/right_half arrays only as observations/commands/labels; one temporary Adam unit step. No teacher queries, no physical integration. Temporary checkpoint is removed automatically. An initial offline revision used zero-masked full238 features; inspection identified dead weights, so it was revised to selecting112 allowed features, then all tests rerun. Each revision had one isolated temporary optimizer test update, two total unit updates across two fresh models; no continuation or trained artifact retained.

Gate limitation: `tanh(sum(u**2))` has zero command gradient at neutral. Existing zero-output OdorBehavior cannot bootstrap through this motor by deterministic end-to-end gradient alone. Teacher command supervision, nonzero behavior initialization or stochastic exploration can provide nonzero commands. A signed-channel multiplicative residual is another architecture, requiring a separately declared revision and checks. The current norm gate is intentional for this scoped candidate, not an asserted best choice.

The copied base `transform_identity` contains historical `all_actor_weights_trainable:true` describing the original training architecture; wrapper `base_frozen:true`, actual requires_grad flags, explicit optimizer coverage and exact before/after checks are authoritative for this candidate.

Parent decides only after active finetune500 results. Any candidate BC/PPO training budget, held-out evaluation and native/physical checks must be preregistered separately. Saved-array MSE, parity and gradient checks establish an interface only; turning quality, slip and food approach remain unestablished.
