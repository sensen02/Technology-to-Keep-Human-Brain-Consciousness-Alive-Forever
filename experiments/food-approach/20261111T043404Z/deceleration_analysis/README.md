# Read-only deceleration analysis: try a small neural-rate hypothesis before another large residual

This directory contains analysis only. No Core was constructed, no physics backend was imported, no optimizer was created, no weight was changed and no new training dataset was generated. Existing teacher arrays were loaded once per analysis run; the bounded second run added teacher-distribution latent-error diagnostics. Both managed analysis jobs exited0 and were collected. All read input hashes remained unchanged. These findings do not qualify neural deceleration or a speed/turn combination.

## Verified data and available physical margin

The qualified speed teachers are under `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_moderate/`, rather than the adjacent `teacher/` directory. Existing `VERIFICATION.json` has63 independent checks and `CURRICULUM_ADMISSION.json` admits all3 actual cases. This analysis authenticated RESULT/SAMPLES/TRACE/IDENTITY against those stored hashes, required original all5/all6/strictNF0, verified15400 trace rows/2880 decisions and native12.5us/42 actuators/nq49/nv48, and compared every saved trace field for the first5800 endpoints and every sample field for the first960 decisions bitexact.

| Actual entire9.6s teacher window | Matched1.00 | .85 | .90 |
|---|---:|---:|---:|
| Signed forward, native/s | .07169613451 | .06011603057 | .06442140627 |
| Margin above original .05 gate | .02169613451 | .01011603057 | .01442140627 |
| Maximum slip | .46213868766 | .46596501313 | .46466051587 |
| COM minimum | 1.25602067719 | 1.25709055717 | 1.25670629515 |
| LF/RF completed-cycle age, seconds | 2.401/3.196 | 2.826/3.760 | 2.669/3.551 |
| Cycles LF,LH,LM,RF,RH,RM | 1,2,2,2,2,2 | 1,2,2,1,1,2 | 1,2,2,1,1,2 |
| Original all5 / strictNF0 / all6 | PASS | PASS | PASS |

The .85 RF cycle has only .240s margin to the existing4s CycleTracker timeout. A neural candidate cannot slow indefinitely and retain the same cycle qualification. No extended window or timeout is justified. All absolute force/SI/biological calibration remains NOT_ESTABLISHED. Teacher nativeD1–D8 status remains UNKNOWN despite real native integration; these are endpoint engineering admissions, not formal native certification.

Both teachers change the entire movement phase over the original1s smooth transition after4.8s priming, keeping full stride. The last command is approximately[-.15,0] or[-.10,0]. Actual target-speed observation stays .075; it must not silently be replaced by .06375/.0675 as a shortcut. Teacher phase and nominal source speed are privileged labels/diagnostic metadata, never actor inputs.

## Timing evidence is unusually specific, but it is not evidence for an ANN rate change

For .85, joint-observation RMS difference against the matched straight teacher is .0561932 at equal wall time, but only .0003514 after interpolation on saved teacher phase. Target-position RMS falls .0670686→.00097065, and action RMS .1056127→.0116718. For .90 the corresponding values are joint .0461640→.00025527, target .0578946→.00062771, action .1067635→.00777331. Full phase-aligned values and exact prefix checks are in `METRICS.json`.

Thus the teacher data directly supports a mostly time-dilated trajectory with a smaller action-amplitude change. It does not show that changing a GRU membrane rate will time-dilate physical locomotion: the network is driven by actual proprioception, its target-position inhibition is instantaneous, and its decoder still emits bounded target increments each fixed5ms decision. Period and actuator rate cannot be assumed to follow ALPHA.

On the actual speed-teacher observations, the frozen eeced parent with sanitized command[0,0] has bounded-action MSE .0100694 (.85) and .00542375 (.90), versus .000614453 on the matched control. Additive pre-tanh residuals needed to exactly match teacher actions have abs p95/p99 of .249927/.508513 for .85 and .173780/.366877 for .90. This suggests that an arbitrarily tiny new output branch is not automatically sufficient. The latent maximum on the control is dominated by very rare saturated teacher actions, so those extrema must not be treated as a physically required correction bound.

## Current neural implementation

The exact chain read is:

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/model_io.py`: eeced positive-turn head2000, rejects nonzero forward input, explicit state[128,64].
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/model_io.py`: frozen-base slowResidual1500 c6b parent.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/residual_motor.py`: 112 permitted physical channels+current base128+command2, slowGRU64, bounded .5 correction gated by tanh(sum(raw command²)).
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/models.py`: baseSlowGRU128; both base and turn residual use ALPHA=.048770575499285984 at5ms, equivalent declared tau=.1s.

The base mean is decoder(h128) minus softplus(raw_gain)*obs[126:168]. Masked ranges are[42:84],[126:168],[195:237], leaving112 physical channels. The target-position inhibition must remain unchanged. It is neither a PD controller added by this analysis nor a new phase generator; it is an already-declared neural connection in the accepted ANN interface. The body/finite-PD implementation is not modified.

The old fast model (`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/fast_motor.py`) has a nonrecurrent242→128→128→42 correction on a frozen recurrent base, normalized forward command by .15 and residual bound1. Its final corrective-fit5000 actual speed085/090 runs went backward (-.0275832/-.0156098), with slip4.77047/.775149 and incomplete cycles. Those failures rule out reusing that checkpoint as a qualified speed controller. They do not prove that all residual architectures fail, nor that memory deficiency is the unique cause.

## Comparing the two proposed neural mechanisms

### A: separately gated slow speedGRU64

A new encoder306→64, GRUCell64 and head64→42 would add47338 parameters and one64-state vector. Features can be actual112 permitted physical channels+CURRENT parent h128+h64+new command2. The parent must always receive[0,turn], with its actual weights and old192-state updates unchanged; the speed correction is activated only by a neural ReLU(-forward) gate. Zero-initialized output gives the original function at initialization. This is an explicitly neuronal speed branch, not a per-step leg rule.

Its advantage is sufficient temporal/output flexibility to learn slower stance progression and amplitude correction while retaining the strong original target inhibition. Its costs are another recurrent policy with only2 qualified speed demonstrations, direct-output overfit risk, a new256-state layout and no proof that a lower teacher MSE improves physics. A bound must be preregistered in light of the measured latent residuals; .25 with raw unnormalized ReLU(-u0) would cap correction near.0375 and can be much too small. A large normalized bound1 merely repeats a prior risky design and is not supported by success evidence.

Current actual/evidence code hardcodes192, not just a generic flatten: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/evaluate_motor.py` flatten requires exactly[128,64] and writes that handover layout; its `verify_actual.py` checks(2880,192), zero192, [128,64] snapshots and hidden-prefix slicing. A needs explicit[128,64,64]/256 audit, row counts, full-chain/HANDOVER hash dimensions and independent checkpoint replay. Ignoring the last64 would invalidate the audit.

### B: command-to-neural-rate modulation

Use ONE learned scalar neuronal connection kappa and the explicit neural activation `alpha(u0)=ALPHA0*2*sigmoid(kappa*u0)`. It changes the recurrent membrane integration in the base128 and residual64 updates, with sanitized parent turn embedding[0,turn]. It does not scale a body trajectory, change dt, resample actions or mutate Python/class ALPHA in a deployment loop. The original decoder, negative target-position connection, all other parent weights and actor observations remain fixed. At u0=0, sigmoid(0)=.5 and alpha equals the old coefficient exactly, so the old operation order and all192 states/outputs can be preserved through any original forward0 history.

Teacher steady scales .85/.90 correspond to kappa about2.015/2.007 in this sigmoid relation; kappa2 gives rate factors approximately .851115/.900332. This is a useful initialization-scale calculation, not a measured ANN speed or a speed-output controller. B has no new recurrent state and very few new weights, but its physiological interpretation is only an engineering hypothesis. It changes the base hidden trajectory at nonzero forward, cannot use frozen-parent hidden caches for training, and must run full BPTT through the modified history. It may have weak physical effect, the wrong effect, or violate the4s cycle deadline.

## One recommended bounded next-stage plan: B first, as a falsifiable small-neural-parameter trial

I recommend B before A. The reason is the specific time-dilation evidence, its small parameter count and its unchanged192-state layout. This is a priority choice for a cheap identifiable experiment, not a prediction that B will work. If it fails, A remains a separate later hypothesis; do not start both routes in the same registration.

A proposed new registration (NOT performed here):

1. Bind the eventual CEM-selected qualified parent checkpoint by its actual SHA and tensor-difference proof, or explicitly name eeced if no new qualified parent exists. Freeze all existing ANN weights. Add only kappa, initialized0; retain parent commanded turn but sanitize forward to0 in its embedding. Implement the neural rate as an explicit torch sigmoid node feeding the two membrane equations, using the original operation order. Never assign a module/class ALPHA dynamically, never read phase/time or teacher geometry in the actor, and do not multiply motor actions or target increments by a scripted speed scalar.
2. Use ONLY the2 qualified actual speed datasets, each complete2880 history, exclude first960 labels from loss and weight the2 post losses equally. Keep .075 in existing target-speed observation. Fresh Adam for ONLY kappa, fixed100updates at lr.025 with projected parameter constraint[0,4] after each optimizer update. Save50/100 for integrity only; physical evaluation ONLY final100. These are proposed fixed engineering values, to be registered before any update rather than selected after teacher/physics outcomes. No frozen hidden cache is valid because kappa changes base history. Verify actual full-sequence gradient, optimizer/RNG/curve, all-state replay and source identity first. An unchanged fitted kappa0 is an allowed negative result, not permission to pick a different hand-tuned checkpoint.
3. Before/after fitting, replay qualified neutral/fullleft/fullright actual histories with forward0 and verify all2880 means/actions/192 states bitexact relative to the selected parent, including continuous teacher withdrawal. The current half-command failures remain failures; adding a forward mechanism does not establish a turn range. Current negative target-position Jacobian must remain exactly the original diagonal at fixed history. At forward<0, altered parent hidden states are expected and must be audited rather than compared to the old trajectory.
4. Physically evaluate exactly the final model at neutral then straight speed090(u0=-.10,u1=0) and speed085(u0=-.15,u1=0), original1s settle+4.8s teacher prime+9.6s ANN with the1s smooth command ramp. Keep all5/all6/strictNF0 and unchanged4s cycle timeout. Require signed forward>=.05 AND <=.95 of the NEW same-model neutral actual window, rather than the teacher control or an old differently timed report. Retain all forces/native71mesh42finite-PD/192-hidden/source/previous-action/full-window/HANDOVER checks and independent checkpoint replay. No full nativeD1–D8 conclusion follows from endpoint engineering alone; add formal native confirmation separately if those cases pass.
5. Do not deploy simultaneous speed+turn. No such qualified teacher or neural trajectory was established here. A subsequent speed→turn or turn→speed transition starts from different192 memory and physical state; u0 returning0 restores the equation, not automatically the historical state. New registrations must test those combinations/transitions before food behavior uses them.

If the action-loss gradient for kappa is numerically negligible, the fixed100-update trial can report that non-effect. It must not be replaced silently with a scripted .85/.90 action multiplier or an external CPG. Only actual signed speed and complete physical gates can establish neural deceleration.

## Conditional rebase to a CEM positive-head bias parent

This analysis replayed ALL2880 observations for EACH of the3 actual straight teacher histories through eeced ReadoutMotor with[0,0] and through its frozen c6b parent: all means and all192 memories matched exactly, max error0. The positive head is `ReLU(turn)*.25*tanh(readout(h64))`; it cannot affect output at turn0 and cannot feed back to the old memory. Therefore if the eventual selected CEM differs ONLY finite readout.bias entries29/32/34, its pure-straight speed fit is algebraically unaffected. This does not authorize automatic checkpoint identity porting.

Before reusing a fitted speed parameter, authenticate both parent identities and EVERY tensor; assert the only differences are the declared positive-head bias scalars, verify all194878 motor tensors and other2730-head entries, and replay both parent versions on the ACTUAL speed-teacher histories with sanitized[0,0] to confirm old192-state/mean equality. Then create a NEW accurately parent-bound speed checkpoint/identity without editing or rehashing old files, and rerun its saved-state proof. For forward0 turns, compare the rate wrapper to the NEW selected parent's output/state, not to eeced's now-different positive-turn outputs. For mixed speed+turn, neither rebase equivalence nor physical qualification follows.

## Outputs and limits

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/deceleration_analysis/METRICS.json` holds exact numbers, source/data hashes, shared-prefix proof and existing-model inference equality. `analyze.py` reproduces them without a physics backend or optimizer. All outputs are inside this new scoped directory. No dataset copy, checkpoint, actual training preregistration or physical run was created. The next-stage plan above remains a proposal for the parent to register; all current turning qualifications and failures remain unchanged.
