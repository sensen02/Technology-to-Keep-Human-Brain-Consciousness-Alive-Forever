# Balanced bilateral-odor neural response — SENSOR_PRETRAIN_ONLY

**The fixed final3000 network passes all four predeclared fresh sensory-response engineering targets. It is not qualified for deployment or physical food approach.** No MuJoCo, Core, body model, physical state writing or physical integration was imported or executed.

## Fresh test result

| Metric | Final3000 fresh result | Predeclared requirement |
|---|---:|---:|
| Aggregate fresh-test turn MSE | .00560797931475 | <=.02 |
| Nonzero-bearing food-only MSE | .00762324997163 | <=.02 |
| Correct direction fraction | .9990403071 over4168 qualified samples | >=.90 |
| No-food turn RMS | .0151462586561 | <=.02 |

All four pass. Food-only MSE was added as a **stricter** requirement before training, so centered/no-food zero-target examples cannot dilute a failing food subset. Full per-episode results and all sixteen mirrored angle pairs are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/EVALUATION/RESULT.json`.

Residuals remain: no-food bias+.0110034254, maximum absolute no-food turn.0383992192 (the registered gate is RMS, not peak), center RMS.0153682064 and bias+.0026735912. No-food per-episode RMS is.0191713386/.0095539563 for the two fresh noise seeds; the two onset metadata values yield repeated traces when amplitude0, so those four controls are not four independent noise realizations. Mirror opposite-sign fraction is at least.97087 across qualified pair samples. The4168 sign samples are correlated timepoints within episodes, not4168 independent experimental replications; no binomial-confidence claim is made. These results do not prove perfect symmetry, zero response at every instant, dynamic on/off adaptation or biological calibration.

Zero-output baseline MSE=.1023085848, food-only=.1406743042. Swapping the two local inputs gives MSE=.4069788024, food-only=.5595245782 and original-direction correct-sign fraction.0002399232. These are sensory-direction controls, **not actual physical causal navigation ablations**.

## Fixed experiment and data provenance

The prior1000 experiment at `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T034309Z/` remains sealed and FAIL: its original heldout MSE.0200442468 and no-food RMS.0816370374 exceeded their targets. Its result informed this separately preregistered correction, so that old heldout is now explicitly development-validation history, not the fresh test.

This experiment preserves the same frozen source class `SteeringBehavior`: two local receptor readings -> encoder32 tanh -> slowGRU32 -> neural turn1, structural forward0. Original source `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/steering_behavior.py` was not modified. The static `identity.training` string is an immutable original architecture declaration, not a claim that this artifact is untrained; `scope_tag`, recorded training steps and deployment flag describe the artifact.

Original44 training episodes and all labels are unchanged. Each new minibatch has exactly10 existing food-amplitude episodes and10 existing no-food episodes, sampled with replacement and shuffled by recorded RNG. Actor sees only receptor240x2 and its explicit recurrent state32; it never sees the balancing category, bearing, source coordinates, distance, onset clock or teacher label. Hidden state resets only at episode start, never at odor onset. Current action uses pending z(t); receptor sampling at t computes only the next interval endpoint. Same Gaussian source, static audited head proxies, saturation,50ms lag,5ms sampling,.2s onset and.002 seeded output noise are retained.

The44 fresh test episodes were frozen **before updates**, using previously unseen angles±.10/±.05, strengths.6/1.25, onsets.30/.60 and noise seeds20261124/25, plus four no-food and eight center controls. Only final3000 was evaluated, once. No hidden-state resets at onset, parameter changes, early-checkpoint selection, threshold changes or additional training followed the fresh results. Physical movement and odor offsets/sources disappearing were not tested.

## Exact continuation and verification

Parent1000 checkpoint SHA `76a4cbb48c7857aaa3a09e568c1a7b50877e0fa266028562a33d451e3c48aeb6` was restored with actual model, all8Adam states/counter1000, RNG and full0..1000 curve. Learning rate changed.001->.0003, with the balanced sampling change explicitly registered. No architecture, input, response equation or label changes occurred. There is no equal-budget old-unbalanced counterfactual, so improvement cannot be causally attributed to balancing alone independently of lowerlr/additional updates.

`TRAIN/RESUME_STATE.pt` and `RESUME_CHECKS.json` record exact restore before any update. Preflight14 checks pass, production-vs-script full240 BPTT gradient error2.7756e-16; one discarded optimizer unit step is separate from persistent training. Fixed2000 additional updates1000->3000 complete as managedjobbash19, exit0 collected,101.315s total. Saved2000/3000 each pass18 independent actual state checks: weights,8Adam tensors/counters/moments/shapes/lr, RNG, parent curve prefix, every new batch10+10, actual saved inference and structural forward0. Live full weights/optimizer/RNG/curve comparisons passed; live RNG restored after constructor checks. No extra updates were made.

An independent array-only final evaluation audit passes11checks, recomputing all registered metrics/labels/masks, full44x240 inputs and state32, all source/data hashes and the entire original sealed29-file failure experiment. Its first report serialization failed on a NumPy bool before writing, then native bool serialization was corrected and the same checks rerun; no model/data/metric changed.

## Public artifact and future boundary

Checkpoint `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/TRAIN/checkpoint_003000.pt`

SHA256 `bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06`

Public API `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/common.py`: `load_behavior(path,sha,require_final=True)->SteeringBehavior,payload` (requires final3000 by default).

Payload retains compatible draft fields format_version1, kindfood_steering_behavior, original identity, state_dict and config.source_hashes/config.data_hashes; added scope_tagSENSOR_PRETRAIN_ONLY and deployment_qualifiedfalse are mandatory in this loader. The old draft physical loader alone does not enforce these additional tags. **Future integration needs an explicit adapter, qualified motor and new physical/cause-control registration; this checkpoint must not automatically enable a failed motor.**

The actual opened chart `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/BALANCED_SENSOR_DIAGNOSTIC.png` and desktop copy show six complete panels, targets and limitations. `/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮4-平衡气味神经反应-纯感觉新测试.png` has the same SHA and title SENSOR RESPONSE ONLY / NO PHYSICAL APPROACH. Sensor-only target PASS does not establish food arrival, stopping, ingestion, satiety, connectome or consciousness.

All managed jobs and foreground commands have been collected. No active training, physical job or temporary unit directory remains. The immutable artifact manifest excludes parent-owned mutable `CURRENT_STATE.json` explicitly.
