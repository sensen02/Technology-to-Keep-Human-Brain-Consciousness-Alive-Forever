# Preregistered direct physical feedback learning

Completed the fixed **53 training episodes plus two5s validation episodes** from the same verified actualtime1s checkpoint. These are separate reset episodes, not a continuous walking trajectory. Selected coefficients remain exactly zero: this search did not improve on the actuator hold baseline.

The actor uses only323 source-identified MNfiltered firing rates through a frozen same-leg SVD basis and168 bounded coefficients; no bias, no teacher, body/root/time/phase input. Raw body state is used only for privileged training reward and gates. The original complete physical body, native12.5us step, liveforward,1ms PD128/caps,5ms coupling, fixed20k/435126-edge conductance circuit,556encoder andzeroDN remain unchanged. Inner synapses/neural parameters were not optimized.

Baseline training reward: **-0.117040201476**. Selected training reward: **-0.117040201476**, selected archive `trial_000`. Feasible training episodes: **17/53**, including the originalbaseline and four explicit incumbent reset controls.

| Generation | Feasible candidates /13 | Incumbent archive | Incumbent reward |
|---|---:|---|---:|
| 0 | 2 | trial_000 | -0.117040201 |
| 1 | 5 | trial_000 | -0.117040201 |
| 2 | 4 | trial_000 | -0.117040201 |
| 3 | 5 | trial_000 | -0.117040201 |

All48 nonzero candidate policies exercised nonzero actor outputs (rateRMS0.359319–0.613591rad/s). Twelve passed the originalsupportgate, but the bestnonzero reward−1.014442992 remained worse thanzero−0.117040201. Registered stance-air-stance coverage among nonzero trials was0legs in15,1leg in17,2legs in12,3legs in2 and4legs in2; none reached5or6. This is a failed limitedsearch, not a proof that no usefulpolicy exists. Sustained mean rates may cause targetdrift; parent separately analyzes savedDCpower rather than treating that possibility as proven. See FAILURE_INTERPRETATION.json.

The optimizer used exactlyseed20261107, four generations of six antithetic Gaussian pairs plusincumbent, initialstd.01, coeffbounds[-.25,.25], feasible-first up-to4elites, mean=.5old+.5elitemean, std=.7old+.3populationelitestd clipped[.004,.05]. Incumbent changes only for feasible strict reward improvement. All candidate arrays, randomdraws, order, before/aftermeans/std and selection evidence are retained. Incumbent reevaluations are explicitly cross-worker/fullreset reproducibility controls; all four passed, maximum error 0.

## Validation and limitations

| Result | Selected5s | Zero5s |
|---|---:|---:|
| Fixed shaping reward | -0.125930445867 | -0.125930445867 |
| Original support gate passed | True | True |
| Forward COM displacement | 0.0202914824159 | 0.0202914824159 |
| Loadweighted material slip distance | 0.0608676172041 | 0.0608676172041 |
| Legs completing cyclecoverage proxy | 0 | 0 |

Reward is fixed forward displacement minus slip/lateral/height/nonfoot/supportdeficit/rate penalties plus a limitedstance-air-stance coverage bonus. The bonus is a scoring proxy, not full coordination acceptance. Support gates passing cannot establish neural necessity because zero-rate PDtarget hold may support this body. Absolute SIforce and biological calibration are NOT_ESTABLISHED; no stable walking or complete brain claim is made.

Training slip is approximate5ms endpoint contactmaterial Jacobian sampling over1s. Finalvalidation uses0.5ms samples over5s. Their raw reward magnitudes are not directly comparable across horizon/sampling and do not imply deterioration duringvalidation. Parent saved-array analysis reports median nonzero rateDCpowerfraction0.237922; do not characterize all policies as constantoutput or lacking dynamics. Native20-columncontact/COM data are retained at every12.5us step for **all55episodes**, alongside 1msBODY/COMMANDS,5msPOLICY,ACTOR,INITIAL,completeNEURAL_INITIAL/FINAL and per-episode result/provenance. No failed physical policy trace was silently removed. No softwareerror occurred during the completed study; pretrial fixture corrections are recorded separately.

Each validation starts at actualtime1s using INITIAL.state; COMMANDS timestamps use actual d.time. Parent owns final independent nativephysics replay and video. Release-only data do not contain a newly executed warmup. Parent plans one selected-zero video, with zeroVIDEO_REFERENCE pointing to the identicaltrajectory, avoiding a redundant secondrender. If the parent joins original0..1s warmupBODY to releaseBODY, it must verify the seam and label shared warm-history initialization provenance. This learning agent opened and inspected the actual STUDY_COMPARISON.png; VISUAL_REVIEW records its scope, not a physicalvideo review.

## Verification and actual calls

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/verify_study.py` independently reconstructs source hashes, canonical physics/neural starts, actorweights/predictions, all53candidate draw/order/thetas, every scoreterm/originalgate, fullreset incumbentcontrols and all generation updates without importing optimizer/runner/scoring or executing physics/neural steps. **11682 checks passed**,55episodes checked. It also confirms frozen selectedtheta was used for5svalidation and the second actor was exactlyzero.

Original study actualcalls: **5040000 physicalsteps**, **126000 neuralsteps**, **5040055 live/initforward calls**. These are study calls, not independentreplay calls. The saved-data verifier adds0physical/0neural steps and0optimization updates. Actual counters and each trial's elapsedtime are retained inCALL_LEDGER andRESULT. Fourmean/std distribution updates are the learning operations; basis is frozen. This agent does not alter trackedfiles, originalbody/core/encoder/checkpoint orfoundation.

Key files:

- Source preregistration: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/PREREG.json`.
- Parent review authorization: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/START_AUTHORIZATION.json`.
- Optimizer and physical runner: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/optimize.py` and `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/episode.py`.
- Full study evidence: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/STUDY_RESULT.json` and `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/CALL_LEDGER.json`.
- Selected coefficients: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/SELECTED.npz`.
- Finalselected native/physics/neural artifacts: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/evaluation/selected`.
- Finalzero native/physics/neural artifacts: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/evaluation/zero`.
- Independent verification: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/INDEPENDENT_STUDY_VERIFICATION.json`.
- Comparison plot: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/STUDY_COMPARISON.png`.

All episodeartifacts also mirror into `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning` with bytehash verification; sourcecode and reports mirror into `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning`. Delivery verification lists both sides. No extra budget extension, posthoc primary selection, modified gate or reward tuning was performed.
