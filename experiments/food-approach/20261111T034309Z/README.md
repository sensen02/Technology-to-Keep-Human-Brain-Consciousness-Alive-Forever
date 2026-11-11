# Bilateral odor neural-response pretraining — SENSOR_PRETRAIN_ONLY

Fixed1000 training and final heldout evaluation are complete. **The overall preregistered sensory targets FAIL.** No motor model, MuJoCo, Core or physical integration was imported or executed. This is a synthetic static-head sensory-response experiment, not navigation, food arrival, ingestion or biological validation.

## Actual results

| Metric | Final1000 | Predeclared target | Result |
|---|---:|---:|---|
| Heldout correct turn sign | .9765068818 over4214 qualified samples | >=.90 | PASS |
| Heldout bounded-turn MSE | .0200442468094 | <=.02 | FAIL |
| No-food turn RMS | .08163703742 | <=.02 | FAIL |

No-food bias is+.0725109250. The network learns substantial directional dependence but emits too much turn without food. Swapped-input MSE=.6949920943 and original-direction signfraction=.02183199, compared with normal MSE=.02004425; zero-output baseline MSE=.1848258040. These controls concern neural sensory response only and do not demonstrate physical causal navigation. All36 per-episode heldout results, sixteen mirrored-pair records and dose/onset controls are saved in EVALUATION/RESULT.json; do not select only passing episodes.

Training data comprises44 frozen episodes and heldout36 episodes, each240 decisions over1.2s. Training angles/doses/onsets and heldout angles/doses/onsets/noise seeds were fixed by ROOTPREREG before generation. Actor inputs are only local receptor[N,240,2], with explicit GRU32 state; source bearings/positions/time/concentrations are environment, target or diagnostic metadata. The exact frozen SteeringBehavior class from191449 was used. Forward output is structurally zero. Head geometry is read from audited JSON with no body construction.

At time t, behavior receives stored pending z(t); current concentration sampled at t advances the receptor only for the next actual5ms sensor decision. Noise.002 is explicitly seeded; noisy pre-onset values are not falsely described as exactzero, although underlying concentration and clean receptor arezero. Initial z0 iszero. A nominal-onset exactzero test initially failed due float accumulation3.55e-15s causing concentration4.29e-30 at the nominal boundary; initial report retained, CLOCK_CHECK_CLARIFICATION.json records exactpre-onsetzero and numerical boundary tolerance without changing data or behavior targets.

## Training and verification

Fixed1000Adam updates, lr.001, CPUfloat64threads1, full240sequence BPTT, batch20uniformepisodes without replacement perupdate. Batch schedule and RNG are saved. Training MSE=.2087005112 to.0194320971, wall52.42s. Twenty preflight checks passed including causal seeded data replay, actual recurrent outputs/states/allgradients versus shared-parameter scripted scan (gradient error8.88e-16), actual Adam update, save/reload and structuralzero speed. Saved500/1000 each independently pass14 actual model/Adam/counter/RNG/schedule/inference checks; live full states were compared before subprocess verification, live RNG restored after model constructor checks. Only1000 was evaluated on heldout; no extra updates or checkpoint shopping occurred after failure.

Jobs: bash17 initialpreflight exit1 (clockinstrumentation only, no persistentupdates), correctedforegroundpreflight exit0, bash18 fixedtraining exit0 and collected; generation, finalheldout, plot commands exit0. No outstanding jobs or temporary unit directories.

## Outputs and loader

Final checkpoint `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T034309Z/TRAIN/checkpoint_001000.pt`

SHA256 `76a4cbb48c7857aaa3a09e568c1a7b50877e0fa266028562a33d451e3c48aeb6`

Public loader `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T034309Z/learning.py`: `load_behavior(path,sha)->SteeringBehavior,payload`.

Schema is compatible with the old draft behavior loader: format_version1, kindfood_steering_behavior, originalidentity, state_dict, config.source_hashes and config.data_hashes. It additionally requires scope_tagSENSOR_PRETRAIN_ONLY and deployment_qualifiedfalse. **Future deployment must enforce motor and behavioral qualification; old draft loader alone does not enforce these extra tags.** Frozen identity.training text is an original architecture declaration, not the artifact's actual training status; scope_tag/steps/config are authoritative. No old class source was modified.

The actual opened plot and hash-verified desktop copy are recorded in DELIVERY.json. Title states SENSOR RESPONSE ONLY / NO PHYSICAL APPROACH and explicitly shows both failed targets.
