# Independent read-only corrective learning review

Reviewed only the requested learning artifacts under `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/correction_fit5000/`, their bound parent/final saved learning states and the two bound recovery-data audit/label-replay JSON reports. No optimization, physical model construction, integration, actual-trajectory re-audit or NPZ decompression was performed. This report is the only newly written artifact. Current actual pipeline234 belongs to the parent; this review does not infer its unfinished results.

## Finding

**No substantive admission, loss-weighting, causal-cache or exact-resume implementation bug was found in the reviewed learning execution.** The new objective is explicitly failure-state corrective supervision, not a reversal of the earlier safe-demonstration rejection. The recorded mixed trajectories remain failed, and the original physical acceptance gates remain separate.

This conclusion is about the learning implementation and provenance. It does not establish that teacher labels are optimal, corrective states are safe demonstrations, additional data caused a physical improvement, or final5000 is qualified. No further experiment or architecture change is recommended by this review while the five actual evaluations remain pending.

## Failed-state eligibility and immutable semantics

`data5000.prepare()` first loads the original four authenticated passing teacher trajectories through the unchanged original loader. Each new left/right trajectory must independently satisfy all of the following:

- Actual saved-trajectory audit `integrity_all_pass` is exactly true and every recorded check true; BOTH collector and audit `dataset_safety_admitted` flags are exactly false.
- Collector is complete without early stop, reports848000 native steps (10600 physical endpoints), and collector/audit bind the exact fixed3000 checkpoint SHA and correct absolute run directory.
- The postprime whole mixed collection contains4800 endpoints, strict nonfoot zero, support≥0.90, COM minimum≥1.20, slip>0.5 and original slip gate false. The mixed failures are required, not silently relabeled passing.
- Exact RESULT, SAMPLES and TRACE digests match the independent audit; original captured runtime/source/asset identity is independently verified.
- Ordered private-IK replay reports1920 labels, all teacher labels bitexact, max label error0, root-balance residuals bitexact, maximum≤0.02 and zero rows above0.02. Its captured source hashes explicitly include the same samples and trace files, so the label report is bound to the actual observed states, not merely to a direction name.
- Actual samples contain1920×238 observations,1920×2 commands and1920×42 teacher labels. The960 priming sources are2;960 postprime rows consist of120 student-source3 and840 recovery-source5 rows. Priming commands are zero, clocks are complete, values finite and labels bounded. Exact source ordering/actual requested actions/force records/original5800-endpoint and960-decision prefixes are certified by the bound actual audit.

The hardcoded final recovery-data manifest SHA is `46e44c767cf9a3ad8d00a340072921307efe815940063dfcce9893776e95f1a2`. The loader verifies that manifest digest and all artifacts it actually admits, including label-source sample/trace hashes and original runtime identity. It need not decompress the complete trace again to trust the already bound independent audit. Both actual reports contain the expected nonempty checks, so theoretical acceptance of an empty checks dictionary is not an issue in this immutable execution.

Actual source data remain failed:

| Corrective trajectory | Original mixed slip | Safety admission | Ordered label error | Maximum root residual |
|---|---:|---|---:|---:|
| left |0.5495288700531669|false|0|0.00007425784462943111|
| right |0.5519080182902079|false|0|0.00010321706343522758|

Both label reports state zero new physical integration. These establish numerical reproduction of this particular teacher's ordered queries; they do not prove globally optimal correction or causal recovery. The explicit role `FAILED_OBSERVATIONS_VALID_TEACHER_LABELS_ONLY` and `passing_demonstration:false` are retained in preflight, checkpoints and saved verification. No code changes either source safety flag or the old admission function.

## Six equal trajectory averages

The training code computes each trajectory's own bounded-action MSE over rows `[960:]`, divides it by6, accumulates six gradients and only then performs one optimizer update. Thus:

`objective = (MSE_old_left + MSE_old_right + MSE_old_085 + MSE_old_090 + MSE_corrective_left + MSE_corrective_right)/6`.

The original four episodes contribute1920 postprime rows each; the corrective two contribute960 each. Each trajectory still has weight1/6, giving corrective data total weight1/3. A row-pooled objective would instead give corrective rows weight1920/9600=1/5; that is not the implemented objective. Every corrective postprime row uses its recorded `teacher_action` target, including both actually student-controlled and teacher-recovery rows. No source mask silently removes either category.

`PREFLIGHT.json` checks the numerical six-way mean and production/cached per-episode gradients. Its check named `not_row_pooled` alone only establishes the9600-row count, not the loss reduction formula. That is a narrow test-name/proof limitation, not a loss bug: `data5000.objective()`, `train5000.py:48–50` and the separate six-equal-trajectory check together explicitly establish the intended weighting.

## Causal base history and actor boundary

For each corrective episode, the frozen original base starts with zero state once, processes all1920 actual observation rows in order, including the entire960-row actual priming history, and caches its actual current base hidden state. The generated hidden sequence is compared bitexactly against the recorded collection hidden sequence, with initial zero and before/after chain checks. Excluding priming from loss therefore does not remove priming from the causal history.

Cached residual features contain only the permitted112 physical observation channels, causal current base hidden128 and normalized command2. Teacher labels, source code, recorded teacher phase, actual time and provenance are checked/read for supervision or diagnosis but never concatenated into actor input. No added recurrence, episode-phase input or hidden reset was introduced.

Preflight reports maximum cached/production mean difference `1.1934897514720433e-15`, hidden error0 and adapter-gradient difference `2.710505431213761e-19`, within the declared1e-12 tolerances. Mean/gradient equivalence is tolerance-based, not a claim that every cached arithmetic operation is bitexact. Recorded base hidden equality is exact.

The frozen original walking base remains checkpoint SHA `246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809`. The original target-position inhibition `−softplus(raw_gain)*obs[126:168]`, physical mask, command gate and bound1 remain unchanged. The residual does not receive target-position, phase or teacher inputs through a new path.

## Exact3000→5000 continuation

A fresh read-only Torch inspection of parent, final and `RESUME_STATE.pt` independently confirmed:

- Actual parent SHA `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8` and final SHA `ddc8d87fa346400ce23b837f3ecc32aa195293fb61f42a69dd3928061f9f5304` match the reports.
- Resume model, complete optimizer state, RNG and entire parent curve are exactly equal to parent3000. All six resume Adam counters are3000.
- Final architecture/gate/bound identity exactly equals the parent identity; all frozen base tensors are exactly equal.
- Optimizer parameter groups, ordering and all hyperparameters are unchanged, including uniform lr0.0001. Final six counters are5000 and moment tensors are finite.
- Final curve contains every integer step0–5000, with the entire0–3000 prefix exactly equal to the parent.
- All48 captured original/new source and data hashes currently match disk. No captured artifact was edited by this review.

The only intervention in the configured learning problem is the added two corrective trajectories and resulting six-way objective, together with the explicitly budgeted2000 continuation updates. No lr, optimizer-momentum reset, residual capacity, bound, gate, feature normalization or frozen-base change was observed.

The new objective is recorded separately in `initial_new_sixway_metrics`; old four metrics are checked against parent final metrics rather than treating the changed six-way scalar loss as the same objective. `RESUME_STATE.pt` is saved before any update, enabling actual equality checks rather than relying solely on later counters. Saved4000 and5000 records each report24 independent checks; their verifier checks full curve, parent prefix, old source/data retention, explicit failed-state roles, real model/Adam/RNG resume and reload, finite moments, neutral parity and checkpoint digest. Preflight and trainer enforce the actual numeric loss construction; the saved verifier is a complementary provenance/state check, not an independent re-execution of all training.

Only final5000 is designated for physical evaluation. Intermediate4000 is retained for integrity, not earlier-checkpoint physical selection. The model-level load helper itself is generic, so final-only physical selection ultimately depends on the separately captured physical evaluator/pipeline guard; this review does not substitute learning metadata for actual execution evidence.

## Interpretive limits

There is no budget-matched old-four-only3000→5000 control in this experiment. Therefore a later difference from3000 can reflect extra2000 updates and the changed six-way objective together; it is not a clean causal estimate of the value of the two new trajectories. Even an eventual physical pass would establish capability under that combined training intervention, not isolated data causality.

Post-update teacher-distribution MSE decreased on all six episodes, including corrective left/right from0.000618916/0.000748259 to0.000133454/0.000143179. These are training-set fit measurements, not held-out generalization or physical success. Mixed recovery examples cover the particular4.8s collection window and current source schedule; they do not establish coverage of every state encountered during9.6s autonomous evaluation. An exact, reproducible teacher label is not a proof that executing it would safely recover that state.

This review makes no claim about incomplete pipeline234 outcomes and requests no additional experiment before all five actual results and independent audits are available.

## Reviewed artifact digests

All paths below are relative to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/correction_fit5000/`.

| Artifact | SHA-256 |
|---|---|
|PREREG.json|4fff5ccfdf6762c3cdb14da5565f994c151b8d14f4928dd57ac4c4af39d96ab6|
|data5000.py|24de33de9658f426a220f3dee399cc4155085be6af166231daeb04f2e3583184|
|check5000.py|f51603f85030a49aeea25b7920071900bd265bd1fae6c928ef240d5be87e375d|
|PREFLIGHT.json|86af8def7d70d9c71d466458b4922734c6dae045d39888265441ba872dd3acc2|
|train5000.py|b56e394fe865db65c4411e39ebe1090d49e077afa4767ca97ee5ad934bd42e04|
|verify_saved5000.py|81e107c36dc4c6e808205125f1773f9bbcb7179c978d6537c58b5021a27feea9|
|TRAIN/RESULT.json|378e96b3800a8b835fd2934b26fb1b26c2788e4ada58c9c207f1a8b87630b663|
|TRAIN/RESUME_CHECKS.json|c866b41d9aeaed46904f4d00562feaedfa84b7a8bde8ca73d8fbe86c36a12fe7|

Communication qualification: attempted `send_message` to the supplied parent id `session-80ccae55-a957-4374-b48e-7ba52c0ab34c` was rejected by the runtime as belonging to another parent session. No delivery success is claimed; the review is returned through this report and final response.
