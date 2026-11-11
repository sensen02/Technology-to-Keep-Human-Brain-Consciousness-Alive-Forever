# Independent read-only fast-residual learning diagnosis

Reviewed only the saved code/data/checkpoints. All new files are confined to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/diagnostics/`. No optimizer step, physical integration, captured-source modification, earlier-checkpoint physical selection, or alternative implementation occurred.

## Recommendation

If the current physical1000 candidate is unqualified, the next controlled experiment should continue **exact checkpoint1000 to fixed global3000 at Adam lr0.0001**, keeping the same four actual teacher episodes, architecture, command gate, residual bound, objective, base freeze, optimizer moments/counters, RNG and complete 0–1000 curve. Register final3000 evaluation before training; do not physically select checkpoint900 because its teacher fit is better. Save/independently validate intermediate checkpoints for integrity only. Lower lr is a measured-stability hypothesis, not a guarantee of physical success.

Do not simultaneously reset Adam, increase the residual bound or add network capacity: that would confound the first informative comparison. Expect some initial retained-momentum transient. The speed gate has a genuine algebraic floor, but current error is sufficiently above it that widening it first is not justified by the present fit deficit. If a stable lower-lr continuation later reaches the floor while physical speed control remains insufficient, a separately registered command-gate/bound intervention becomes reasonable.

## Evidence for optimization oscillation

The six adapter parameter tensors all have one uniform lr0.001 and correct Adam step counters500/600/900/1000. Moments remain finite. The entire training curve has no clipping event: max raw norm0.06057264 < clip1.0. Every saved after-update metric at100…900 equals the next iteration's pre-update metric exactly. The600 spike begins several iterations before saving: gradient norm grows from0.00309 at592 to0.05858 at600. Therefore the evidence does not support a serialization, checkpoint reload, RNG discontinuity or gradient-clipping cause.

Saved1000 loss gradients across the four episodes are strongly aligned (pairwise cosine0.862–0.942), rather than mutually opposing at that snapshot. Yet stored Adam first moment points almost opposite the current full-batch gradient (cosine−0.996355); its stored preconditioned direction has cosine−0.606618. These measurements support an overshoot/momentum oscillation interpretation. They do not prove that lr is the only cause. `GRADIENT_REVIEW.json` also records exact next-moment algebra at1001 without mutating model/optimizer; the linear loss estimate is not an actual evaluated update.

## Saved1000 fit versus algebraic floor

| Teacher | Actual post-update action MSE | Reachability floor | Floor/observed |
|---|---:|---:|---:|
| left_full | 0.000741510 | 0.00000146259 | 0.00197245 |
| right_full | 0.000765372 | 0.00000128475 | 0.00167860 |
| straight_085 | 0.001757067 | 0.00016004967 | 0.0910891 |
| straight_090 | 0.001460654 | 0.00016829285 | 0.1152175 |

The floor is computed per actual input from `[tanh(base_mean−gate), tanh(base_mean+gate)]`, so it includes command ramp and the tighter0.90 speed gate. Turn floors concentrate in the initial1s ramp and become nearly negligible later; speed floors persist in later teacher phases, especially large front-leg action transitions. This is a real gate/capacity restriction for exact teacher imitation. It does not establish that those particular teacher commands are necessary for successful physical deceleration.

The saved900 teacher-distribution fit is better on every case than saved1000: turn MSE0.00027305/0.00025440 and speed0.00108747/0.00133545. Those are **offline diagnostics**, not held-out performance or permission to shop earlier checkpoints physically.

## Error concentration and actual-load association

At saved1000, turn absolute action-error p95 is0.05832/0.05777, maximum0.19845/0.20985; speed p95 is0.08475/0.07189, maximum0.61759/0.51746. Leg groups use the frozen actuator/contact ordering `LF,LH,LM,RF,RH,RM` in seven-joint blocks.

For left_full, RF/RH/RM together account for0.66955 of squared error; for right_full they account for0.66473. RH error occurs during measured load≥0.1W for0.80878/0.82893 of that leg's error energy, RM for0.86152/0.87902. RF is primarily unloaded-error energy instead. Thus an undifferentiated 'all error is swing' or 'all error is stance' account is unsupported. Speed errors are especially strong in LF/RF/RH; worst local moments are around post5.02s for0.85 and4.245/9.58s for0.90. All timing is teacher provenance used for diagnosis only, never policy input or deployment action scheduling.

Loads are sampled at the actual immediately preceding1ms physical endpoint, with exact frozen native `WEIGHT=24.446354755878005`. Each compressed sample/trace is read once per analysis invocation; no repeated NPZ decompression inside decision loops. `DIAGNOSIS.json` contains all per-leg values, worst joint indices, event times and hashes.

## Base-hidden observability: what is and is not established

The instantaneous residual has112 permitted physical channels + actual causal base hidden128 + normalized command2. The base executes the original slowGRU update on the real previous-state sequence; no teacher phase/time/future is supplied. The baseline checkpoint remains SHA`246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809`; its151636 parameters are frozen. Direct target-position inhibition `−softplus(raw_gain)*obs[126:168]` is unchanged, and the residual's physical selection excludes those target channels. Existing `FAST_CHECKS.json` verifies the direct target Jacobian remains exact negative diagonal.

Offline nearest-neighbor tests exclude±0.25s and compare standardized physical+command, physical+basehidden+command, and hidden+command. None of the four teacher cases contains an exact duplicate full-feature row, or a nearest full-feature RMS distance below0.05. Neighbor target differences are small for many repeated turn states but have larger tails for speed transitions. This does not show an exact observational contradiction requiring an additional recurrent state, nor does it prove that the frozen hidden state is sufficient under a new physical trajectory. Adding128 hidden dimensions changes the distance metric; comparisons between those metrics cannot by themselves measure information gain. No decoder was trained and no deployment phase input was added.

## Limits and reproducibility

This analysis is entirely on teacher-distribution observations. It cannot prove the cause of closed-loop slip, establish a safe command range, or certify body/biology. Parent owns actual pipeline211 and all further learning authorization. Keep original all5 gates/all6 cycles and independent saved-inference/force audit.

Outputs: `DIAGNOSIS.json`, `GRADIENT_REVIEW.json`, the two analysis scripts, and this report. Captured source/data hashes are checked before load and after analysis. An initial diagnostics-only script variable shadowed the imported `load` function and exited1 before data analysis; corrected locally and rerun successfully as jobs213/214, with zero physical or optimization steps. The final gradient algebra rerun is job215, collected separately.
