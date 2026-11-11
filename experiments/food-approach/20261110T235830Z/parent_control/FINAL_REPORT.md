# Matched parent500 one-second-ramp controls

Exactly two preregistered physical controls were executed for the previously named parent checkpoint:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/TRAIN_RESIDUAL_500/checkpoint_000500.pt`

SHA-256: `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`.

The new root amendment and `parent_control/PREREG.json` were written before these control outcomes. Commands match the new1500 evaluation: forward0, left/right signed turn multiplied by a fixed one-second smoothstep after neural withdrawal. No optimization, other weights, intermediate1000 selection, neutral repeat or extra configurations occurred.

| Parent500 matched control | Signed forward | Maximum slip, limit0.5 | Whole-window yaw rad | Six-leg cycles | Result |
|---|---:|---:|---:|---|---|
| Left | 0.0709642956915 | 0.552491916079 | +0.0624247262577 | LF1 LH2 LM2 RF2 RH2 RM2 | FAIL: slip |
| Right | 0.0722796656776 | 0.514034062284 | −0.0782577615754 | LF1 LH2 LM2 RF2 RH2 RM2 | FAIL: slip |

Both conditions passed original support, nonfoot, COM and forward gates, strict nonfoot zero, six-leg completion and proper yaw sign/magnitude. Left support mean1.00004878556, minimumCOM1.25553497993; right1.00005289007 and1.25557912431. Both recorded nonfoot force0. The original full9.6-second window includes the one-second ramp; no threshold, denominator, timeout or failed segment was changed.

Each completed saved run passed92 independent checks: full15400 physical endpoints, all2880 actual checkpoint means/actions, explicit192-dimensional base128/residual64 states and chain, actual source counts/actions/previous-action observations, immediate handover hashes, original5800 physical prefix/960 neural prime, zero extra forces, unchanged compiled71-mesh opaque body, geometry and whole-window summary/cycle/yaw recomputation. The original parent neutral pass and its independent audit are explicitly hash-bound with separately preserved source identities. No new neutral physical trajectory was produced.

## Interrupted aggregation recovery

Managed job `bash-239` was started for the two controls. After session interruption its handle became unknown; no terminal pipeline exit code is invented. The original `SUMMARY.json` records the completed left evaluation and audit as exit0 but remained `RUNNING` before aggregating right. Both complete right result and original92-check audit were already saved, and the parent confirmed no live control process.

The original summary is unchanged and copied byte-for-byte to `SUMMARY_INTERRUPTED_ORIGINAL.json`. `recover_summary.py` reconstructs aggregation from the existing complete results, replays both saved-run audits without new physical integration, and writes `RECOVERED_SUMMARY.json` plus fresh `RECOVERED_VERIFY_left.json` and `RECOVERED_VERIFY_right.json`. Its recovery exit code is recorded separately; original right/pipeline terminal codes remain unavailable. This is aggregation recovery, not a repeated physical attempt or a changed result.

## Reuse and limits

`evaluate_parent.py` explicitly binds the frozen new evaluation body to the strict old500 `load_candidate` API, verifying original training source/data hashes, actual step500 and exact checkpoint digest. The physical loop, ramp formula and scoring remain the prepared new1500 implementation. `verify_actual.py` binds the same parent loader and output/source paths. New wrappers, amendment, preregistration and all old executed source files enter the actual control identity. No training/evaluation/ or previous evidence files were edited.

Both matched parent controls remain ineligible because slip exceeds0.5. These results permit an honest matched comparison against the predeclared final1500 candidate; they do not establish a qualified parent motor or food deployment. Old abrupt-command failures remain unchanged. Any later command-switch/initial-state validation requires its own fixed registration.

CLI/API retained for review only; existing outputs cannot be overwritten:

- Model: `evaluate_parent.checked_load(io, absolute_checkpoint, exact_sha)` returns the original parent500 model/payload. `step(obs[B,238], command[B,2], (base[B,128], residual[B,64])) -> (mean[B,42], state)`; apply tanh once.
- Commands: `evaluate_parent.command_at(condition, decision)` matches all2880 new1500 command rows; both states persist through withdrawal.
- Saved audit: `verify_actual.py --run ABS_EVAL --out ABS_FRESH_AUDIT` performs no Core construction or physical integration.

These are replacement-ANN/declared-derived-body finite-PD fixed-condition engineering controls, not full nativeD1–D8, connectome, biological or food certification.
