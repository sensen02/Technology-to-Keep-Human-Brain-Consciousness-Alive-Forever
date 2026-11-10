# Independent saved readout and neural-input verification

Result: **PASS_ENGINEERING_REPLAY_NOT_CALIBRATION**. Independent readout verification passed 278 checks and replayed all 1817 samples / 18170 neural steps. No new learning, optimization, physical steps, MuJoCo calls or tracked-file edits occurred.

Source/data provenance, exact q42/qvel42/footload rows, five-row applied-rate label means, qualified encoder currents, start-of-interval MN rates and MN voltages were checked. Recorded sensory currents were then held for each 5ms neural interval with zero descending current, total conductance .02uS and dt .5ms. The verifier imports frozen LocalCircuit and SensoryEncoder but never imports the trainer or physical runner. This is deterministic implementation replay, not a second independently implemented neural integrator; separate dynamics verification remains the source for the latter.

The same-leg-only 323x42 weight mask, [-1,1] coefficient bounds, exact zero bias, silent-MN zero action, actual42 names, root888 IDs, local indices and leg labels pass. Predictions, aggregate and per-joint RMS were independently recalculated. The constrained ridge objective uses lambda1; no fit was repeated. The maximum projected KKT gradient residual is 1.14787554462e-11, below the preregistered 1e-6 bound. Raw active-set gradient residuals and each joint's objective are retained in VERIFY_READOUT.json.

| Metric | Recomputed rad/s |
|---|---:|
| train_RMS_rad_s | 3.1983161147166013 |
| temporal_RMS_rad_s | 3.6848873470228107 |
| zero_baseline_temporal_RMS_rad_s | 3.6618457263912414 |

Temporal prediction improves on zero: **False**. The source is the failed slow4 reference and both splits belong to the same episode. These are engineering fit diagnostics, not independent-animal validation or successful movement labels.

The readout replay minimum interval-end voltage is **-108.574384866 mV**, which fails the original -90mV sanity threshold. Frozen configuration remains **CALIBRATION_FAILED**, using the preregistered .02 fixed fallback. Exact numerical replay does not establish calibrated physiology, stable walking or complete brain function. The plot shows recorded MN start voltages only; the reported full-network interval-end minimum is a separate measurement and must not be inferred from that plot alone.

Closed-loop replay: `PASS_ENGINEERING_REPLAY_NOT_CALIBRATION`, 600 policy samples, 6000 additional neural steps, 128 checks. Full saved final voltage, excitatory/inhibitory conductance, adaptation, refractory state, delay ring/index, spike counts, filtered rates and step count compare exactly. Saved handover neural fields compare exactly. Minimum interval-end voltage -109.834052072 mV; -90mV sanity passes: False. The parent owns independent native physical replay; this verifier performed no physical steps.

Output-hold control: `PASS_ENGINEERING_REPLAY_NOT_CALIBRATION`, 600 samples / 6000 neural steps / 128 checks. Qualified inputs, start MN states, predicted readout, saved handover and full final neural state compare exactly. Applied commands after1s are exactly zero while neural processing continues. Voltage minimum -108.146734437 mV also fails -90mV sanity. Total verification replay steps including readout and both pilots: 30170; no new physical steps or learning.

## Outputs and reproduction

Primary verifier: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/verify_saved.py`.

Preregistration: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/PREREG.json`. Readout result: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/VERIFY_READOUT.json`. Replay-derived readout final state: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/REPLAY_READOUT_FINAL.npz`. The trainer did not archive a full readout final state, so that snapshot is not claimed equal to an unavailable original.

Desktop report: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/verification/REPORT.md`. Plot: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/verification/READOUT_VERIFICATION.png`. Byte-identical delivery hashes: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/DELIVERY_VERIFICATION.json`.

Reproduction command (writes protected fresh result paths; existing result prevents accidental overwrite):

```bash
PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/verify_saved.py readout
```

For the two pilot replays use `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/verify_pilots.py` with argument `closed_loop` or `output_hold` under the same interpreter. Scope extension was saved before replay in `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/verification/PILOT_EXTENSION_PREREG.json`. Original MODEL/TRAIN_TRACE/POLICY/source artifacts are read-only to this task. System python3 lacked scipy during an initial dependency-only probe; the existing venv_body interpreter was used without installing packages. All three verification jobs exited0. The plotted readout figure was actually opened and inspected; it displays reference/prediction mismatch, sustained MN activity and the MN-only voltage trace with a separately labeled sanity threshold.
