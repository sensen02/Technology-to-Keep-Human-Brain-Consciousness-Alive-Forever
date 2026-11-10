# Fixed four-case mechanism and causal MN-history comparison

Primary **newhistory** was fixed in preregistration before fitting; it was not selected on temporal results. All four cases fail to improve on the temporal zero-output baseline **3.661845726 rad/s**. The same failed slow4 source remains a failed reference, and the temporal split is part of the same episode, not independent-animal generalization. This work serves S3 local learning/S4 interface discipline.

| Case | New fits | Eligible coefficients | Training RMS rad/s | Temporal RMS rad/s |
|---|---:|---:|---:|---:|
| oldinstant | 0 | 2261 | 3.198316115 | 3.684887347 |
| oldhistory | 42 | 9044 | 2.952735332 | 3.826357337 |
| newinstant | 42 | 2261 | 3.159207147 | 3.734605986 |
| newhistory | 42 | 9044 | 2.767699683 | 4.036152110 |

Oldinstant weights are exactly cached from185910Z and were not refitted. The other three cases each use42 fixed boundedridge fits: **126 new solves**, lambda1, bounds[-1,1], tol1e-9, maxiter100, no bias. No neural steps were used to generate learning features here; both old and new traces were reused. No physical execution was performed by this agent.

The new dynamics trace was frozen by its owning agent at SHA256 `8a73c8a221b0f752e7c53f9e129fcf9747c42dc3ead6209a7ce912f430e70847`; module SHA256 `c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250`. Mechanism: adaptation conductance with EA−80mV, spike increment .002/15uS and tau100ms, replacing the old subtractive adaptation current. Neuron identities, original source edge masks, old556 encoder and zeroDN drive remain unchanged. The old voltage sanity failures are preserved historical evidence; the new mechanism is an engineering hypothesis, not physiology calibration.

## Causal runtime and identities

Implementation: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/runtime.py`. `Readout(model_path).observe(mn_rates323)` appends the current START-bin MNfilteredHz; `.action()` returns `30*tanh(feature@W)` in actual42 joint order. Feature order is lagmajor,323 canonical MN within each block, /100Hz. History lags are[0,1,4,10] bins, corresponding to0,5,20,50ms. Pre-start inputs are exactzero and there is no future wrapping. Every output joint uses only same-leg MN features. Model roots/local indices/leg labels and all42 joint names are checked against frozen source files at initialization; malformed identities, crossleg weights or nonzero bias are rejected.

The runtime keeps11 samples for history models and1 forinstant. `.state_dict()` returns history_rates_hz, observations_seen, lags_bins and bin_ms as numeric arrays; `.load_state_dict()` validates and restores them. Fresh or fully silent history produces exactzero action. Immediatezero current input with nonzero past history can retain a remembered action. A last nonzero sample at t0 is still eligible at exactaget50ms and absent by t55ms; no feature refers to data older than50ms. Parent was informed of this endpoint convention before results.

Primary model: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/cases/newhistory/MODEL.npz`; SHA256 `0474b070a8576229a2878f890225f4b081684ca85cef9e6dca9beb8c240118c6`. Desktop equivalent: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/learning/cases/newhistory/MODEL.npz`. Models include MNroot/local/leg/joint identity, lags, scales, source hashes and fixedprimary metadata.

## Independent verification and runner

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/verify_cases.py` does not import the fitter, neural module, optimization library or physical runner. **1203 checks passed**, rebuilding lag features by explicit per-row past indexing; comparing full1817-sample runtime features/actions; checking all model identities, same-leg masks, bounds, zero bias, exactsource labels, causal prefix invariance, stationary inputs, impulse/drain behavior, invalid inputs and source hashes. Ridge objectives/KKT were independently recomputed without optimization. Largest projected residual across all cases is 1.14787554462e-11, below preregistered1e-6. There was no performance-based refit or primary substitution. Temporary invalid-model fixtures were removed immediately after checks.

Physical runner: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/run_closed_loop.py`. It was written and syntax-checked, then handed to parent; this agent never executes it. Default is fixednewhistory learnedcontrol, optional `--control hold` remains available but is not launched automatically. It preserves original sourceINITIAL/fullbody/nativephysics/PD128/caps and1s warmup+2s posthandover schedule. START MN rates enter readout history before action, history survives handover; physical sensory currents are held and advanced neurally after the same physical5ms interval. HANDOVER stores `neural_g_adapt_uS` and `predictor_*` state; NEURAL_FINAL stores explicit `g_adapt_uS`; READOUT_FINAL preserves predictor history. No adaptationconductance is mislabeled as nA. Old BODY/NATIVE/COMMANDS/POLICY schemas are preserved with extra readout feature/history metadata.

Parent produced the learned pilot; this task independently replayed its saved neural/input/history trace: **PASS_NEW_NEURAL_HISTORY_REPLAY_NOT_WALKING**, 184 checks, 6000 neural steps and zero physical steps. Exact local inputs/encoder currents, start MN rates/volts, causal features, handover and full neural/predictor final states match; matrix predictions allow1e-10 accumulation tolerance. Voltage minimum -76.5965768397mV, -90mV sanity pass=True; biological calibration remains NOT_ESTABLISHED. Old185910Z hold INITIAL/body/dt/warmup-command contract matches exactly and its postwarm commands arezero; it is a historical matched body-only control, not a new-adaptation neuralhold experiment. Parent retains native physical replay and physical interpretation.

Runner reproduction (parent scope only):

```bash
PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/run_closed_loop.py --control learned
```

Independent fit/runtime verification:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/learning/verify_cases.py --phase all
```

Completed outputs are protected against accidental overwrite by fit/verifier entry points. Numerical plot `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/learning/CASE_COMPARISON.png` compares the four fixed fits and the primary LF knee trace; the actual generated image was opened and inspected, with scope recorded in VISUAL_REVIEW.json. All sources/models remain engineering artifacts; no stable walking, causal support dependence or complete brain claim is made. Only this learning subtree and its Desktop mirror were modified; tracker remains parent-owned.
