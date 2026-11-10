# BANC local conductance dynamics: executed engineering handoff

The requested API is implemented and executed. Source identity/topology, numerical math, deterministic replay and legal-input propagation are verified. Calibration acceptance is **FAILED** under the original conservative voltage bound; fixed deployment candidate is the preregistered **0.02 uS fallback**, not a falsely successful calibration. No physical steps, training updates, fresh brain training, external research, surgical or lesion experiments were run.

Primary API: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/module.py`.

Desktop delivery: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/dynamics/`.

## API and identity

Load `module.py` using `importlib.util.spec_from_file_location`; construction works independently of cwd and imports the original `/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py` directly. `LocalCircuit()` exposes `root_ids` (20,000 int64 IDs), `allowed_sensory_local` (5,103), `descending_local` (1,327), `ascending_local` (2,359), `mn_local` (323 in exact eligible exported CSV order), `mn_leg` (lowercase lf/lm/lh/rf/rm/rh), `net`, `filtered_rates_hz` (20,000), and `reset()`.

`advance(sensory_current_nA, descending_current_nA, duration_ms=5.)` takes two finite (20000,) vectors and returns a new (323,) motor filtered-Hz array. Each role independently rejects any nonzero current outside its identity mask. It accepts no body/global/target/reference/time arguments. Thirty guard rejection cases were verified without advancing a neural step. All 391 candidate motor neurons are outside both input roles. Sensory and descending roles are disjoint; 112 anatomical sensory/ascending identities overlap and remain correctly represented. Four ambiguous sensory records are excluded from generic sensory injection while their neurons and eligible source edges remain intact.

The parent physical encoder further qualifies exactly 556 identities; its currents are a subset of this API's mask. Module permission is not a claim that all 5,103 cells have measured modalities or outgoing strict edges. Bristle/noxious inputs have no strict outgoing edges, and parent leaves bristle/club/noxious currents zero.

## Declared model

Actual `ConductanceNetwork` and `ConductanceParams`, unchanged: 20,000 spiking conductance point neurons, 705,068 original induced anatomical edges, 435,126 eligible active edges (207,166 ACh excitation; 227,960 GABA/glutamate inhibition). Unknown/mixed/modulatory source labels exclude 269,942 edges from strict dynamics. No new synapses or hidden current/CPG/MLP/VNC replacement.

Receptor signs are **illustrative assumptions**, not receptor measurements; glutamate inhibition is not claimed universally biological. Node parameters are the original uncalibrated defaults: C .02 nF; leak .001 uS; rest/reset -65 mV; threshold -50 mV; reversals 0/-70 mV; exc/inh decay 3/8 ms; adaptation tau100 ms and increment .002 nA; refractory2 ms; delay2 ms. Existing/historical physiology remains uncalibrated.

Each edge weight is `0.02 uS * original_synapse_count / sum(strict incoming synapse counts at same post)`, with the denominator shared across receptor types. Thus summed incoming conductance increments equal .02 uS for each connected target; raw count ratios are preserved. Backward Euler at .5 ms, sensitivity .25 ms; no physics changes. Endpoint spike filter is `r <- exp(-dt/20)*r + 50*spike`, a causal exponential point-process rate in Hz. End-step spikes arrive at interval start exactly2 ms later (ring slots5 at .5 ms,9 at .25 ms).

## Executed outcomes and preserved failures

All tests are explicit constant or single-pulse engineering currents, **not brain outputs**. Broad initial pulse is .05 nA on allowed sensory plus descending ports during25–125 ms in a200 ms run. Three predeclared gains were tried only after prior failure:

| Incoming total uS | MN recruited | MN spikes | Full-window mean Hz | Highest cell mean Hz | Recorded voltage min mV | Original calibration |
|---|---:|---:|---:|---:|---:|---|
| .02 first probe | 152 | 1205 | 18.6533 | 155 | -95.2018 | FAIL: voltage below -90 |
| .005 | 3 | 5 | .0774 | 15 | -84.5278 | FAIL: fewer than16 MN |
| .08 | 258 | 3488 | 53.9938 | 270 | -95.1830 | FAIL: high cell rate and voltage |

All failed first/full probes, input arrays, traces and rules are retained. No post-hoc threshold change or further search. Broad probe extrema above were recorded every5 ms; all steps checked for finite state. Supplemental encoder probes record extrema every neural step.

At the fixed .02 fallback: zero input gives exactly zero spikes/rates; zero-connectivity software control with the identical legal input produces57,870 directly driven sensory/DN spikes but **zero MN spikes**. These are numerical controls, not surgical/lesion experiments. Sensory-only broad pulse recruits147MN; DN-only151MN. Broad .5/.25 ms recorded-input replay gives1205/1255MN spikes and152 recruited at both timesteps; relative aggregate rate change .04149. Same-dt replay is bitwise equal for complete saved final neural state and MN rates.

The earlier559-cell source-modality union pulse with zeroDN recruited131MN. The exact556-cell encoder pulse recruits137/139MN at .5/.25 ms (1023/1015spikes), including every leg. Constant midpoint-angle/zero-velocity/equal-load encoder fixture recruits135/137MN. These fixtures are not actual simulated standing. Perstep voltage minima cross the original -90 bound in several exact-encoder tests; propagation does not erase that failure.

Parent then selected the actual saved physical sample1000 at t1.0 s, without selecting on neural response, from `/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics/`: qpos/qvel using INITIAL.qa/va and NATIVE.rows[79999,1:7]. Its verified encoder-generated currents were held constant200 ms with **zero DN current**, no body steps:

| dt ms | MN spikes | Recruited MN | Mean Hz | Max cell mean Hz | Perstep minimum mV |
|---|---:|---:|---:|---:|---:|
| .5 | 1287 | 144 | 19.9226 | 170 | -91.3591 |
| .25 | 1337 | 141 | 20.6966 | 180 | -91.8991 |

Nominal recruitment LF17, LM24, LH28, RF20, RM28, RH27; all six legs propagate. Relative aggregate rate difference .03885; nominal replay bitwise exact. All states finite, but original conservative voltage floor still fails. Adaptation is explicitly present in the existing model; values below the inhibitory reversal can occur under adaptation and are not clipped away. This numerical observation does not establish physiological validity. A held real sensory sample is not ongoing closed-loop feedback or walking. Many MN remain silent and no claim of all-cell controllability is made.

## Independent verification and accounting

Independent verifier `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/independent_verify.py` reconstructs selected identities, all705068 edges and strict weights from the original cache, then uses `np.bincount` edge scatter, separately written backward Euler and absolute-timestamp pending spikes, not the module's sparse multiplication or delay ring. All source supports/weights at all3 gains match. Two step sizes include128 scalar sampled-row comparisons and212 full-vector reference iterations. Maximum voltage difference3.19744e-13 mV; adaptation/refractory/spikes/counts/filter recurrence exact; closed-form20ms filter maximum difference3.41061e-13 Hz. Exact2 ms delay verified at each step. All53 named checks passed on first execution, exit0. Source/module/engine/original-prereg hashes remained unchanged.

Actual neural steps owned by this task: **9,422** = initial absolute-cwd smoke10 + main bounded probes5200 + independent verification212 + exact encoder probes2400 + held recorded sample/replay1600. Manual reference iterations212 and scalar row checks128 are separate, not extra network calls. Physical steps0; training updates0. Parent training or other agents' independent work is outside this count.

Commands used existing `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B` with absolute script paths for `run_validation.py`, `independent_verify.py`, `probe_encoder.py`, and `probe_recorded_sample.py`; all completed exit0. Two directory-discovery `ls` attempts returned exit2 for initially absent/wrong paths; these were resolved with discovery and did not execute neural steps. No simulation execution failures were hidden. All modifications stay inside the two owned dynamics directories; tracked AGENTS/ROADMAP/TRACKER and engine are untouched by this subtask.

`CHOSEN_CONFIG.json` keeps status CALIBRATION_FAILED and fixes the predeclared .02 fallback. Parent may study its explicitly limited restricted-input behavior, but must not relabel this as accepted physiology, full-brain training, calibrated all-port dynamics, or walking. This serves S3/S4 local identity/port implementation and subsequent S5 engineering readout work; it does not complete the broader platform.
