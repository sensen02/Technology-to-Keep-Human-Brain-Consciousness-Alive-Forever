# Fixed conductance-adaptation hypothesis: executed and verified

The new version passes its preregistered engineering voltage, propagation and rate checks on the held physical-input sample and the complete 9.085 s recorded sensory trace. It is an **illustrative mechanism hypothesis, not biological calibration**. The previous current-adaptation model, its failure below −90 mV, and all frozen source files remain unchanged. No gains or thresholds were searched, no voltage clipping was added, and no physical or training steps were run.

API: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py`.

Desktop delivery: `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/dynamics/`.

## What changed and what is explicitly assumed

The original immutable identity loader supplies the same20,000 real root IDs, original435,126 strict edges, .02uS per-post incoming normalization based on source synapse count ratios, sensory/descending masks and323MN in exact exported CSV order. Unknown/mixed edges remain inactive. ACh excitation and GABA/glutamate inhibition remain illustrative receptor assumptions, not measurements.

The new `AdaptationConductanceNetwork` reuses original network initialization/connectivity conventions and the existing membrane/receptor parameters. Its active adaptation state is **`net.g_adapt` in uS**. The old independent subtractive current is replaced by `g_adapt * (EA - V)`, with illustrative `EA=-80mV`, tau100ms and per-spike increment`.002nA/15mV=.00013333333333333334uS`. This matches the old .002nA outward increment only atV−65mV. No receptor/reversal measurement is claimed.

Backward Euler includes gA in the denominator and gA×EA in the numerator; decay then increment use the original end-step convention. No voltage clip is used. The separate frozen `AdaptationConductanceParams` and `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/PARAMETER_CONTRACT.json` give explicit units. `net.p` retains original `ConductanceParams` as source metadata, but its historical adaptation-current increment does not participate in the new equations. There is no `net.adaptation` alias pretending that uS is nA.

API remains `LocalCircuit()` with `root_ids`, `allowed_sensory_local`, `descending_local`, `ascending_local`, `mn_local`, `mn_leg`, `net`, `filtered_rates_hz`, `reset()`, and `advance(sensory_current_nA, descending_current_nA, duration_ms=5.)`. Return is323MN filteredHz. Each current vector has20,000 entries, rejects nonfinite or negative values and any nonzero outside its role. Lower-level `net.step` also rejects negative currents. Total incoming conductance is fixed.02uS; nominaldt.5ms, sensitivitydt.25ms, delay2ms, filter20ms. No target/time/body/reference input is exposed. Actual recorded runs use exactly556 qualified encoder identities and zeroDN.

## Bound argument

For `a=C/dt>0`, nonnegative conductances and `I>=0`, the candidate voltage is

`V*=(aV + gL*EL + ge*EE + gi*EI + gA*EA + I)/(a+gL+ge+gi+gA)`.

Subtracting `L=-80mV` yields a sum of nonnegative terms over a positive denominator wheneverV≥L, since all reversals and initial/reset voltages are≥L. Decay/increments preserve nonnegative conductances; reset/refractory voltage−65mV preserves the bound. This proves the reset-origin poststep lower bound by induction in real arithmetic. Threshold/reset gives poststepV≤−50mV. Numerical residual tolerance is1e−10mV; the original engineering acceptance band remains **[−90,−49]mV**, unchanged. Full assumptions, public-state mutation limits and finite-arithmetic caveats are documented in `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/DOMAIN_PROOF.md`.

## Fixed-condition results

Preregistration preceded tests. The input sample is the unchanged prior encoded slow4 physical samplek1000, held200ms; all tests use zero descending current. There was no adjustment after seeing outcomes.

| Condition | Actual neural steps | MN spikes | Recruited MN | Mean MN Hz | Max cell mean Hz | Minimum voltage mV |
|---|---:|---:|---:|---:|---:|---:|
| Held sample,.5ms | 400 | 1053 | 142 | 16.3003 | 150 | −74.6265 |
| Held sample,.25ms | 800 | 1105 | 140 | 17.1053 | 150 | −74.3004 |
| Full9085ms,.5ms | 18170 | 47583 | 209 | 16.2153 | 182.2785 | −76.5959 |

Every condition recruitsMN in all six legs and passes the preregistered≥16MN, mean(0,100)Hz and maxcell<200Hz checks. Full-trace counts by leg: LF8128spikes/22MN, LM4616/35, LH11295/48, RF7648/21, RM7047/37, RH8849/46. All neural-step states are finite; every recorded step passes the original voltage band and derived−80mV lower bound.

Held-sample aggregate rate sensitivity is.0493827 between timesteps; recruitment differs by2. This passes the declared sensitivity check, not a proof of continuous-time convergence. An additional nominal held replay is bitwise equal in MN rates and complete final state.

Zero-input200ms control remains exactly quiet at−65mV. Legal-input/no-connectivity200ms numerical control produces3171 directly driven sensory spikes and **zero MN spikes**. This is a software numerical control, not a lesion/surgical experiment. The new state has no hidden tonic source or externalCPG.

The new global bound does not mean every neuron's voltage increased. Matched START-bin MN minima were−71.9581mV in the old trace and−75.2494mV in the new one; the prior extreme negative-voltage failure was elsewhere in the network. Matched mean filteredMN rates changed from25.2722 to16.3957Hz. A mechanism change alters coding and needs the parent's separate readout comparison; lower rates or greater recruitment are not by themselves behavioral improvement.

## Full trace handoff and verification

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz` contains1817bins×5ms. Sensory currents,556column indices and source times are exactly copied from frozen priorTRAIN_TRACE. No teacher/target is passed to the circuit. Features `mn_rates_hz` and `mn_voltages_mV` are sampled **at the START of each bin, before its current advances the network**. This avoids same-bin lookahead.

The archive additionally saves every-step clock, globalVmin/Vmax, gAmax, all/MN spike counts and packed20,000-cell spike vectors (`spikes_packed`,2500bytes/step, little bit order). Root IDs andMN indices/leg labels are included. Learning code can reuse this artifact without duplicating the neural replay; labels/masks remain in the old training trace.

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/FULL_FINAL_STATE.npz` saves full20kV/ge/gi/gA/refractory/count/filter arrays, lastspikes, delayring, ring index, clock, steps/dt/delay count. No final state is falsely reconstructed and presented as original; these are directly saved states.

Independent solver verification reconstructed all sourcegraph supports/weights and identities from the original cache, then used separately written `np.bincount` event scatter, increment-form backwardEuler, gA equations and absolute timestamp queues. At both timesteps it checked100ms of actual saved input plus randomized numerical diagnostics and quiet controls:84named checks and64guard cases passed. MaximumVdifference7.81597e−13mV; gA/filter/refractory/spikes/counts/queue matched exactly. Derived lower-bound algebra error≤1.42109e−14mV. Original module/engine and all14recorded sources stayed unchanged.

A separate **zero-network-step** saved-event verifier reconstructs all18170steps' start-binMN rates, finalfull20kfilter/gA/count/refractory arrays, perstepgAmax/spikecounts, all5delay slots and clock from packed events. All compared values are exactly equal; source currents/index/time are exact. This validates full saved-event integrity but does not independently re-solve the entire9085ms voltage trajectory; the independent full equation solver covers the bounded100ms traces at bothdt. Reports clearly distinguish these scopes.

## Ledger and failures

Completed neural steps: **21,212** =10API smoke +20,570main probes/replays +632independent solver steps. There were21,236`net.step` invocations including24expected rejected direct-network guard calls. Additional API-level rejections performed no network steps. Separate work:632manual full-vector reference iterations,256scalar-row comparisons,18,170saved-event reconstruction iterations. Physical steps0; training updates0.

Existing runtime `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B` executed all scripts. Initial main-runner bash206 failed at parse time, exit1, due to an unused walrus expression in a comprehension iterable: **0neural/physical steps**. `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/FAILED_FIRST_RUN.json` preserves the exact error/repair. Removing that unused assignment changed only the source-hash list syntax. Successful mainbash207, independentbash208 and saved-eventbash211 exited0. No model, parameter or preregistration changed to rescue a result. Completed progressJSON was deleted after the authoritative results were saved; failure evidence was retained.

ModuleSHA`c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250`; sourceTraceSHA`8a73c8a221b0f752e7c53f9e129fcf9747c42dc3ead6209a7ce912f430e70847`; finalStateSHA`24d514b21ca53e0971e97ae92be47c19286264ef027017a32557a04b2ecec4f7`. These bytes were frozen for parent/readout reuse before final reporting. Full perfile hashes and byte-verified Desktop mirror are inDELIVERY_MANIFEST.

This satisfies the scoped new hypothesis implementation, bounded engineering tests and full recorded-input handoff. It establishes no biological adaptation reversal, measured receptor signs, physical feedback success, full brain, behavior improvement or walking. Prior185910Z failure history is untouched. Tracked project files, original engine and frozen prior artifacts were not edited.
