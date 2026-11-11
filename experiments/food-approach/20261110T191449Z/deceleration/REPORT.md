# Two-case halfspeed teacher diagnostic: six-leg completion FAIL preserved

Exactly two preregistered actual-physics cases finished with no early stop. Both passed the separately declared halfspeed forward, original-denominator slip, support, COM and strict nonfoot-zero gates, but both failed all-six-leg cycle completion. Neither is eligible as a passing deceleration curriculum trajectory. No training, neural checkpoint, food field or food behavior ran here; algorithm teacher results do not establish neural autonomy.

| Modified19.2s window | Straight yaw0 | Smallleft yaw+.004 |
|---|---:|---:|
| Signed world-X forward native/s | .03663420393402432 | .03668078339495214 |
| Original forward>=.05 | FAIL | FAIL |
| Explicit halfspeed forward>=.025 | PASS | PASS |
| Original max per-leg slip<=.5 | .46625129530390946 PASS | .4775510223246901 PASS |
| Foot support mean/W>=.90 | 1.0000243116118754 PASS | 1.00002334297619 PASS |
| Minimum COMz>=1.20 native | 1.258632494608017 PASS | 1.2586231991222896 PASS |
| Strict nonfoot peak native==0 | 0 PASS | 0 PASS |
| LF,LH,LM,RF,RH,RM complete counts | 0,2,2,0,2,2 FAIL | 0,2,2,0,2,2 FAIL |
| Actual thorax signed yaw rad | -.009954829526463213 | +.062261241152576634 |
| Original all-five | FAIL | FAIL |
| All explicit halfspeed gates | FAIL | FAIL |

All metrics above use the full19200 modified1ms endpoints with unchanged baseline formulas, including original load-weighted tangential-foot-speed numerator divided by netXY displacement/time. The physical window is19.2s; unchanged last-minus-first endpoint metric duration is19.199s. Native force/length units only, independent absolute SI not established. Strict force-zero and nonfoot results here are1ms endpoints, not a full12.5us peak/D1-D8 certification.

## Exact front-leg failure predicate

Both front legs lift and advance sufficiently within the unchanged sensed cycle detector: peak height about.109–.111 native exceeds.015, and event advance.250–.291 exceeds.03. They return near/below the ground threshold, but the event-window load never reaches the required.1W, so loaded-and-grounded dwell is exactly0ms against the10ms requirement before timeout.

| Case/leg | Abort reasons | Maximum load/W inside failed events | Later actual10ms loaded-grounded dwell age from event start |
|---|---|---:|---|
| Straight LF | 1 timeout +1 terminal partial | .09696297589369428 | 4.805s after first event start; last event ends at25s before reload |
| Straight RF | 2 timeouts | .09470644606834418 | 6.389s each |
| Smallleft LF | 1 timeout +1 terminal partial | .09648423209644816 | 4.815s after first event start; last event ends at25s before reload |
| Smallleft RF | 2 timeouts | .0956611976192711 | 6.390s each |

Whole-window front-leg load peaks are higher (.2004/.2068W straight, .2013/.2078W left), but occur outside the failed active event budget. These whole-window maxima cannot be substituted for required within-event completion. Both LF/RF are grounded for many samples; the missing conjunction is sufficient reloading within the fixed4s event budget, not insufficient lift/advance. This is a descriptive physical timing/reload finding, not a proof of a unique root cause or infeasibility. The globally longer19.2s window alone does not establish complete halfspeed cycles under the current detector. All thresholds and timeout remain unchanged.

## Evidence and independent verification

All paths below are absolute:

- Preregistration, created before physics: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/PREREG.json.
- Actual per-case outputs: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/halfspeed_straight/ and /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/halfspeed_smallleft/. Each contains entire25000-row1ms trace,4800-row X238/Y42/command2/actualtime sample archive, continuous yaw, runtime identity, and RESULT.json.
- Fresh-process saved replay: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/VERIFICATION.json:46/46 integrity checks PASS. All original and halfspeed gates, exact slip/load denominators, all cycle counts/events and analytic command schedule independently recomputed. This replays saved evidence without a third physical trajectory.
- Additional alignment checks: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/ALIGNMENT_VERIFICATION.json:12/12 PASS. All5 ticks per action decision, actual previous-action channels, every sample array and full-body shapes checked. Smallleft's entire first15.4s trace and2880 teacher sample rows are bitexact to the original failed halfspeed-left case, preserving the old failure rather than replacing it.
- Front-leg exact threshold analysis: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/FRONT_PREDICATE_DIAGNOSIS.json.
- Actual path/yaw/speed plot: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/ACTUAL_DIAGNOSTIC.png. Actually opened and inspected: roughly.7 native forward travel over19.2s, persistent low mean speed, smallleft positive yaw and straight small negative drift; axes and original-versus-halfspeed thresholds clearly labeled. No video requested or generated.
- Frozen imports/source hashes: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/INPUT_IDENTITY.json and per-case IDENTITY.json include actual imports, source/asset hashes and compiled arrays/finite-PD parameters. Previous teacher source/all direct saved evidence and baseline source hashes unchanged before/after.

The unchanged previous teacher was explicitly imported from /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/turning_teacher.py; actual modules came from /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z. Rest calibrated to actual post-settle state, teacher private FK only; no physical root writes, no reset at transition, adhesion=False and actual applied body/generalized forces0 at every recorded1ms tick. Native timestep12.5us,49qpos/48qvel/42actuators unchanged. Each case covers exactly2,000,000 native steps and25000ms:1000actualzero+4800baseline+19200continuous modified. Sample records contain960 neutral warmup+3840 conditional decisions, all4800 continuous actual simulator start times, no trimmed failures.

## Resources, scope and parent APIs

Managed physical job bash-188 (PID333382) completed exit0 and was collected. CPU affinity restricted to one allowed CPU0; OPENBLAS/OMP/MKL/NUMEXPR threads1; provided interpreter /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B. Case wall times280.8332s and279.2236s. Fresh-process independent verification bash-192 completed exit0 and was collected; alignment and front diagnosis commands exit0. No managed job remains relevant/running.

Parent-facing saved-sample admission: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/curriculum.py `load_verified_samples(case_name,post_only=False)` requires the independent recomputed case pass and authenticates both RESULT/SAMPLES hashes. It correctly rejects BOTH cases. Frozen /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/api.py still provides explicit-path `make_teacher`, `command2`, halfspeed gate helper and saved cycle replay; its weaker original sample loader must not be used for parent admission. Full API contract: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/API.md.

Read-only static audit found a generic exception-persistence limitation in the frozen runner. Neither run hit it: both complete traces/results exist and independently verify. The audit and handling remain documented in /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/STATIC_AUDIT.md rather than modifying running source. No new files outside the authorized deceleration directory were created or changed; no temporary/dead duplicate artifacts need removal. Master tracker update is delegated to parent due to explicit new-directory-only scope. No further configuration or training was attempted; any future reloading/timing hypothesis needs a separate preregistered experiment.
