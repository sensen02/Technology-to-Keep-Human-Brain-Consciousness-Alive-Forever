# Saved-data onset mismatch diagnosis

The train/test command-onset difference is substantial. All four training episodes smoothly increase the command over 1 s after the 4.8 s teacher prefix. Actual fixed1000 evaluations apply the full command immediately. No label-formula error was found: saved commands match `smoothstep(t/1s) * final_command` to maximum absolute error 1.11e-15 for turns and 2.08e-16 for speed.

This report reads immutable saved data and runs offline neural inference only. It does not establish that ramping fixes physical failures. No physics backend was imported, no physical model constructed, no integration or optimization executed, and no captured source changed.

## Exactly matched starting state

For each condition, the teacher and actual evaluation share bitexact first5800 physical rows of qpos/qvel/COM/action/target/actualctrl, all960 priming observation/action rows, and the first autonomous X238. The teacher-sequence causal frozenbase hidden at handover matches the actual production hidden bitexactly. One-row production-sized offline abrupt inference matches the actual saved first policy action bitexactly.

At the first autonomous decision teacher command is `[0,0]`; actual commands are `[0,+1]`, `[0,-1]`, `[-.15,0]`, `[-.10,0]`. After command normalization `[.15,1]`, the first mismatch magnitude is 1 for left/right/speed085 and 2/3 for speed090. During the first1s, mean command-gap norm is .5025 for the first three and .335 for speed090. Turn teacher reaches .15625 at250ms, .5 at500ms, .99992525 at995ms.

## Offline same-teacher-state comparison

For every teacher stream the frozen base was advanced causally through the actual960 priming rows and first200 autonomous rows. Both command variants use the SAME teacher observation and SAME causal base hidden at each decision. This isolates their neural action difference without simulating an alternative body. Numbers are RMS over normalized42 actions, with the 1s window including 200 decisions.

| Condition | First action step−ramp RMS | First200ms ramp target RMS | First200ms abrupt target RMS | First1s ramp target RMS | First1s abrupt target RMS |
|---|---:|---:|---:|---:|---:|
| Left | .0619193 | .0085196 | .0467408 | .0180637 | .0303660 |
| Right | .0649300 | .0085166 | .0521330 | .0178734 | .0378519 |
| Speed .85 | .0948472 | .0085280 | .0853879 | .0164193 | .0576816 |
| Speed .90 | .0461221 | .0085250 | .0420817 | .0164098 | .0294183 |

The neutral first action already differs from teacherY: RMS .00469510937, maximum .01534208014, identical across all4. With gate=0 at command0 that boundary residual cannot be corrected by adapter weights. It is small relative to the abrupt-command jump and is present even under the offline ramp. No weak-observability or body-impossibility claim follows.

A 200-row batched forward differs from production one-row arithmetic at roughly1e-16 due BLAS rounding. The initial strict batch equality check exited1 before report creation. It was corrected to verify the actual one-row inference bitexactly; batched-vs-one-row deltas remain explicitly reported (maximum observed5.27e-16), not treated as a policy discrepancy.

## Saved physical divergence and contacts

The actual bodies below all used abrupt commands. Joint-angle relative RMS first exceeds .01 at .586s(left),1.830s(right),.327s(speed085),1.967s(speed090). qvel relative RMS exceeds .01 already at the first1ms endpoint in all4. Relative RMS is RMS(actual−teacher)/RMS(teacher) at each time; absoluteRMS and translation/orientation diagnostics are supplied in JSON because world-origin normalization and near-zero rates can magnify relative values. These thresholds are diagnostics, never acceptance criteria.

| Condition | Actual max segment slip first1s | Teacher first1s | Actual after1s | Teacher after1s | Actual nonfoot-positive ms first1s / after1s |
|---|---:|---:|---:|---:|---:|
| Left | 1.02147 | .61596 | 18.32509 | .46013 | 0 / 0 |
| Right | 1.01759 | .70520 | 4.47777 | .43553 | 0 / 0 |
| Speed .85 | 13.76433 | .53676 | 4.79646 | .46159 | 0 / 7714 |
| Speed .90 | 1.11959 | .52826 | 1.52107 | .45787 | 0 / 0 |

Segment slip is the original load-weighted tangential-speed numerator divided by that segment's netXY speed. These short windows can have low progress/turning transients; even the passing teacher has first1s ratios above .5. This neither changes the whole-window gate nor means teacher failed its full-window validation. JSON includes absolute load-weighted pad speeds, COM displacement-derived speed, support, carrying legs, nonfoot and COMmin so denominator effects remain visible. The first200ms window is also recorded. All saved failures and whole-window gates remain intact.

## Next experiment recommendation only

After the selected3000 checkpoint's existing five fixed actual reports finish, separately preregister a matched **actual** 1s smooth highlevel command-onset diagnostic using that same selected checkpoint, physicalprefix, continuous neural state and totalwindow. Keep the abrupt results, exact original physical/cycle/turn criteria and failures. Do not shorten scored windows, change gates, substitute labels or claim fooddeployment qualification. Such a schedule supplies a diagnostic neural input; it does not specify gait, joints, root motion or external support.

The present offline comparison supports this controlled experiment but cannot establish causality: body feedback and hidden states during an actual ramp would differ from the teacher replay. Do not infer the entire late failure is caused by onset, or that remaining target-fit error is irrelevant.

## Files and reproduction

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/onset_diagnosis/DIAGNOSIS.json`: all4 quantitative paired results, input/source SHA256, exactness checks and limits.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/onset_diagnosis/diagnose.py`: cached-once reproducible analysis; refuses existing DIAGNOSIS output.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/onset_diagnosis/ANALYSIS_PLAN.json`: preanalysis scope and methods.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/onset_diagnosis/ARTIFACT_SHA256.json`: final artifact identities and run record.

Interpreter `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B`, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Final run exit0; every NPZ loaded once/materialized/cached per run, then released. Fixed1000 checkpoint SHA256 `3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28`. No3000 inference is mixed into these1000 measurements.
