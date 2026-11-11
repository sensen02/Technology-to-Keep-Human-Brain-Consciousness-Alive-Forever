# Fixed-cadence stride-shortening diagnostic: six-leg completion restored, slip FAIL

The two preregistered cases both completed the entire15.4s physical trajectory without exception or early stop. Both completed all six sensed leg cycles under the unchanged4s detector, but neither passed slip<=.5. Both remain failed overall candidates, and the authenticated curriculum loader rejects them. No training, food behavior or neural autonomy was performed or established.

| Full9.6s modified window | stride.18 scale.5 | stride.27 scale.75 |
|---|---:|---:|
| SignedworldX forward native/s | .0369217144137807 | .05424607724227364 |
| Original forward>=.05 | FAIL | PASS |
| Explicit forward>=.025/.04 | PASS | PASS |
| Maximum original perlegslip<=.5 | .9460449446457037 FAIL | .6204679142089786 FAIL |
| Supportmean/W>=.90 | 1.0000363503311787 PASS | 1.0000444420411811 PASS |
| COMminimum>=1.20 native | 1.2570894747824912 PASS | 1.2570132331051547 PASS |
| Strictnonfoot0 native | 0 PASS | 0 PASS |
| LF,LH,LM,RF,RH,RM complete counts | 1,2,2,2,2,2 PASS | 1,2,2,2,2,2 PASS |
| Original allfive | FAIL | FAIL |
| All explicit stridegates | FAIL | FAIL |

Perlegslip LF,LH,LM,RF,RH,RM is .527096,.744240,.823369,.588049,.946045,.836222 forstride.18; .349328,.509108,.540467,.405200,.620468,.550988 forstride.27. All original denominators and failures remain unchanged. Metrics use all9600modified1ms endpoints, duration9.599s under original last-first convention; actual physicalwindow9.6s. Nativeunits only; absoluteSI not established.

## What the saved slip diagnosis supports

Exact saved1ms comparison against the existing fullspeed leftteacher: netXY progress denominator is .037032081366751936 forstride.18 versus .07162064860670896 forfull, ratio .5170587265986168. Absolute loadweighted padtangent velocity vt did not halve: half/full vt ratios LF1.0281,LH1.0178,LM1.0963,RF.9415,RH1.0303,RM1.0151. RH vt is .03503401336672403 versus .03400357751600388; consequently RH slip rose from .47477338138513653 to .9460449446457037 with the smaller progress denominator. This is a numerical explanation of the ratio, not a changed metric.

Loaded contacts dominate: stride.18 numerator fractions at load>=.1W are LF.71855,LH.96150,LM.92692,RF.62684,RH.97128,RM.95373; unloaded<=.05W fractions are .04457,.02047,.01502,.03893,.01520,.01541. Intermediate load supplies the remainder. Thus the failed slip is mostly actual loaded-foot motion rather than unweighted swing motion.

A separate private reconstruction uses only saved qpos/qvel/actualctrl and mj_forward at every5ms endpoint,1920rows percase, zero integration and no third configuration. It reproduces saved contactloads within5.33e-15 native and speeds within2.22e-16. RH absolute worldX velocity is .0239271 versusfull.0246813; absolute worldY is .0200978 versusfull.0181289. Both components remain substantial. Across legs the xcomponent half/full ratios are .921–1.141 and ycomponents .994–1.109. These observations support persistent loaded tangential motion, including lateral motion; they do not isolate unchanged lateral reference as the unique cause. The existing comparison teacher turns yaw+.008 while stride.18 is straight, an explicit confound. No new fullstraight control was run.

Full quantitative evidence: /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/SLIP_DIAGNOSIS.json and CASE_SUMMARY.json.

## Implementation and validation

All outputs are ONLY under /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/. Previous /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/ and teacher/ files are unchanged and before/after hashed. Actual baseline imports are pinned to /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z, previous turning_teacher explicitly imported. Original StaticTeacher constructed once afteractual1szero settle; original WaveReference phase t/.8legcadence/.36baseline/.12lift/.01contacttransition retained. Only sagittal foot offset is smoothly scaled over1s postprime to.5 or.75; y/z/stance unchanged; vx includes scale_dot*delta+scale*basevx. No physical root/state writes, resets, extra forces or adhesion; unchanged42finitePD,49qpos/48qvel, gravity/friction/native12.5us.

- /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/PREREG.json existed before actual physics, exactly2configs. REFERENCE_CHECKS.json12/12preflight checksPASS, zero physicssteps.
- Per-case stride_018/ and stride_027/ retain entire15400row1mstraces and2880 X238/Y42/command2/actualtime records,960neutral+1920post, plus identity/yaw/results. Each trajectory exactly1,232,000native steps,1sactualzero+4.8sphysicalbaseline+9.6continuousmodified.
- /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/VERIFICATION.json49/49freshprocess saved-evidence checksPASS: original/stridegates, exactslip and load denominators, all6counts/events, independent analytic schedule, everyheldactiontick, previousactionX238, times/sources/fullbody/forcezero, savedhashes. Both actual5800prefix physics and960decision samples are bitexact to original teacher. No third physical trajectory.
- /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/REFERENCE_REPLAY.json10/10additionalchecksPASS: all2880actualrest reference targets andvx match independent analytic schedule exactly0error, lateral/z/stance/sourcephase unchanged. Original preflight file was not frozen by runner beforephysics; this limitation and poststart association are explicitly retained in STATIC_AUDIT.md, not retroactively claimed frozen.
- ACTUAL_DIAGNOSTIC.png actually opened: saved paths show~.35/.52native forwardtravel; speed periodic at originalcadence; small straightyaw drift; full window and thresholds labeled. No video required or generated.
- CURRICULUM_ADMISSION.json confirms both cases refused. Parent API /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/api.py provides explicit make_teacher/reference/command2 and hash-authenticated load_verified_samples; contract in API.md.

Physical managedjob bash-193 completedexit0/collected, casewalls179.3452+172.9303s. Independentreplay bash-197 completedexit0/collected; forwardonly slipdiagnosis bash-198 completedexit0/collected; referencechecks/offlinereplay/summarize exit0. Provided /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B, CPUaffinityone allowedCPU, all numericalthreads1. No active relevantjobs, temporaryfiles or redundant artifacts. Parent owns mastertracker update because all writes restricted to this new directory.1msengineering diagnostics do not establish fullnativeD1-D8, originalbody/connectome or neural behavior.

## Next hypothesis without further execution here

Measured fullteacher RF eventages3.195–3.197s and halfphase actual reloading6.389–6.390s support a falsifiable moderate phase-speed hypothesis: .85 projects RF~3.76s below4s, .75~4.26s exceeds4s. This is a projection with~.24s margin, not a passing candidate; actual slip/support/loading can differ. Details in /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_stride/MODERATE_SPEED_HYPOTHESIS.md. These two stride configurations are complete; any moderate-speed experiment is separately preregistered outside this directory. Prior failures are not body-impossibility conclusions.
