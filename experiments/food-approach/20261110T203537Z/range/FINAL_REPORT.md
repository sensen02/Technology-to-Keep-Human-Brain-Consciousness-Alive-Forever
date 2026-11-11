# Four fixed restricted-range diagnostics: all fail slip

All new outputs reside under `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/range/`. Old frozen sources and evidence were not edited. Managed `bash-207` finished and was collected with exit0; four physical child subprocesses and one independent saved-evidence verifier each exited0. No training or food behavior occurred.

The exact residual500 checkpoint SHA is `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`. Previously verified same-checkpoint neutral actual control is reused (slip.49278883265); no neutral integration repeated. Preregistration explicitly waives previous fullrange prerequisite only for diagnostic mapping of the already failed full±1 candidate. No food/deployment qualification is inferred.

| Profile | Maximum original slip (limit .5) | First4.8s yaw rad | Second4.8s yaw rad | Frozen yaw criterion | Original all5 |
|---|---:|---:|---:|---|---|
| halfleft+.5 | .5227798735301911 | +.005348450305468964 | +.005052586887251406 | PASS whole+.0104010372 | FAIL slip |
| halfright-.5 | .5227988150589508 | -.014066064893379953 | -.01451300608716451 | PASS whole-.0285790710 | FAIL slip |
| left_to_right | .5220005470783844 | +.005348450305468964 | -.014218509858355304 | PASS bothsegments | FAIL slip |
| right_to_left | .5192207238614788 | -.014066064893379953 | +.00554205002633982 | PASS bothsegments | FAIL slip |

All four runs pass support, nonfoot, COM and forward gates. All six sensed cycles pass in each: LF1,LH2,LM2,RF2,RH2,RM2. Direction changes respond to actual neural commands with continuous192hidden memory, but contact precision fails unchanged originalslip<=.5. Halfcommand is not monotonically better: halfright slip exceeds previously fullright.51516. These results do not establish a qualified restricted turn range, impossibility, ingestion, biological validity or fullnativeD1–D8 acceptance.

Each actual run contains1s zero settle+4.8s originalteacher neutralprime+9.6s pureANN,15400 endpoints and2880 decisions. Underlying integration is unchanged12.5us,42finitePD,zeroextra forces/rootwrites/adhesion. New wrapper hook adds captured prereg/source identity and NumPy-scalar JSON normalization only, without editing old physical/scoring code.

`VERIFICATION.json`: independent96/96 checksPASS, no integration/optimization. Each archive was cached once. Checked complete15400 timecoverage, sources, actionholding and previousaction, unchanged5800physical prefix against actualneutral, command schedule, exactall2880 checkpointmean/action and192hidden before/after chain, cycleevents, originalgate/slip and yaw recomputation, handover/switch snapshot hashes, source/artifacthashes and compiledruntime identity. No time interval was dropped and no failure hidden.

`ACTUAL_RANGE_DIAGNOSTIC.png` was generated solely from saved traces and actually opened: allfour slipbars remain above red.5limit; switching yaw signs visibly respond. Figure is diagnostic only and explicitly says nofood/deploymentqualification. No desktop writes were made because this task's write scope is only this new directory; parent can deliver if desired.

Primary files: `PREREG.json`, `SOURCE_IDENTITY.json`, `ALL4_RESULT.json`, `VERIFICATION.json`, fourcondition result/identity/trace/neural/geometry archives, `ACTUAL_RANGE_DIAGNOSTIC.png`. All necessary failure evidence is retained. No relevant jobs remain running. Parent owns subsequent architecture work and mastertracker updates.
