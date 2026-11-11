# Fixed diagonal CEM physical-objective experiment: 16 left runs, no passing candidate

The complete preregistered four-generation, four-candidate budget finished. All16 actual LEFT rollouts and all16 independent audits completed; each saved-array audit passed92/92 checks. Every candidate passed support/nonfoot/COM/forward, strict nonfoot zero, six sensed cycles (LF1, other legs2), and positive yaw≥.01rad. Every candidate failed the unchanged maximum-slip≤.5 requirement.

| Generation | Candidate0 slip | Candidate1 slip | Candidate2 slip | Candidate3 slip |
|---|---:|---:|---:|---:|
| 1 | .5155999796 | .5063484049 | .5119092233 | .5091957099 |
| 2 | .5070298609 | .5078949677 | .5080286463 | .5141522580 |
| 3 | .5087780225 | .5063177892 | .5070850219 | .5079886075 |
| 4 | .5057359624 | .5082252993 | .5072885010 | .5064372551 |

Best observed case g04_c00 remains FAIL at0.50573596241638. Its delta3 is[-.04787023063760513,.032534641726104006,.000912322708612634], signed forward0.0716083635662815, yaw+.0631163657289116. Its checkpoint SHA is6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f. It is descriptive best-failed evidence, **not selected or qualified**. FINAL_SUMMARY selected=null and confirmations=[]; SELECTED.json does not exist. Thus no neutral/right confirmation or selected native run is authorized by this experiment.

## Exact neural intervention and optimizer

Only readout.bias[29,32,34] differs from immutable head2000 parent SHAeeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df. Names joint_RHCoxa_pitch/joint_RHTrochanter_pitch/joint_RHTibia_pitch were verified against sealed zero-integration runtime mapping. All other197608-state-dictionary tensors remain exactly the parent, as reconstructed and checked on every candidate load. The original ReadoutMotor, positive-ReLU*.25*tanh branch, observations,128+64 memories and physical42 outputs are unchanged. Biases are constant ANN parameters throughout each rollout; optimization happens only between full runs. No gait controller, phase/time/food coordinate input, root writes, extra forces, adhesion, geometry or PD/torque/rate changes were added.

Each generation draws two standard-normal3-vectors from default_rng(20261111)/PCG64, uses the exact registered antithetic order and±.25 clipping, and freezes all four checkpoints plus proposal RNG/mean/sigma before any generation physics. Only independently audited whole9.6s slip is used when all other constraints pass; otherwise the registered penalty applies. All16 here satisfied those other constraints, so all objectives equal measured slip. Elite2 updates and tie breaks follow the fixed declared formula; no extra generation, candidate, bound or unrun-mean score was added.

`OPTIMIZER_AUDIT.json` independently passes68 checks: all frozen source hashes, generation proposals and RNG states, all candidate/report/audit digests, all92 actual audit flags per case, exact objectives, elite identities and mean/sigma updates, and absence of a false selected candidate. This audit uses JSON and hashes only, no physical integration or NN updates.

Preflight bash24 returned0 and was collected, with24 checks including actual sealed joints, PCG64 roundtrip/antithetic/update/penalty tests and all four initial candidate neutral/right2880 means+192hidden histories bitexact to the parent. Later-generation candidate loaders independently reconstruct the exact immutable RNG proposal and exact three-scalar changes. All source paths exist before Core. CURRENT_STATE.json/LIVE_PROGRESS.json are mutable pointers excluded from all scientific source seals.

## Actual evidence

Every physical case retains full1s zero settle+4.8s actual teacher priming+9.6s pure ANN1s smooth turn, continuous handover and both hidden states, native12.5us/target1ms/policy5ms, complete71 opaque meshes and42 finite PD, original full-window slip denominator and gates. Each audit replays all2880 candidate means/actions/192hidden chains and verifies15400 endpoint/source clocks, actual previous-action and held-action consistency, original prefix, zero extra forces, compiled/runtime geometry, immediately persisted HANDOVER and full-window summary/cycles/yaw. No score from partial or unverifiable data entered CEM.

## Orchestration interruption and recovery

Original managed bash25 became unavailable after an interrupted stream; no original terminal code is fabricated. Generation3 candidates2/3 had complete actual RESULT+VERIFY records with all92 checks true, but printing their summaries raised BrokenPipeError32. GEN03_EXECUTION_ERRORS.json is preserved. Process inspection confirmed no surviving run. RESUME_AFTER_BROKEN_PIPE.json documents recovery.

Recovery managed bash1 ran the **unchanged** pipeline with stdout/stderr redirected to RECOVERY_PIPELINE.log. Its saved-result branch verified and reused all12 completed actual cases without re-integrating them, reconstructed the exact generation3 update, froze generation4 and ran only the four remaining preregistered actual cases. Recovery bash1 finished exit0 and was collected. All actual/audit subprocess exit codes for the complete16 are recorded as successful; no extra rollout was caused by recovery. Every subprocess exits before its slot is reused, and only bounded JSON summaries/checksums are retained by the orchestration, not all historical full arrays.

## Public API and artifact pointers

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/model_io.py` provides load_checked(candidate_path,expected_sha,require_final=True) returning original ReadoutMotor and payload. Payload kind=CEM3bias includes generation/index/delta/proposal path+SHA/parentSHA/identity/state_dict/calibration_source_hashes. Loading reconstructs proposal from immutable PCG64 state, verifies bounds/order and exact three-bias-only tensor intervention. No candidate is deployment-qualified here.

Each GENxx_PROPOSAL/GENxx_FROZEN/GENxx_RESULT records proposal state, all frozen checkpoints and actual measurements, elites and exact updates. FINAL_SUMMARY.json is COMPLETE/4generations/16LEFT/selectednull. Artifact manifests seal only explicit completed inputs/results and exclude mutable CURRENT_STATE/LIVE_PROGRESS, native preparation and deceleration preparation subdirectories. All prior timestamped sources, checkpoints and scientific failures remain unchanged.

The experiment improved the best measured slip relative to the starting parent but did not cross the original gate. It does not prove body infeasibility or identify a unique cause. No continuous command range, switching, initial-state robustness, neural deceleration, full-native/body-biological/connectome validity or food approach is claimed.
