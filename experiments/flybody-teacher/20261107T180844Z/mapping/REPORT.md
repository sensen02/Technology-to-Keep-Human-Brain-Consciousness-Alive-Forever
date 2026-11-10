# Official Flybody → retained current body: measured mapping audit

The deliverable is complete as a **structural/FK audit**, not a trained adapter or accepted walking controller. **No scalar joint-angle map or action-transfer map is verified.** The next student should learn on the retained current complete body from anatomically registered motion targets and physical feedback. An arbitrary 59→42 trim, name permutation, angle-offset subtraction, or reuse of teacher controls would be unsupported.

This is S2/S5 support for the approved teacher→connectome platform. The present task does not change the current body, train a policy, simulate a trajectory, or perform surgery. Feeding/choice remain part of the full-platform objective, and surgery remains deferred until that platform and its preoperative parameters are frozen. No tracked project file was edited by this audit.

Primary workspace: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/`.
Desktop copy: `/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/mapping/`.

## What was actually loaded

| Quantity | Official archived teacher model | Actual current baseline |
|---|---:|---:|
| Existing MuJoCo runtime | 3.1.6 | 3.9.0 |
| nq / nv / nu / na | 116 / 114 / 59 / 59 | 49 / 48 / 42 / 0 |
| All joints / physical fly hinge joints | 104 / 102 | 43 / 42 |
| Bodies / geoms / meshes | 135 / 241 / 170 | 73 / 72 / 71 |
| Free joints | 2: walker + visualization ghost | 1: thorax |
| Tendons / equality constraints | 8 / 0 | 90 / 0 |
| Physical fly native mass | 0.0009846214672177625 | 0.0024942714779999996 |
| Gravity z | −981 | −9801 |
| Native integration dt | 0.0002 s | 0.0000125 s |
| Friction cone | elliptic | pyramidal |
| noslip_iterations | 5 | 0 |

Official 170 meshes include the duplicate ghost assets; they do not mean the real fly has 170 distinct anatomical parts. The official ghost adds a free joint but no hinge joints. The current 90 retained tendons do not mean the baseline is muscle-driven: all 42 active transmissions are direct unit-gear, unit-gain stateless hinge motors.

Official model: `/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2/MODEL.mjb`, SHA256 `5c9e55c8db9811c75e0d2e24766643f021e0830d0f7c824dae4347685c6631b7`. Its accompanying `/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2/MODEL_IDENTITY.json` supplies the independently recorded public action order and bounds; verification cross-checks them against compiled actuator identities. The MJB was loaded only in its matching 3.1.6 environment, without constructing/resetting an environment.

Current construction was traced by importing `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T123634Z/run_residual.py` and its real wrapper chain, not by choosing another similarly named body. It resolves through the walking wrappers to `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T073258Z/neural_walk.py`, `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/plant_probe.py`, and `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py`. The latter compiles `/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml`, SHA256 `517cec0eed476f825535b56858b434c949bcbf8ed18ce954189689c784d1d2d1`, through the existing FlyGym asset resolver in MuJoCo 3.9.0. No cross-version MJB loading was attempted. The loader's existing opaque-mesh visualization changes are retained in memory; no XML was rewritten.

The exact imported runtime contract is `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/current_runtime_contract.json`. Source and imported dependency hashes, including both canonical source identities, are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/SOURCE_HASHES.json`. Desktop inputs resolve through an existing symlink, so resolved hash keys can begin `/run/media/sensen/Data2/root-space-relocation/20261022-215857/Desktop/`; the user-facing Desktop source paths above identify the same bytes.

## Verified interface differences

The official 59 actions comprise **6 adhesion + 3 head + 2 abdominal tendon + 18 coxa + 12 femur + 6 tibia + 6 first-tarsus + 6 distal-tarsal tendon** controls. There are **45 joint, 8 tendon and 6 body transmissions**. Only 36 official actions have a regional anatomical candidate among current joints; 23 have no counterpart in the current actuator interface. This is a coverage count, not 36 verified angle mappings.

Official policy indices **0–5 map to MuJoCo ctrl indices 53–58** (adhesion). Policy indices 6–58 map to ctrl indices 0–52. The mapping is verified against archived action names and compiled names. Directly assigning the policy-order vector to ctrl would be wrong. The canonical policy outputs are normalized to [−1,1]; official `canonical2real` computes `low + 0.5*(clip(a,-1,1)+1)*(high-low)` before the walker's action-class permutation. See `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/official-flybody/flybody/tasks/task_utils.py:96` and `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/official-flybody/flybody/fruitfly/fruitfly.py:532`.

Each abdominal tendon sums seven hinge coordinates, with coefficients all 1. Each `tarsus2_*` action targets the fixed tendon coordinate `q_tarsus2 + 0.5*q_tarsus3 + 0.5*q_tarsus4 + 0.5*q_tarsus5`. Those names do not designate isolated tarsus2 motors, and tendon actuation does not impose a kinematic equality between the four angles. This is sourced at `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/official-flybody/flybody/fruitfly/assets/fruitfly.xml:762` and verified from the compiled wrap coefficients.

Official joint/tendon targets use a first-order `filter` activation (`dyntype=2`), time constant 0.010 s; adhesion uses 0.007 s. They are not `filterexact`. For a general actuator, force is determined by compiled gain/bias and filtered activation: `gain*activation + bias0 + bias1*length + bias2*velocity`, followed by enabled force limits. Most leg actuators here have gain 0.8 or 0.4 and positional bias −gain, but no explicit forcelimit; head actuators have ±0.1 native force limits. All exact arrays and limit flags are exported, so `[0,0]` force ranges with `forcelimited=0` must not be read as zero torque. Adhesion has gain 0.985 and command [0,1]; it is a body/contact transmission, not a joint torque or a biological force calibration. Teacher control period is 0.002 s, from archived model identity.

Current output semantics are **42 angular rates in actuator order**, clipped to ±30 rad/s, integrated once every 0.001 s into bounded targets. The policy period is 0.005 s. Every 12.5 µs physical substep uses `clip(kp*(target−q)−kd*qvel, ±native_limit)`, with `kp=native_limit/0.2*128` and `kd=native_limit*0.005*sqrt(128)` in this imported baseline. It has no MuJoCo activation filter (`na=0`), but does have target-integrator memory and finite local PD feedback. Zero rate holds a target and is not equivalent to zero force. Exact 42 limits and gains are in the candidate CSV. This audit did not call the integrator update, policy, or torque loop.

Current actuator leg blocks are LF, LH, LM, RF, RH, RM, not the diagnostic LF, LM, LH, RF, RM, RH convention. The RF coxa block also starts **roll, yaw, pitch**, whereas other coxa blocks are yaw, pitch, roll. Consumers must use saved indices/names, not synthesize an ordering.

## Anatomical candidates and measured coordinate differences

The proposed candidates use the ordered anatomical chain, body ownership and rotation groups in addition to labels. They remain hypotheses across these different model anatomies/frames:

| Anatomical region | Official | Current | Transfer status |
|---|---|---|---|
| Thorax→coxa | three hinges, local ordered axes z,y,x | three hinges, generally x,y,z; RF z,x,y | group orientation candidate; no scalar permutation |
| Coxa→femur/trochanter region | two hinges, local y,x | three named trochanter hinges, local x,y,z | 2→3 rotation-group candidate; redundancy needs a declared objective |
| Femur→tibia | one hinge, local x | tibia_pitch, local y | candidate hinge region; sign/zero need registered anatomy |
| Tibia→first tarsus | one hinge | fixed body | no actuator counterpart |
| Distal tarsi/claw | four hinge coordinates under one tendon actuator | fixed serial tarsal bodies | no actuator counterpart |
| Claw adhesion | six body/contact actuators | none | no transfer |
| Head / abdomen | head3 + two multi-joint abdominal controls | physically present but fixed | no transfer |

These are axes in each joint's own body frame. They are not axes in one globally shared anatomical frame. The exact local body quaternions, joint anchors, and state-dependent thorax-frame axes are exported. A simple illustration: official LF tibia local axis `[1,0,0]`, limits `[−1.35,1.3]` rad; current LF tibia local axis `[0,1,0]`, limits `[0.4789,2.502]` rad, keyframe/spring reference `1.2246235578147273` rad. Official LF femur extension limits `[−0.15,2]`; current LF trochanter_pitch limits `[−3.242,−1.217]`, keyframe `−2.115804366019112` rad. The latter cannot be silently interpreted as the same angle with a sign change.

The current front legs have explicit Trochanter bodies followed by fixed Femur bodies. Middle/hind `Trochanter_*` hinges live on the Femur bodies themselves. Therefore body name matching would also misplace the rotational center. Candidate landmark pairing uses current Trochanter for the front coxa→femur center and current Femur for middle/hind; this is explicitly recorded, not presented as externally validated anatomy.

The audit evaluates `qpos0` and native `key0`. Official key0 and qpos0 are identical in this archive; therefore its two named checks are a repeat reference pose, not two independent gait poses. Current key0 is its noncrossing warmup pose. Current zero pose violates some joint ranges and is used only to expose the reference coordinate system. Neither set is an equilibrium, matched biological pose or dynamic feasibility test.

For every candidate joint pair, `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/candidate_axes_and_offsets.csv` records the two local-pose thorax axes, dot product, unsigned angular difference, anchors and raw coordinate difference. **Raw `q_current−q_official` is not a calibrated mapping offset.** `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/landmark_frame_offsets.csv` records full `R_currentᵀ R_official` matrices and positions for 48 candidate landmarks in each reference pose. These are exact descriptive frame differences under the stated numeric thorax-frame convention; no cross-animal rigid registration is claimed.

## Units, shape and contact geometry

Official length is cm, mass g, time seconds, angles radians. The generator explicitly converts mm→cm at `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/official-flybody/flybody/fruitfly/build_fruitfly/make_fruitfly.py:149`; its multiplication of compiled masses by 1000 to display mg is at line 1219. Its internally consistent force/torque dimensions are g·cm/s² and g·cm²/s². This dimensional interpretation does not establish biological force validity.

Current angles and time are explicitly radians/seconds. Geometry and gravity numerically support a nominal mm length interpretation, but this branch itself states absolute SI force and physical unit calibration **NOT_ESTABLISHED**. The native mass and torque numbers must not be equated to official quantities, nor converted by the same factor as length. The measured gravity ratio is 9801/981, not exactly 10. This audit uses **official cm ×10 versus current native length only as an explicit nominal geometry comparison**, never as a validated dynamical unit conversion.

Even with that nominal conversion, 42 candidate inter-anchor length ratios range **0.6383364336645597–1.1471807298163597**. Examples: LM coxa→femur official 0.0281 cm vs current 0.1814001739 native (ratio after ×10 = 0.6455522); LF tibia→tarsus 0.05103001568 cm vs 0.5105231703 native (1.0004370); LF tarsus2→tarsus3 0.01361182960 cm vs 0.1561522861 native (1.1471807). These are candidate joint/body-origin separations, not independently measured biological bone lengths. They already disprove one common global geometry scale for these selected anchors. Exact rows: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/segment_lengths.csv`.

Official visual meshes coexist with simplified capsule/ellipsoid collision geometry; its six claw collision capsules define the official ground-contact feet. Current collision surfaces are its physical mesh geoms and its baseline foot classifier includes all tarsal geoms. Their foot-origin trajectories and contact locations are therefore not interchangeable. Official `noslip_iterations=5`, elliptic cone, collision masks, friction, margins and gap differ from current `noslip_iterations=0`, pyramidal cone and mesh contacts. Per-geom local pose, size/type, friction, solref/solimp, margin/gap, masks and condim are in the two geometry CSVs; body masses/inertias and local quaternions are in body-frame CSVs. Both model JSONs include compiled vertex hashes and mesh bounding boxes. No contact/friction/solver parameter was changed to manufacture agreement.

## Measurements and independent verification

The final retained extraction used **410 official + 170 current explicit `mj_forward` calls**, **204 + 84 `mj_jac` calls**, and one scratch keyframe reset per model. All 102 official physical hinge joints and all 42 current hinges were checked at both named reference poses, with central differences ±1e−6 rad on a descendant material point. Maximum absolute Jacobian disagreement: official **5.49589679321727e−11 cm/rad**, current **4.0163627978984096e−10 native-length/rad**, both below the preregistered 1e−7 tolerance.

A first successful extraction omitted the transmission-kind decoder for tendon names; the raw transmission integers and numerical FK were already correct. After fixing that decoder and adding tendon coefficients, both extractions were rerun. Superseded JSON was replaced. **Cumulative actual explicit calls across all four extraction processes: 1160 mj_forward, 576 mj_jac, 4 scratch keyframe resets; mj_step/mj_step1/mj_step2 = 0; environment steps = 0; training updates = 0.** Compiler-internal model construction calculations are not represented as intercepted Python forward calls. There were two official MJB loads and two current model compiles. No teacher policy was invoked.

Independent `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/verify_audit.py` uses only NumPy serial rigid transforms and Rodrigues rotations, without importing MuJoCo. It reconstructs all bodies' reference transforms and hinge axes/anchors from the saved compiled arrays. Maximum disagreement over all reported arrays is **1.887379141862766e−15**. It also verifies complete action/ctrl permutation coverage, every tendon target/coefficient, all 42 current action identities, candidate-only flags, and unchanged source hashes. Evidence: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/VERIFICATION.json`.

The no-step extraction entry points guard `mj_step`, `mj_step1` and `mj_step2` by raising on any attempted call. Scratch data always remains time zero. The 3D reference skeleton image was generated from these arrays and opened for inspection; six legs and the distinct native reference poses are visible. `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/KINEMATIC_REFERENCE.png` is a reference-geometry illustration, not a motion/video/contact success result.

## Recommended next student interface while retaining the current body

1. Keep the current complete 71-mesh free body, original contacts/gravity/limits and 42 bounded incremental-rate actuator interface. Store the **actual ordered names** as part of every dataset/checkpoint. The student emits 42 rates in rad/s, not official normalized actions or filtered position controls.
2. Treat the official measured 102-joint motion and teacher trajectories as separately labeled sources. Build thorax-relative anatomical landmarks and segment orientations using the full official chain, including distal passive/tendon-coupled coordinates. Register coordinate conventions and body proportions explicitly; retain uncertainty and residuals. Do not supervise current angles directly with official angle columns.
3. On scratch current-body FK, solve a bounded continuous retargeting problem for current joint targets using corresponding leg landmarks/orientations. Use current range and rate limits, a declared smoothness/redundancy objective for its third proximal rotation, and a held-out trajectory split. Preserve the original trajectory time base. Report unreachable residuals and label retiming/projection as engineering transformations. This is the next concrete preparation step; it has not been executed here.
4. Use retargeted motion as an auxiliary training reference, then train/evaluate the local student in the actual current-body physics. Inputs should be explicit local proprioception/contact and a constrained descending command; privileged teacher root/reference information belongs in training labels/critic only. A learned motion module is a replacement/translation network until real connectome identity and pathway constraints are installed and tested. Do not let the brain bypass the neck to read all joints or issue 42 direct commands.
5. Do not add adhesion to imitate the official policy. Measure the current-body student's own support, material contact slip, clearance, nonfoot contact, finite torque saturation and generalization under its original physics. No-step geometric agreement cannot predict no-adhesion dynamic feasibility. Retain unsuccessful teacher episodes and reference residuals rather than giving the student false success labels.
6. Record a separate feeding/choice interface requirement: the current complete head, rostrum, haustellum and abdominal segments are **fixed**. This walking student interface cannot actively extend the proboscis or drive abdominal articulation. Retain the body during this audit and locomotor learning; design an explicit future ingestion/environment interaction and internal-state contract, or a separately reviewed anatomical articulation extension if needed. Mere presence of mouthpart meshes is not embodied feeding, and no platform-complete claim is justified yet. Surgery still waits for completion and freeze of the full platform.

## Deliverable index and reproduction

All files below are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/` and mirrored to `/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/mapping/`:

- `official59_actions.csv`: all 59 policy indices, ctrl indices, transmission types, target joints/tendon coefficients, exact control/filter/force parameters, source lines and candidate/missing status.
- `current42_candidates.csv`: every retained actuator/joint, regional candidate set, exact axes/angle limits/keyframe/springref/torque caps/PD gains and source line.
- `official_joints.csv`, `current_joints.csv`: all compiled joints including passive joints and free roots; full hinge axes, limits, damping, stiffness, armature and coordinates. For free roots, scalar qpos fields are address samples only; complete positions/quaternions are in model JSON.
- `official_actuators.csv`, `current_actuators.csv`, `official_tendons.csv`, `current_tendons.csv`: complete compiled mechanical interface.
- `official_geometry.csv`, `current_geometry.csv`, `official_body_frames.csv`, `current_body_frames.csv`: exact geometry/contact/body/inertia differences.
- `candidate_axes_and_offsets.csv`, `landmark_frame_offsets.csv`, `segment_lengths.csv`, `articulation_gaps.csv`: measured correspondence evidence and missing articulation.
- `official_model.json`, `current_model.json`: source-identifiable arrays, meshes, both pose frames, finite-difference measurements and counted calls.
- `SUMMARY.json`, `SOURCE_HASHES.json`, `VERIFICATION.json`, `PREREGISTRATION.json`: outcome, provenance, checks and scope; `MANIFEST.json` checks all delivered file bytes.
- `extract_models.py`, `build_audit.py`, `verify_audit.py`: reproducible extraction, tables/plot and independent validation.

Run with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```sh
/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/.venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/extract_models.py official
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/extract_models.py current
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/build_audit.py
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/mapping/verify_audit.py
```

Historical cumulative counts in `SUMMARY.json` describe this audit's actual execution, including the two corrected reruns; rerunning these commands does not retroactively update that historical ledger. Preserve/archive the delivered directory before reproduction if its current manifest is needed. Model source files and old experiment outputs are only read.
