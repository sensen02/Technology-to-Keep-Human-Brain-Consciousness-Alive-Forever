# Executed no-step teacher→current42 retargeting

This S2/S5 preparation was implemented and executed for both sources. It is offline engineering supervision, not a controller, neural result, physical standing or walking. The primary is recorded **physical teacher-derived motion, NOT raw animal motion**. The current full42 body, PD128, ±30rad/s and1ms target integrator contract are retained; no adhesion, source-angle trimming, root trajectory deployment, simulation stepping or training was performed.

Primary: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/TARGETS.npz`. Separate measured-reference artifact: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/MEASURED_TARGETS.npz`. Both contain1112 frames at .002s, inclusive0..2.222s. Source measured FK retains all1176 frames in `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/SOURCE_LANDMARKS.npz`. Primary first pose: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/FIRST_POSE.npz`. Desktop mirror: `/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/kinematics`.

## Fixed gate and actual outcome

```
{
  "teacher": {
    "independent_fk_max_abs_error": 2.220446049250313e-15,
    "independent_target_max_abs_error": 1.3322676295501878e-15,
    "max_rate_rad_s": 30.000000000003304,
    "finite_limits_rate_pass": true,
    "geometry_gate": false,
    "max_normalized_landmark_error": 0.19501574985124895,
    "rms_normalized_landmark_error": 0.03724495430183113,
    "holdout_max": 0.17427491835469838
  },
  "measured": {
    "independent_fk_max_abs_error": 2.220446049250313e-15,
    "independent_target_max_abs_error": 1.3322676295501878e-15,
    "max_rate_rad_s": 30.000000000003304,
    "finite_limits_rate_pass": true,
    "geometry_gate": false,
    "max_normalized_landmark_error": 0.2576742107864666,
    "rms_normalized_landmark_error": 0.04555656872890276,
    "holdout_max": 0.19449630932822798
  }
}
```

Geometry failure is retained, not hidden or used to loosen the budget. Finite/inbounds/rate validity permits a separate parent free-body physical diagnostic even if geometry fails. Nothing here establishes dynamic support or foot friction feasibility. The last222 frames are a temporal diagnostic holdout with frozen settings, not independent animal/recording generalization.

## Anatomical geometry and assumptions

Source and current anatomical bases use fixed coxa-anchor pairs: anterior=(LF+RF−LH−RH)/2, left=mean(left three)−mean(right three), orthogonalized against anterior; dorsal=anterior×left. This assumes named front/hind/left/right attachments identify corresponding anatomical axes. Proper rotation R=C Sᵀ maps source thorax vectors into current thorax coordinates; it is a declared anatomical registration, not external biological calibration. Each current coxa supplies the positional origin, avoiding a false global scale or root deployment.

R source→current:
```
[[ 0.97391872 -0.00484831 -0.22684538]
 [ 0.00625183  0.9999655   0.00546907]
 [ 0.22681104 -0.00674463  0.97391543]]
```
Determinant=0.9999999999999999. Complete fixed anchors/bases are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/GEOMETRY.json`. Current leg lengths (LF,LH,LM,RF,RH,RM)=[2.265131796099363, 2.523901591168466, 2.352414879009001, 2.2672997734726508, 2.5225593875456416, 2.3723509288975064] native. Every segment length, normalized proportion and nominal current/(official cm×10) ratio is in `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/SEGMENT_LENGTHS.csv`. Current SI calibration remains NOT_ESTABLISHED; nominal cm×10 is only a descriptive geometric comparison.

Nine landmarks per leg: coxa, femur-center, tibia, tarsus1–4, claw-origin, material-foot. Front current femur-center uses Trochanter; middle/hind uses Femur, per established mapping. Source foot is a fixed body-local point on its actual claw collision capsule chosen lowest along thorax-Z at key0. Current foot is a fixed vertex of its Tarsus5 physical mesh chosen lowest along thorax-Z at key0. These are material landmarks, not a claim that the two contact surfaces or moving contact patches coincide. Body-local coordinates and current vertex indices are saved. This current distal-body foot does not represent all tarsal collision geoms.

Targets reconstruct cumulative registered source segment directions with eight current inter-landmark lengths, with no scalar joint mapping. Position and directed-segment orientation tasks are explicit; axial twist is unobserved and resolved only by regularization. Official distal hinges cannot all be reproduced by the current fixed tarsi, and current proximal rotational DOFs differ; retained residuals quantify the result.

## Optimization and deployment boundary

Preregistration `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/PREREGISTRATION.json` was written before extraction/solve. One deterministic sequential pass per dataset; each leg has7 bounded variables; frame0 starts at native key0 with80 maximum function evaluations, later frames previous solution with16. TRF ftol/xtol/gtol1e−7. Residuals: position/leglength weight1, directed segment-vector difference .05, previous-angle continuity .02/rad, native-key0 redundancy .002/rad. Hard range bounds and every adjacent delta≤.06rad. There was no broad tuning, source-angle offset fitting or retiming. Max-evaluation terminations remain in per-frame solver_status and quality reports; they are not silently declared converged.

`targets_rad` is (1112,42) in saved actual actuator order LF,LH,LM,RF,RH,RM; RF proximal ordering differs, so use ordered_joint_names exactly. `first_pose_targets_rad` is a one-time parent physical-initialization proposal, not a root write. Linear interpolation from2ms knots to1ms gives ≤.03rad per command tick. A parent may replay these targets solely as a separately labeled physics diagnostic with unchanged finite PD128/native torque bounds; replay is not a deployable neural policy. `source_root_qpos` and source/reference identities are privileged provenance only. No current root-trajectory command exists.

Full NPZ shapes/dtypes: `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/TARGET_SCHEMA.json`. Target/current achieved landmarks are current-thorax/native; root quaternions retain source wxyz in official units and never share that position coordinate convention.

## Independent validation and actual calls

`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/verify_saved.py` reconstructs every current saved frame through independent NumPy serial transforms/Rodrigues rotations from audited compiled geometry, reconstructs source FK at10 predetermined samples, independently rebuilds targets, validates all42 names/angle limits/rates plus1ms interpolation, and checks all42 material-point Jacobians at5 fixed poses by central differences in current MuJoCo. Source hashes are rechecked. Independent verification passed=True; Jacobian max absolute disagreement=8.046847632670051e-10; source FK max error=9.43689570931383e-16.

Cumulative explicit no-step calls (including extraction, optimizer and independent verifier):
```
{
  "mj_forward": 2290,
  "mj_kinematics": 102753,
  "mj_comPos": 102333,
  "mj_jac": 919272,
  "mj_resetDataKeyframe": 2,
  "mj_step": 0,
  "mj_step1": 0,
  "mj_step2": 0
}
```
Official MJB loads1; current full-model compiles2. Independent NumPy serial FK evaluations2234. Environment steps0, training updates0. Compiler-internal evaluations are not counted as intercepted Python calls. All MuJoCo data remain scratch at time0; step/step1/step2 entry points raise. Preview `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/PREVIEW.png` is a kinematic overlay only.

## Reproduction

Run with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The entrypoints overwrite only files in this kinematics output directory, so preserve the current manifest before reproducing.

```
/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/.venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/retarget.py official
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/retarget.py current
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/verify_saved.py
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics/build_report.py
```

No tracked file or other experiment was modified. Project tracker updates belong to the parent because this delegated scope owns only the workspace and desktop kinematics directories.
