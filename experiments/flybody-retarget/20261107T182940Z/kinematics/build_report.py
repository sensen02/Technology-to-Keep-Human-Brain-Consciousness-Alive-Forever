"""Produce geometry tables, no-step preview, self-contained report and desktop mirror."""
import json,hashlib,csv,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
DESK=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/kinematics')
load=lambda n:json.loads((P/n).read_text())
g=load('GEOMETRY.json');v=load('VERIFICATION.json');a=load('OFFICIAL_EXTRACTION.json');b=load('CURRENT_EXECUTION.json');pre=load('PREREGISTRATION.json')
z=np.load(P/'TARGETS.npz');zz=np.load(P/'MEASURED_TARGETS.npz')
with (P/'SEGMENT_LENGTHS.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['leg','from_landmark','to_landmark','official_cm','current_native','current_div_nominal_official_cm10','current_fraction_of_leg'])
    for l,leg in enumerate(z['leg_names']):
        for k in range(8):w.writerow([leg,z['landmark_names'][k],z['landmark_names'][k+1],g['source_segment_lengths_cm'][l][k],g['current_segment_lengths_native'][l][k],g['current_div_source_nominal_cm10'][l][k],g['current_segment_lengths_native'][l][k]/g['current_leg_lengths_native'][l]])
fig=plt.figure(figsize=(16,10));colors=plt.cm.tab10(np.arange(6))
for panel,t in enumerate([0,250,555,1111]):
    ax=fig.add_subplot(2,3,panel+1,projection='3d')
    for l,leg in enumerate(z['leg_names']):
        for key,ls,alpha in [('target_landmarks','--',.55),('current_achieved_landmarks','-',1.)]:
            p=z[key][t,l];ax.plot(*p.T,ls,color=colors[l],alpha=alpha,label=str(leg) if key=='current_achieved_landmarks' else None)
    ax.set_title(f't={z["time_s"][t]:.3f}s: solid achieved / dashed target');ax.set_xlabel('thorax x native');ax.set_ylabel('y');ax.set_zlabel('z');ax.set_box_aspect((1,1,1));ax.view_init(22,35)
    pts=np.concatenate([z['target_landmarks'][t].reshape(-1,3),z['current_achieved_landmarks'][t].reshape(-1,3)]);mid=(pts.max(0)+pts.min(0))/2;half=np.max(np.ptp(pts,axis=0))*.55
    ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[1]-half,mid[1]+half);ax.set_zlim(mid[2]-half,mid[2]+half)
    if panel==0:ax.legend(fontsize=7,ncol=2)
ax=fig.add_subplot(2,3,5)
for l,leg in enumerate(z['leg_names']):ax.plot(z['time_s'],z['normalized_landmark_error'][:,l].max(1),label=str(leg),color=colors[l])
ax.axhline(.1,color='k',ls='--',label='fixed max budget .1');ax.set_xlabel('time s');ax.set_ylabel('max landmark error / leglength');ax.set_title('Primary physical teacher-derived residual');ax.legend(fontsize=7,ncol=2)
ax=fig.add_subplot(2,3,6)
for data,label in [(z,'teacher actual'),(zz,'measured reference')]:ax.plot(data['time_s'][1:],np.max(abs(np.diff(data['targets_rad'],axis=0)),axis=1)/.002,label=label)
ax.axhline(30,color='k',ls='--');ax.set_xlabel('time s');ax.set_ylabel('max joint rate rad/s');ax.set_title('Hard rate bounds; source timing preserved');ax.legend()
fig.suptitle('NO-STEP RETARGETING ONLY — offline targets, not physical walking\nPrimary teacher-derived geometry gate: '+str(v['datasets']['teacher']['geometry_gate']),fontsize=15);fig.tight_layout();fig.savefig(P/'PREVIEW.png',dpi=150);plt.close(fig)
counts={}
for ledger in [a['calls'],b['calls'],v['current_runtime_calls']]:
    for k,n in ledger.items():counts[k]=counts.get(k,0)+n
summary=dict(scope='S2/S5 offline kinematics only',primary=str(P/'TARGETS.npz'),secondary=str(P/'MEASURED_TARGETS.npz'),verification=v,datasets={tag:load(tag.upper()+'_QUALITY.json') for tag in ['teacher','measured']},cumulative_explicit_mujoco_calls=counts,model_loads_official=1,current_model_compiles=2,independent_numpy_fk_calls=v['independent_serial_fk_calls'],environment_steps=0,training_updates=0,geometry_units='current native SI NOT_ESTABLISHED',preview_opened=(P/'PREVIEW_INSPECTION.json').exists())
(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(P/'FIRST_POSE.npz',targets_rad=z['first_pose_targets_rad'],ordered_joint_names=z['ordered_joint_names'],source_id=z['source_id'],joint_limits_rad=z['joint_limits_rad'])
schema={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
(P/'TARGET_SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n')
report=f'''# Executed no-step teacher→current42 retargeting

This S2/S5 preparation was implemented and executed for both sources. It is offline engineering supervision, not a controller, neural result, physical standing or walking. The primary is recorded **physical teacher-derived motion, NOT raw animal motion**. The current full42 body, PD128, ±30rad/s and1ms target integrator contract are retained; no adhesion, source-angle trimming, root trajectory deployment, simulation stepping or training was performed.

Primary: `{P/'TARGETS.npz'}`. Separate measured-reference artifact: `{P/'MEASURED_TARGETS.npz'}`. Both contain1112 frames at .002s, inclusive0..2.222s. Source measured FK retains all1176 frames in `{P/'SOURCE_LANDMARKS.npz'}`. Primary first pose: `{P/'FIRST_POSE.npz'}`. Desktop mirror: `{DESK}`.

## Fixed gate and actual outcome

```
{json.dumps(v['datasets'],indent=2)}
```

Geometry failure is retained, not hidden or used to loosen the budget. Finite/inbounds/rate validity permits a separate parent free-body physical diagnostic even if geometry fails. Nothing here establishes dynamic support or foot friction feasibility. The last222 frames are a temporal diagnostic holdout with frozen settings, not independent animal/recording generalization.

## Anatomical geometry and assumptions

Source and current anatomical bases use fixed coxa-anchor pairs: anterior=(LF+RF−LH−RH)/2, left=mean(left three)−mean(right three), orthogonalized against anterior; dorsal=anterior×left. This assumes named front/hind/left/right attachments identify corresponding anatomical axes. Proper rotation R=C Sᵀ maps source thorax vectors into current thorax coordinates; it is a declared anatomical registration, not external biological calibration. Each current coxa supplies the positional origin, avoiding a false global scale or root deployment.

R source→current:
```
{np.array(g['registration_source_to_current_rotation'])}
```
Determinant={g['determinant']}. Complete fixed anchors/bases are in `{P/'GEOMETRY.json'}`. Current leg lengths (LF,LH,LM,RF,RH,RM)={g['current_leg_lengths_native']} native. Every segment length, normalized proportion and nominal current/(official cm×10) ratio is in `{P/'SEGMENT_LENGTHS.csv'}`. Current SI calibration remains NOT_ESTABLISHED; nominal cm×10 is only a descriptive geometric comparison.

Nine landmarks per leg: coxa, femur-center, tibia, tarsus1–4, claw-origin, material-foot. Front current femur-center uses Trochanter; middle/hind uses Femur, per established mapping. Source foot is a fixed body-local point on its actual claw collision capsule chosen lowest along thorax-Z at key0. Current foot is a fixed vertex of its Tarsus5 physical mesh chosen lowest along thorax-Z at key0. These are material landmarks, not a claim that the two contact surfaces or moving contact patches coincide. Body-local coordinates and current vertex indices are saved. This current distal-body foot does not represent all tarsal collision geoms.

Targets reconstruct cumulative registered source segment directions with eight current inter-landmark lengths, with no scalar joint mapping. Position and directed-segment orientation tasks are explicit; axial twist is unobserved and resolved only by regularization. Official distal hinges cannot all be reproduced by the current fixed tarsi, and current proximal rotational DOFs differ; retained residuals quantify the result.

## Optimization and deployment boundary

Preregistration `{P/'PREREGISTRATION.json'}` was written before extraction/solve. One deterministic sequential pass per dataset; each leg has7 bounded variables; frame0 starts at native key0 with80 maximum function evaluations, later frames previous solution with16. TRF ftol/xtol/gtol1e−7. Residuals: position/leglength weight1, directed segment-vector difference .05, previous-angle continuity .02/rad, native-key0 redundancy .002/rad. Hard range bounds and every adjacent delta≤.06rad. There was no broad tuning, source-angle offset fitting or retiming. Max-evaluation terminations remain in per-frame solver_status and quality reports; they are not silently declared converged.

`targets_rad` is (1112,42) in saved actual actuator order LF,LH,LM,RF,RH,RM; RF proximal ordering differs, so use ordered_joint_names exactly. `first_pose_targets_rad` is a one-time parent physical-initialization proposal, not a root write. Linear interpolation from2ms knots to1ms gives ≤.03rad per command tick. A parent may replay these targets solely as a separately labeled physics diagnostic with unchanged finite PD128/native torque bounds; replay is not a deployable neural policy. `source_root_qpos` and source/reference identities are privileged provenance only. No current root-trajectory command exists.

Full NPZ shapes/dtypes: `{P/'TARGET_SCHEMA.json'}`. Target/current achieved landmarks are current-thorax/native; root quaternions retain source wxyz in official units and never share that position coordinate convention.

## Independent validation and actual calls

`{P/'verify_saved.py'}` reconstructs every current saved frame through independent NumPy serial transforms/Rodrigues rotations from audited compiled geometry, reconstructs source FK at10 predetermined samples, independently rebuilds targets, validates all42 names/angle limits/rates plus1ms interpolation, and checks all42 material-point Jacobians at5 fixed poses by central differences in current MuJoCo. Source hashes are rechecked. Independent verification passed={v['passed']}; Jacobian max absolute disagreement={v['jacobian_fd_max_abs_error']}; source FK max error={v['source_sample_fk_max_error']}.

Cumulative explicit no-step calls (including extraction, optimizer and independent verifier):
```
{json.dumps(counts,indent=2)}
```
Official MJB loads1; current full-model compiles2. Independent NumPy serial FK evaluations{v['independent_serial_fk_calls']}. Environment steps0, training updates0. Compiler-internal evaluations are not counted as intercepted Python calls. All MuJoCo data remain scratch at time0; step/step1/step2 entry points raise. Preview `{P/'PREVIEW.png'}` is a kinematic overlay only.

## Reproduction

Run with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The entrypoints overwrite only files in this kinematics output directory, so preserve the current manifest before reproducing.

```
/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/.venv/bin/python {P/'retarget.py'} official
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {P/'retarget.py'} current
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {P/'verify_saved.py'}
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {P/'build_report.py'}
```

No tracked file or other experiment was modified. Project tracker updates belong to the parent because this delegated scope owns only the workspace and desktop kinematics directories.
'''
(P/'REPORT.md').write_text(report)
DESK.mkdir(parents=True,exist_ok=True)
for f in P.iterdir():
    if f.is_file() and f.name!='MANIFEST.json':shutil.copy2(f,DESK/f.name)
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name!='MANIFEST.json'}
(P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(P/'MANIFEST.json',DESK/'MANIFEST.json')
print(json.dumps(summary,indent=2))
