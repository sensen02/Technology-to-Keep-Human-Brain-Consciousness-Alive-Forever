"""Fixed-pose model torque authority, not an equilibrium or autonomous controller.

All signs are generalized-coordinate signs, NOT anatomical flexion/extension.
Uses active difference at same qpos/qvel, retaining passive force baseline.
No stepping, posture solve, muscle-file modification or command prescription.
engine/embodied/force_channel.py: forces are uncalibrated model quantities.
"""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')

def main():
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'outputs/autonomous_authority_round8';out.mkdir(parents=True,exist_ok=True)
    xml=ROOT/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
    mp=ROOT/'outputs/muscle_anatomy_resolution/resolution.json'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    original_hash=sha(xml);manifest=json.loads(mp.read_text())
    m=_load_mjcf(str(xml)).compile();d=mj.MjData(m)
    assert m.nu==90 and m.na==90 and m.neq==0
    assert sum(m.jnt_type==mj.mjtJoint.mjJNT_FREE)==1
    assert np.all(m.actuator_dyntype==mj.mjtDyn.mjDYN_MUSCLE)
    mj.mj_resetDataKeyframe(m,d,0)
    d.qvel[:]=0;d.ctrl[:]=0;d.act[:]=0
    q0=d.qpos.copy();mj.mj_forward(m,d)
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
    allowed_names={r['actuator'] for r in manifest['actuators'] if r['status']=='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE'}
    allowed=np.array([i for i,n in enumerate(names) if n in allowed_names]);assert len(allowed)==44
    hinges=np.flatnonzero(m.jnt_type==mj.mjtJoint.mjJNT_HINGE)
    assert len(hinges)==42
    dofs=m.jnt_dofadr[hinges];joints=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j)) for j in hinges]
    moment=np.zeros((m.nu,m.nv));mj.mju_sparse2dense(moment,d.actuator_moment,d.moment_rownnz,d.moment_rowadr,d.moment_colind)
    baseline_force=d.actuator_force.copy();baseline_qforce=d.qfrc_actuator.copy()
    A=np.zeros((m.nv,m.nu));delta_force=np.zeros(m.nu)
    for i in range(m.nu):
        d.act[:]=0;d.act[int(m.actuator_actadr[i])]=1
        mj.mj_forward(m,d)
        assert np.array_equal(d.qpos,q0) and not np.any(d.qvel)
        diff=d.actuator_force-baseline_force
        other=diff.copy();other[i]=0
        assert np.max(abs(other))<1e-10
        A[:,i]=d.qfrc_actuator-baseline_qforce;delta_force[i]=diff[i]
    reconstruction=moment.T*delta_force[None,:]
    err=float(np.max(abs(A-reconstruction)));assert err<1e-8
    joint_A=A[dofs,:]
    tol=max(float(np.max(abs(joint_A)))*1e-9,1e-12)
    geomtol=max(float(np.max(abs(moment[:,dofs])))*1e-9,1e-12)
    def summary(cols):
        mat=joint_A[:,cols];s=np.linalg.svd(mat,compute_uv=False)
        rank_tol=max(mat.shape)*np.finfo(float).eps*s[0] if s.size and s[0] else 0
        rec=[]
        for k,j in enumerate(joints):
            pos=[names[cols[i]] for i in np.flatnonzero(mat[k]>tol)]
            neg=[names[cols[i]] for i in np.flatnonzero(mat[k]<-tol)]
            rec.append({'joint':j,'positive_active_columns':pos,'negative_active_columns':neg,
                        'sign_coverage':'both' if pos and neg else 'positive_only' if pos else 'negative_only' if neg else 'none',
                        'positive_sum':float(np.maximum(mat[k],0).sum()),'negative_sum':float(np.minimum(mat[k],0).sum())})
        return {'columns':len(cols),'rank':int(np.sum(s>rank_tol)),'rank_tolerance':rank_tol,
                'singular_values':s.tolist(),'joint_sign_coverage':rec,
                'unsupported_joints':[r['joint'] for r in rec if r['sign_coverage']=='none'],
                'geometry_rank':int(np.linalg.matrix_rank(moment[cols][:,dofs].T)),
                'geometry_unsupported_joints':[joints[k] for k in range(42) if np.max(abs(moment[cols][:,dofs[k]]))<=geomtol]}
    full=summary(np.arange(90));sel=summary(allowed)
    lost=[{'joint':a['joint'],'all90':a['sign_coverage'],'allowed44':b['sign_coverage']} for a,b in zip(full['joint_sign_coverage'],sel['joint_sign_coverage']) if a['sign_coverage']!=b['sign_coverage']]
    manifest_rows={r['actuator']:r for r in manifest['actuators']}
    for row in lost:
        k=joints.index(row['joint'])
        missing_positive=row['allowed44']=='negative_only'
        supplying=np.flatnonzero(joint_A[k]>tol if missing_positive else joint_A[k]<-tol)
        row['missing_coordinate_sign']='positive' if missing_positive else 'negative'
        row['excluded_source_names']=[{'actuator':names[i], 'mapping_status':manifest_rows[names[i]]['status'],
            'active_generalized_difference':float(joint_A[k,i])} for i in supplying if i not in allowed]
        assert row['excluded_source_names']
    report={'status':'FIXED_POSE_PHYSICAL_AUTHORITY_DIAGNOSTIC_NOT_CONTROL',
        'sources':{str(p):sha(p) for p in (Path(__file__),xml,mp)},'mujoco_version':mj.__version__,
        'pose':'standing keyframe0; qvel=0; each act_i=1 compared with all act=0; no steps',
        'zero_excitation_note':'ctrl lower bound unchanged; mj_forward does not integrate activation. act assigned explicitly; baseline passive muscle force retained.',
        'all90':full,'allowed44':sel,'lost_sign_coverage':lost,'generalized_active_matrix':A.tolist(),
        'moment_matrix':moment.tolist(),'actuator_names':names,'leg_joints':joints,
        'allowed_indices':allowed.tolist(),'active_force_delta':delta_force.tolist(),
        'zero_activation_actuator_force':baseline_force.tolist(),'zero_activation_qfrc_actuator':baseline_qforce.tolist(),
        'moment_reconstruction_max_abs_error':err,'sign_tolerance':tol,
        'limitations':['No generalized bias/contact balance solve. Rank is not support/equilibrium/stability.','Positive/negative means compiled coordinate sign, not anatomical flexion/extension.','Independent nonnegative activation cannot arbitrarily realize vectors implied by ordinary rank.','Single pose and zero velocity only; lengths/velocities can change active authority.','No calibrated physiological force interpretation or anatomical correspondence validation.','Missing46 need not be sole cause of collapse; all90 may also lack directions.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    assert sha(xml)==original_hash
    fig,ax=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[3,1]})
    scale=max(np.max(abs(joint_A)),1e-12)
    im=ax[0].imshow(joint_A/scale,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
    ax[0].set_yticks(range(42),joints,fontsize=6);ax[0].set_xlabel('Actuator column (all90), allowed marked below')
    for c in allowed:ax[0].plot(c,42,'k.',markersize=3,clip_on=False)
    ax[0].set_title('Active generalized-force differences / global absolute max')
    codes={'none':0,'positive_only':1,'negative_only':-1,'both':2}
    cov=np.array([[codes[r['sign_coverage']] for r in s['joint_sign_coverage']] for s in (full,sel)]).T
    ax[1].imshow(cov,aspect='auto',vmin=-1,vmax=2,cmap='viridis');ax[1].set_xticks([0,1],['all90','allowed44']);ax[1].set_yticks(range(42),['']*42)
    ax[1].set_title('Signs: -1 negative,0 none\n1 positive,2 both',fontsize=9)
    for axis in ax:axis.set_ylim(41.5,-.5)
    fig.suptitle(f'Fixed pose, NOT equilibrium: joint rank all90={full["rank"]}, allowed44={sel["rank"]}\nCoordinate signs only; no flexion labels or controller')
    fig.colorbar(im,ax=ax[0],shrink=.5);fig.subplots_adjust(left=.20,wspace=.3,top=.89,bottom=.08)
    fig.savefig(out/'authority.png',dpi=140);plt.close(fig)
    print(json.dumps({'all90_rank':full['rank'],'allowed44_rank':sel['rank'],'all90_unsupported':full['unsupported_joints'],'allowed44_unsupported':sel['unsupported_joints'],'lost_sign_coverage':lost,'maxerr':err},indent=2))
if __name__=='__main__':main()
