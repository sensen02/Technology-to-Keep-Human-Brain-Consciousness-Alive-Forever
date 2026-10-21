"""Scaled Kabsch (Umeyama) alignment of BBNPYX 3D keypoints onto the model skeleton.

Kinematics only (mj_kinematics). Answers exactly two questions:
  1) is the dataset scale factor uniquely determined (are per-leg chain ratios consistent)?
  2) what is the body-frame -> model-frame rigid+scale transform residual, per leg and per point?
No pose is generated, no XML is changed, no body/neural experiment is run, nothing is tuned.
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
FITTED=Path('/home/sensen/Desktop/engine修复验收/S2-keypoint-pose-fit-20261015T101928/candidate_pose.npz')
KEYS=['ThCx','CxTr','TrFe','FeTi','TiTar','Tar']
BODY={'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def umeyama(X,Y):
    """X: dataset (n,3) -> Y: model (n,3); returns scale,R,t and per-point residual."""
    mx,my=X.mean(0),Y.mean(0);Xc,Yc=X-mx,Y-my
    C=Yc.T@Xc/len(X);U,D,Vt=np.linalg.svd(C)
    S=np.eye(3);S[2,2]=np.sign(np.linalg.det(U@Vt)) or 1.0
    R=U@S@Vt;scale=float(np.trace(np.diag(D)@S)/np.mean(np.sum(Xc**2,axis=1)))
    t=my-scale*R@mx;P=(scale*X@R.T)+t;res=np.linalg.norm(P-Y,axis=1)
    return scale,R,t,res
def main():
    import mujoco
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-kabsch-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='SCALED_KABSCH_ALIGNMENT_ONLY_NO_POSE_GENERATED',
           questions=['is the dataset scale uniquely determined (per-leg chain ratio consistency)?','what is the body-frame to model-frame rigid+scale residual, per leg and per point?'],
           caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset readme declares no length unit','model keypoints are body origins, dataset keypoints are tracked markers (declared correspondence)','model keypoints evaluated at the canonical declared pose (primary) and at the previously fitted candidate (secondary); neither is an animal record'])
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,FITTED,cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    fk=0
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        assert np.isfinite(V).all() and V.shape[0]==334
        def col(name,f):return np.array([V[f,idx[f'{name}_{a}']] for a in 'xyz'])
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        with np.load(FITTED) as z:fitted=z['projected_seated_qpos'].copy()
        def bid(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
        def model_keys(qpos):
            nonlocal_holder[0]+=1
            d.qpos[:]=qpos;mujoco.mj_kinematics(m,d)
            RT=d.xmat[bid('Thorax')].reshape(3,3);oT=d.xpos[bid('Thorax')]
            return {ds:np.array([RT.T@(d.xpos[bid(side+bn)]-oT) for bn in BODY.values()]) for ds,side in LEGS.items()}
        nonlocal_holder=[0]
        mk_canon=model_keys(canon);mk_fit=model_keys(fitted);fk=nonlocal_holder[0]
        land={('Th-lWing' if n=='LWing' else 'Th-rWing'):(d.xmat[bid('Thorax')].reshape(3,3).T@(d.xpos[bid(n)]-d.xpos[bid('Thorax')])) for n in ('LWing','RWing')}
        # ---- Q1: scale estimability from inter-keypoint distances (rigid-invariant)
        seg_names=[f'{KEYS[i]}->{KEYS[i+1]}' for i in range(len(KEYS)-1)]
        per_leg_scale={}
        ratio_rows=[]
        for ds,side in LEGS.items():
            dm=np.array([np.linalg.norm(mk_canon[ds][i+1]-mk_canon[ds][i]) for i in range(5)]+[np.linalg.norm(mk_canon[ds][5]-mk_canon[ds][0])])
            dsig=[]
            for f in range(V.shape[0]):
                P=np.array([col(f'{ds}-{k}',f) for k in KEYS])
                dsig.append(np.array([np.linalg.norm(P[i+1]-P[i]) for i in range(5)]+[np.linalg.norm(P[5]-P[0])]))
            dsig=np.array(dsig);dmean=dsig.mean(0)
            ratios=dmean/dm
            s_ls=float(np.sum(dmean*dm)/np.sum(dm*dm))          # least-squares scale through origin
            per_leg_scale[ds]=dict(model_distances=dm.tolist(),dataset_mean_distances=dmean.tolist(),ratios=dict(zip(seg_names+['total_span'],ratios.round(4).tolist())),ratio_mean=float(ratios.mean()),ratio_std=float(ratios.std()),ratio_cv=float(ratios.std()/ratios.mean()),least_squares_scale=s_ls,shape_residual_rms=float(np.sqrt(np.mean((dmean-s_ls*dm)**2))))
            ratio_rows.append(ratios)
        scale_vals=np.array([per_leg_scale[ds]['least_squares_scale'] for ds in LEGS])
        allseg=np.concatenate(ratio_rows)
        r['q1_scale_estimability']=dict(per_leg=per_leg_scale,across_legs=dict(least_squares_scales={ds:per_leg_scale[ds]['least_squares_scale'] for ds in LEGS},mean=float(scale_vals.mean()),std=float(scale_vals.std()),min=float(scale_vals.min()),max=float(scale_vals.max()),max_over_min=float(scale_vals.max()/scale_vals.min())),all_segments=dict(mean=float(allseg.mean()),std=float(allseg.std()),cv=float(allseg.std()/allseg.mean()),min=float(allseg.min()),max=float(allseg.max())),glossary=dict(mean_msd='mean over frames of the dataset distance',least_squares_scale='sum(dm*dd)/sum(dm^2) through the origin',shape_residual_rms='residual of the distance set with the best single scale removed'))
        # ---- Q2: scaled Kabsch residual per leg (dataset frame -> model thorax frame)
        frames=list(range(0,V.shape[0],20))
        def kabsch_table(mk,label):
            per_leg={}
            allres=[]
            for ds,side in LEGS.items():
                scales=[];res_pts=[];shape=[]
                for f in frames:
                    X=np.array([col(f'{ds}-{k}',f) for k in KEYS]);Y=mk[ds]
                    s,R,t,res=umeyama(X,Y);scales.append(s);res_pts.append(res)
                    Xn=(X-X.mean(0))/np.linalg.norm(X-X.mean(0))
                    Yn=(Y-Y.mean(0))/np.linalg.norm(Y-Y.mean(0))
                    s2,R2,t2,res2=umeyama(Xn*1.0,Yn*np.linalg.norm(Y-Y.mean(0))/1.0)
                    shape.append(float(np.sqrt(np.mean(res2**2))))
                res_pts=np.array(res_pts)
                per_leg[ds]=dict(median_scale=float(np.median(scales)),scale_std=float(np.std(scales)),scale_cv=float(np.std(scales)/np.median(scales)),rms_residual=float(np.sqrt(np.mean(res_pts**2))),max_point_residual=float(res_pts.max()),per_point_median=dict(zip(KEYS,res_pts.mean(0).round(4).tolist())),shape_only_rms=float(np.median(shape)))
            # global alignment over all six legs at once
            Xg=[];Yg=[]
            for ds in LEGS:
                for f in frames[:5]:
                    Xg.append(np.array([col(f'{ds}-{k}',f) for k in KEYS]));Yg.append(mk[ds])
            Xg=np.concatenate(Xg);Yg=np.concatenate(Yg);s,R,t,res=umeyama(Xg,Yg)
            glob=dict(scale=float(s),rms_residual=float(np.sqrt(np.mean(res**2))),max_point_residual=float(res.max()),n_points=int(len(Xg)))
            return dict(pose_label=label,frames=frames,per_leg=per_leg,global_alignment=glob)
        r['q2_kabsch']=dict(canonical=kabsch_table(mk_canon,'canonical declared pose'),fitted=kabsch_table(mk_fit,'previously fitted candidate pose'))
        r['landmark_transform']=dict(formula='R from the apex/lWing/rWing triangle via Kabsch; translation maps Th-apex to the model thorax origin; scale swept separately in the fit round',model_lWing=land['Th-lWing'].tolist(),model_rWing=land['Th-rWing'].tolist(),model_apex_proxy='thorax frame origin (declared assumption)')
        r['fk_calls']=fk
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        can=r['q2_kabsch']['canonical']['per_leg']
        w=.4;xs=np.arange(len(KEYS))
        for i,(ds,side) in enumerate(LEGS.items()):
            axes[0].bar(xs+(i-2.5)*w/3,list(can[ds]['per_point_median'].values()),width=w/3,label=f'{ds}->{side}')
        axes[0].set_xticks(xs);axes[0].set_xticklabels(KEYS,fontsize=7);axes[0].set_ylabel('median Kabsch point residual (model units)');axes[0].set_title('per-point residual after per-leg scaled Kabsch (canonical pose)');axes[0].legend(fontsize=6,ncol=2)
        ax=axes[1]
        segs=[f'{KEYS[i]}->{KEYS[i+1]}' for i in range(5)]+['total_span']
        for ds in LEGS:ax.plot(range(6),[r['q1_scale_estimability']['per_leg'][ds]['ratios'][s] for s in segs],'o-',label=ds)
        ax.axhline(1.0,color='k',lw=.8);ax.set_xticks(range(6));ax.set_xticklabels(segs,fontsize=6,rotation=30);ax.set_ylabel('dataset / model segment length');ax.set_title('scale estimability: per-segment ratios (flat line = consistent)');ax.legend(fontsize=6,ncol=3)
        fig.suptitle('BBNPYX keypoints vs model skeleton — alignment diagnostics only, NOT standing',fontsize=9);fig.tight_layout();fig.savefig(out/'kabsch-deviation.png');plt.close(fig)
        r['status']='PASS_ALIGNMENT_DIAGNOSTICS_REPORTED'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
