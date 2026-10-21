"""Keypoint fit v5: NOMINAL scale only, hard-fail body resolution, 6x6 name->id table, start census.

RESEARCH SCRIPT, NOT A PRODUCT. Kinematics only (mj_kinematics). No dynamics, no contact
integration, no neural simulation, no canonical XML change, no pose/support search.

Defect fixed here (separate from any wording change):
  mid/hind legs have no <side>Trochanter body; the trochanter joints are declared on
  <side>Femur (XML 319/346). v1 looked the body up by name without checking the id, so
  mj_name2id returned -1 and d.xpos[-1] (=RWing, a thorax-fixed point) was silently used
  for those legs' CxTr keypoint. v5 resolves the trochanter keypoint to the child body that
  carries the trochanter joint, asserts id >= 0 AND that the body belongs to that leg
  (never thorax/wing), and prints the full 6x6 name->id->body-name table.
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
NOMINAL=1.0;STRIDE=20;N_STARTS=21
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
# keypoint -> body suffix. 'Trochanter' is resolved to the child body that actually carries
# the trochanter joint: a dedicated <side>Trochanter body for LF/RF, <side>Femur for LM/LH/RM/RH.
MAP={'LF':{'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RF':{'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'LM':{'ThCx':'Coxa','CxTr':'Femur','TrFe':None,'FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'LH':{'ThCx':'Coxa','CxTr':'Femur','TrFe':None,'FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RM':{'ThCx':'Coxa','CxTr':'Femur','TrFe':None,'FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RH':{'ThCx':'Coxa','CxTr':'Femur','TrFe':None,'FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from scipy.optimize import least_squares
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-fit-v5-nominal-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='KEYPOINT_FIT_V5_NOMINAL_SCALE_HARD_FAIL_RESOLUTION_PLUS_START_CENSUS',research_script_not_product=True,
        caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset readme declares no length unit; only the nominal scale 1.0 is fitted this round',
                 'the animal is not this model specimen: a data-driven candidate posture, not a measured stance','fit target is keypoint residual only; no support/duration/contact criterion was used'],
        nominal_scale=NOMINAL,frame_stride=STRIDE,n_starts=N_STARTS,
        start_policy='start 0 = canonical keyframe joints; start k (1..20) = clip(canonical + 0.4*sin(2*pi*k/20 + joint_index), joint_range) - deterministic, no RNG')
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    fk=[0]
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        col=lambda n,f:np.array([V[f,idx[f'{n}_{a}']] for a in 'xyz'])
        assert V.shape==(334,len(cols)) and np.isfinite(V).all()
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        BAD={'LWing','RWing','Thorax'}
        # ---- the 6 x 6 name -> id -> body-name table (required deliverable)
        table=[]
        for ds,side in LEGS.items():
            for key,suffix in MAP[side].items():
                if suffix is None:
                    table.append(dict(leg=ds,model_side=side,keypoint=key,requested_body=None,resolved_body=None,id=None,used=False,
                        reason='DECLARED_MISSING: the model has no separate trochanter-femur body for this leg type, so the dataset TrFe keypoint is not constrained'))
                    continue
                name=side+suffix;bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
                if bid<0:raise ValueError(f'{ds}/{key}: body {name} does not exist (id={bid}) - refusing to continue')
                actual=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)
                if actual!=name:raise ValueError(f'{ds}/{key}: resolved id {bid} is body {actual}, expected {name}')
                if actual in BAD or not actual.startswith(side):raise ValueError(f'{ds}/{key}: resolved body {actual} does not belong to leg {side}')
                table.append(dict(leg=ds,model_side=side,keypoint=key,requested_body=name,resolved_body=actual,id=int(bid),used=True,reason=None))
        r['keypoint_resolution_table_6x6']=table
        r['keypoint_resolution_check']=dict(rows=len(table),used=sum(1 for t in table if t['used']),declared_missing=sum(1 for t in table if not t['used']),
            assertions='id >= 0, resolved name equals requested name, body belongs to the leg prefix, body is not thorax/wing',
            trochanter_child_body_evidence='LF/RF: <body name="LFTrochanter"> with joint_LFTrochanter_* pos="0 0 0" (XML 223/225-227). LM/LH/RM/RH: joint_<side>Trochanter_* declared on <side>Femur with pos="0 0 0" (XML 319/346), so the Femur origin is that joint anchor.')
        print('resolution table built:',r['keypoint_resolution_check'],flush=True)
        used_map={side:{t['keypoint']:t['resolved_body'] for t in table if t['model_side']==side and t['used']} for side in MAP}
        def leg_joints(side):
            js=[]
            for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
                for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
                    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{side}{grp}_{a}')
                    if j<0:raise ValueError(f'joint missing {side}{grp}_{a}')
                    js.append(j)
            return js
        def keypoints(side,angles):
            fk[0]+=1;d.qpos[:]=canon
            for j,a in zip(leg_joints(side),angles):d.qpos[m.jnt_qposadr[j]]=a
            mujoco.mj_kinematics(m,d)
            bidT=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax');RT=d.xmat[bidT].reshape(3,3);oT=d.xpos[bidT]
            pts=[]
            for key,bn in used_map[side].items():
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,bn)
                if bid<0:raise ValueError(f'{side}/{key}: body {bn} unresolved at fit time')
                pts.append(RT.T@(d.xpos[bid]-oT))
            return np.array(pts)
        frames=list(range(0,V.shape[0],STRIDE))
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
        bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
        land={k:d.xmat[bid('Thorax')].reshape(3,3).T@(d.xpos[bid(b)]-d.xpos[bid('Thorax')]) for k,b in (('Th-lWing','LWing'),('Th-rWing','RWing'))}
        def R_of(frame):
            a=col('Th-lWing',frame);bb=col('Th-rWing',frame);X=np.stack([a,np.zeros(3),bb]);Y=np.stack([land['Th-lWing'],np.zeros(3),land['Th-rWing']])
            Xc=X-X.mean(0);Yc=Y-Y.mean(0);U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T))
            return Vt.T@np.diag([1,1,dd])@U.T,float(np.sqrt(np.mean(np.sum((X@(Vt.T@np.diag([1,1,dd])@U.T).T-Y)**2,axis=1))))
        fit={};census={}
        for ds,side in LEGS.items():
            js=leg_joints(side);lo=[];hi=[]
            for j in js:
                a0,a1=m.jnt_range[j]
                if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                lo.append(a0);hi.append(a1)
            lo=np.array(lo);hi=np.array(hi);x0=canon[[m.jnt_qposadr[j] for j in js]]
            keys=list(used_map[side]);per_frame={}
            for frame in frames:
                R,lres=R_of(frame);tgt=np.array([NOMINAL*R@col(f'{ds}-{k}',frame) for k in keys])
                rms=[]
                for k in range(N_STARTS):
                    start=x0 if k==0 else np.clip(x0+0.4*np.sin(2*np.pi*k/20+np.arange(len(x0))),lo,hi)
                    sol=least_squares(lambda x:(keypoints(side,x)-tgt).ravel(),start,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
                    rms.append(dict(k=k,rms=float(np.sqrt(np.mean(sol.fun**2))),angles=sol.x.tolist(),nfev=int(sol.nfev)))
                best=min(rms,key=lambda x:x['rms']);per_frame[frame]=dict(canonical_start=rms[0],best_over_starts=best,all_start_rms=[round(x['rms'],4) for x in rms])
            A=np.array([per_frame[f]['canonical_start']['angles'] for f in frames])
            med=np.median(A,axis=0)
            fit[ds]=dict(model_side=side,keypoints=keys,n_keypoints=len(keys),frames=frames,
                per_frame_rms=[per_frame[f]['canonical_start']['rms'] for f in frames],
                per_frame_rms_best_over_starts=[per_frame[f]['best_over_starts']['rms'] for f in frames],
                all_start_rms_per_frame={str(f):per_frame[f]['all_start_rms'] for f in frames},
                median_angles=med.tolist(),rms_median_frame=float(np.median([per_frame[f]['canonical_start']['rms'] for f in frames])),
                rms_worst_frame=float(np.max([per_frame[f]['canonical_start']['rms'] for f in frames])),
                rms_median_frame_best_over_starts=float(np.median([per_frame[f]['best_over_starts']['rms'] for f in frames])),
                rms_worst_frame_best_over_starts=float(np.max([per_frame[f]['best_over_starts']['rms'] for f in frames])),
                angle_spread_max_deg=float(np.degrees(np.max(A.max(0)-A.min(0)))),
                joint_range_violations=int(np.sum((med<lo-1e-9)|(med>hi+1e-9))))
            print('leg',ds,'canonical-start median',round(fit[ds]['rms_median_frame'],4),'worst',round(fit[ds]['rms_worst_frame'],4),
                  '| best-over-starts median',round(fit[ds]['rms_median_frame_best_over_starts'],4),'worst',round(fit[ds]['rms_worst_frame_best_over_starts'],4),flush=True)
        r['fit']=fit;r['fk_calls']=fk[0]
        r['local_minimum_disclosure']=dict(
            statement='the reported residuals are LOCAL solutions from the canonical start; the start census (21 deterministic starts) shows better local minima exist, so the canonical-start values are upper bounds, not global optima.',
            canonical_start_median={ds:round(fit[ds]['rms_median_frame'],4) for ds in LEGS},
            canonical_start_worst={ds:round(fit[ds]['rms_worst_frame'],4) for ds in LEGS},
            best_over_starts_median={ds:round(fit[ds]['rms_median_frame_best_over_starts'],4) for ds in LEGS},
            best_over_starts_worst={ds:round(fit[ds]['rms_worst_frame_best_over_starts'],4) for ds in LEGS})
        cand=canon.copy();viol=[]
        for ds,side in LEGS.items():
            for j,val in zip(leg_joints(side),fit[ds]['median_angles']):
                a0,a1=m.jnt_range[j]
                if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                if val<a0-1e-9 or val>a1+1e-9:viol.append(dict(joint=m.joint(j).name,value=float(val)))
                cand[m.jnt_qposadr[j]]=min(max(val,a0),a1)
        r['candidate_pose']=dict(qpos=cand.tolist(),joint_range_violations=viol,source='v5 canonical-start median angles at nominal scale')
        np.savez_compressed(out/'v5_candidate_pose.npz',projected_seated_qpos=cand)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        xs=np.arange(6);w=.35
        axes[0].bar(xs-w/2,[fit[ds]['rms_median_frame'] for ds in LEGS],width=w,label='canonical start')
        axes[0].bar(xs+w/2,[fit[ds]['rms_median_frame_best_over_starts'] for ds in LEGS],width=w,label='best over 21 starts')
        axes[0].set_xticks(xs);axes[0].set_xticklabels([f'{d}->{LEGS[d]}' for d in LEGS],fontsize=7)
        axes[0].set(ylabel='median-frame keypoint RMS (model units)',title='v5 nominal scale 1.0, corrected trochanter resolution');axes[0].legend(fontsize=7)
        for ds in LEGS:
            axes[1].plot(fit[ds]['frames'],fit[ds]['per_frame_rms'],'o-',ms=3,lw=1,label=ds)
            axes[1].plot(fit[ds]['frames'],fit[ds]['per_frame_rms_best_over_starts'],'x--',ms=3,lw=.8)
        axes[1].set(xlabel='dataset frame',ylabel='keypoint RMS (model units)',title='per-frame RMS: canonical start (solid) vs best of 21 starts (dashed)');axes[1].legend(fontsize=6,ncol=3)
        fig.suptitle('v5 keypoint fit after the trochanter-body fix - kinematics only; NOT standing, NOT a measured stance',fontsize=9)
        fig.tight_layout();fig.savefig(out/'keypoint-fit-v5.png');plt.close(fig)
        r['status']='PASS_V5_NOMINAL_WITH_RESOLUTION_TABLE'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
