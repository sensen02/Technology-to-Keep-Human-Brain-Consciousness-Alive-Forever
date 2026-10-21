"""Keypoint pose fit v3: explicit leg->body lookup table with existence asserts.

RESEARCH SCRIPT, NOT A PRODUCT. Kinematics only (mj_kinematics), no dynamics, no contact,
no neural simulation, no canonical XML change, no pose/support search.

Fix over v1: v1 looked up body names by string prefix and never checked mj_name2id, so for the
mid/hind legs (which have NO <side>Trochanter body) the model CxTr keypoint silently resolved to
d.xpos[-1] = RWing. v3 uses an explicit table and asserts every id >= 0.
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
SCALES=[0.1,0.25,0.4,0.5,0.75,1.0,1.25,1.5,2.0]
NOMINAL=1.0;STRIDE=20
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
# Explicit table. Front legs have a separate Trochanter body; mid/hind legs carry the
# joint_LM/LH/RM/RHTrochanter_* joints on their Femur body (pos="0 0 0"), so the Femur origin
# IS the coxa-trochanter joint anchor there and the dataset TrFe keypoint has no counterpart.
KEYPOINT_BODY={
 'LF':{'ThCx':'LFCoxa','CxTr':'LFTrochanter','TrFe':'LFFemur','FeTi':'LFTibia','TiTar':'LFTarsus1','Tar':'LFTarsus5'},
 'RF':{'ThCx':'RFCoxa','CxTr':'RFTrochanter','TrFe':'RFFemur','FeTi':'RFTibia','TiTar':'RFTarsus1','Tar':'RFTarsus5'},
 'LM':{'ThCx':'LMCoxa','CxTr':'LMFemur','FeTi':'LMTibia','TiTar':'LMTarsus1','Tar':'LMTarsus5'},
 'LH':{'ThCx':'LHCoxa','CxTr':'LHFemur','FeTi':'LHTibia','TiTar':'LHTarsus1','Tar':'LHTarsus5'},
 'RM':{'ThCx':'RMCoxa','CxTr':'RMFemur','FeTi':'RMTibia','TiTar':'RMTarsus1','Tar':'RMTarsus5'},
 'RH':{'ThCx':'RHCoxa','CxTr':'RHFemur','FeTi':'RHTibia','TiTar':'RHTarsus1','Tar':'RHTarsus5'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from scipy.optimize import least_squares
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-pose-fit-v3-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='DATA_DRIVEN_KEYPOINT_FIT_V3_EXPLICIT_BODY_TABLE',
        research_script_not_product=True,
        caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset readme declares no length unit, hence the scale sweep',
                 'the animal is not this model specimen: a data-driven candidate posture, not a measured stance','fit target is keypoint residual only; support margin/duration/contact count were NOT used to select anything'],
        declared_keypoint_body_table=KEYPOINT_BODY,scales=SCALES,nominal_scale=NOMINAL,frame_stride=STRIDE)
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    fk=[0]
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        assert V.shape==(334,len(cols)) and np.isfinite(V).all()
        def col(name,f):return np.array([V[f,idx[f'{name}_{a}']] for a in 'xyz'])
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        ids={}
        for side,tab in KEYPOINT_BODY.items():
            for key,bname in tab.items():
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,bname)
                assert bid>=0,f'missing body {bname} for {side}/{key}'   # fail loudly, never fall back
                ids[(side,key)]=bid
        r['resolved_body_ids']={f'{s}:{k}':int(v) for (s,k),v in ids.items()}
        def keypoints(side,angles):
            fk[0]+=1;d.qpos[:]=canon
            for j,a in zip(leg_joints(side),angles):d.qpos[m.jnt_qposadr[j]]=a
            mujoco.mj_kinematics(m,d)
            RT=d.xmat[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')].reshape(3,3)
            oT=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')]
            return np.array([RT.T@(d.xpos[ids[(side,k)]]-oT) for k in KEYPOINT_BODY[side]])
        def leg_joints(side):
            js=[]
            for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
                for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
                    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{side}{grp}_{a}')
                    assert j>=0,f'joint missing {side}{grp}_{a}';js.append(j)
            return js
        frames=list(range(0,V.shape[0],STRIDE))
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d)   # REQUIRED before reading xpos/xmat for the landmarks
        land={k:d.xmat[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')].reshape(3,3).T@(
                 d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b)]-d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')])
              for k,b in (('Th-lWing','LWing'),('Th-rWing','RWing'))}
        land['Th-apex']=np.zeros(3)
        def transform(scale,frame):
            a=col('Th-lWing',frame);bb=col('Th-rWing',frame)
            X=np.stack([a,np.zeros(3),bb]);Y=np.stack([land['Th-lWing'],np.zeros(3),land['Th-rWing']])
            Xc=X-X.mean(0);Yc=Y-Y.mean(0)
            U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T));R=Vt.T@np.diag([1,1,dd])@U.T
            lres=float(np.sqrt(np.mean(np.sum((scale*X@R.T-Y)**2,axis=1))))
            return R,np.zeros(3),lres
        fit={}
        for scale in SCALES:
            fit[str(scale)]={}
            for ds,side in LEGS.items():
                js=leg_joints(side);lo=[];hi=[]
                for j in js:
                    a0,a1=m.jnt_range[j]
                    if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                    lo.append(a0);hi.append(a1)
                lo=np.array(lo);hi=np.array(hi);x0=canon[[m.jnt_qposadr[j] for j in js]]
                per=[]
                for frame in frames:
                    R,t,lres=transform(scale,frame)
                    keys=list(KEYPOINT_BODY[side]);tgt=np.array([scale*R@col(f'{ds}-{k}',frame)+t for k in keys])
                    sol=least_squares(lambda x:(keypoints(side,x)-tgt).ravel(),x0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
                    res=sol.fun.reshape(-1,3)
                    per.append(dict(frame=frame,rms=float(np.sqrt(np.mean(res**2))),max=float(np.abs(res).max()),nfev=int(sol.nfev),status=int(sol.status),angles=sol.x.tolist(),landmark_residual=lres))
                A=np.array([p['angles'] for p in per]);med=np.median(A,axis=0)
                fit[str(scale)][ds]=dict(model_side=side,keypoints=list(KEYPOINT_BODY[side]),n_keypoints=len(KEYPOINT_BODY[side]),
                    frames=[p['frame'] for p in per],per_frame_rms=[p['rms'] for p in per],per_frame_max=[p['max'] for p in per],
                    per_frame_nfev=[p['nfev'] for p in per],per_frame_status=[p['status'] for p in per],
                    median_angles=med.tolist(),median_angles_deg=[float(np.degrees(x)) for x in med],
                    rms_median_frame=float(np.median([p['rms'] for p in per])),rms_worst_frame=float(np.max([p['rms'] for p in per])),
                    angle_spread_max_deg=float(np.degrees(np.max(A.max(0)-A.min(0)))),landmark_residual_median=float(np.median([p['landmark_residual'] for p in per])),
                    joint_range_violations=int(np.sum((med<lo-1e-9)|(med>hi+1e-9))))
            print('scale',scale,'done',flush=True)
        r['fit']=fit;r['fk_calls']=fk[0]
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
        span={ds:float(np.linalg.norm(d.xpos[ids[(side,'Tar')]]-d.xpos[ids[(side,'ThCx')]])) for ds,side in LEGS.items()}
        dspan={ds:float(np.linalg.norm(col(f'{ds}-Tar',0)-col(f'{ds}-ThCx',0))) for ds in LEGS}
        r['chain_span_diagnostic']=dict(model=span,dataset_frame0=dspan,ratio={k:round(dspan[k]/span[k],4) for k in span})
        nom=fit[str(NOMINAL)];cand=canon.copy();viol=[]
        for ds,side in LEGS.items():
            for j,val in zip(leg_joints(side),nom[ds]['median_angles']):
                a0,a1=m.jnt_range[j]
                if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                if val<a0-1e-9 or val>a1+1e-9:viol.append(dict(joint=m.joint(j).name,value=float(val),range=[float(a0),float(a1)]))
                cand[m.jnt_qposadr[j]]=min(max(val,a0),a1)
        r['candidate_pose']=dict(qpos=cand.tolist(),joint_range_violations=viol)
        np.savez_compressed(out/'candidate_pose.npz',projected_seated_qpos=cand)
        d.qpos[:]=cand;mujoco.mj_kinematics(m,d)
        feet=[ids[(side,'Tar')] for side in LEGS.values()]
        r['candidate_fact_report']=dict(tarsus5_z_min_max=[float(min(d.xpos[f][2] for f in feet)),float(max(d.xpos[f][2] for f in feet))],note='fact only; no support search, no contact count used for selection')
        # per-scale summary table for the report
        r['median_rms_by_scale']={s:{ds:round(fit[s][ds]['rms_median_frame'],4) for ds in LEGS} for s in fit}
        r['worst_frame_rms_by_scale']={s:{ds:round(fit[s][ds]['rms_worst_frame'],4) for ds in LEGS} for s in fit}
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ds in LEGS:axes[0].plot(SCALES,[fit[str(s)][ds]['rms_median_frame'] for s in SCALES],'o-',label=f'{ds}->{LEGS[ds]}')
        axes[0].set(xlabel='declared dataset scale',ylabel='median-frame keypoint RMS (model units)',title='v3 fit residual vs declared scale (whole curve, corrected lookup)');axes[0].legend(fontsize=7)
        frame=frames[len(frames)//2];R,t,_=transform(NOMINAL,frame)
        tgt=np.array([NOMINAL*R@col(f'L2-{k}',frame)+t for k in KEYPOINT_BODY['LM']]);pts=keypoints('LM',np.array(nom['L2']['median_angles']))
        axes[1].plot(tgt[:,1],tgt[:,2],'o-',color='tab:blue',label='dataset L2 keypoints (transformed)')
        axes[1].plot(pts[:,1],pts[:,2],'x--',color='tab:red',label='model LM keypoints (v3 fit)')
        axes[1].set(xlabel='thorax-frame y',ylabel='thorax-frame z',title=f'LM keypoints, scale={NOMINAL}, frame={frame} (corrected CxTr->Femur)');axes[1].legend(fontsize=7)
        fig.suptitle('v3 data-driven keypoint fit (kinematics only) - NOT standing, NOT a measured stance',fontsize=9);fig.tight_layout();fig.savefig(out/'keypoint-fit.png');plt.close(fig)
        r['status']='PASS_KEYPOINT_FIT_V3_ALL_SCALES'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print('fk',r.get('fk_calls'));print(json.dumps(r.get('median_rms_by_scale',{}).get('1.0',{})));print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
