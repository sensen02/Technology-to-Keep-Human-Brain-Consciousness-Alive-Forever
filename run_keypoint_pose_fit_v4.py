"""Keypoint pose fit v4: three declared keypoint-map variants, hard-fail name resolution.

RESEARCH SCRIPT, NOT A PRODUCT. Kinematics only (mj_kinematics); no dynamics, no contact,
no neural simulation, no canonical XML change, no pose or support search.

Variants (all declared BEFORE looking at residuals):
  V0_defective_prefix : v1's behaviour - <side>Trochanter looked up by name; for the mid/hind legs
                        that body does not exist and the original code silently used d.xpos[-1]
                        (=RWing). Reproduced here ON PURPOSE only as the comparison baseline.
  V1_cxtr_femur       : all six legs constrain CxTr; mid/hind map CxTr -> <side>Femur because the
                        model's joint_<side>Trochanter_* joints are declared on that body with
                        pos="0 0 0" (XML 319/346), so the Femur origin IS the joint anchor.
                        The dataset TrFe point then has no counterpart there (declared missing, 5 pts).
  V2_trfe_femur       : mid/hind treat <side>Femur as the dataset TrFe point and DROP CxTr
                        (5 pts) - the independent reviewer's alternative reading.

Any unresolved body name raises immediately; nothing may fall back to xpos[-1].
"""
from pathlib import Path
import json,time,hashlib,zipfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
R30=ROOT/'outputs/stance_joint_feasible_round30/stance_for_runner.json'
SCALES=[0.1,0.25,0.4,0.5,0.75,1.0,1.25,1.5,2.0]
NOMINAL=1.0;STRIDE=20
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
FRONT={'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
VARIANTS={
 'V0_defective_prefix':{'LF':FRONT,'RF':FRONT,'LM':FRONT,'LH':FRONT,'RM':FRONT,'RH':FRONT},
 'V1_cxtr_femur':{'LF':FRONT,'RF':FRONT,
   'LM':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'LH':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'RM':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'RH':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}},
 'V2_trfe_femur':{'LF':FRONT,'RF':FRONT,
   'LM':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'LH':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'RM':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
   'RH':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from scipy.optimize import least_squares
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-pose-fit-v4-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='KEYPOINT_FIT_V4_THREE_DECLARED_VARIANTS',research_script_not_product=True,
        caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset readme declares no length unit, hence the scale sweep',
                 'animal is not this model specimen: a data-driven candidate posture, not a measured stance','fit target is keypoint residual only; no support/duration/contact criterion was used'],
        declared_variants={k:{s:dict(v) for s,v in tab.items()} for k,tab in VARIANTS.items()},scales=SCALES,nominal=NOMINAL,stride=STRIDE)
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML,R30]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    fk=[0]
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        assert V.shape==(334,len(cols)) and np.isfinite(V).all()
        col=lambda n,f:np.array([V[f,idx[f'{n}_{a}']] for a in 'xyz'])
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        # ---- self-check: resolve every body name for every variant BEFORE any fitting
        resolution={}
        for vname,tab in VARIANTS.items():
            resolution[vname]={}
            for side,spec in tab.items():
                resolution[vname][side]={}
                for key,bn in spec.items():
                    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+bn)
                    if vname=='V0_defective_prefix' and bid<0:
                        resolution[vname][side][key]=dict(name=side+bn,id=int(bid),
                            note='NEGATIVE ID on purpose: this reproduces v1, which then indexed d.xpos[-1] (=RWing). This variant is a baseline only and is NOT delivered.')
                    else:
                        if bid<0:raise ValueError(f'{vname}: body {side+bn} for keypoint {key} does not exist (id={bid}) - refusing to fit')
                        resolution[vname][side][key]=dict(name=side+bn,id=int(bid))
        r['body_resolution_selfcheck']=resolution
        print('self-check: all non-baseline body names resolved',flush=True)
        def keypoints(side,spec,angles,variant=None):
            fk[0]+=1;d.qpos[:]=canon
            for j,a in zip(leg_joints(side),angles):d.qpos[m.jnt_qposadr[j]]=a
            mujoco.mj_kinematics(m,d)
            RT=d.xmat[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')].reshape(3,3)
            oT=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')]
            out_pts=[]
            for key,bn in spec.items():
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+bn)
                if bid<0 and variant!='V0_defective_prefix':
                    raise ValueError(f'unresolved body {side+bn} at fit time - refusing xpos[-1] fallback')
                out_pts.append(RT.T@(d.xpos[bid]-oT))   # for V0 bid=-1 reproduces v1's d.xpos[-1] on purpose
            return np.array(out_pts)
        def leg_joints(side):
            js=[]
            for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
                for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
                    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{side}{grp}_{a}')
                    assert j>=0,f'joint missing {side}{grp}_{a}';js.append(j)
            return js
        frames=list(range(0,V.shape[0],STRIDE))
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
        bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
        land={k:d.xmat[bid('Thorax')].reshape(3,3).T@(d.xpos[bid(b)]-d.xpos[bid('Thorax')]) for k,b in (('Th-lWing','LWing'),('Th-rWing','RWing'))}
        def R_of(frame):
            a=col('Th-lWing',frame);bb=col('Th-rWing',frame);X=np.stack([a,np.zeros(3),bb]);Y=np.stack([land['Th-lWing'],np.zeros(3),land['Th-rWing']])
            Xc=X-X.mean(0);Yc=Y-Y.mean(0);U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T))
            lres=float(np.sqrt(np.mean(np.sum((X@(Vt.T@np.diag([1,1,dd])@U.T).T-Y)**2,axis=1))))
            return Vt.T@np.diag([1,1,dd])@U.T,lres
        fit={}
        for vname,tab in VARIANTS.items():
            fit[vname]={}
            for scale in SCALES:
                fit[vname][str(scale)]={}
                for ds,side in LEGS.items():
                    spec=tab[side];js=leg_joints(side);lo=[];hi=[]
                    for j in js:
                        a0,a1=m.jnt_range[j]
                        if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                        lo.append(a0);hi.append(a1)
                    lo=np.array(lo);hi=np.array(hi);x0=canon[[m.jnt_qposadr[j] for j in js]]
                    per=[]
                    for frame in frames:
                        R,lres=R_of(frame);tgt=np.array([scale*R@col(f'{ds}-{k}',frame) for k in spec])
                        sol=least_squares(lambda x:(keypoints(side,spec,x,vname)-tgt).ravel(),x0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
                        res=sol.fun.reshape(-1,3)
                        per.append(dict(frame=frame,rms=float(np.sqrt(np.mean(res**2))),max=float(np.abs(res).max()),nfev=int(sol.nfev),status=int(sol.status),angles=sol.x.tolist()))
                    A=np.array([p['angles'] for p in per]);med=np.median(A,axis=0)
                    fit[vname][str(scale)][ds]=dict(model_side=side,keypoints=list(spec),n_keypoints=len(spec),
                        frames=[p['frame'] for p in per],per_frame_rms=[p['rms'] for p in per],per_frame_max=[p['max'] for p in per],
                        per_frame_nfev=[p['nfev'] for p in per],per_frame_status=[p['status'] for p in per],median_angles=med.tolist(),
                        rms_median_frame=float(np.median([p['rms'] for p in per])),rms_worst_frame=float(np.max([p['rms'] for p in per])),
                        angle_spread_max_deg=float(np.degrees(np.max(A.max(0)-A.min(0)))),
                        joint_range_violations=int(np.sum((med<lo-1e-9)|(med>hi+1e-9))))
                print('variant',vname,'scale',scale,'done',flush=True)
        r['fit']=fit;r['fk_calls']=fk[0]
        r['summary_scale1']={v:{ds:dict(median=round(fit[v]['1.0'][ds]['rms_median_frame'],4),worst=round(fit[v]['1.0'][ds]['rms_worst_frame'],4),n_keypoints=fit[v]['1.0'][ds]['n_keypoints']) for ds in LEGS} for v in VARIANTS}
        # chain span ratios: model at corrected candidate pose and at canonical; dataset frame0 and 334-frame median
        def span_model(side,qpos):
            d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);RT=d.xmat[bid('Thorax')].reshape(3,3);oT=d.xpos[bid('Thorax')]
            a=RT.T@(d.xpos[bid(side+'Coxa')]-oT)
            tar=None
            for cand in ('Tarsus5','Tarsus1'):
                if bid(side+cand)>=0:tar=RT.T@(d.xpos[bid(side+cand)]-oT)
            return float(np.linalg.norm(tar-a))
        def span_dataset(ds,mode):
            vals=[]
            fr=[0] if mode=='frame0' else range(V.shape[0])
            for f in fr:vals.append(float(np.linalg.norm(col(f'{ds}-Tar',f)-col(f'{ds}-ThCx',f))))
            return float(np.median(vals))
        cands={}
        for vname in ('V1_cxtr_femur','V2_trfe_femur'):
            c=canon.copy()
            for ds,side in LEGS.items():
                for j,val in zip(leg_joints(side),fit[vname]['1.0'][ds]['median_angles']):
                    a0,a1=m.jnt_range[j]
                    if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                    c[m.jnt_qposadr[j]]=min(max(val,a0),a1)
            cands[vname]=c
            np.savez_compressed(out/(vname+'_candidate_pose.npz'),projected_seated_qpos=c)
        np.savez_compressed(out/'canonical_pose.npz',projected_seated_qpos=canon)
        r['chain_span_ratios']={v:{ds:dict(ratio_frame0=round(span_dataset(ds,'frame0')/span_model(side,cands[v]),4),
                                          ratio_median334=round(span_dataset(ds,'median334')/span_model(side,cands[v]),4)) for ds,side in LEGS.items()} for v in cands}
        r['candidate_poses']={v:str(out/(v+'_candidate_pose.npz')) for v in cands}
        r['delivered_variant']=dict(chosen='V1_cxtr_femur',
            reason='the XML declares joint_<side>Trochanter_* on <side>Femur with pos="0 0 0" (lines 319/346; front legs have <side>Trochanter at 223-227), so the Femur origin IS the coxa-trochanter joint anchor. This mapping choice was fixed from the XML BEFORE any residuals were inspected; V2 (drop CxTr, keep TrFe) and V0 (defective prefix lookup) are reported for transparency and are not used to pick the candidate.',
            v0_note='V0 reproduces v1 exactly (including the xpos[-1] fallback) and is a baseline only; it must never be delivered.')
        r['status']='PASS_KEYPOINT_FIT_V4_THREE_VARIANTS'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r.get('summary_scale1',{}),indent=1));print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
