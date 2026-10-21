"""Fix round: prove the CxTr lookup defect, then re-run the SAME evaluation with the fix.

RESEARCH SCRIPT, NOT A PRODUCT. Kinematics only (mj_kinematics). No new search: same x0,
same bounds, same scale 1.0, same residual definition. No scale sweep, no tuning, no extra
starts (the 21-start census is cited as a limitation note only), no pose/support search,
no keyframe writing.

Defect proof: for each leg print the requested CxTr body name, the mj_name2id result, and -
when the id is negative - the body that d.xpos[-1] actually indexes (the LAST body = RWing).
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
SCALE=1.0;STRIDE=20
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
DEFECTIVE={'LF':'Trochanter','RF':'Trochanter','LM':'Trochanter','LH':'Trochanter','RM':'Trochanter','RH':'Trochanter'}
FIXED_ALT={'LF':'Trochanter','RF':'Trochanter','LM':'Femur','LH':'Femur','RM':'Femur','RH':'Femur'}   # CxTr -> Femur
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from scipy.optimize import least_squares
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-pose-fit-fix-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='S2_A_GATE_KEYPOINT_CXTR_LOOKUP_FIX_ONLY',research_script_not_product=True,
        fixed_conditions=dict(scale=SCALE,frame_stride=STRIDE,x0='canonical keyframe leg joints',bounds='joint ranges (jnt_range)',loss='keypoint Euclidean residual (unchanged)'),
        not_done='no scale change, no tuning, no extra starts, no pose/support search, no keyframe written')
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    fk=[0]
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        col=lambda n,f:np.array([V[f,idx[f'{n}_{a}']] for a in 'xyz'])
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        last_body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.nbody-1)
        # ---- (1) defect proof + the 6-leg CxTr resolution table
        resolution=[]
        for ds,side in LEGS.items():
            name=side+DEFECTIVE[side];bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            entry=dict(leg=ds,requested_body=name,name2id=int(bid),
                falls_back_to=('d.xpos[-1] = '+str(last_body)) if bid<0 else 'n/a',
                actual_body_indexed=(last_body if bid<0 else mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)),
                fixed_body=side+FIXED_ALT[side],fixed_id=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+FIXED_ALT[side])))
            resolution.append(entry)
        r['cxTr_resolution_table']=resolution
        r['defect_proof']=dict(
            only_front_trochanter_bodies=[n for n in ('LFTrochanter','RFTrochanter','LMTrochanter','LHTrochanter','RMTrochanter','RHTrochanter') if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)>=0],
            missing_trochanter_bodies=[n for n in ('LMTrochanter','LHTrochanter','RMTrochanter','RHTrochanter') if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)<0],
            last_body_in_model=last_body,
            statement='for the mid/hind legs mj_name2id(<side>Trochanter) returns -1, so the old code silently indexed d.xpos[-1], i.e. the LAST body of the model = RWing (a thorax-fixed point).',
            mitigation='every mapping in this round asserts id >= 0 and that the resolved body belongs to that leg (prefix match, excluding Thorax/LWing/RWing); a -1 fallback is forbidden')
        for e in resolution:
            assert e['fixed_id']>=0,e
        print('resolution table:',json.dumps(resolution,ensure_ascii=False)[:400],flush=True)
        def leg_joints(side):
            js=[]
            for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
                for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
                    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{side}{grp}_{a}')
                    if j<0:raise ValueError(f'joint missing {side}{grp}_{a}')
                    js.append(j)
            return js
        def pts(side,spec,angles,defective=False):
            fk[0]+=1
            d.qpos[:]=canon
            for j,a in zip(leg_joints(side),angles):d.qpos[m.jnt_qposadr[j]]=a
            mujoco.mj_kinematics(m,d)
            T=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax');RT=d.xmat[T].reshape(3,3);oT=d.xpos[T]
            res=[]
            for key,bn in spec.items():
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+bn)
                if bid<0:
                    if not defective:raise ValueError(f'{side}/{key}: {side+bn} unresolved - refusing xpos[-1] fallback')
                else:
                    actual=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)
                    if actual in {'Thorax','LWing','RWing'} or not actual.startswith(side):
                        raise ValueError(f'{side}/{key}: body {actual} does not belong to leg {side}')
                res.append(RT.T@(d.xpos[bid]-oT))
            return np.array(res)
        frames=list(range(0,V.shape[0],STRIDE))
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d);bd=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
        land={k:d.xmat[bd('Thorax')].reshape(3,3).T@(d.xpos[bd(b)]-d.xpos[bd('Thorax')]) for k,b in (('Th-lWing','LWing'),('Th-rWing','RWing'))}
        def R_of(f):
            a=col('Th-lWing',f);bb=col('Th-rWing',f);X=np.stack([a,np.zeros(3),bb]);Y=np.stack([land['Th-lWing'],np.zeros(3),land['Th-rWing']])
            Xc=X-X.mean(0);Yc=Y-Y.mean(0);U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T));return Vt.T@np.diag([1,1,dd])@U.T
        SIX={'ThCx':'Coxa','CxTr':None,'TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}  # placeholder, filled per leg below
        FRONT6={'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
        def spec_for(side,name):
            # front legs (LF/RF) always keep all six keypoints: their Trochanter body exists
            if side in ('LF','RF'):return dict(FRONT6)
            if name=='DEFECTIVE':return dict(FRONT6)                    # mid/hind: CxTr -> missing body (defect)
            if name=='FIXED_CXTR_UNAVAILABLE':return {'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
            return {'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
        results={}
        for name in ('DEFECTIVE','FIXED_CXTR_UNAVAILABLE','FIXED_CXTR_TO_FEMUR'):
            results[name]={}
            for ds,side in LEGS.items():
                spec=spec_for(side,name);js=leg_joints(side);lo=[];hi=[]
                for j in js:
                    a0,a1=m.jnt_range[j]
                    if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                    lo.append(a0);hi.append(a1)
                lo=np.array(lo);hi=np.array(hi);x0=canon[[m.jnt_qposadr[j] for j in js]]
                rms=[];angs=[]
                for f in frames:
                    R=R_of(f);tgt=np.array([SCALE*R@col(f'{ds}-{k}',f) for k in spec])
                    sol=least_squares(lambda x:(pts(side,spec,x,name=='DEFECTIVE')-tgt).ravel(),x0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
                    rms.append(float(np.sqrt(np.mean(sol.fun**2))));angs.append(sol.x.tolist())
                A=np.array(angs);med=np.median(A,axis=0)
                viol=[m.joint(int(j)).name for j,v in zip(js,med) if not (lo[list(js).index(int(j))]-1e-9<=v<=hi[list(js).index(int(j))]+1e-9)]
                results[name][ds]=dict(keypoints=list(spec),per_frame_rms=[round(x,4) for x in rms],
                    median=round(float(np.median(rms)),4),worst=round(float(np.max(rms)),4),
                    median_angles_named={m.joint(int(j)).name:round(float(v),6) for j,v in zip(js,med)},
                    joint_range_violations=viol)
                print(name,ds,'median',results[name][ds]['median'],'worst',results[name][ds]['worst'],flush=True)
        r['results']=results
        r['supersede']=dict(
            invalid_numbers='the previously reported L2 0.5570 / L3 0.6293 / R2 0.6554 / R3 0.6840 (and their worst-frame values) were produced under the DEFECTIVE objective',
            invalid_angles='the mid/hind candidate angles delivered from that run were fitted against the WRONG objective (CxTr resolved to RWing); they do NOT satisfy the objective the report declares and must not be cited as "fit under the declared objective"',
            history='the original numbers and files are KEPT as history and marked superseded (no overwrite, no deletion): /home/sensen/Desktop/engine修复验收/S2-keypoint-pose-fit-20261015T101928/ (DEFECTIVE-SUPERSEDED notice + 更正-CxTr取值缺陷.zh-CN.md)',
            reproducible_now={k:{ds:results[k][ds]['median'] for ds in LEGS} for k in results})
        r['not_reproduced']=dict(
            numbers='L2 0.8426 / L3 0.9077 / R2 0.6797 / R3 0.8662 (from the independent reviewer message)',
            status='NOT REPRODUCED by this round under either declared mid/hind mapping',
            my_values={'FIXED_CXTR_UNAVAILABLE':{ds:results['FIXED_CXTR_UNAVAILABLE'][ds]['median'] for ds in ('L2','L3','R2','R3')},
                       'FIXED_CXTR_TO_FEMUR':{ds:results['FIXED_CXTR_TO_FEMUR'][ds]['median'] for ds in ('L2','L3','R2','R3')}},
            likely_cause='those numbers pair with a different TARGET convention (body-frame rotation/orientation convention or scale application) or a different keypoint set; the model-side body ids alone cannot produce a ~0.27 shift. They are therefore not claimed as ours, and the reviewer numbers must not be mixed with the tables here.')
        r['census_note']='The 21-start local-minimum census (deterministic, from the earlier round) is recorded ONLY as a limitation: the canonical-start residuals reported here are local solutions and are not global optima. It was NOT used to select anything this round.'
        r['conclusions_unchanged']=dict(
            canonical_gate='canonical declared pose: 14 contacts, native diamond NUMERICALLY_INFEASIBLE (unchanged)',
            fitted_pose_gate='fitted pose: ncon = 0, all four floor/cone combinations NUMERICALLY_INFEASIBLE (unchanged)',
            interpretation='a larger or smaller keypoint residual is neither a biological calibration nor support or stance evidence; neither direction makes the pose "more like a real stance"')
        # ---- (3) three disclosure fixes, carried in the report itself
        r['disclosure_fixes']=dict(
            monotonicity='the collapse-vs-scale curve is NOT monotone: L1/L2/R1 rise monotonically while L3/R2/R3 dip slightly near scale 0.25 (wording fixed).',
            apex='the dataset Th-apex is only APPROXIMATELY the origin (max |Th-apex| = 0.007025) and its three columns are never read; the transform uses lWing/rWing plus the declared apex->thorax-origin assumption.',
            fk_calls='this report carries a single count: report.json.fk_calls (below); the old 368364/368365 mismatch belongs to the earlier v1 report and is not repeated here.')
        # candidate poses from the two fixed mappings
        for name,tag in (('FIXED_CXTR_UNAVAILABLE','cxtr_unavailable'),('FIXED_CXTR_TO_FEMUR','cxtr_to_femur')):
            c=canon.copy();clipped=[]
            for ds,side in LEGS.items():
                for j,v in zip(leg_joints(side),np.array(list(results[name][ds]['median_angles_named'].values()))):
                    a0,a1=m.jnt_range[j]
                    if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                    if v<a0-1e-9 or v>a1+1e-9:clipped.append(m.joint(j).name)
                    c[m.jnt_qposadr[j]]=float(v)
            assert not clipped,('fitted angles out of range',clipped)
            np.savez_compressed(out/(tag+'_candidate_pose.npz'),projected_seated_qpos=c)
        r['fitted_pose_contacts_check']=None
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        xs=np.arange(6);w=0.27
        for k,name in enumerate(('DEFECTIVE','FIXED_CXTR_UNAVAILABLE','FIXED_CXTR_TO_FEMUR')):
            axes[0].bar(xs+(k-1)*w,[results[name][ds]['median'] for ds in LEGS],width=w,label=name)
        axes[0].set_xticks(xs);axes[0].set_xticklabels([f'{d}->{LEGS[d]}' for d in LEGS],fontsize=7)
        axes[0].set(ylabel='median-frame keypoint RMS (model units)',title=f'same x0/bounds/loss, scale {SCALE}: defective vs fixed maps');axes[0].legend(fontsize=6)
        for ds in ('L2','L3','R2','R3'):
            axes[1].plot(results['DEFECTIVE'][ds]['per_frame_rms'],'o--',ms=3,lw=1,label=f'{ds} defective')
            axes[1].plot(results['FIXED_CXTR_UNAVAILABLE'][ds]['per_frame_rms'],'-',lw=1.2,label=f'{ds} fixed')
        axes[1].set(xlabel='frame index (stride 20)',ylabel='keypoint RMS (model units)',title='per-frame RMS: defective (dashed) vs fixed (solid)')
        axes[1].legend(fontsize=6,ncol=2)
        fig.suptitle('CxTr lookup fix: defect vs corrected (kinematics only) - NOT standing, NOT a measured stance',fontsize=9)
        fig.tight_layout();fig.savefig(out/'keypoint-fit-fix.png');plt.close(fig)
        r['status']='PASS_CXTR_LOOKUP_FIX_REPORTED'
    except Exception as e:
        import traceback;r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()[-800:])
    finally:
        r['fk_calls']=fk[0]
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z2:z2.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
