"""Keypoint fit v6: corrected lookup, corrected angles re-issued, ground-contact condition solved.

RESEARCH SCRIPT, NOT A PRODUCT. Kinematics only for the fit (mj_kinematics); ONE mj_forward per
ground-contact evaluation via the project's existing seat_root bisection. No dynamics, no
contact integration, no neural simulation, no canonical XML change, no joint clipping.

Item 2 of this round: the raw fitted pose has ncon = 0 (no foot on the ground), so it is not a
valid support input. Here the ground-contact CONDITION is re-solved by adjusting ONLY the root
height (monotone bisection, the project's existing seat_root), never by clipping joint angles.
"""
from pathlib import Path
import json,time,hashlib,zipfile,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
PREV=Path('/home/sensen/Desktop/engine修复验收/S2-keypoint-pose-fit-20261015T101928')
NOMINAL=1.0;STRIDE=20
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
# PRIMARY (compliant with the instruction): mid/hind CxTr marked UNAVAILABLE, no substitute point.
MAP={'LF':{'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RF':{'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'LM':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'LH':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RM':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RH':{'ThCx':'Coxa','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}}
ALT={'LM':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'LH':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RM':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'},
     'RH':{'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from scipy.optimize import least_squares
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-fit-v6-seated-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='KEYPOINT_FIT_V6_CORRECTED_LOOKUP_PLUS_GROUND_CONTACT_CONDITION',research_script_not_product=True,
        caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset readme declares no length unit; scale fixed at 1.0 this round',
                 'the animal is not this model specimen: a data-driven candidate posture, not a measured stance','no joint was clipped to obtain contact; only the root height was solved'],
        nominal_scale=NOMINAL,
        primary_mapping='mid/hind CxTr UNAVAILABLE (no substitute point, because the model has no mid/hind trochanter body)',
        alternative_mapping='mid/hind CxTr -> Femur (reported for transparency only)')
    paths=[Path(__file__),DATA/'BodyCoords3D_fly01_t001.csv',DATA/'readme.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML,ROOT/'run_s2_support_gate.py']
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    try:
        lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
        V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2);idx={c:i for i,c in enumerate(cols)}
        col=lambda n,f:np.array([V[f,idx[f'{n}_{a}']] for a in 'xyz'])
        m=cp.load_model();d=mujoco.MjData(m)
        with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
        BAD={'LWing','RWing','Thorax'}
        table=[]
        for ds,side in LEGS.items():
            for key,suffix in list(MAP[side].items())+[('CxTr','Trochanter')]:
                if key=='CxTr' and suffix not in MAP[side].values():
                    pass
                name=side+suffix;bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
                if bid<0:
                    table.append(dict(leg=ds,keypoint=key,requested_body=name,id=int(bid),resolved_body=None,used=False,
                        reason='body does not exist for this leg type: mid/hind trochanter joints are declared on <side>Femur, so CxTr has no body here'))
                    continue
                actual=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)
                if actual in BAD or not actual.startswith(side):raise ValueError(f'{ds}/{key}: resolved to foreign body {actual}')
                table.append(dict(leg=ds,keypoint=key,requested_body=name,id=int(bid),resolved_body=actual,used=key in MAP[side],reason=None))
        r['name_to_id_table']=table
        def leg_joints(side):
            js=[]
            for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
                for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
                    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{side}{grp}_{a}')
                    if j<0:raise ValueError(f'joint missing {side}{grp}_{a}')
                    js.append(j)
            return js
        def keypoints(side,spec,angles):
            d.qpos[:]=canon
            for j,a in zip(leg_joints(side),angles):d.qpos[m.jnt_qposadr[j]]=a
            mujoco.mj_kinematics(m,d)
            T=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax');RT=d.xmat[T].reshape(3,3);oT=d.xpos[T]
            pts=[]
            for key,bn in spec.items():
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+bn)
                if bid<0:raise ValueError(f'{side}/{key}: {side+bn} unresolved - refusing xpos[-1] fallback')
                actual=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)
                if actual in BAD or not actual.startswith(side):raise ValueError(f'{side}/{key}: foreign body {actual}')
                pts.append(RT.T@(d.xpos[bid]-oT))
            return np.array(pts)
        frames=list(range(0,V.shape[0],STRIDE))
        d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
        bd=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
        land={k:d.xmat[bd('Thorax')].reshape(3,3).T@(d.xpos[bd(b)]-d.xpos[bd('Thorax')]) for k,b in (('Th-lWing','LWing'),('Th-rWing','RWing'))}
        def R_of(f):
            a=col('Th-lWing',f);bb=col('Th-rWing',f);X=np.stack([a,np.zeros(3),bb]);Y=np.stack([land['Th-lWing'],np.zeros(3),land['Th-rWing']])
            Xc=X-X.mean(0);Yc=Y-Y.mean(0);U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T));return Vt.T@np.diag([1,1,dd])@U.T
        fit={}
        for ds,side in LEGS.items():
            spec=MAP[side];alt=ALT.get(side,spec)
            js=leg_joints(side);lo=[];hi=[]
            for j in js:
                a0,a1=m.jnt_range[j]
                if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                lo.append(a0);hi.append(a1)
            lo=np.array(lo);hi=np.array(hi);x0=canon[[m.jnt_qposadr[j] for j in js]]
            entry=dict(model_side=side,primary_keypoints=list(spec),alt_keypoints=(list(alt) if alt is not spec else None))
            for label,sp in (('primary',spec),)+((('alt_cxtr_femur',alt),) if alt is not spec else ()):
                rms=[];angs=[]
                for f in frames:
                    R=R_of(f);tgt=np.array([R@col(f'{ds}-{k}',f) for k in sp])
                    sol=least_squares(lambda x:(keypoints(side,sp,x)-tgt).ravel(),x0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
                    rms.append(float(np.sqrt(np.mean(sol.fun**2))));angs.append(sol.x.tolist())
                A=np.array(angs);med=np.median(A,axis=0)
                viol=[dict(joint=m.joint(int(j)).name,value=float(v)) for j,v in zip(js,med) if v<lo[list(js).index(int(j))]-1e-9 or v>hi[list(js).index(int(j))]+1e-9]
                entry[label]=dict(per_frame_rms=[round(x,4) for x in rms],median=round(float(np.median(rms)),4),worst=round(float(np.max(rms)),4),
                    median_angles_rad=med.tolist(),median_angles_named={m.joint(int(j)).name:round(float(v),6) for j,v in zip(js,med)},
                    joint_range_violations=viol,angle_spread_max_deg=float(np.degrees(np.max(A.max(0)-A.min(0)))))
            fit[ds]=entry
            print(ds,'primary median',entry['primary']['median'],'worst',entry['primary']['worst'],flush=True)
        r['fit_corrected']=fit
        r['previous_candidate_declaration']=dict(
            invalid_for=['L2','L3','R2','R3'],superseded_dir=str(PREV),
            statement='the previously delivered candidate angles for the mid/hind legs (L2/L3/R2/R3) were fitted against the WRONG objective (the model-side CxTr keypoint resolved to body -1 -> d.xpos[-1] = RWing). Those angles are VOID and are re-issued in this directory.',
            still_usable='L1/R1 (front legs): their lookup was already correct')
        # ---- item 2: ground-contact CONDITION (root height only, no joint clipping)
        cand=canon.copy();clip_used=[]
        for ds,side in LEGS.items():
            for j,v in zip(leg_joints(side),fit[ds]['primary']['median_angles_rad']):
                a0,a1=m.jnt_range[j]
                if not m.jnt_limited[j]:a0,a1=-2.5,2.5
                if v<a0-1e-9 or v>a1+1e-9:clip_used.append(dict(joint=m.joint(j).name,value=float(v),range=[float(a0),float(a1)]))
                cand[m.jnt_qposadr[j]]=float(v)
        assert not clip_used,'fitted angles out of range: refusing to clip'
        d.qpos[:]=cand;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
        raw_contacts=int(d.ncon)
        tarsus=cp.tarsus_geom_ids(m)
        lowest_raw=float(min(d.geom_xpos[g][2] for g in tarsus))
        r['ground_contact_condition']=dict(
            raw_pose=dict(contacts=raw_contacts,lowest_tarsus_z=lowest_raw,thorax_z=float(d.xpos[bd('Thorax'),2]),
                verdict=('NO_CONTACT: this dataset-derived posture cannot be used as a stance/support input as-is' if raw_contacts==0 else 'has contacts')),
            method='root HEIGHT solved by the project existing seat_root bisection toward the declared seating penetration; joint angles untouched, nothing clipped')
        seated=cand.copy();height=cp.seat_root(m,d,seated)
        seated=d.qpos.copy();d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
        lowest_seated=float(min(d.geom_xpos[g][2] for g in tarsus))
        r['ground_contact_condition']['seated_pose']=dict(contacts=int(d.ncon),thorax_z=float(d.xpos[bd('Thorax'),2]),
            lowest_tarsus_z=lowest_seated,root_z=float(seated[2]),seat_root_return=float(height),
            tarsus_only=bool(all('Tarsus' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '') for c in d.contact for g in (c.geom1,c.geom2) if m.geom_type[int(g)]!=mujoco.mjtGeom.mjGEOM_PLANE)))
        np.savez_compressed(out/'v6_candidate_raw_pose.npz',projected_seated_qpos=cand)
        np.savez_compressed(out/'v6_candidate_seated_pose.npz',projected_seated_qpos=seated)
        # ---- Gate A on the seated pose (existing pipeline; cone labels recorded)
        run=subprocess.run([sys.executable,'-B',str(ROOT/'run_s2_support_gate.py'),str(out/'v6_candidate_seated_pose.npz')],capture_output=True,text=True,timeout=600)
        lines2=run.stdout.strip().splitlines();gd=Path(lines2[-1]) if lines2 else None
        if gd is not None and (gd/'report.json').exists() and not (gd/'metadata.json').exists():
            grej=json.loads((gd/'report.json').read_text())
            r['gate_a_on_seated_pose']=dict(dir=str(gd),returncode=run.returncode,contacts=None,
                classification='REJECTED_BY_GATE_INPUT_QUALIFICATION',gate_status=grej.get('status'),gate_error=grej.get('error'),
                meaning='the existing pipeline refuses this pose before any LP: its input qualification requires foot-only contacts')
        elif gd and (gd/'report.json').exists():
            gr=json.loads((gd/'report.json').read_text());gm=json.loads((gd/'metadata.json').read_text())
            r['gate_a_on_seated_pose']=dict(dir=str(gd),returncode=run.returncode,contacts=int(gm['ncon']),
                classifications={f"{e['floor']}/{e['contact_model']}":e['classification'] for e in gr['results']},
                cone_source='run_s2_support_gate.py blocks: diamond = native MuJoCo pyramidal cone; outer and union are relaxations beyond it')
        else:
            r['gate_a_on_seated_pose']=dict(dir=None,returncode=run.returncode,stderr=run.stderr[-300:])
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        xs=np.arange(6)
        axes[0].bar(xs-0.2,[fit[ds]['primary']['median'] for ds in LEGS],width=0.2,label='v6 primary (CxTr unavailable on mid/hind)')
        axes[0].bar(xs+0.2,[fit[ds].get('alt_cxtr_femur',{}).get('median',fit[ds]['primary']['median']) for ds in LEGS],width=0.2,label='v6 alt (CxTr->Femur)')
        axes[0].set_xticks(xs);axes[0].set_xticklabels([f'{d}->{LEGS[d]}' for d in LEGS],fontsize=7)
        axes[0].set(ylabel='median-frame keypoint RMS (model units)',title='corrected fit, scale 1.0');axes[0].legend(fontsize=6)
        zs=np.linspace(-3,3,121);heights=[]
        for z in zs:
            q=cand.copy();q[2]+=z;d.qpos[:]=q;mujoco.mj_kinematics(m,d);heights.append(float(min(d.geom_xpos[g][2] for g in tarsus)))
        axes[1].plot(zs,heights,label='lowest tarsus z vs root offset');axes[1].axhline(0,color='k',lw=.8,label='floor plane z=0')
        axes[1].axvline(float(seated[2]-cand[2]),color='r',ls='--',label='solved root offset (seat_root)')
        axes[1].set(xlabel='root z offset (model units)',ylabel='lowest tarsus z',title='ground-contact condition: root height only, joints untouched');axes[1].legend(fontsize=6)
        fig.suptitle('v6 keypoint fit: corrected lookup + ground-contact condition - NOT standing, NOT a measured stance',fontsize=9)
        fig.tight_layout();fig.savefig(out/'v6-fit-and-contact.png');plt.close(fig)
        r['status']='PASS_V6_CORRECTED_AND_CONTACT_CONDITION_SOLVED'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z2:z2.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r.get('ground_contact_condition',{}),indent=1)[:700]);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
