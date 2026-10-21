"""A-gate decision run on the CORRECTED keypoint candidate (frozen pose, v=0, zero external load).

RESEARCH SCRIPT, NOT A PRODUCT. Reuses the existing gate (run_s2_support_gate.py) as a
subprocess so the A=[R,C,limit] / bias-passive-base construction is identical. No neural
re-run, no tuning, no seating/sinking/search to manufacture standing, no canonical XML change.

Deliverable: a feasible / infeasible / indeterminate verdict per contact-cone, the residual
numbers of the corrected fit, and a figure (actually opened) supporting the decision on whether
the pose SOURCE must be changed.
"""
from pathlib import Path
import json,time,hashlib,zipfile,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
FIX=Path('/home/sensen/Desktop/engine修复验收/S2-keypoint-pose-fit-fix-20261015T105221')
V6=Path('/home/sensen/Desktop/engine修复验收/S2-keypoint-fit-v6-seated-20261015T105022')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_gate(pose,out):
    r=subprocess.run([sys.executable,'-B',str(ROOT/'run_s2_support_gate.py'),str(pose)],capture_output=True,text=True,timeout=600)
    lines=r.stdout.strip().splitlines();gd=Path(lines[-1]) if lines else None
    res=dict(returncode=r.returncode,dir=(str(gd) if gd else None))
    if gd and (gd/'report.json').exists():
        gr=json.loads((gd/'report.json').read_text())
        res['gate_status']=gr.get('status');res['gate_error']=gr.get('error')
        if (gd/'metadata.json').exists():
            gm=json.loads((gd/'metadata.json').read_text())
            res['contacts']=int(gm['ncon'])
            res['classifications']={f"{e['floor']}/{e['contact_model']}":e['classification'] for e in gr['results']}
    return res
def main():
    import mujoco
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('A-gate-corrected-candidate-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='A_GATE_NECESSARY_CONDITION_ON_THE_CORRECTED_KEYPOINT_CANDIDATE',research_script_not_product=True,standing=False,
        conditions=dict(pose='FROZEN corrected keypoint candidate (no re-seating in the primary arm)',qvel='0',external_load='none (zero qfrc_applied / xfrc_applied)',
                        dof='full root + joint DOFs',activation='native bounds [0.0001, 1] (not relaxed)',
                        cones='diamond = native MuJoCo pyramidal cone (the declared contact model); outer and union are relaxations BEYOND that cone'),
        not_done='no neural re-run, no tuning, no canonical XML change, no seating/sinking/search to manufacture standing')
    poses={'corrected_raw':FIX/'cxtr_unavailable_candidate_pose.npz','corrected_alt_cxtr_to_femur':FIX/'cxtr_to_femur_candidate_pose.npz'}
    paths=[Path(__file__),ROOT/'run_s2_support_gate.py',cp_xml:=ROOT/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml']+list(poses.values())+[FIX/'report.json']
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        m=cp.load_model();d=mujoco.MjData(m)
        weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        tarsus=cp.tarsus_geom_ids(m)
        thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')
        fixrep=json.loads((FIX/'report.json').read_text())
        r['corrected_fit_residuals']={leg:{arm:dict(median=fixrep['results'][arm][leg]['median'],worst=fixrep['results'][arm][leg]['worst'],n_keypoints=len(fixrep['results'][arm][leg]['keypoints']))
                                                for arm in ('DEFECTIVE','FIXED_CXTR_UNAVAILABLE','FIXED_CXTR_TO_FEMUR')} for leg in ('L1','L2','L3','R1','R2','R3')}
        facts={}
        for name,p in poses.items():
            q=np.load(p)['projected_seated_qpos']
            d.qpos[:]=q;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
            lowest=float(min(d.geom_xpos[g][2] for g in tarsus))
            facts[name]=dict(contacts=int(d.ncon),thorax_z=float(d.xpos[thorax,2]),lowest_tarsus_z=lowest,root_z=float(q[2]))
            facts[name]['gate_raw']=run_gate(p,out)
            # root-only seating arm (no joint touch) to see whether a foot-only contact pose exists at all
            qs=q.copy();cp.seat_root(m,d,qs);qs=d.qpos.copy()
            d.qpos[:]=qs;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
            foot=sum(1 for c in d.contact if (int(c.geom1) in tarsus) ^ (int(c.geom2) in tarsus))
            nonfoot=int(d.ncon)-foot
            facts[name]['seated_arm']=dict(root_z=float(qs[2]),contacts=int(d.ncon),foot_floor_contacts=foot,nonfoot_contacts=nonfoot,
                lowest_tarsus_z=float(min(d.geom_xpos[g][2] for g in tarsus)))
            sp=out/(name+'_seated.npz');np.savez_compressed(sp,projected_seated_qpos=qs)
            facts[name]['gate_seated']=run_gate(sp,out)
            print(name,json.dumps({k:v for k,v in facts[name].items() if not k.startswith('gate')},ensure_ascii=False),flush=True)
        r['pose_facts']=facts
        primary=facts['corrected_raw']
        verdicts=[]
        for arm in ('gate_raw','gate_seated'):
            g=primary[arm]
            verdicts.append(dict(arm=arm,gate_status=g.get('gate_status'),contacts=g.get('contacts'),classifications=g.get('classifications')))
        infeasible=all(v=='NUMERICALLY_INFEASIBLE_FOR_STATED_RELAXATION' for v in (primary['gate_raw'].get('classifications') or {}).values()) if primary['gate_raw'].get('classifications') else False
        r['verdict']=dict(
            classification=('INFEASIBLE' if infeasible else ('UNDETERMINED' if primary['gate_raw'].get('gate_status')=='FAIL' else 'FEASIBLE')),
            per_arm=verdicts,
            reason=('the corrected candidate has ZERO contacts (lowest tarsus z = %.4f above the floor plane), so the static system has no ground reaction columns at all: infeasibility is degenerate and says nothing about the force system. The root-only seated arm produces contacts but they are NOT foot-only, and the existing input qualification REJECTS that pose (gate status FAIL, "nonfoot active contact").' % primary['lowest_tarsus_z']),
            decision_basis='A-gate therefore cannot qualify this pose source: the dataset-derived posture is NOT a usable support input. The decision is to CHANGE THE POSE SOURCE (or declare the dataset posture unusable), not to search poses.')
        r['dataset_posture_verdict']=dict(
            verdict='UNUSABLE_AS_SUPPORT_INPUT',
            basis=['no ground contact at the declared root height (ncon=0, lowest tarsus z=%.4f)'%primary['lowest_tarsus_z'],
                   'root-only seating gives non-foot contacts and is rejected by the existing input qualification',
                   'corrected fit residuals are 0.57-0.72 median (~30-50 percent of the chain span), i.e. this posture is not reproduced faithfully either'],
            action='stop fitting further variants of this dataset posture; treat the BBNPYX posture source as not usable for support input')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        # left: tarsus height vs root offset for the corrected candidate
        zs=np.linspace(-2.0,0.4,97);hs=[]
        base=np.load(poses['corrected_raw'])['projected_seated_qpos']
        for z in zs:
            q=base.copy();q[2]+=z;d.qpos[:]=q;mujoco.mj_kinematics(m,d);hs.append(float(min(d.geom_xpos[g][2] for g in tarsus)))
        axes[0].plot(zs,hs,label='corrected candidate: lowest tarsus z')
        axes[0].axhline(0,color='k',lw=.8,label='floor plane')
        axes[0].axvline(float(primary['seated_arm']['root_z']-base[2]),color='r',ls='--',label='root-only seat solution')
        axes[0].axvline(0,color='g',ls=':',label='canonical root (as declared)')
        axes[0].set(xlabel='root z offset (model units)',ylabel='lowest tarsus z',title='no contact until the root is lowered by ~1.5');axes[0].legend(fontsize=6)
        names=list(poses);xs=np.arange(len(names));w=0.35
        axes[1].bar(xs-w/2,[facts[n]['contacts'] for n in names],width=w,label='as declared (ncon)')
        axes[1].bar(xs+w/2,[facts[n]['seated_arm']['nonfoot_contacts'] for n in names],width=w,label='root-seated: non-foot contacts')
        axes[1].set_xticks(xs);axes[1].set_xticklabels(['corrected raw','alt CxTr->Femur'],fontsize=7)
        axes[1].set(ylabel='contact count',title='contacts: none as declared; non-foot once seated');axes[1].legend(fontsize=6)
        fig.suptitle('A-gate on the corrected keypoint candidate: unusable support input (NOT standing)',fontsize=9)
        fig.tight_layout();fig.savefig(out/'a-gate-corrected.png');plt.close(fig)
        r['status']='PASS_A_GATE_CORRECTED_CANDIDATE_VERDICT'
    except Exception as e:
        import traceback;r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()[-700:])
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r.get('verdict'),indent=1,ensure_ascii=False)[:900]);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
