"""Necessary-feasibility of a pre-registered declared pose family (A-gate machinery reused).

Reuses run_s2_support_gate.py unchanged in substance (identical A=[R,C,limit], b, LP) by
calling it as a subprocess with an explicit pose file, and reuses the exact rational
separation tool for infeasible members. No search, no tuning, no bound relaxation is used
for the conclusion (the native lo=0.0001/hi=1 floor is primary).
"""
from pathlib import Path
import json,time,hashlib,zipfile,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
GATE=ROOT/'run_s2_support_gate.py'
CERT=ROOT/'tools_s2_support_certificate.py'
R37=ROOT/'outputs/stance_margin_round37/stance_margin.json'
R30=ROOT/'outputs/stance_joint_feasible_round30/stance_for_runner.json'
SAVED=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
PARAM_J=('Coxa_yaw','Coxa_pitch','Coxa_roll','Trochanter_yaw','Trochanter_pitch','Trochanter_roll','Tibia_pitch')
LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_family(model,data):
    import mujoco
    from engine import canonical_physics as cp
    key=model.key_qpos[0].copy()
    fam={}
    def seat(q):
        q=q.copy();cp.seat_root(model,data,q);return data.qpos.copy()
    fam['M1_keyframe_seated']=seat(key)
    r30=json.loads(R30.read_text())['joints_rad'];q=key.copy()
    for nm,val in r30.items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,nm);assert j>=0,nm
        q[model.jnt_qposadr[j]]=float(val)
    fam['M2_round30_named_seated']=seat(q)
    best=np.asarray(json.loads(R37.read_text())['best']['x'],float)
    params=[(leg,j) for leg in LEGS for j in PARAM_J]
    assert len(best)==len(params),(len(best),len(params))
    for label,scale in (('M3_round37_best_seated',1.0),('M4_round37_best_scale0.5_seated',0.5),('M5_round37_best_scale1.5_seated',1.5)):
        q=key.copy()
        for k,(leg,j) in enumerate(params):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{leg}{j}');assert jid>=0,f'joint_{leg}{j}'
            adr=model.jnt_qposadr[jid];lo,hi=model.jnt_range[jid]
            q[adr]=float(np.clip(key[adr]+scale*best[k],lo,hi))
        fam[label]=seat(q)
    with np.load(SAVED) as z:fam['M6_saved_projected_pose']=z['projected_seated_qpos'].copy()
    return fam
def main():
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-pose-family-feasibility-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='NECESSARY_FEASIBILITY_OF_PREREGISTERED_DECLARED_POSE_FAMILY',
           primary_floor='native lo=0.0001 hi=1 (not relaxed); infeasible verdict now requires the union (8-ray) cone to be infeasible too',secondary_reference_only='zero floor lo=0 (NOT used for the conclusion)',
           semantics='infeasible => no static support for that declared pose (necessary condition fails); feasible => necessary condition passes only, NOT support, NOT stability, NOT standing',
           protocol_sha256=sha(ROOT/'S2_POSE_FAMILY_FEASIBILITY_PROTOCOL.zh-CN.md'),members={})
    hashes={str(p):sha(p) for p in (GATE,CERT,R37,R30,SAVED,ROOT/'S2_POSE_FAMILY_FEASIBILITY_PROTOCOL.zh-CN.md',Path(__file__))}
    try:
        import mujoco
        m=cp.load_model();d=mujoco.MjData(m)
        fam=build_family(m,d)
        for name,qpos in fam.items():
            pdir=out/'poses';pdir.mkdir(exist_ok=True)
            ppath=pdir/(name+'.npz');np.savez_compressed(ppath,projected_seated_qpos=qpos)
            # The gate names its output directory with a second-resolution timestamp, so two
            # subprocess runs inside one second collide. Retry on that (infrastructure only).
            attempts=[]
            for attempt in range(6):
                run=subprocess.run([sys.executable,'-B',str(GATE),str(ppath)],capture_output=True,text=True,timeout=300)
                attempts.append(dict(attempt=attempt,returncode=run.returncode,collision='FileExistsError' in run.stderr))
                if 'FileExistsError' not in run.stderr:break
                time.sleep(1.2)
            entry=dict(pose_file=str(ppath),gate_returncode=run.returncode,gate_attempts=attempts,gate_stdout=run.stdout.strip().splitlines()[-3:] if run.stdout else [],gate_stderr=run.stderr.strip()[-300:])
            gd=Path(run.stdout.strip().splitlines()[-1]) if run.returncode in (0,1) and run.stdout.strip() else None
            if gd is not None and not (gd/'report.json').exists(): gd=None
            if run.returncode==0 and gd and (gd/'report.json').exists():
                gr=json.loads((gd/'report.json').read_text());meta=json.loads((gd/'metadata.json').read_text())
                entry['gate_dir']=str(gd);entry['contacts']=meta['ncon']
                native=[e for e in gr['results'] if e['floor']=='native'];zero=[e for e in gr['results'] if e['floor']=='zero']
                entry['native']={e['contact_model']:e['classification'] for e in native}
                entry['zero_reference_only']={e['contact_model']:e['classification'] for e in zero}
                infeasible=all(e['classification']=='NUMERICALLY_INFEASIBLE_FOR_STATED_RELAXATION' for e in native)
                feasible_witness=any(e['classification']=='RELAXED_WITNESS' for e in native)
                entry['primary_verdict']='INFEASIBLE' if infeasible else ('FEASIBLE_NECESSARY_ONLY' if feasible_witness else 'INDETERMINATE')
                if infeasible:
                    c=None
                    for attempt in range(6):
                        c=subprocess.run([sys.executable,'-B',str(CERT),str(gd),'union'],capture_output=True,text=True,timeout=600)
                        if 'FileExistsError' not in c.stderr:break
                        time.sleep(1.2)
                    entry['certificate_returncode']=c.returncode;entry['certificate_stdout']=c.stdout.strip().splitlines()[-2:] if c.stdout else [];entry['certificate_stderr']=c.stderr.strip()[-300:]
                    cd=Path(c.stdout.strip().splitlines()[-2]) if len(c.stdout.strip().splitlines())>=2 else None
                    if cd and (cd/'certificate.json').exists():
                        cr=json.loads((cd/'certificate.json').read_text());entry['certificate_dir']=str(cd)
                        entry['certificate_status']={e['floor']:e['status'] for e in cr['results']}
                        entry['certificate_gap_float']={e['floor']:e.get('gap_float') for e in cr['results']}
            else:entry['primary_verdict']='REJECTED_BY_RANGE_CHECK_OR_GATE_FAILURE'
            r['members'][name]=entry;print(name,entry['primary_verdict'],entry.get('native'),entry.get('certificate_status'),flush=True)
        vals=[e['primary_verdict'] for e in r['members'].values()]
        r['summary']=dict(members=len(vals),infeasible=vals.count('INFEASIBLE'),feasible_necessary_only=vals.count('FEASIBLE_NECESSARY_ONLY'),rejected_or_indeterminate=len(vals)-vals.count('INFEASIBLE')-vals.count('FEASIBLE_NECESSARY_ONLY'),fraction_infeasible=round(vals.count('INFEASIBLE')/len(vals),4))
        r['conclusion']=dict(within_family=('no member of the pre-registered declared family passes the necessary static-support condition under unchanged parameters' if r['summary']['infeasible']==len(vals) else 'at least one member passes the necessary condition only'),
            limit='a family of %d declared poses cannot establish impossibility over the whole pose space; failure here means these declared poses cannot statically support, not that no pose can'%len(vals),
            not_claimed='no standing, no stability, no biological calibration, no dynamic behaviour')
        r['status']='PASS_POSE_FAMILY_NECESSARY_FEASIBILITY'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in (Path(__file__),ROOT/'S2_POSE_FAMILY_FEASIBILITY_PROTOCOL.zh-CN.md'):z.write(p,p.name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r.get('summary'),indent=2));print(json.dumps(r.get('conclusion'),indent=2,ensure_ascii=False));print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
