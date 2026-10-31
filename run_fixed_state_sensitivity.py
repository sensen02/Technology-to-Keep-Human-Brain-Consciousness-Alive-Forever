"""S156 bounded static activation counterfactual. No integration or neural imports.
Run only once in a fresh directory, with venv_body/bin/python -B.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=Path('/home/sensen/Desktop/engine修复验收')
SOURCE=BASE/'s154-activation-rate/20261024T215751775489Z'
MAP=BASE/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS=['lf','lm','lh','rf','rm','rh']; POINTS=[12,60]; ATOL=1e-9

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def extract(path,name,ns):
    tree=ast.parse(path.read_text()); node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    text=ast.unparse(node)+'\n'; exec(compile(text,str(path),'exec'),ns); return text

def main(out):
    assert out.is_absolute() and not out.exists()
    out.mkdir(parents=True); freeze=out/'SOURCE_FREEZE'; freeze.mkdir()
    ledger=dict(model_load=0,forward=0,step=0,neural_ticks=0,baseline_forward=0,perturb_forward=0)
    dump(out/'CALL_LEDGER.json',ledger)
    def reserve(kind):
        ledger[kind]+=1
        if kind in ('baseline_forward','perturb_forward'): ledger['forward']+=1
        assert ledger['forward']<=26
        dump(out/'CALL_LEDGER.json',ledger)
    z=load(SOURCE/'init80ms/TRAJECTORY.npz'); rows=load(SOURCE/'init80ms/CONSTRAINT_ROWS.npz')
    mapping=json.loads(MAP.read_text()); meta=json.loads((SOURCE/'init80ms/COM_CONTACT_METADATA.json').read_text())
    proof=json.loads((SOURCE/'init80ms/MODEL_PROVENANCE.json').read_text())
    functions={}; texts=[]
    for path,name in [(ROOT/'run_com_contact_probe.py','com_contact_record'),(ROOT/'run_constraint_row_recording.py','constraint_record'),(ROOT/'run_matched_state_counterfactual.py','snapshot')]:
        texts.append(extract(path,name,dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())))
    extracted='\n'.join(texts); (freeze/'INERT_RECORDERS.py').write_text(extracted)
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()); exec(extracted,ns)
    inputs={str(Path(__file__).resolve()):sha(__file__),str(MAP):sha(MAP),str(XML):sha(XML)}
    for p in [SOURCE/'INPUT_HASHES.json',SOURCE/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py',*[SOURCE/'init80ms'/n for n in ['TRAJECTORY.npz','CONSTRAINT_ROWS.npz','MODEL_PROVENANCE.json','NATIVE_MODEL_ARRAYS.npz','COM_CONTACT_METADATA.json']],ROOT/'run_com_contact_probe.py',ROOT/'run_constraint_row_recording.py',ROOT/'run_matched_state_counterfactual.py']:
        inputs[str(p)]=sha(p)
    # Bind every original asset/source; no imports or execution of the original runner.
    original=json.loads((SOURCE/'INPUT_HASHES.json').read_text())
    original=original.get('hashes',original)
    for p,h in original.items():
        if isinstance(h,str) and len(h)==64:
            assert sha(p)==h, 'Original input changed: '+p
            inputs[p]=h
    dump(out/'INPUT_HASHES.json',inputs)
    shutil.copyfile(__file__,freeze/'run_fixed_state_sensitivity.py')
    shutil.copyfile(SOURCE/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py',freeze/'S154_PRIVATE_RUNNER_SOURCE.py')
    dump(out/'PREREG.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='S156/P1/S2 static explicit act overwrite counterfactual, NOT neural controller or trajectory',source=str(SOURCE),selected_indices=POINTS,selected_ms=[.300,1.500],max_forward=26,baseline_forward=2,perturb_forward=24,steps=0,neural_ticks=0,policy='Both baseline gates must pass before ANY perturbation. Any failure stops all sensitivities; no retry or historical trajectory rerun.',absolute_tolerance=ATOL,relative_tolerance=None,exact_gates=['model numeric snapshot','contact IDs/dim/addresses/efc types/IDs','unchanged state/input fields'],baseline_numeric_gates=['all48 qacc/qfrc_constraint/qfrc_actuator/qfrc_bias/qfrc_passive/Ma','COM acceleration','six leg loads','raw contact force/position/frame','efc_force/J'],intervention='Fresh exact restored state before every call; only act for one S150-confirmed leg times .95 or 1.05; no clipping, no accumulated changes',unchanged=['qpos','qvel','ctrl','time','external inputs','qacc_warmstart','all model parameters'],model=dict(xml=str(XML),mujoco='3.9.0',integrator='RK4',dt=.000025,tau_act=.001,tau_deact=.0004),missing_state_policy='Missing nonempty native input state => MISSING_STATE.json and no forwards; never guess/fill',units='native; physical units NOT_ESTABLISHED',actuator_map=mapping,source_freeze_sha256={str(p):sha(p) for p in freeze.iterdir()}))
    inventory={k:list(v.shape) for k,v in z.items()}
    dump(out/'ARCHIVE_INVENTORY.json',inventory)
    import mujoco as mj
    assert mj.__version__=='3.9.0'
    for name in ['control','act_dyn','act_gain','act_bias','passive','sensor']:
        getter=getattr(mj,'get_mjcb_'+name,None); assert getter is None or getter() is None
    reserve('model_load'); m=mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m)==proof['before']
    m.actuator_dynprm[:,0]=.001; m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
    assert ns['snapshot'](m)==proof['after']; assert np.all(m.actuator_dynprm[:,1]==.0004)
    missing=[]
    for n,count in [('mocap_pos',m.nmocap),('mocap_quat',m.nmocap),('eq_active',m.neq),('plugin_state',m.npluginstate),('userdata',m.nuserdata)]:
        if count and n not in z: missing.append(n)
    required=['time','qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied','native_qacc_warmstart']
    missing.extend(n for n in required if n not in z)
    # These dimensions are fixed by the fully matched model, not zero-filled guesses.
    stateaudit=dict(missing=missing,nmocap=m.nmocap,neq=m.neq,nplugin=m.nplugin,npluginstate=m.npluginstate,nuserdata=m.nuserdata,solver=int(m.opt.solver),disableflags=int(m.opt.disableflags),iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance),warmstart_archive='native_qacc_warmstart',model_snapshot_match=True,all_archived_inputs=required)
    dump(out/'STATE_SUFFICIENCY.json',stateaudit)
    if missing:
        dump(out/'MISSING_STATE.json',stateaudit); finish(out,inputs,ledger,'MISSING_STATE',[]); return
    # Same full compiled actuator structure as S149, except intended tau change.
    prior=json.loads((BASE/'s149-half-gain-physical/20261024T213108429114Z/init80ms/MODEL_PROVENANCE.json').read_text())['after']
    for k,v in proof['after'].items():
        if k.startswith('model.actuator_') and k!='model.actuator_dynprm': assert prior[k]==v
    for entry in mapping:
        assert mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,entry['actuator'])==entry['name']
    dump(out/'ACTUATOR_MAP_PROOF.json',dict(source=str(MAP),same_compiled_actuator_structure_except_dynprm=True,entries=mapping))
    geomleg={int(k):LEGS.index(v) for k,v in meta['geom_map'].items()}; floors=set(meta['floor_geom_ids'])
    def restore(i,leg=None,gain=1.):
        d=mj.MjData(m)
        for n in required:
            if n=='time': d.time=float(z[n][i])
            elif n=='native_qacc_warmstart': d.qacc_warmstart[:]=z[n][i]
            else: getattr(d,n)[:]=z[n][i]
        for n in ['mocap_pos','mocap_quat','eq_active','plugin_state','userdata']:
            if n in z: getattr(d,n)[:]=z[n][i]
        if leg is not None:
            ids=[e['actuator'] for e in mapping if e['leg']==leg]; adr=m.actuator_actadr[ids]; assert np.all(adr>=0); d.act[adr]*=gain
        return d
    def call(i,leg=None,gain=1.):
        d=restore(i,leg,gain); before={n:getattr(d,n).copy() for n in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied']}; warm=d.qacc_warmstart.copy(); t=d.time
        reserve('baseline_forward' if leg is None else 'perturb_forward'); mj.mj_forward(m,d)
        assert d.time==t and all(np.array_equal(v,getattr(d,n)) for n,v in before.items())
        c=ns['com_contact_record'](mj,m,d,floors); r=ns['constraint_record'](mj,m,d,i)
        c.update(r); c.update(before); c['qacc_warmstart_input']=warm; c['qacc_warmstart_after']=d.qacc_warmstart.copy()
        for n in ['qacc','qfrc_actuator','qfrc_constraint','qfrc_bias','qfrc_passive','actuator_force','actuator_length','actuator_velocity','solver_niter']: c[n]=getattr(d,n).copy()
        M=np.empty((m.nv,m.nv)); mj.mj_fullM(m,M,d.qM); c['M']=M; c['Ma']=M@d.qacc
        c['RHS']=d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint+d.qfrc_applied-d.qfrc_bias
        c['mass_residual']=c['Ma']-c['RHS']; c['JTF_residual']=r['efc_J_dense'].T@d.efc_force-d.qfrc_constraint
        c['COM_force_residual']=c['com_mass_times_acceleration_world']-c['external_force_world']
        foot=np.zeros((6,3)); positive=np.zeros(6,int); negative=np.zeros(6,int)
        for ci,(g1,g2) in enumerate(c['contact_geom_ids']):
            if not (g1 in floors or g2 in floors): continue
            footid=int(g2 if g1 in floors else g1)
            if footid in geomleg:
                li=geomleg[footid]; f=(1 if g1 in floors else -1)*c['contact_force_on_geom2_world'][ci]; foot[li]+=f; positive[li]+=int(f[2]>0); negative[li]+=int(c['contact_dist'][ci]<0)
        c['foot_force']=foot; c['positive_load_contact_counts']=positive; c['negative_distance_contact_counts']=negative
        np.savez_compressed(out/f'state{i:03d}_{leg or "baseline"}_gain{gain:.2f}.npz',**c)
        return c
    def check(a,b,exact=False):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape: return dict(passed=False,shapes=[list(a.shape),list(b.shape)])
        delta=a-b; err=float(np.max(np.abs(delta),initial=0))
        return dict(passed=bool(np.array_equal(a,b) if exact else err<=ATOL),max_abs=err)
    bases={}; gates=[]
    for i in POINTS:
        c=call(i); bases[i]=c; n=int(z['contact_ncon'][i]); p=f'sample{i:03d}_'; checks={}
        for k in ['qacc','qfrc_actuator','qfrc_constraint','qfrc_bias','qfrc_passive','Ma','com_acceleration_world','foot_force']:
            checks[k]=check(c[k],z[k][i])
        for k in ['contact_force_raw6','contact_pos_world','contact_frame','contact_geom_ids','contact_dim']:
            checks[k]=check(c[k],z[k][i,:n],k in ['contact_geom_ids','contact_dim'])
        for k in ['efc_type','efc_id','efc_force','efc_J_dense','contact_efc_address']:
            checks[k]=check(c[k],rows[p+k],k in ['efc_type','efc_id','contact_efc_address'])
        for k in ['mass_residual','JTF_residual','COM_force_residual']: checks[k]=check(c[k],np.zeros_like(c[k]))
        passed=all(v['passed'] for v in checks.values()); gates.append(dict(sample=i,passed=passed,checks=checks)); dump(out/'BASELINE_GATES.json',gates)
        np.savez_compressed(out/f'BASELINE_DIFFERENCES_{i:03d}.npz',**{k:c[k]-z[k][i] for k in ['qacc','qfrc_constraint','qfrc_actuator','com_acceleration_world','foot_force','Ma']})
        if not passed:
            finish(out,inputs,ledger,'BASELINE_FAILED_STOP_ALL_SENSITIVITIES',gates); return
    result=[]
    for i in POINTS:
        b=bases[i]; matrix=np.zeros((6,6)); comresponse=np.zeros(6); asym=np.zeros((6,6))
        for j,leg in enumerate(LEGS):
            low=call(i,leg,.95); high=call(i,leg,1.05)
            matrix[:,j]=(high['foot_force'][:,2]-low['foot_force'][:,2])/.1
            comresponse[j]=(high['com_acceleration_world'][2]-low['com_acceleration_world'][2])/.1
            asym[:,j]=high['foot_force'][:,2]+low['foot_force'][:,2]-2*b['foot_force'][:,2]
            for gain,c in [(.95,low),(1.05,high)]:
                topology=all(np.array_equal(c[k],b[k]) for k in ['contact_geom_ids','contact_dim','contact_efc_address','efc_type','efc_id'])
                for target in range(6): result.append(dict(sample=i,time_ms=float(z['time'][i]*1000),source_leg=leg,gain=gain,target_leg=LEGS[target],delta_Fz=float(c['foot_force'][target,2]-b['foot_force'][target,2]),delta_COMaz=float(c['com_acceleration_world'][2]-b['com_acceleration_world'][2]),same_contact_set=topology,max_Ma_residual=float(np.max(np.abs(c['mass_residual']))),max_JTF_residual=float(np.max(np.abs(c['JTF_residual'])))))
        np.savez_compressed(out/f'RESPONSE_{i:03d}.npz',response6x6=matrix,COMaz_response=comresponse,nonlinear_asymmetry6x6=asym,leg_order=np.asarray(LEGS))
    with (out/'SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    finish(out,inputs,ledger,'COMPLETED_STATIC_COUNTERFACTUAL',gates)

def finish(out,inputs,ledger,status,gates):
    unchanged=all(sha(p)==h for p,h in inputs.items()); assert unchanged
    dump(out/'INPUT_HASHES_AFTER.json',inputs)
    dump(out/'SUMMARY.json',dict(status=status,ledger=ledger,baseline_gates=gates,input_hashes_unchanged=unchanged,units='native; physical units NOT_ESTABLISHED',neural_controller=False,trajectory=False,physical_prewarm=False,autonomy=False))
    (out/'REPORT.zh-CN.md').write_text('# S156 固定状态静态敏感性\n\n状态：'+status+'\n\n仅显式激活覆盖反事实，不是神经控制器、轨迹、预热或自主成功。两基线全部通过前不执行任何扰动；容差绝对1e-9，不因结果放宽。所有原始状态、接触、48DOF约束闭合与模型质量矩阵保存在逐调用NPZ。原始单位生物标定NOT_ESTABLISHED。\n\n'+json.dumps(ledger,indent=2)+'\n')
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    print(str(out)); print(json.dumps(dict(status=status,ledger=ledger)))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path); a=p.parse_args()
    main(a.output or BASE/'s156-fixed-state-sensitivity'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
