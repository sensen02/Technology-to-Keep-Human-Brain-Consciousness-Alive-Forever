"""S154/P1 ONE bounded S149 activation-tau contrast; imports/preparation inert.
Only dynprm[:,0] changes .0005 -> .001. New numeric config; frozen XML untouched.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT=Path(__file__).resolve().parent
S149=ROOT/'run_half_gain_physical_probe.py'
S142=ROOT/'run_free_physical_prewarm_probe.py'
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical/20261024T213108429114Z')
SELECTION=Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z')
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate')
HEADER=ROOT/'venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mujoco.h'
SCALAR=ROOT/'verify_neural_activation_initialized_probe.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inert(path,name,source=None):
    m=types.ModuleType(name); m.__file__=str(path)
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),m.__dict__)
    return m

def semantics():
    import numpy as np
    import mujoco as mj
    assert mj.__version__=='3.9.0'
    assert '// Muscle activation dynamics, prm = (tau_act, tau_deact, smoothing_width).' in HEADER.read_text()
    tree=ast.parse(SCALAR.read_text()); node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scalar_rate')
    namespace={}; exec(compile(ast.Module(body=[node],type_ignores=[]),str(SCALAR),'exec'),namespace)
    rows=[]
    for a,u in [(0.,.5),(.2,.5),(.8,.1),(.5,.5)]:
        for tau in [.0005,.001]:
            prm=np.array([tau,.0004,0.]); native=float(mj.mju_muscleDynamics(u,a,prm)); scalar=float(namespace['scalar_rate'](a,u,prm))
            assert abs(native-scalar)<1e-10
            rows.append(dict(a=a,u=u,tau_act=tau,tau_deact=.0004,native_rate=native,scalar_rate=scalar))
    return dict(mujoco_version=mj.__version__,header=str(HEADER),header_sha256=sha(HEADER),scalar_source=str(SCALAR),scalar_sha256=sha(SCALAR),field_semantics='prm0=tau_act; prm1=tau_deact; prm2=smoothing_width',checks=rows,model_loads=0)

PREMODEL='''    # S154 new derived numeric configuration, saved and bound BEFORE model load.
    s154_ref = Path(s154_reference)
    with np.load(s154_ref/'init80ms/NATIVE_MODEL_ARRAYS.npz',allow_pickle=False) as z:
        s154_old = z['actuator_dynprm'].copy()
    assert s154_old.shape == (90,10) and np.all(s154_old[:,0] == .0005) and np.all(s154_old[:,1] == .0004)
    s154_new = s154_old.copy(); s154_new[:,0] = .001
    atomic_npz(out/'DERIVED_NUMERIC_CONFIG.npz',actuator_dynprm=s154_new)
    atomic_json(out/'DERIVED_NUMERIC_CONFIG.json',dict(schema='new derived numeric configuration; not a modified old XML',base_xml=str(MODEL),base_xml_sha256=sha(MODEL),artifact=str(out/'DERIVED_NUMERIC_CONFIG.npz'),artifact_sha256=sha(out/'DERIVED_NUMERIC_CONFIG.npz'),reference=str(s154_ref),old_prm_sha256=hashlib.sha256(s154_old.tobytes()).hexdigest(),new_prm_sha256=hashlib.sha256(s154_new.tobytes()).hexdigest(),sole_change='all90 actuator_dynprm[:,0] .0005 -> .001; deactivation .0004 and all other tokens unchanged',private_runner_sha256=sha(out/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')))
    (out/'INPUT_DELTA.csv').write_text('actuator,field,old,new\\n'+''.join(f'{i},actuator_dynprm[0],0.0005,0.001\\n' for i in range(90)))
    atomic_json(out/'PREANALYSIS.json',dict(reference=str(s154_ref),sole_model_change='activation tau .0005 -> .001 all90',interface='unchanged ctrl=.5 raw decoded all90; no act overwrite/filter/replay',initial='S149 same qpos and sensory reference; zero qvel/act/ctrl; genuine PRE160',duration_max_ms=10,body_dt_s=.000025,first_fail_stop=True,original_six_gates_and_original_height_reference=True,success_rule='early lower work/vz ALONE NOT support success; compare later first failure, load/sink/min vz/height/angle/nonfoot/limits; no regression allowed; no autonomous success claim',sensory_rule='live time0 and5ms only if reached; otherwise time0 only NOT closed-loop verification',semantics=s154_semantics,physical_units='NOT_ESTABLISHED'))
'''
PATCH='''            s154_native_before = numeric_model_snapshot(m)
            s154_ref_proof = json.loads((Path(s154_reference)/'init80ms/MODEL_PROVENANCE.json').read_text())
            assert s154_native_before == s154_ref_proof['before']
            assert np.array_equal(m.actuator_dynprm,s154_old)
            atomic_npz(out/'NATIVE_PREPATCH.npz',actuator_dynprm=m.actuator_dynprm.copy())
            with np.load(out/'DERIVED_NUMERIC_CONFIG.npz',allow_pickle=False) as z:
                m.actuator_dynprm[:,0] = z['actuator_dynprm'][:,0]
            assert np.array_equal(m.actuator_dynprm,s154_new)
            s154_native_after = numeric_model_snapshot(m)
            assert [k for k in s154_native_before if s154_native_before[k] != s154_native_after[k]] == ['model.actuator_dynprm']
            atomic_npz(out/'NATIVE_POSTPATCH.npz',actuator_dynprm=m.actuator_dynprm.copy())
            atomic_json(out/'NATIVE_INPUT_DELTA_PROOF.json',dict(before=s154_native_before,after=s154_native_after,changed=['model.actuator_dynprm'],all_other_native_arrays_equal=True,only_dynprm_column0_changed=True,config_sha256=sha(out/'DERIVED_NUMERIC_CONFIG.npz')))
'''

def prepare(selection):
    s149=inert(S149,'_s154_s149'); s142,binding,_,_=s149.prepare(selection)
    refs=[REFERENCE/n for n in ('PREREG.json','INPUT_HASHES.json','COMPLETE.json','HALF_GAIN_INDEPENDENT_CHECK.json','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py','SOURCE_FREEZE/S146_run_half_gain_physical_probe.py','init80ms/MODEL_PROVENANCE.json','init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/INITIALIZATION.npz','init80ms/TRAJECTORY.npz')]
    assert sha(S149)==sha(REFERENCE/'SOURCE_FREEZE/S146_run_half_gain_physical_probe.py')
    proof=json.loads((REFERENCE/'init80ms/MODEL_PROVENANCE.json').read_text()); assert proof['before']['model.actuator_dynprm']==proof['after']['model.actuator_dynprm']
    sem=semantics()
    binding['hashes'].update({str(p):sha(p) for p in [*refs,HEADER,SCALAR,Path(__file__).resolve()]})
    prior=s142.private_source; reasons=s142.stop_reasons
    module_source,_=s149.replace_exact(S142.read_text(),[("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'","protocol='S154 ONE bounded activation tau contrast versus S149; no assistance or activation assignment', EXPERIMENT='ACTIVATION_RATE', candidate=False, S119_same_condition_candidate=False, half_interface_gain=True, activation_tau_intervention_s=.001")])
    s142=inert(S142,'_s154_private_s142',module_source); s142.stop_reasons=reasons
    def transform(text):
        source,changes=prior(text)
        source,extra=s149.replace_exact(source,[
            ("    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,",PREMODEL+"    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,"),
            ('            before = numeric_model_snapshot(m)','            before = numeric_model_snapshot(m)\n'+PATCH),
            ("assert changed == ['opt.integrator','opt.timestep']","assert changed == ['model.actuator_dynprm','opt.integrator','opt.timestep']\n            assert {k:v for k,v in after.items() if k != 'model.actuator_dynprm'} == {k:v for k,v in s154_ref_proof['after'].items() if k != 'model.actuator_dynprm'}"),
            ("numerical_match_except_timestep=True, xml_sha256=sha(MODEL)","numerical_match_except_timestep=False, sole_extra_model_change='actuator_dynprm[:,0]', derived_numeric_config_sha256=sha(out/'DERIVED_NUMERIC_CONFIG.npz'), xml_sha256=sha(MODEL)"),
            ("no act overwrite; no tau change","no act overwrite; activation tau .001 only"),
            ("str(ROOT/'run_half_gain_physical_probe.py')]:","str(ROOT/'run_half_gain_physical_probe.py'), str(ROOT/'run_activation_rate_probe.py')]:"),
            ("clocks='continuous neural/FIR .08s offset; source END at 83ms; physical START at 80ms operator split'","clocks='PRE160 genuine neural-only0..80ms; physical0..<=10ms; .5ms neural/25us body/5ms live sensory if reached'")])
        source='import hashlib\ns154_reference='+repr(str(REFERENCE))+'\ns154_semantics='+repr(sem)+'\n'+source
        ast.parse(source); compile(source,str(s142.ORIGINAL),'exec')
        assert source.count('m.actuator_dynprm[:,0] =')==1 and 'd.ctrl[:] = 0.5 * u' in source
        return source,changes+extra
    s142.private_source=transform
    source,changes=transform(s142.ORIGINAL.read_text())
    return s142,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selection',type=Path,default=SELECTION); p.add_argument('--output',default='auto'); a=p.parse_args()
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume')
    s142,binding,source,changes=prepare(a.selection)
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; s142.main()
    finally: sys.argv=old
if __name__=='__main__': main()
