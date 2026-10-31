"""S155/P1 ONE actual activation-dose contrast: S154 .001 -> .002.
Inert preparation reuses S154; native XML still loads .0005 before one patch.
No edits to old producers or evidence; no retries/search/assistance.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, sys
import run_activation_rate_probe as base
ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s155-activation-dose')

PREMODEL='''    # S155: bind original native and prior intervention separately BEFORE model load.
    s154_ref = Path(s154_reference)
    s155_ref = Path(S155_REFERENCE)
    with np.load(s154_ref/'init80ms/NATIVE_MODEL_ARRAYS.npz',allow_pickle=False) as z:
        s154_old = z['actuator_dynprm'].copy()
    with np.load(s155_ref/'init80ms/NATIVE_MODEL_ARRAYS.npz',allow_pickle=False) as z:
        s155_prior = z['actuator_dynprm'].copy()
    assert s154_old.shape == (90,10) and np.all(s154_old[:,0] == .0005) and np.all(s154_old[:,1] == .0004)
    assert np.all(s155_prior[:,0] == .001)
    assert np.array_equal(s155_prior[:,1:],s154_old[:,1:])
    s154_new = s154_old.copy(); s154_new[:,0] = .002
    atomic_npz(out/'DERIVED_NUMERIC_CONFIG.npz',actuator_dynprm=s154_new)
    atomic_json(out/'DERIVED_NUMERIC_CONFIG.json',dict(schema='S155 new numeric config; original XML unchanged',base_xml=str(MODEL),base_xml_sha256=sha(MODEL),artifact=str(out/'DERIVED_NUMERIC_CONFIG.npz'),artifact_sha256=sha(out/'DERIVED_NUMERIC_CONFIG.npz'),baseline=str(s154_ref),reference=str(s155_ref),baseline_array_file_sha256=sha(s154_ref/'init80ms/NATIVE_MODEL_ARRAYS.npz'),reference_array_file_sha256=sha(s155_ref/'init80ms/NATIVE_MODEL_ARRAYS.npz'),baseline_prm_sha256=hashlib.sha256(s154_old.tobytes()).hexdigest(),reference_prm_sha256=hashlib.sha256(s155_prior.tobytes()).hexdigest(),new_prm_sha256=hashlib.sha256(s154_new.tobytes()).hexdigest(),reference_delta='S154 all90 .001 -> .002',native_prepatch_delta='original S149/baseXML all90 .0005 -> .002; native prepatch is NOT .001',other_parameters='deactivation .0004 and all remaining entries unchanged',private_runner_sha256=sha(out/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')))
    (out/'INPUT_DELTA.csv').write_text('actuator,field,baseline_old,reference_old,new\\n'+''.join(f'{i},actuator_dynprm[0],0.0005,0.001,0.002\\n' for i in range(90)))
    atomic_json(out/'PREANALYSIS.json',dict(reference=str(s155_ref),baseline=str(s154_ref),sole_model_change='all90 activation tau .002 vs S154 .001 vs base .0005',interface='all90 ctrl=.5 raw decode; no filter/act overwrite/replay/assistance/pose or velocity correction',initial='same S149/S154 initial state and original sensory/height reference; zero qvel act ctrl; real PRE160',duration_max_ms=10,body_dt_s=.000025,first_fail_stop=True,original_six_gates_and_original_height_reference=True,preregistered_tests=[dict(id='T1',window_s=[0,.00025],prediction='active muscle work and COM vz lower than S154; compare exact pairwise common window, work native units'),dict(id='T2',prediction='first original-six-gate failure strictly after S154 2.1ms; retain first failing sample; no continuation'),dict(id='T3',prediction='no sinking/earlier load failure or worse LM/RM unloading versus S154; parent reports full common-window curves and minima, no compensation by early work benefit')],comparison='parent independent pairwise S155 vs S154 and S149 on each exact common saved window; include failure samples',sensory_rule='live physical0 and5ms only if reached; earlier stop explicitly NOT refreshed 5ms feedback',success_rule='early benefit alone NOT support success; retain all failures; no stable support/preatwarm/autonomous claim',semantics=s154_semantics,physical_units='NOT_ESTABLISHED'))
'''
PATCH='''            s154_native_before = numeric_model_snapshot(m)
            s154_ref_proof = json.loads((Path(s154_reference)/'init80ms/MODEL_PROVENANCE.json').read_text())
            s155_ref_proof = json.loads((Path(S155_REFERENCE)/'init80ms/MODEL_PROVENANCE.json').read_text())
            assert s154_native_before == s154_ref_proof['before'] == s155_ref_proof['before']
            assert np.array_equal(m.actuator_dynprm,s154_old)
            atomic_npz(out/'NATIVE_PREPATCH.npz',actuator_dynprm=m.actuator_dynprm.copy())
            with np.load(out/'DERIVED_NUMERIC_CONFIG.npz',allow_pickle=False) as z:
                m.actuator_dynprm[:,0] = z['actuator_dynprm'][:,0]
            assert np.array_equal(m.actuator_dynprm,s154_new)
            s154_native_after = numeric_model_snapshot(m)
            assert [k for k in s154_native_before if s154_native_before[k] != s154_native_after[k]] == ['model.actuator_dynprm']
            atomic_npz(out/'NATIVE_POSTPATCH.npz',actuator_dynprm=m.actuator_dynprm.copy())
            atomic_json(out/'NATIVE_INPUT_DELTA_PROOF.json',dict(before=s154_native_before,after=s154_native_after,changed=['model.actuator_dynprm'],all_other_native_arrays_equal=True,only_dynprm_column0_changed=True,native_prepatch_tau=.0005,reference_S154_tau=.001,new_tau=.002,config_sha256=sha(out/'DERIVED_NUMERIC_CONFIG.npz')))
'''

def prepare(selection=base.SELECTION):
    # Specific count-checked source tokens only. Never replace numeric literals globally.
    replacements=[
      ('for tau in [.0005,.001]:','for tau in [.0005,.001,.002]:'),
      ("protocol='S154 ONE bounded activation tau contrast versus S149; no assistance or activation assignment', EXPERIMENT='ACTIVATION_RATE'", "protocol='S155 ONE bounded activation dose contrast versus S154 and S149; no assistance or activation assignment', EXPERIMENT='ACTIVATION_DOSE'"),
      ('activation_tau_intervention_s=.001','activation_tau_intervention_s=.002'),
      ('no act overwrite; activation tau .001 only','no act overwrite; activation tau .002 only'),
      ("str(ROOT/'run_activation_rate_probe.py')]:", "str(ROOT/'run_activation_rate_probe.py'), str(ROOT/'run_activation_dose_probe.py')]:"),
      ("    assert sha(S149)==sha(REFERENCE/'SOURCE_FREEZE/S146_run_half_gain_physical_probe.py')", "    refs += [Path("+repr(str(REFERENCE))+ ")/n for n in ('PREREG.json','PREANALYSIS.json','INPUT_HASHES.json','COMPLETE.json','DERIVED_NUMERIC_CONFIG.npz','DERIVED_NUMERIC_CONFIG.json','init80ms/MODEL_PROVENANCE.json','init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/INITIALIZATION.npz','init80ms/TRAJECTORY.npz','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')]\n    assert sha(S149)==sha(REFERENCE/'SOURCE_FREEZE/S146_run_half_gain_physical_probe.py')"),
      ("s154_ref_proof['after'].items()", "s155_ref_proof['after'].items()"),
    ]
    import run_half_gain_physical_probe as helper
    text,_=helper.replace_exact((ROOT/'run_activation_rate_probe.py').read_text(),replacements)
    module=base.inert(ROOT/'run_activation_rate_probe.py','_s155_s154',text)
    module.PREMODEL='    S155_REFERENCE = '+repr(str(REFERENCE))+'\n'+PREMODEL
    module.PATCH=PATCH
    runner,binding,source,changes=module.prepare(selection)
    ast.parse(source); compile(source,str(runner.ORIGINAL),'exec')
    assert source.count('m.actuator_dynprm[:,0] =')==1
    assert 's154_new[:,0] = .002' in source
    assert "'model.actuator_dynprm'} == {k:v for k,v in s155_ref_proof['after']" in source
    return runner,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='auto'); a=p.parse_args()
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume')
    runner,binding,source,changes=prepare()
    assert binding['hashes']=={n:base.sha(n) for n in binding['hashes']}
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; runner.main()
    finally: sys.argv=old
if __name__=='__main__': main()
