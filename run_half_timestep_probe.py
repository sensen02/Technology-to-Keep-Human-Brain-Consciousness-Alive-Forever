"""S160/P1/P2: ONE S154 half-body-timestep contrast. Preparation is inert.
Only actual model change versus S154 is opt.timestep 25us -> 12.5us.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT = Path(__file__).resolve().parent
S154 = ROOT/'run_activation_rate_probe.py'
S142 = ROOT/'run_free_physical_prewarm_probe.py'
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s160-half-timestep')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inert(path, name, source=None):
    m=types.ModuleType(name); m.__file__=str(path)
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),m.__dict__)
    return m

def replace_exact(source, replacements):
    changes=[]
    for old,new in replacements:
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new,1); changes.append(dict(old=old,new=new))
    ast.parse(source)
    return source,changes

def prepare(selection=None):
    s154=inert(S154,'_s160_s154')
    old142,binding,_,_=s154.prepare(selection or s154.SELECTION)
    refs=[REFERENCE/n for n in ('PREREG.json','INPUT_HASHES.json','COMPLETE.json',
        'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py','init80ms/MODEL_PROVENANCE.json',
        'init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/INITIALIZATION.npz','init80ms/TRAJECTORY.npz')]
    producer=[Path(__file__).resolve(),S154,*refs]
    assert all(p.is_file() for p in producer)
    binding['hashes'].update({str(p):sha(p) for p in producer})
    # Do not bind parent analysis/checkers or any newly created parent file.
    module_source,module_changes=replace_exact(S142.read_text(),[
        ("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'",
         "protocol='S160 ONE bounded S154 half-timestep contrast; no assistance or activation assignment', EXPERIMENT='HALF_TIMESTEP', candidate=False, S119_same_condition_candidate=False, reference_s154="+repr(str(REFERENCE))+", sole_change_vs_s154='opt.timestep 25us to12.5us', half_interface_gain=True, activation_tau_intervention_s=.001"),
        ("physical_duration_ms=10,physical_steps_per_arm={'init80ms':400}","physical_duration_ms=10,physical_steps_per_arm={'init80ms':800}"),
        ("physical_samples_per_arm={'init80ms':401}","physical_samples_per_arm={'init80ms':801}"),
        ('20 substeps/tick; stop may be mid-tick','40 substeps/tick; stop may be mid-tick'),
        ('check release and every25us sample','check release and every12.5us sample'),
        ("        constraint = recorder.constraint_record(mj,m,d,index)","""        constraint = recorder.constraint_record(mj,m,d,index)
        # Passive metadata correction only: actual 40 body steps per neural tick.
        actual_tick = -1 if index == 0 else (index-1)//40
        constraint['physical_tick'] = np.asarray(actual_tick)
        constraint['source_start_s'] = np.asarray(.08 if index == 0 else .08+actual_tick*.0005)
        constraint['source_end_s'] = np.asarray(.08 if index == 0 else .08+(actual_tick+1)*.0005)
        assert abs(float(d.time)-index*.0000125) < 1e-12""")])
    s142=inert(S142,'_s160_private_s142',module_source)
    s142.stop_reasons=old142.stop_reasons
    prior=old142.private_source
    def transform(text):
        source,changes=prior(text)
        source,extra=replace_exact(source,[
            ("TIMESTEPS = {'init80ms': .000025}","TIMESTEPS = {'init80ms': .0000125}"),
            ("SUBSTEPS = {'init80ms': 20}","SUBSTEPS = {'init80ms': 40}"),
            ("EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=401, step=400, neural_ticks=180)}","EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=801, step=800, neural_ticks=180)}"),
            ('EXPECTED = dict(model_load=1, forward=401, step=400, neural_ticks=180)','EXPECTED = dict(model_load=1, forward=801, step=800, neural_ticks=180)'),
            ("assert ledger['step'] <= 400","assert ledger['step'] <= 800"),
            ("assert {k:v for k,v in after.items() if k != 'model.actuator_dynprm'} == {k:v for k,v in s154_ref_proof['after'].items() if k != 'model.actuator_dynprm'}", """assert {k:v for k,v in after.items() if k not in ('model.actuator_dynprm','opt.timestep')} == {k:v for k,v in s154_ref_proof['after'].items() if k not in ('model.actuator_dynprm','opt.timestep')}
            s160_reference_after = json.loads((Path(s160_reference)/'init80ms/MODEL_PROVENANCE.json').read_text())['after']
            s160_changed = sorted(k for k in after if after[k] != s160_reference_after[k])
            assert s160_changed == ['opt.timestep'] and m.opt.timestep == .0000125
            assert {k:v for k,v in after.items() if k != 'opt.timestep'} == {k:v for k,v in s160_reference_after.items() if k != 'opt.timestep'}
            with np.load(Path(s160_reference)/'init80ms/NATIVE_MODEL_ARRAYS.npz',allow_pickle=False) as s160_native:
                for s160_key in s160_native.files:
                    assert np.array_equal(np.asarray(getattr(m,s160_key)),s160_native[s160_key]), s160_key
            atomic_json(out/'S154_TIMESTEP_DELTA_PROOF.json',dict(reference=str(s160_reference),reference_after=s160_reference_after,actual_after=after,changed_vs_s154=s160_changed,all_other_native_numeric_fields_equal=True,all_saved_native_arrays_equal=True,old_timestep_s=.000025,new_timestep_s=.0000125))"""),
            ("    identities = {str(s): sha(s) for s in sources}","    sources = list(dict.fromkeys([*sources,*[Path(p) for p in s160_producer_bindings]]))\n    identities = {str(s): sha(s) for s in sources}"),
            ("str(ROOT/'run_activation_rate_probe.py')]:","str(ROOT/'run_activation_rate_probe.py'), str(ROOT/'run_half_timestep_probe.py')]:"),
            ("sole_model_change='activation tau .0005 -> .001 all90'", "sole_model_change='versus S154 ONLY opt.timestep .000025 -> .0000125; inherited XML-to-S154 tau derivation unchanged'"),
            ("interface='unchanged ctrl=.5 raw decoded all90; no act overwrite/filter/replay'", "reference_s154=str(s160_reference), interface='unchanged S154 immediate ctrl=.5 raw decoded all90; no act overwrite/filter/replay', substeps_per_tick=40, native_sampling_s=.0000125, first_fail_halfgrid_retained=True"),
            ('body_dt_s=.000025','body_dt_s=.0000125'),
            ('.5ms neural/25us body/5ms live sensory','.5ms neural/12.5us body/5ms live sensory')])
        source='s160_reference='+repr(str(REFERENCE))+'\ns160_producer_bindings='+repr([str(p) for p in producer])+'\n'+source
        compile(source,str(s142.ORIGINAL),'exec')
        assert 'PRE, TICKS = 160, 20' in source and 'd.ctrl[:] = 0.5 * u' in source
        assert "SUBSTEPS = {'init80ms': 40}" in source
        return source,changes+module_changes+extra
    s142.private_source=transform
    source,changes=transform(s142.ORIGINAL.read_text())
    return s142,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='auto'); p.add_argument('--prepare-only',action='store_true'); a=p.parse_args()
    s142,binding,source,changes=prepare()
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    if a.prepare_only:
        print(json.dumps(dict(status='INERT_COMPILE_OK',model_load=0,forward=0,step=0,neural_ticks=0,source_sha256=hashlib.sha256(source.encode()).hexdigest(),bound_inputs=len(binding['hashes'])))); return
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume or retry')
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; s142.main()
    finally: sys.argv=old
if __name__=='__main__': main()
