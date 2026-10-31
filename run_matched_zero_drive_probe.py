"""S161/P1/S2: one <=10ms S154-matched all90 zero-interface control.
Import/--stage-only do not construct models or advance neural/physical time.
No historical source is edited. Native passive muscles/contact remain enabled.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys
import run_activation_rate_probe as base
ROOT=Path(__file__).resolve().parent
ARCHIVE=Path('/home/sensen/Desktop/engine修复验收')
REFERENCE=ARCHIVE/'s154-activation-rate/20261024T215751775489Z'
S148=ARCHIVE/'s148-zero-control-ablation/20261024T212511892865Z'
S159=ARCHIVE/'s159-command-timing/20261024T222612451265673Z'
OUTPUT_PARENT=ARCHIVE/'s161-matched-zero-drive'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def existing_audit():
    import numpy as np
    inputs={}; scanned=[]; matched=[]
    def load(p):
        inputs[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    ref=load(REFERENCE/'init80ms/NATIVE_MODEL_ARRAYS.npz')
    init=load(REFERENCE/'init80ms/INITIALIZATION.npz')
    ref_prov=json.loads((REFERENCE/'init80ms/MODEL_PROVENANCE.json').read_text())['after']
    # Archive-wide existing dynamic records, excluding our new output family.
    for p in sorted(ARCHIVE.glob('**/init80ms/TRAJECTORY.npz')):
        if 's161-matched-zero-drive' in p.parts: continue
        with np.load(p,allow_pickle=False) as z:
            if 'ctrl' not in z.files: continue
            zero=not np.any(z['ctrl']); samples=len(z['time'])
        entry=dict(trajectory=str(p),samples=samples,all_saved_ctrl_zero=zero)
        if zero:
            inputs[str(p)]=sha(p); arm=p.parent; run=arm.parent
            mp=arm/'NATIVE_MODEL_ARRAYS.npz'; ip=arm/'INITIALIZATION.npz'; pp=arm/'MODEL_PROVENANCE.json'
            if mp.exists() and ip.exists() and pp.exists():
                m=load(mp); i=load(ip); inputs[str(pp)]=sha(pp)
                md=[k for k in ref if k not in m or not np.array_equal(ref[k],m[k])]
                idiff=[k for k in init if k not in i or not np.array_equal(init[k],i[k])]
                prov=json.loads(pp.read_text())['after']
                pdiff=[k for k in ref_prov if prov.get(k)!=ref_prov[k]]
                prereg=json.loads((run/'PREREG.json').read_text()); inputs[str(run/'PREREG.json')]=sha(run/'PREREG.json')
                duration=prereg.get('physical_duration_ms')
                entry.update(model_array_differences=md,full_native_snapshot_differences=pdiff,initialization_differences=idiff,requested_duration_ms=duration,complete=(run/'COMPLETE.json').exists())
                if not md and not idiff and not pdiff and duration==10 and entry['complete']: matched.append(str(run))
            else: entry['eligibility']='missing full-model/full-initial/safety provenance; cannot prove full match'
        scanned.append(entry)
    a=load(S148/'init80ms/NATIVE_MODEL_ARRAYS.npz')
    delta=np.argwhere(a['actuator_dynprm']!=ref['actuator_dynprm'])
    assert delta.shape==(90,2) and np.all(delta[:,1]==0)
    t=load(S159/'init80ms/TRAJECTORY.npz')
    return dict(reference=str(REFERENCE),matched_existing=matched,scanned=scanned,
        S148=dict(reusable=False,reason='activation tau all90 .0005 vs S154 .001; requested250us not max10ms first-failure protocol; mathematical zero-active invariance does not meet identical-parameter request',differing_dynprm_indices=delta.tolist(),old_tau=.0005,required_tau=.001,deactivation_tau=.0004),
        S159=dict(reusable=False,reason='initial tick zero then delayed active commands; not a full zero arm',first_ctrl_zero=not np.any(t['ctrl'][1:21]),later_ctrl_nonzero=bool(np.any(t['ctrl'][21:]))),input_hashes=inputs)

def prepare():
    audit=existing_audit()
    if audit['matched_existing']: return None,None,None,audit
    old,binding,_,_=base.prepare(base.SELECTION)
    assert sha(ROOT/'run_activation_rate_probe.py')==sha(REFERENCE/'SOURCE_FREEZE/S146_run_activation_rate_probe.py')
    refs=[REFERENCE/n for n in ('PREREG.json','COMPLETE.json','init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/INITIALIZATION.npz','init80ms/TRAJECTORY.npz','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')]
    binding['hashes'].update(audit['input_hashes'])
    binding['hashes'].update({str(p):sha(p) for p in [*refs,Path(__file__).resolve()]})
    prior=old.private_source
    replacer=base.inert(base.S149,'_s161_replace').replace_exact
    def transform(text):
        source,changes=prior(text)
        source,extra=replacer(source,[
            ('                d.ctrl[:] = 0.5 * u','                d.ctrl[:] = s161_interface_gains * u'),
            ('applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),','applied=np.asarray(commands)*s161_interface_gains,interface_gains=s161_interface_gains.copy(),decoded=np.asarray(commands),current=np.asarray(currents),'),
            ("    # S154 new derived numeric configuration, saved and bound BEFORE model load.","    s161_interface_gains = np.zeros(90,dtype=float)\n    atomic_npz(out/'INTERFACE_CONFIG.npz',interface_gains=s161_interface_gains)\n    atomic_json(out/'HISTORICAL_CONTROL_COMPARISON.json',s161_audit)\n    atomic_json(out/'S161_PREANALYSIS.json',dict(reference=s161_reference,sole_contrast='all90 interface gains .5 -> 0 throughout release; preserve raw decoded',model_tau_act_s=.001,model_tau_deact_s=.0004,body_dt_s=.000025,integrator='RK4',PRE=160,max_duration_ms=10,first_failure_stop=True,original_six_gates=True,original_height_reference=True,activation_overwrite=False,assistance=False,passive_muscles_and_contacts='UNCHANGED',sensory='genuine time0 then5ms if reached; no PRE fill/no replay',interface_config_sha256=sha(out/'INTERFACE_CONFIG.npz'),historical_audit_sha256=sha(out/'HISTORICAL_CONTROL_COMPARISON.json')))\n    # S154 new derived numeric configuration, saved and bound BEFORE model load."),
            ("str(ROOT/'run_activation_rate_probe.py')]:","str(ROOT/'run_activation_rate_probe.py'),str(ROOT/'run_matched_zero_drive_probe.py')]:"),
            ("applied_interface='all90 ctrl=0.5*decoded raw u after decode; no act overwrite; activation tau .001 only'","applied_interface='S161 all90 ctrl=0*decoded raw u throughout release; raw preserved; no act overwrite; tau .001/.0004 same S154'"),
            ("sole_model_change='activation tau .0005 -> .001 all90'","sole_model_change='NONE versus S154; XML-derived tau patch exactly S154 .001/.0004',sole_contrast='all90 interface gains .5 -> 0 for entire physical arm'"),
            ("interface='unchanged ctrl=.5 raw decoded all90; no act overwrite/filter/replay'","interface='all90 ctrl=0*rawdecoded entire physical arm; no act overwrite/filter/replay'"),
        ])
        source='s161_audit='+repr(audit)+'\ns161_reference='+repr(str(REFERENCE))+'\n'+source
        ast.parse(source); compile(source,str(old.ORIGINAL),'exec')
        assert 'd.ctrl[:] = 0.5 * u' not in source
        return source,changes+extra
    module_text=base.S142.read_text().replace("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'","protocol='S161 ONE S154-matched zero-drive arm; passive muscles/contact unchanged',EXPERIMENT='MATCHED_ZERO_DRIVE',candidate=False,S119_same_condition_candidate=False,interface_gain_all90=0,reference="+repr(str(REFERENCE)))
    module=base.inert(base.S142,'_s161_private',module_text)
    module.private_source=transform; module.stop_reasons=old.stop_reasons
    source,changes=transform(module.ORIGINAL.read_text())
    return module,binding,source,audit

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='auto'); p.add_argument('--stage-only',action='store_true'); a=p.parse_args()
    module,binding,source,audit=prepare()
    if audit['matched_existing']:
        print(json.dumps(dict(status='REUSE_EXISTING_NO_NEW_RUN',proof=audit))); return
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    if a.stage_only:
        print(json.dumps(dict(status='INERT_COMPILE_OK',model_load=0,forward=0,step=0,neural_ticks=0,source_sha256=hashlib.sha256(source.encode()).hexdigest(),existing_matching_arms=audit['matched_existing'],scanned=len(audit['scanned']),S148=audit['S148'],S159=audit['S159']))); return
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume/retry')
    previous=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; module.main()
    finally: sys.argv=previous
if __name__=='__main__': main()
