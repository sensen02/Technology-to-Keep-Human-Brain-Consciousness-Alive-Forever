"""Read-only offline actual-rest reference replay, independent analytic schedule."""
import json,hashlib
from pathlib import Path
import numpy as np
from api import HERE,CONDITIONS,FixedCadenceStrideReference
from wave_reference import WaveReference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    original=HERE.parent/'teacher'/'left_full'/'TRACE_1MS.npz'
    with np.load(original,allow_pickle=False) as z:rest=z['relativefoot'][999].copy()
    base=WaveReference(rest,stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    checks={};metrics={}
    preflight=json.loads((HERE/'REFERENCE_CHECKS.json').read_text())
    checks['recorded_preflight_all12_named_checks_true']=len(preflight['checks'])==12 and all(preflight['checks'].values()) and preflight['all_passed'] and preflight['actual_physics_steps']==0
    for cfg in CONDITIONS:
        ref=FixedCadenceStrideReference(base,cfg['stride_scale']);worstx=worstv=0.;same=True;phase=True
        for decision in range(2880):
            t=decision*.005;b=base.sample(t);s=ref.sample(t)
            u=min(1.,max(0.,t-4.8));scale=1+(cfg['stride_scale']-1)*u*u*(3-2*u);rate=(cfg['stride_scale']-1)*6*u*(1-u)
            expectedx=rest[:,0]+scale*(b['target_xy'][:,0]-rest[:,0]);expectedv=scale*b['velocity_xy'][:,0]+rate*(b['target_xy'][:,0]-rest[:,0])
            worstx=max(worstx,float(np.max(np.abs(s['target_xy'][:,0]-expectedx))))
            worstv=max(worstv,float(np.max(np.abs(s['velocity_xy'][:,0]-expectedv))))
            same=same and all(np.array_equal(s[k],b[k]) for k in ('target_z','velocity_z','stance')) and np.array_equal(s['target_xy'][:,1],b['target_xy'][:,1]) and np.array_equal(s['velocity_xy'][:,1],b['velocity_xy'][:,1])
            phase=phase and s.get('source_time_s')==b.get('source_time_s') and s.get('phase')==b.get('phase')
        checks[cfg['name']+'_all2880_independent_analytic_x']=worstx<1e-14
        checks[cfg['name']+'_all2880_independent_productrule_vx']=worstv<1e-14
        checks[cfg['name']+'_all2880_lateral_vertical_stance_bitexact']=same
        checks[cfg['name']+'_all2880_original_sourcephase']=phase
        metrics[cfg['name']]=dict(max_x_error=worstx,max_vx_error=worstv)
    inputs=json.loads((HERE/'INPUT_IDENTITY.json').read_text())
    checks['audited_actual_reference_source_hashes_still_frozen']=all(sha(HERE/name)==inputs['source_hashes'][str(HERE/name)] for name in ('api.py','run.py','verify.py','check_reference.py','PREREG.json'))
    result=dict(all_passed=all(checks.values()),checks=checks,metrics=metrics,
        files={str(p):sha(p) for p in (HERE/'REFERENCE_CHECKS.json',HERE/'api.py',HERE/'check_reference.py',original,Path(__file__).resolve())},
        scope='Offline2880times actualsavedsettledrest reference replay, zero physics; preflight result association captured after start, not claimed frozen beforerun')
    (HERE/'REFERENCE_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['all_passed']
if __name__=='__main__':main()
