"""Offline schedule/index/criterion checks only; never imports the physical backend."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from evaluate_profile import PROFILES,command_at,yaw_report


def main():
    here=Path(__file__).resolve().parent
    checks={}
    prereg=json.loads((here/'PREREG.json').read_text())
    checks['exact_five_preregistered_profiles']=prereg['profiles']==list(PROFILES) and len(PROFILES)==5
    for name,turns in PROFILES.items():
        commands=np.array([command_at(name,k) for k in range(1920)])
        checks[name+'_forward_always_zero']=bool(np.all(commands[:,0]==0.))
        checks[name+'_first960_exact']=bool(np.all(commands[:960,1]==turns[0]))
        checks[name+'_last960_exact']=bool(np.all(commands[960:,1]==turns[1]))
        checks[name+'_switch_boundary']=commands[959,1]==turns[0] and commands[960,1]==turns[1]
        # Full trace endpoints:5.8s index5799,10.6s10599,15.4s15399.
        angle=np.zeros(15400)
        first=.003*np.sign(turns[0]);second=.003*np.sign(turns[1])
        angle[5800:10600]=np.linspace(first/4800,first,4800)
        angle[10600:]=first+np.linspace(second/4800,second,4800)
        report=yaw_report(name,angle)
        checks[name+'_exact_split_indexing']=abs(report['first4p8_rad']-first)<1e-14 and abs(report['second4p8_rad']-second)<1e-14
        checks[name+'_matching_sign_passes']=report['passed']
        if name!='neutral':checks[name+'_wrong_sign_fails']=not yaw_report(name,-angle)['passed']
        # Wrap across +/-pi to ensure no spurious full rotations.
        wrapped=(angle+np.pi-.001+np.pi)%(2*np.pi)-np.pi
        checks[name+'_unwrap_crossing']=abs(yaw_report(name,wrapped)['whole_rad']-report['whole_rad'])<1e-12
    for invalid in (-1,1920):
        try:command_at('neutral',invalid)
        except ValueError:checks['reject_decision_'+str(invalid)]=True
        else:checks['reject_decision_'+str(invalid)]=False
    zero=np.zeros(15400)
    checks['intermediate_zero_yaw_fails']=not yaw_report('halfleft',zero)['passed']
    checks['switching_zero_yaw_fails']=not yaw_report('left_to_right',zero)['passed']
    checks['offline_import_no_physics']=not any(n in sys.modules for n in ('physics','mujoco','student_wave','torch'))
    for path in here.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
        checks['syntax:'+path.name]=True
    report=dict(all_passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},physical_steps=0,checkpoint_loads=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.iterdir()) if p.suffix in ('.py','.json') and p.name!='OFFLINE_CHECKS.json'},
        limitations='Offline schedule, yaw index/criteria and syntax only; no physical candidate evaluated.')
    out=here/'OFFLINE_CHECKS.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,out=str(out))))
    if not report['all_passed']:raise RuntimeError('offline profile checks failed')

if __name__=='__main__':main()
