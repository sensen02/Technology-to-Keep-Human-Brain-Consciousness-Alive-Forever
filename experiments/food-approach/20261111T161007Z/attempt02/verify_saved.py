from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent.parent.parent/'banc-walking-recovery/20261110T152714Z'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
sys.path.insert(0,str(BASE))
import identity
from physics import summarise
from cycle_reward import CycleTracker,CycleConfig

def verify(folder):
    r=json.loads((folder/'RESULT.json').read_text());manifest=json.loads((folder/'IDENTITY.json').read_text());identity.verify(manifest)
    with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:t={k:z[k] for k in z.files}
    with np.load(folder/'SAMPLES_5MS.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    with np.load(folder/'YAW_1MS.npz',allow_pickle=False) as z:y={k:z[k] for k in z.files}
    checks={}
    def check(k,v):checks[k]=bool(v)
    check('trace_hash',identity.sha256(folder/'TRACE_1MS.npz')==r['trace']['sha256'])
    check('samples_hash',identity.sha256(folder/'SAMPLES_5MS.npz')==r['samples']['sha256'])
    check('full15400',len(t['time'])==15400)
    check('finite_all',all(np.isfinite(a).all() for group in (t,s,y) for a in group.values()))
    check('full2880samples',s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42) and s['command2'].shape==(2880,2))
    check('sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,5)]))
    check('clock1ms',np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8)
    check('clock5ms',np.max(np.abs(s['actual_time']-(1+np.arange(2880)*.005)))<1e-8)
    check('actions_equal',np.array_equal(t['action'],t['actual_requested_action']))
    check('samples_actions_held',np.array_equal(t['action'][1000:],np.repeat(s['Y42'],5,axis=0)))
    check('no_extras',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('normalization',np.max(np.abs(s['command2'][:,1]-s['command_speed_yaw'][:,1]/.06))<1e-14 and np.max(np.abs(s['command2'][:,0]))<1e-14)
    expected=np.clip((np.arange(2880)-960)*.005,0,1);expected=expected**2*(3-2*expected)*np.sign(r['condition']['yaw_rate'])
    check('ramp',np.max(np.abs(s['command2'][:,1]-expected))<1e-12)
    heading=np.unwrap(np.r_[float(y['boundary_yaw']),y['yaw_world'][4800:]])
    actual=float(heading[-1]-heading[0]);check('yaw_metric',abs(actual-r['actual_signed_yaw_rad'])<1e-12)
    rec=[(t['time'][i],t['com'][i],t['loads'][i],t['nonfoot'][i],t['speed'][i]) for i in range(5800,15400)]
    metrics=summarise(rec,len(rec)*80)
    for k in ('slip_ratio_max','com_height_min','foot_support_mean','forward_signed_native_per_s','nonfoot_force_max_native'):
        check('metric_'+k,abs(metrics[k]-r[k])<1e-10)
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    counts={leg:0 for leg in r['cycles']['complete_per_leg']}
    for i in range(5800,15400):
        step=observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],foot=t['relativefoot'][i]))
        for event in step['events']:
            if event['type']=='complete':counts[event['leg']]+=1
    check('all_cycle_counts_recomputed',counts==r['cycles']['complete_per_leg'])
    check('native_count',r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000)
    eligible=bool(r['all_gates'] and r['cycles']['all_legs_completed'] and r['native']['post_nonfoot_peak_native']==0 and abs(actual)>=np.pi/6 and actual*r['condition']['yaw_rate']>0 and r['early_stop'] is None)
    result=dict(integrity_all_pass=all(checks.values()),checks=checks,teacher_eligible=eligible,yaw_deg=np.degrees(actual),gates=r['gates'],slip=r['slip_ratio_max'],cycles=r['cycles']['complete_per_leg'],native_peak=r['native']['post_nonfoot_peak_native'],scope='saved-array independent audit and captured native summaries; no physics constructed')
    (folder/'VERIFY.json').write_text(json.dumps(result,indent=2)+'\n');assert all(checks.values()),checks
    return result
if __name__=='__main__':
    rows={name:verify(HERE/name) for name in ('left_yaw_p060','right_yaw_m060')}
    (HERE/'VERIFIED_SUMMARY.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
