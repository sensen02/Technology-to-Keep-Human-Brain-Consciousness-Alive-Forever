"""Cached saved-evidence independent replay; zero integration, no training."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from run_one import HERE,OLD,prepare,native
sys.path.insert(0,str(OLD/'residual_candidate'))
from evaluate_candidate import load_checked
from residual_motor import sha256,BASE
sys.path.insert(0,str(BASE))
import student_wave as sw
from identity import verify,verify_runtime
from cycle_reward import CycleTracker,CycleConfig

def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    torch.set_num_threads(1)
    reg=json.loads((HERE/'PREREG.json').read_text());motor,_=load_checked(reg['checkpoint'],reg['checkpoint_sha256'])
    original=prepare();neutral=archive(OLD/'residual_candidate/EVAL_RESIDUAL500_neutral/trace_1ms.npz')
    checks={};summaries=[]
    for profile in reg['profiles']:
        path=HERE/profile;r=json.loads((path/'RESULT.json').read_text());ident=json.loads((path/'IDENTITY.json').read_text())
        verify(ident)
        c=sw.Core(adhesion=False);verify_runtime(ident,c)
        z=archive(path/'trace_1ms.npz');d=archive(path/'neural_decisions.npz');g=archive(path/'geometry_1ms.npz')
        def check(name,value):
            checks[profile+':'+name]=bool(value)
            if not value:raise ValueError(profile+':'+name)
        check('trace15400',z['qpos'].shape==(15400,49) and z['qvel'].shape==(15400,48))
        check('all_finite',all(np.isfinite(a).all() for a in [*z.values(),*d.values(),*g.values()]))
        check('all_times',np.all(np.diff(z['time'])>0) and np.allclose(np.diff(z['time']),.001,atol=1e-9,rtol=0))
        check('sources',np.array_equal(z['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]))
        check('zero_extra',all(np.max(np.abs(z[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
        check('all5800_prefix_exact',all(np.array_equal(z[k][:5800],neutral[k][:5800]) for k in z if k in neutral))
        check('actions_held',np.array_equal(z['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)))
        check('previous_action',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]))
        check('actor_actual_after_withdrawal',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]))
        expected=np.vstack([np.zeros((960,2)),np.asarray([original.command_at(profile,k) for k in range(1920)])])
        check('command_schedule',np.array_equal(d['command'],expected))
        state=motor.initial_state(1);out=[];hb=[];ha=[]
        with torch.no_grad():
            for x,u in zip(d['X238'],d['command']):
                hb.append(torch.cat(state,dim=-1)[0].numpy().copy())
                mean,state=motor.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
                out.append(mean[0].numpy().copy());ha.append(torch.cat(state,dim=-1)[0].numpy().copy())
        check('all2880_mean_exact',np.array_equal(np.asarray(out),d['mean']))
        check('all2880_hidden_before192_exact',np.array_equal(np.asarray(hb),d['hidden_before']))
        check('all2880_hidden_after192_exact',np.array_equal(np.asarray(ha),d['hidden_after']))
        check('all2880_policy_exact',np.array_equal(torch.tensor(np.asarray(out)).tanh().numpy(),d['policy_action']))
        check('hidden_chain',np.array_equal(d['hidden_after'][:-1],d['hidden_before'][1:]))
        rec=[(z['time'][i],z['com'][i],z['loads'][i],z['nonfoot'][i],z['speed'][i]) for i in range(5800,15400)]
        a=sw.summarise(rec,768000,initial_heading=c.initial_heading)
        check('original5_exact',a['gates']==r['autonomous']['gates'] and a['slip_ratio_max']==r['autonomous']['slip_ratio_max'])
        tracker=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(z['time'][5799]));events=[]
        for i in range(5800,15400):events.extend(tracker.update(dict(time=z['time'][i],loads=z['loads'][i],padz=z['padz'][i],foot=z['relativefoot'][i]))['events'])
        cr=sw.cycle_report(tracker,events)
        check('cycles_events_exact',cr==r['autonomous_cycles'])
        yr=original.yaw_report(profile,g['thorax_yaw']);check('yaw_exact',yr==r['yaw'])
        check('handover_and_switch_snapshots',all(v['all_exact'] and v['before_sha256']==v['after_sha256'] for v in r['boundaries'].values()))
        check('trace_hash',sha256(path/'trace_1ms.npz')==r['trace']['sha256'])
        check('neural_hash',sha256(path/'neural_decisions.npz')==r['neural_decisions_sha256'])
        check('geometry_hash',sha256(path/'geometry_1ms.npz')==r['geometry_sha256'])
        check('new_prereg_captured',str(HERE/'PREREG.json') in ident['files'] and str(HERE/'run_one.py') in ident['files'])
        check('compiled_identity',True)
        summaries.append(dict(profile=profile,gates=a['gates'],slip=a['slip_ratio_max'],cycles=cr['complete_per_leg'],yaw=yr,diagnostic_only=True,eligible=r['eligible']))
        del c,z,d,g
    report=dict(integrity_all_pass=all(checks.values()),checks=checks,profiles=summaries,physics_steps=0,food_qualification=False,source_sha256=sha256(__file__))
    (HERE/'VERIFICATION.json').write_text(json.dumps(native(report),indent=2)+'\n');print(json.dumps(dict(all_passed=report['integrity_all_pass'],checks=len(checks))),flush=True)
if __name__=='__main__':main()
