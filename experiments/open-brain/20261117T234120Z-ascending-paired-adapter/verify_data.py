"""Independent stored-data checks; no new neural or physical integration."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import *
from adapter import AscendingRateAdapter


def load(name):
    path=HERE/'DATA'/name
    meta=read(path/'RESULT.json')
    verify(meta['source_sha256']); verify(meta['runtime_source_hashes'])
    assert sha(path/'INPUTS.npz')==meta['inputs_sha256']
    with np.load(path/'INPUTS.npz',allow_pickle=False) as a:
        values={k:a[k].copy() for k in a.files}
    return values,meta


def main():
    torch.set_num_threads(1)
    records={}
    for name in ('train_left','train_right','held_left','held_right','held_left_cut_left','held_left_cut_right'):
        a,m=load(name)
        assert all(np.isfinite(v).all() for v in a.values())
        assert 'native_teacher_rate_hz' not in a
        checks={}
        if m['cut_side']=='none':
            for c,d in enumerate(m['port_delay_ticks']):
                checks[f'delay_exact_channel{c}']=bool(np.array_equal(a['emitted'][:-d,c],a['arrivals'][d:,c]) and not a['arrivals'][:d,c].any())
            checks['both_channels_transmitted']=bool(a['arrivals'].sum(0).min()>0)
        else:
            base,_=load('held_left'); side=0 if m['cut_side']=='left' else 1
            checks['cut_arrivals_zero']=not bool(a['arrivals'][:,side].any())
            checks['cut_features_zero']=not bool(a['features_hz'][:,side].any())
            checks['uncut_arrivals_exact']=bool(np.array_equal(base['arrivals'][:,1-side],a['arrivals'][:,1-side]))
            checks['uncut_features_exact']=bool(np.array_equal(base['features_hz'][:,1-side],a['features_hz'][:,1-side]))
        records[name]=dict(checks=checks,passed=all(checks.values()),arrivals=m['arrival_counts'])
    replays={}
    for name,reference in (('held_left_replay','held_left'),('held_left_cut_left_replay','held_left_cut_left')):
        a,_=load(name); b,_=load(reference)
        checks={k:bool(np.array_equal(a[k],b[k])) for k in a}
        replays[name]=dict(array_equal=checks,passed=all(checks.values()))
    model=AscendingRateAdapter(2)
    z=torch.zeros((100,2,3),dtype=torch.float64)
    mask=torch.zeros(2,dtype=torch.bool)
    silent=bool((model(z,mask)==0).all())
    z[:,0,:]=100.
    rate=model(z,mask)
    isolated=bool((rate[:,1]==0).all() and (rate[:,0]>0).all())
    bounded=bool((rate>=0).all() and (rate<=300).all())
    mask[0]=True
    interrupted=bool((model(z,mask)==0).all())
    gate=read(HERE/'FIT_GATE/RESULT.json')
    gate_ok=gate['optimizer_updates']==0 and not gate['weights_written'] and not list((HERE/'FIT_GATE').glob('*.pt'))
    all_ok=all(r['passed'] for r in records.values()) and all(r['passed'] for r in replays.values()) and all((silent,isolated,bounded,interrupted,gate_ok))
    result=dict(data_preparation_pass=all_ok,cases=records,replays=replays,
                adapter_untrained_constraints=dict(silent_zero=silent,channel_isolation=isolated,rate_bound=bounded,cut_zero=interrupted),
                training_refusal_verified=gate_ok,optimizer_updates=0,neural_ticks=160000,
                new_physical_steps=0,reused_body_force_samples=80000,brain_steps=0,
                original_teacher_training='NOT_RUN_MISSING_TEACHER_AND_BOUNDARY_REVIEW',
                full_loop='NOT_IMPLEMENTED',source_audit_sha256=sha(HERE/'SOURCE_AUDIT.json'))
    write(HERE/'PREPARATION_ACCEPTANCE.json',result)
    fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
    normal,_=load('held_left');t=normal['time_s']
    for c,color in enumerate(('tab:blue','tab:orange')):
        label=('BANC left','BANC right')[c]
        axes[0].plot(t,normal['normal_load_native'][:,c],color=color,label=label)
        axes[1].plot(t,normal['features_hz'][:,c,1],color=color,label=label)
        axes[2].plot(t,normal['target_voltage_mv'][:,c],color=color,label=label)
        cut,_=load('held_left_cut_left')
        axes[3].plot(t,cut['features_hz'][:,c,1],color=color,label=label)
    for ax,label in zip(axes,('Native normal load','Post-neck causal rate (Hz)','BANC first-hop mean (mV)','Left cut: post-neck rate (Hz)'),strict=True):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc='upper right')
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle('Actual fixed BANC neck replay; original Eon teacher absent; adapter UNTRAINED')
    fig.tight_layout();fig.savefig(HERE/'ASCENDING_PREPARATION.png',dpi=130);plt.close(fig)
    print('data preparation pass',all_ok,'training updates',0,flush=True)
    if not all_ok:
        raise SystemExit(2)


if __name__=='__main__':
    main()
