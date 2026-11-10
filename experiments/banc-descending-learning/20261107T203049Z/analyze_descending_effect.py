"""Parent saved-array analysis of the six fixed descending probes; no simulation."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z');D=SRC/'descending'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation/BASE.npz')
LEGS=['LF','LH','LM','RF','RH','RM']
base=np.load(BASE);leg=np.asarray(base['mn_leg']);allmn=None
res={'source_runs':{},'effect':{}}
for mode in ['held_first','recorded100']:
    z={d:np.load(D/'conditions'/f'{mode}_dose{d:03d}'/'TRACE.npz') for d in (0,25,50)}
    zero=z[0];allmn=np.asarray(zero['all_mn_local']);pos=np.searchsorted(allmn,np.asarray(base['mn_local']))
    assert np.array_equal(allmn[pos],np.asarray(base['mn_local']))
    res['source_runs'][mode]={'start_state_identical_across_doses':bool(np.array_equal(zero['mn_start_hz'][0],z[50]['mn_start_hz'][0])),
        'first_diverging_bin':int(np.argmax(~(zero['mn_start_hz']==z[50]['mn_start_hz']).all(1))),
        'encoder_currents_identical':bool(np.array_equal(zero['sensory_current_nA'],z[50]['sensory_current_nA'])),
        'descending_current_difference_nA':float(np.abs(zero['descending_current_nA']-z[50]['descending_current_nA']).max())}
    win=bins=slice(20,100)  # 100ms onward, past the transient
    r0=zero['mn_start_hz'][win][:,pos].mean(0);r5=z[50]['mn_start_hz'][win][:,pos].mean(0)
    drift=zero['mn_start_hz'][20:60][:,pos].mean(0)-zero['mn_start_hz'][60:][:,pos].mean(0)
    perleg={L:dict(zero_hz=float(r0[leg==L.lower()].mean()),on_hz=float(r5[leg==L.lower()].mean()),
                   delta_hz=float((r5[leg==L.lower()]-r0[leg==L.lower()]).mean())) for L in LEGS}
    res['effect'][mode]={'per_leg':perleg,'mean_abs_delta_hz':float(np.abs(r5-r0).mean()),
        'signed_mean_delta_hz':float((r5-r0).mean()),'mean_abs_baseline_drift_hz':float(np.abs(drift).mean()),
        'delta_to_drift_ratio':float(np.abs(r5-r0).mean()/np.abs(drift).mean()),'per_MN_delta':r5-r0}
a=res['effect']['held_first']['per_MN_delta'];b=res['effect']['recorded100']['per_MN_delta']
res['cross_mode']={'pearson_r':float(np.corrcoef(a,b)[0,1]),'sign_agreement':float(np.mean(np.sign(a)==np.sign(b))),
    'per_leg_r':{L:float(np.corrcoef(a[leg==L.lower()],b[leg==L.lower()])[0,1]) for L in LEGS}}
runs=json.loads((D/'RESULT.json').read_text())
res['probe_counts']={c['condition']:{k:c[k] for k in ('amplitude_nA','DN_spikes','all391_MN_spikes','exported323_MN_spikes','all391_active_MN','voltage_min_mV','neural_steps','physical_steps')} for c in runs['conditions']}
res['limits']='Saved arrays from fixed probes only; no new steps and no physical trial in this file. DNp09 is a candidate identity from the local catalogue; the constant current is an engineering input, not a brain-generated command, an optogenetic equivalent, or proof of function. Local MN-rate change being smaller than the network own drift means a constant command is not amplified into a coordinated motor programme by this module; it does not show the source animal lacks such a pathway.'
for m in res['effect']:res['effect'][m].pop('per_MN_delta')
with (OUT/'DESCENDING_EFFECT.json').open('x') as f:json.dump(res,f,indent=2,allow_nan=False)
fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
w=.35;x=np.arange(6)
for i,mode in enumerate(['held_first','recorded100']):
    z=json.loads(json.dumps(res))['effect'][mode]['per_leg']
    ax[0].bar(x+(i-.5)*w,[z[L]['delta_hz'] for L in LEGS],w,label=mode)
ax[0].axhline(0,color='k',lw=.8);ax[0].set_xticks(x,LEGS);ax[0].set(ylabel='MN rate change Hz',title='Leg-mean effect of .05nA DN drive');ax[0].legend()
z0=np.load(D/'conditions/held_first_dose000'/'TRACE.npz');z5=np.load(D/'conditions/held_first_dose050'/'TRACE.npz')
t=z0['mn_start_hz'].shape[0];tt=np.arange(t)*.005+0.1
ax[1].plot(tt,z0['mn_start_hz'].mean(1),label='DN 0');ax[1].plot(tt,z5['mn_start_hz'].mean(1),label='DN .05nA')
ax[1].set(xlabel='time s',ylabel='mean MN rate Hz',title='All-MN activity (held input)');ax[1].legend()
ax[2].bar(x,[res['cross_mode']['per_leg_r'][L] for L in LEGS],color='tab:blue');ax[2].axhline(0,color='k',lw=.8);ax[2].set_xticks(x,LEGS);ax[2].set(ylabel='per-MN delta correlation r',title='Reproducibility across two input modes');ax[2].set_ylim(-1,1)
fig.suptitle('Fixed descending probe diagnostics | constant candidate-DN current, not a learned command')
fig.savefig(OUT/'DESCENDING_EFFECT.png',dpi=150);plt.close(fig)
print(json.dumps({m:{k:round(v,3) for k,v in res['effect'][m].items() if k!='per_leg'} for m in res['effect']},indent=2))
print(json.dumps(res['cross_mode'],indent=2))
print(json.dumps(res['source_runs'],indent=2))
