"""Saved population behavior diagnostics; not model selection or optimization."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z');L=OUT/'learning'
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
rows=[]
for folder in sorted((L/'training').glob('trial_*')):
    r=json.loads((folder/'RESULT.json').read_text());policy=load(folder/'POLICY.npz');commands=load(folder/'COMMANDS.npz')['rows'];init=load(folder/'INITIAL.npz');a=policy['predicted_rate'];power=float(np.mean(a*a));meanpower=float(np.mean(a.mean(0)**2));mn=policy['mn_rates_hz'];bounds=init['joint_bounds'];target=commands[:,43:85]
    row=dict(trial=r['episode_id'],valid=r['valid'],feasible=r['feasible'],reward=r['reward'],**r['reward_terms'],mean_abs_rate=float(np.abs(a).mean()),rate_total_mean_square=power,rate_temporal_mean_square=float(np.mean((a-a.mean(0))**2)),rate_DC_power_fraction=meanpower/power if power else 0.,MN_mean_rate=float(mn.mean()),MN_temporal_std_mean=float(mn.std(0).mean()),target_bound_entries=int(np.count_nonzero((np.abs(target-bounds[:,0])<1e-9)|(np.abs(target-bounds[:,1])<1e-9))),target_total_entries=int(target.size),theta_nonzero=int(np.count_nonzero(load(folder/'ACTOR.npz')['theta'])))
    rows.append(row)
assert len(rows)==53
with (OUT/'SEARCH_DIAGNOSIS.csv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
nonzero=[r for r in rows if r['theta_nonzero']];feasible=[r for r in nonzero if r['feasible']]
summ=dict(training_trials=53,nonzero_candidates=len(nonzero),nonzero_feasible=len(feasible),nonzero_completing_any_cycle=sum(r['completed_cycle_legs']>0 for r in nonzero),nonzero_completing_all6cycles=sum(r['completed_cycle_legs']==6 for r in nonzero),max_completed_cycle_legs=max(r['completed_cycle_legs'] for r in rows),DC_power_fraction_nonzero_median=float(np.median([r['rate_DC_power_fraction'] for r in nonzero])),baseline_reward=rows[0]['reward'],best_nonzero_feasible_reward=max((r['reward'] for r in feasible),default=None),notes='Temporal DC power is a saved-action diagnostic, not proof of neuralrhythm absence. Cycle coverage proxy does notestablishcoordination. No newselection, fitting orphysicalsteps.')
with (OUT/'SEARCH_DIAGNOSIS.json').open('x') as f:json.dump(summ,f,indent=2)
fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
colors=['tab:green' if r['feasible'] else 'tab:red' for r in rows]
ax[0].scatter(range(53),[r['reward'] for r in rows],c=colors,s=24);ax[0].axhline(rows[0]['reward'],color='k',ls='--',label='zero baseline');ax[0].set(xlabel='Trial index',ylabel='Fixed training score',title='Green: original support screen passed');ax[0].legend()
ax[1].scatter([r['weighted_slip_distance'] for r in rows],[r['forward_displacement'] for r in rows],c=colors,s=24);ax[1].set(xlabel='Load-weighted slip distance(native)',ylabel='Forward COM displacement(native)',title='Slip and forward motion')
ax[2].scatter([r['rate_DC_power_fraction'] for r in nonzero],[r['completed_cycle_legs'] for r in nonzero],c=['tab:green' if r['feasible'] else 'tab:red' for r in nonzero],s=24);ax[2].set(xlabel='DC rate power / total rate power',ylabel='Legs with stance-air-stance',title='Saved-action and contact diagnostics');ax[2].set_ylim(-.3,6.3)
fig.suptitle('Fixed53-episode search | diagnostics do notchange selection orwalking acceptance');fig.savefig(OUT/'SEARCH_DIAGNOSIS.png',dpi=150);plt.close(fig)
print(json.dumps(summ,indent=2))
