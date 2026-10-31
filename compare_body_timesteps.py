"""Saved-only common-grid comparison. No MuJoCo calls."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_body_timestep_diagnostic import OUT,BASE
from run_one_second_motor_pair import dump,sha

def metric(x):return dict(max_abs=float(np.max(abs(x))),rms=float(np.sqrt(np.mean(x*x))))
def main():
 grids={};full={};stats={};arms=('normal','motor_cut');tags=('baseline100us','dt25us','dt12p5us');common=np.arange(501)*.0001
 for tag,ratio in zip(tags,(1,4,8)):
  for arm in arms:
   p=BASE/arm/'BODY_FLOOR_VERIFIED.npz' if tag==tags[0] else OUT/tag/arm/'BODY.npz';z=np.load(p);n=500*ratio+1;s={k:z[k][:n].copy() for k in z.files};full[tag,arm]=s;g={k:v[::ratio] for k,v in s.items()};grids[tag,arm]=g
   np.testing.assert_allclose(g['time'],common,atol=1e-13,rtol=0);np.testing.assert_array_equal(g['qpos'][0],grids[tags[0],arms[0]]['qpos'][0]);assert not np.any(s['warnings']);assert not np.any(s['external_force'])
   crossing=np.flatnonzero(s['thorax_position'][:,2]<.9*s['thorax_position'][0,2]);stats[tag+'/'+arm]=dict(max_activation=float(s['activation'].max()),activation_gt1_scalar_count=int((s['activation']>1).sum()),activation_lt0_scalar_count=int((s['activation']<0).sum()),first_below90_s=None if not len(crossing) else float(s['time'][crossing[0]]),end_thorax=s['thorax_position'][-1],contact_samples_per_leg=s['foot_contact'].sum(0))
   if tag!=tags[0]:
    i=np.load(OUT/tag/arm/'INTERVALS.npz');np.testing.assert_allclose(i['interval_start_s'],np.arange(10)*.005,atol=1e-14);np.testing.assert_allclose(i['interval_end_s'],np.arange(1,11)*.005,atol=1e-13);np.testing.assert_array_equal(s['command'][1:],np.repeat(i['commands'],50*ratio,axis=0))
    if arm=='motor_cut':assert not np.any(i['commands'][3:])
 # pairs coincide through15ms state, commands diverge first following substep
 for tag in tags:np.testing.assert_array_equal(grids[tag,'normal']['qpos'][:151],grids[tag,'motor_cut']['qpos'][:151])
 diffs={};paired={};neural={}
 for arm in arms:
  for label,left,right in [('coarse_medium',tags[0],tags[1]),('medium_fine',tags[1],tags[2])]:
   a,b=grids[left,arm],grids[right,arm];r={k:metric(a[k]-b[k]) for k in ['activation','thorax_position','qpos','qvel','foot_normal_force']}
   qa=a['qpos'][:,3:7];qb=b['qpos'][:,3:7];dot=abs(np.sum(qa*qb,axis=1)/(np.linalg.norm(qa,axis=1)*np.linalg.norm(qb,axis=1)));r['root_quaternion_geodesic_rad']=metric(2*np.arccos(np.clip(dot,0,1)));r['foot_contact_mismatch_count']=int((a['foot_contact']!=b['foot_contact']).sum());diffs[arm+'/'+label]=r
  for tag in tags:
   a,b=grids[tag,'normal'],grids[tag,'motor_cut'];paired[tag]={k:metric(a[k]-b[k]) for k in ['qpos','activation','thorax_position']}
  for tag in tags:
   p=BASE/arm/'INTERVALS.npz' if tag==tags[0] else OUT/tag/arm/'INTERVALS.npz';z=np.load(p);sp=z['spikes'][:10];cmd=z['commands'][:10];neural[tag+'/'+arm]=dict(total_emitted_spike_count=int(sp.sum()),commands_sum=float(cmd.sum()))
   if tag==tags[0]:bs=sp.copy();bc=cmd.copy()
   else:neural[tag+'/'+arm].update(spike_count_diff_from_baseline=metric(sp.astype(float)-bs),command_diff_from_baseline=metric(cmd-bc))
 fig,axs=plt.subplots(3,2,figsize=(14,12))
 for j,arm in enumerate(arms):
  for tag in tags:
   s=full[tag,arm];t=s['time']*1000;axs[0,j].plot(t,s['activation'].max(1),label=tag);axs[1,j].plot(t,s['thorax_position'][:,2],label=tag);axs[2,j].step(t,s['foot_contact'].sum(1),where='post',label=tag)
  for ax,title in zip(axs[:,j],['Maximum activation','Thorax z / model units','True floor-contact leg count']):ax.set_title(arm+' / '+title);ax.set_xlabel('physical ms');ax.axvline(15,color='grey',ls=':');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('S2/P3 fixed dt refinement; fresh feedback, no replay / not a convergence or walking certificate');fig.tight_layout();fig.savefig(OUT/'TIMESTEP_COMPARISON.png',dpi=140);plt.close(fig)
 Path('/home/sensen/Desktop/S2_TIMESTEP_COMPARISON.png').write_bytes((OUT/'TIMESTEP_COMPARISON.png').read_bytes())
 dump(OUT/'COMPARISON.json',dict(stats=stats,differences=diffs,paired_arm_differences=paired,neural=neural,common_grid_samples=501,initial_and_cut_prefix_exact=True,status='TIMESTEP_INDEPENDENCE_NOT_ESTABLISHED',comparison_exit_code=0))
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(stats=stats,differences=diffs),default=lambda x:x.tolist()));print('saved comparison exit0')
if __name__=='__main__':main()
