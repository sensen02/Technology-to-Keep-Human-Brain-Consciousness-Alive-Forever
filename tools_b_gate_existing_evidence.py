"""Read frozen P1 local motor evidence; not an isolated-pool causal experiment."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
P=ROOT/'outputs/experiment_platform/20261013_233237_7be93a1d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=json.loads((P/'result.json').read_text());mp=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/native_floor/metadata.json';m=json.loads(mp.read_text());assert sha(P/'traces.npz')==r['traces_sha256']
 rows=[x for x in r['motor_manifest']['rows'] if x['actuator'].startswith('LFTibia_')];assert len(rows)==2
 joint=next(x for x in m['dof_labels'] if x['joint']=='joint_LFTibia_pitch');v=joint['dof'];j=m['dof_jntid'][v];q=m['jnt_qposadr'][j]
 out=Path('/home/sensen/Desktop/engine修复验收/B-local-existing-20261014-v1');out.mkdir(exist_ok=False);arms=['normal','normal_repeat','replay_baseline','motor_cut','output_silence'];data={};summaries={}
 with np.load(P/'traces.npz',allow_pickle=False) as z:
  base=z['normal_trajectory'];assert base.shape[1]==49+48+90
  for arm in arms:
   tr=z[arm+'_trajectory'];end=z[arm+'_interval_end_s'];cmd=z[arm+'_controls'];act=z[arm+'_activation'];sp=z[arm+'_spike_counts'];entries=[]
   for row in rows:
    ai=row['actuator_index'];idx=row['local_indices'];counts=sp[:,idx].sum(axis=1);entries.append(dict(actuator=row['actuator'],actuator_index=ai,pool_status=row['status'],pool_size=len(idx),pool_roots=row['tier_root_ids'],interval_counts=counts.tolist(),control=cmd[:,ai].tolist(),end_activation=act[:,ai].tolist()))
   data[arm+'_tibia_qpos_rad']=tr[:,q];data[arm+'_tibia_qvel_rad_s']=tr[:,49+v];data[arm+'_controls']=cmd[:,[x['actuator_index'] for x in rows]];data[arm+'_activation']=act[:,[x['actuator_index'] for x in rows]];data[arm+'_end_s']=end
   summaries[arm]=dict(pools=entries,tibia_angle_rad=tr[:,q].tolist(),tibia_velocity_rad_s=tr[:,49+v].tolist(),max_angle_difference_from_normal=float(np.max(np.abs(tr[:,q]-base[:,q]))),max_velocity_difference_from_normal=float(np.max(np.abs(tr[:,49+v]-base[:,49+v]))))
  assert np.array_equal(data['normal_tibia_qpos_rad'],data['normal_repeat_tibia_qpos_rad']);assert np.array_equal(data['normal_tibia_qpos_rad'],data['replay_baseline_tibia_qpos_rad'])
 np.savez_compressed(out/'local-traces.npz',**data)
 paths=[P/'result.json',P/'traces.npz',mp,Path(__file__)]
 result=dict(scope='FROZEN_P1_LOCAL_EXTRACTION_NOT_NEW_EXPERIMENT',selection='LF tibia flex/ext named before reading numeric results; not effect-ranked',joint=joint,qpos_address=q,qvel_address=v,arms=summaries,missing=['per-actuator force','moment at matching endpoint','pool-specific intervention with other commands fixed'],causal_scope='all-output cut/replay only, not isolated-pool effect or biological validity',hashes={str(p):sha(p) for p in paths})
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
 for arm in ('normal','motor_cut','output_silence'):
  axs[0].plot(data[arm+'_end_s']*1000,data[arm+'_tibia_qpos_rad'],label=arm)
 for i,row in enumerate(rows):axs[1].step(data['normal_end_s']*1000,data['normal_controls'][:,i],where='pre',label=row['actuator'])
 axs[0].set_ylabel('LF tibia angle (rad)');axs[0].legend();axs[1].set_ylabel('normal applied control');axs[1].set_xlabel('saved interval end (ms)');axs[1].legend();fig.suptitle('Existing 50ms P1 local view; whole-output controls, NOT isolated-pool causality');fig.tight_layout();fig.savefig(out/'local-view.png');plt.close(fig)
 print(out);print([(a,summaries[a]['max_angle_difference_from_normal']) for a in arms]);print([(x['actuator'],x['interval_counts']) for x in summaries['normal']['pools']])
if __name__=='__main__':main()
