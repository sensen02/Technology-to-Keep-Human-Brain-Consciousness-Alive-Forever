"""Independent saved-only modality-cut verification, after generic continuous integrity check."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--cut',choices=['contact','proprioception'],required=True);args=ap.parse_args();p=Path(args.directory)
 base=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z');rec=json.loads((p/'NEURAL_RECORDS.json').read_text());old=json.loads((base/'NEURAL_RECORDS.json').read_text())
 cut=['bristle_touch','campaniform_strain'] if args.cut=='contact' else ['claw_position','hook_direction','hairplate_angle'];keep=['claw_position','hook_direction','hairplate_angle'] if args.cut=='contact' else ['bristle_touch','campaniform_strain'];seen_positive_retained=False
 for r in rec:
  for leg,channels in r['channels']['per_leg'].items():
   assert all(channels[k]['weight']==0 for k in cut)
   assert all(channels[k]['weight']==1 for k in keep)
   if any(channels[k]['value']>0 and channels[k]['n_cells']>0 for k in keep):seen_positive_retained=True
 assert seen_positive_retained
 # At the initial boundary physical measurements must remain identical; only declared weights change.
 for leg,channels in rec[0]['channels']['per_leg'].items():
  for name,value in channels.items():assert value['value']==old[0]['channels']['per_leg'][leg][name]['value']
 with np.load(p/'BODY.npz') as b,np.load(base/'BODY.npz') as a:
  n=int(np.flatnonzero(b['time']>=1-1e-9)[0]);assert n==1000
  for k in ['time','qpos','qvel','activation']:np.testing.assert_array_equal(b[k][:n+1],a[k][:n+1])
  delta=np.max(np.abs(b['qpos']-a['qpos']),axis=1);div=np.flatnonzero(delta>1e-12);first_div=None if not len(div) else float(b['time'][div[0]]-b['time'][n])
 result=dict(passed=True,cut_modality=args.cut,cut_channels=cut,retained_channels=keep,initial_physical_sensory_values_unchanged=True,entire_prewarm_and_boundary_exact_to_061=True,first_qpos_divergence_relative_s=first_div,motor_spikes_first15ms=sum(r['motor_spikes'] for r in rec if r['relative_start_s']<.015-1e-12),motor_spikes_250ms=sum(r['motor_spikes'] for r in rec),new_physics_steps=0,new_neural_steps=0,scope='Validates modality weights and initial inputs. Does not equate spiking or static interface fidelity with autonomous walking.')
 out=p/'PARENT_SENSORY_CUT_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
