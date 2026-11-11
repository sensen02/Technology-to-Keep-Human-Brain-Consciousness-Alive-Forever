"""Static audited-head sensory episodes. No body construction or motion."""
import itertools,json
import numpy as np
from common import HERE,GEOMETRY,field_module as ff,frozen_sources,digest,save_json,guard

def episode(angle,amplitude,onset,seed,positions,midpoint,forward,left):
 center=midpoint+.6*forward+.6*np.tan(angle)*left
 source=ff.FoodSource(tuple(center),.1,float(amplitude),.5,float(onset),.2)
 sensory=ff.BilateralOdorSensory(source,K=1.,tau_s=.05,noise_std=.002,seed=int(seed))
 pending=np.zeros(2);obs=[];labels=[];concentration=[];clock=0.
 for k in range(240):
  current=pending.copy()  # z(t), not END t+dt.
  obs.append(current)
  factor=0. if amplitude==0 else np.clip(angle/.15,-1,1)
  labels.append(float(factor*np.clip(current.mean()/.15,0,1)))
  concentration.append(source.concentration(positions,clock))
  pending=sensory.sample(positions,clock,.005);clock+=.005
 return np.asarray(obs),np.asarray(labels),np.asarray(concentration),dict(angle=float(angle),amplitude=float(amplitude),onset_s=float(onset),seed=int(seed),source_center_xy=center.tolist(),no_food=bool(amplitude==0),source_forward_offset=.6)

def main():
 guard();audit=json.loads(GEOMETRY.read_text())
 if audit['status']!='PASS':raise ValueError('audited geometry unavailable')
 pos=np.asarray(audit['head']['xpos']);rot=np.asarray(audit['head']['xmat']);offset=audit['sensor_offsets']
 positions=ff.bilateral_positions(pos,rot,offset['forward_offset'],offset['lateral_offset'])
 midpoint=positions.mean(0);forward=rot[:2,0].copy();forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]])
 train=[(a,s,o,20261111+i) for i,(a,s,o) in enumerate(itertools.product([-.15,-.08,-.04,0,.04,.08,.15],[.5,1.,2.],[.25,.45]))]
 train +=[(0.,0.,o,20262111+i) for i,o in enumerate([.25,.45])]
 heldout=[(a,s,o,seed) for a,s,o,seed in itertools.product([-.12,-.06,.06,.12],[.75,1.5],[.35,.55],[20261120,20261121])]
 heldout +=[(0.,0.,o,seed) for o,seed in itertools.product([.35,.55],[20261120,20261121])]
 metadata={}
 for split,configs in [('train',train),('heldout',heldout)]:
  rows=[episode(*cfg,positions,midpoint,forward,left) for cfg in configs]
  x=np.stack([r[0] for r in rows]);y=np.stack([r[1] for r in rows]);c=np.stack([r[2] for r in rows]);meta=[r[3] for r in rows]
  path=HERE/(split+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,receptors=x,teacher_turn=y,environment_concentration=c,time_s=np.arange(240)*.005)
  metadata[split]=dict(path=str(path),sha256=digest(path),episodes=len(rows),actor_shape=list(x.shape),label_shape=list(y.shape),episodes_config=meta)
 sources=frozen_sources();sources[str(HERE/'common.py')]=digest(HERE/'common.py');sources[str(HERE/'generate_data.py')]=digest(HERE/'generate_data.py')
 report=dict(kind='SENSOR_PRETRAIN_ONLY',physical_steps=0,body_constructions=0,static_head=True,geometry=dict(head_position=pos.tolist(),head_rotation=rot.tolist(),sensor_xy=positions.tolist(),sampling_midpoint_xy=midpoint.tolist()),splits=metadata,source_hashes=sources,
             observation_contract='Only receptors[N,240,2] crosses actor boundary; environment_concentration and meta bearing/position/time are targets/diagnostics only',
             timing='Read pending z(t), label current receptors, integrate current concentration solely for next step. Initial pending zero. Hidden reset only episode start.',
             noise='Output noise .002; concentration/clean receptor before onset0 but noisy outputs maynonzero; no-food teacherlabels0 by explicitcontrol. No falsely asserted noisyzero input.')
 save_json(HERE/'DATASET.json',report);guard();print(json.dumps({k:v['episodes'] for k,v in metadata.items()}))
if __name__=='__main__':main()
