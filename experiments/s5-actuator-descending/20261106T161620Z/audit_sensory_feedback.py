import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.embodied.sensory_channels import channel_index_map
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'neural_tendon'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json');A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
groups,_=channel_index_map(12000,np.load(H)['root_ids'],lookup_path=LOOK);p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());graph=(sparse.load_npz(A/'ACTUAL_We.npz')+sparse.load_npz(A/'ACTUAL_Wi.npz')).T.tocsr();graph.data[:]=1
# directed adjacency is pre,post here, unlike stored We/Wi.
channels={}
for ch in ['claw_position','hook_direction','hairplate_angle']:
 ids=groups['lf'][ch];dist=shortest_path(graph,directed=True,indices=ids,unweighted=True)
 channels[ch]=dict(ids=ids,count=len(ids),reach_flex=int(np.isfinite(dist[:,p['flex']]).any(1).sum()),reach_ext=int(np.isfinite(dist[:,p['ext']]).any(1).sum()),reach_either_within4=int((dist[:,p['flex']+p['ext']]<=4).any(1).sum()))
pairs=[]
for r in s['records']:
 if r['condition']!='normal':continue
 match=next(t for t in s['records'] if t['arm']==r['arm'] and t['dt_ms']==r['dt_ms'] and t['pulse_ms']==r['pulse_ms'] and t['condition']=='sensory_cut')
 z=np.load(O/(r['tag']+'.npz'));cut=np.load(O/(match['tag']+'.npz'));sp=np.unpackbits(z['spikes_packed'],axis=1)[:,:12000];sc=np.unpackbits(cut['spikes_packed'],axis=1)[:,:12000]
 pairs.append(dict(tag=r['tag'],physics_identical=bool(np.array_equal(z['rows'],cut['rows'])),motor_spikes_identical=bool(np.array_equal(sp[:,p['flex']+p['ext']],sc[:,p['flex']+p['ext']])),network_spike_bit_difference=int(np.count_nonzero(sp!=sc)),max_sensory_abs_nA=float(z['commands'][:,3].max()),lf_channel_spikes={ch:int(sp[:,channels[ch]['ids']].sum()) for ch in channels}))
report=dict(importance='核心必需',channels=channels,normal_sensorycut_pairs=pairs,scope='actual directed reachability ignoring weight/sign for topology only; negative signed current hyperpolarizes cells, no baseline encoded; no conclusion of missing connection or biological tuning',hashes={str(k):hashlib.sha256(k.read_bytes()).hexdigest() for k in [Path(__file__),H,LOOK,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz']})
(O/'SENSORY_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
