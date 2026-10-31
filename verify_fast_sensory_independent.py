"""S162 independent saved-data mechanics and first refreshed-neural differences."""
from pathlib import Path
import argparse,types,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text().replace('HALF_GAIN_INDEPENDENT_CHECK.json','FAST_SENSORY_INDEPENDENT_CHECK.json')
 mod=types.ModuleType('_s162_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF;oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 proof=json.loads((p/'init80ms/MODEL_PROVENANCE.json').read_text());base=json.loads((REF/'init80ms/MODEL_PROVENANCE.json').read_text());assert proof['after']==base['after']
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 # First tick includes resulting body endpoint before the new .5ms observation.
 for k in ('qpos','qvel','act','ctrl','foot_force','com_acceleration_world'):assert np.array_equal(a[k][:21],b[k][:21]),k
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:f[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(f-a['foot_force']).max()<1e-10;ticks=min(len(a['decoded']),len(b['decoded']));diff=[]
 for k in range(ticks):diff.append(dict(tick=k,physical_sample_ms=k*.5,current_maxabs=float(abs(a['current'][k]-b['current'][k]).max()),changed_spike_cells=int(np.count_nonzero(a['spikes'][k]!=b['spikes'][k])),features_maxabs=float(abs(a['features'][k]-b['features'][k]).max()),decoded_maxabs=float(abs(a['decoded'][k]-b['decoded'][k]).max())))
 with (p/'FAST_SENSORY_CONTACT_NEURAL_CHECK.json').open('x') as h:json.dump(dict(passed=True,pre_refresh_physics_bitexact=True,contact_maxabs=float(abs(f-a['foot_force']).max()),common_neural_ticks=diff,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),h,indent=2)
 print(json.dumps(diff))
if __name__=='__main__':main()
