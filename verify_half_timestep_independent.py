"""S160 independent saved-array check, never part of acquisition guard."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text();src=src.replace('n<=401','n<=801').replace('np.arange(n)*25e-6','np.arange(n)*12.5e-6').replace('(k-1)//20','(k-1)//40').replace("m['actuator_dynprm'][j,:3])","m['actuator_dynprm'][j,:3],h=.0000125)").replace('HALF_GAIN_INDEPENDENT_CHECK.json','HALF_TIMESTEP_INDEPENDENT_CHECK.json')
 mod=types.ModuleType('_s160_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF;oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 import hashlib
 identity={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'verify_half_gain_physical_probe.py']}
 with (p/'PARENT_CHECKER_IDENTITY.json').open('x') as f:json.dump(identity,f,indent=2)
 proof=json.loads((p/'init80ms/MODEL_PROVENANCE.json').read_text());baseproof=json.loads((REF/'init80ms/MODEL_PROVENANCE.json').read_text());delta={k:(baseproof['after'][k],v) for k,v in proof['after'].items() if baseproof['after'][k]!=v};assert delta=={'opt.timestep':(.000025,.0000125)},delta
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:f[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(f-a['foot_force']).max()<1e-10;ticks=min(len(a['decoded']),len(b['decoded']));diff=[dict(tick=k,spikes_equal=bool(np.array_equal(a['spikes'][k],b['spikes'][k])),decoded_maxabs=float(abs(a['decoded'][k]-b['decoded'][k]).max()),current_maxabs=float(abs(a['current'][k]-b['current'][k]).max())) for k in range(ticks)]
 schedule=json.loads((p/'init80ms/SCHEDULE.json').read_text())
 for k,r in enumerate(schedule):assert abs(r['physical_start_s']-k*.0005)<1e-12 and abs(r['source_end_s']-(.08+(k+1)*.0005))<1e-12
 with (p/'HALF_TIMESTEP_CONTACT_CLOCK_CHECK.json').open('x') as h:json.dump(dict(passed=True,model_delta=delta,contact_maxabs=float(abs(f-a['foot_force']).max()),common_ticks=diff,actual_max_activation=float(a['act'].max()),halfgrid_first_failure=bool((len(a['time'])-1)%2)),h,indent=2)
 print('Half timestep independent contact clock check passed')
if __name__=='__main__':main()
