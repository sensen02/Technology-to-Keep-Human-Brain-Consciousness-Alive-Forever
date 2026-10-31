"""S157 fixed paired-foreleg interface independent native audit."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
MAP=Path('/home/sensen/Desktop/engine修复验收/s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;mapping=json.loads(MAP.read_text());gains=np.array([.45 if x['leg'] in ('lf','rf') else .5 for x in mapping]);assert np.count_nonzero(gains==.45)==30
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text();old="np.array_equal(a['applied'],.5*a['decoded'])";assert old in src;src=src.replace(old,"np.array_equal(a['applied'],a['decoded']*S157_GAINS)")
 old="assert 'd.ctrl[:] = .5*u' in source or 'd.ctrl[:] = 0.5*u' in source or 'd.ctrl[:] = 0.5 * u' in source";assert old in src;src=src.replace(old,"assert 'd.ctrl[:] = foreleg_gains * u' in source")
 src=src.replace('HALF_GAIN_INDEPENDENT_CHECK.json','FORELEG_GAIN_INDEPENDENT_CHECK.json').replace('actual_gain=.5','actual_gains=S157_GAINS.tolist()')
 mod=types.ModuleType('_s157_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF;mod.S157_GAINS=gains;oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);force=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:force[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(force-a['foot_force']).max()<1e-10;ticks=min(len(a['decoded']),len(b['decoded']));tickdiff=[dict(tick=j,raw_decoded_maxabs=float(abs(a['decoded'][j]-b['decoded'][j]).max()),spikes_equal=bool(np.array_equal(a['spikes'][j],b['spikes'][j]))) for j in range(ticks)]
 with (p/'FORELEG_CONTACT_COMMAND_CHECK.json').open('x') as f:json.dump(dict(passed=True,contact_maxabs=float(abs(force-a['foot_force']).max()),mapping_source=str(MAP),mapping_sha256=mod.sha(MAP),gains=gains.tolist(),common_neural_ticks=tickdiff),f,indent=2)
 print('Independent foreleg contact and per-tick neural command audit passed')
if __name__=='__main__':main()
