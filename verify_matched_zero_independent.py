"""S161 independent zero-interface saved-array audit (not acquisition input)."""
from pathlib import Path
import argparse,types,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text();src=src.replace("np.array_equal(a['applied'],.5*a['decoded'])","np.array_equal(a['applied'],np.zeros_like(a['decoded']))")
 old="assert 'd.ctrl[:] = .5*u' in source or 'd.ctrl[:] = 0.5*u' in source or 'd.ctrl[:] = 0.5 * u' in source";assert old in src;src=src.replace(old,"assert 'd.ctrl[:]' in source and 'zero' in source.lower()")
 src=src.replace('HALF_GAIN_INDEPENDENT_CHECK.json','MATCHED_ZERO_INDEPENDENT_CHECK.json').replace('actual_gain=.5','actual_gain=0.')
 mod=types.ModuleType('_s161_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF;oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 proof=json.loads((p/'init80ms/MODEL_PROVENANCE.json').read_text());base=json.loads((REF/'init80ms/MODEL_PROVENANCE.json').read_text());assert proof['after']==base['after']
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 assert not a['ctrl'].any() and not a['act'].any()
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:f[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(f-a['foot_force']).max()<1e-10;ticks=min(len(a['decoded']),len(b['decoded']));diff=[dict(tick=k,spikes_equal=bool(np.array_equal(a['spikes'][k],b['spikes'][k])),decoded_maxabs=float(abs(a['decoded'][k]-b['decoded'][k]).max()),current_maxabs=float(abs(a['current'][k]-b['current'][k]).max())) for k in range(ticks)]
 with (p/'MATCHED_ZERO_CONTACT_NEURAL_CHECK.json').open('x') as h:json.dump(dict(passed=True,contact_maxabs=float(abs(f-a['foot_force']).max()),common_neural_ticks=diff,max_native_muscle_force=float(abs(a['actuator_force']).max()),max_actual_activation=float(abs(a['act']).max()),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),h,indent=2)
 print('Matched zero interface, contact and neural check passed')
if __name__=='__main__':main()
