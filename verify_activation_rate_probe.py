"""S154 independent activation-time-constant audit using saved native dynamics."""
from pathlib import Path
import argparse,json,types,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical/20261024T213108429114Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text();old=" for k in mb:assert np.array_equal(m[k],mb[k]),k";assert old in src;src=src.replace(old," for k in mb:\n  if k=='actuator_dynprm':\n   expected=mb[k].copy();assert np.all(expected[:,0]==.0005);expected[:,0]=.001;assert np.array_equal(m[k],expected),k\n  else:assert np.array_equal(m[k],mb[k]),k")
 src=src.replace("HALF_GAIN_INDEPENDENT_CHECK.json","ACTIVATION_RATE_INDEPENDENT_CHECK.json").replace("initial_state_model_PRE160_exact=True","initial_state_PRE160_exact=True,model_only_act_tau_changed=True")
 mod=types.ModuleType('_s154_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF
 old=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=old
 proof=json.loads((p/'NATIVE_INPUT_DELTA_PROOF.json').read_text());assert proof['changed']==['model.actuator_dynprm'] and proof['all_other_native_arrays_equal']
 with np.load(p/'DERIVED_NUMERIC_CONFIG.npz') as z:derived=z['actuator_dynprm'].copy()
 with np.load(p/'init80ms/NATIVE_MODEL_ARRAYS.npz') as z:assert np.array_equal(derived,z['actuator_dynprm'])
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force']);counts=np.zeros_like(a['contacts'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:
    j=legs.index(gm[foot]);f[t,j]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3]);counts[t,j]+=int(a['contact_dist'][t,c]<0)
 assert abs(f-a['foot_force']).max()<1e-10;assert np.array_equal(counts,a['contacts']);ticks=min(len(a['decoded']),len(b['decoded']));comparison={k:bool(np.array_equal(a[k][:ticks],b[k][:ticks])) for k in ('spikes','features','decoded','applied')}
 with (p/'CONTACT_COMMAND_INDEPENDENT_CHECK.json').open('x') as h:json.dump(dict(passed=True,contact_maxabs=float(abs(f-a['foot_force']).max()),common_neural_ticks=ticks,reference_comparison=comparison,actual_activation_max=float(a['act'].max()),activation_residual_is_not_actual_amplitude=True),h,indent=2)
 print('Contact reconstruction and common neural command comparison passed',comparison)
if __name__=='__main__':main()
