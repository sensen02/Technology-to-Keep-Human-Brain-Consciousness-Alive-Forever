"""S151 independent saved-array audit, sharing tested S149 scalar audit mechanics."""
from pathlib import Path
import argparse,json,types,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical/20261024T213108429114Z')
MAP=Path('/home/sensen/Desktop/engine修复验收/s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 gains=np.full(90,.5);gains[4]=.25
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text()
 old="np.array_equal(a['applied'],.5*a['decoded'])";assert src.count(old)==1;src=src.replace(old,"np.array_equal(a['applied'],a['decoded']*np.array([.25 if j==4 else .5 for j in range(90)]))")
 start=src.index(' source=') if ' source=' in src else -1
 old="assert 'd.ctrl[:] = .5*u' in source or 'd.ctrl[:] = 0.5*u' in source or 'd.ctrl[:] = 0.5 * u' in source";assert old in src;src=src.replace(old,"assert 'd.ctrl[:]' in source and ('s151_gains' in source or 'interface_gains' in source)")
 src=src.replace("actual_gain=.5","actual_gains=[.25 if j==4 else .5 for j in range(90)]").replace("HALF_GAIN_INDEPENDENT_CHECK.json","LF_CHANNEL_INDEPENDENT_CHECK.json")
 module=types.ModuleType('_s151_independent');module.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),module.__dict__);module.REF=REF
 # Compare originals to S149: initial dynamics and PRE160/model unchanged.
 m=json.loads(MAP.read_text())[4];assert m['actuator']==4 and m['leg']=='lf' and m['name']=='LFC_sternal_anterior_rotator'
 oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];module.main()
 finally:sys.argv=oldargs
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:f[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(f-a['foot_force']).max()<1e-10
 mapping_hash=module.sha(MAP);gain_actual=np.full(90,.5);gain_actual[4]=.25;assert np.array_equal(a['applied'],a['decoded']*gain_actual)
 with (p/'LF_CONTACT_MAPPING_CHECK.json').open('x') as h:json.dump(dict(passed=True,zero_based_index=4,mapping=m,mapping_source=str(MAP),mapping_sha256=mapping_hash,contact_maxabs=float(abs(f-a['foot_force']).max()),only_changed_gain_index=4,actual_activation_max=float(a['act'].max()),activation_reconstruction_residual_is_not_actual_amplitude=True),h,indent=2)
 print('Independent per-leg contact reconstruction and channel mapping passed')
if __name__=='__main__':main()
