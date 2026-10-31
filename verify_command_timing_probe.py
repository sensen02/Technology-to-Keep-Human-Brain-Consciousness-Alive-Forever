"""S159 delayed application independent saved-data audit."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_half_gain_physical_probe.py').read_text();old="np.array_equal(a['applied'],.5*a['decoded'])";assert old in src;src=src.replace(old,"np.array_equal(a['applied'],np.concatenate((np.zeros((1,90)),.5*a['decoded'][:-1]),axis=0))")
 old="assert 'd.ctrl[:] = .5*u' in source or 'd.ctrl[:] = 0.5*u' in source or 'd.ctrl[:] = 0.5 * u' in source";assert old in src;src=src.replace(old,"assert 'pending' in source and 'd.ctrl[:] = applied_actual' in source")
 src=src.replace('HALF_GAIN_INDEPENDENT_CHECK.json','COMMAND_TIMING_INDEPENDENT_CHECK.json')
 mod=types.ModuleType('_s159_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__);mod.REF=REF;oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 meta=json.loads((p/'init80ms/COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);f=np.zeros_like(a['foot_force'])
 for t in range(len(a['time'])):
  for c in range(int(a['contact_ncon'][t])):
   g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
   if foot in gm:f[t,legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3])
 assert abs(f-a['foot_force']).max()<1e-10;ticks=min(len(a['decoded']),len(b['decoded']));diff=[dict(tick=k,spikes_equal=bool(np.array_equal(a['spikes'][k],b['spikes'][k])),raw_decoded_maxabs=float(abs(a['decoded'][k]-b['decoded'][k]).max())) for k in range(ticks)]
 schedules=list((p/'init80ms').glob('*SCHEDULE*.json'));assert len(schedules)==1
 schedule=json.loads(schedules[0].read_text())
 if isinstance(schedule,dict):schedule=schedule.get('rows',schedule.get('schedule'))
 assert isinstance(schedule,list) and len(schedule)==len(a['decoded'])
 for k,row in enumerate(schedule):
  assert abs(row['source_start_s']-(.08+k*.0005))<1e-12 and abs(row['decoder_validfrom_s']-row['source_end_s'])<1e-12
  assert abs(row['command_available_physical_s']-(row['decoder_validfrom_s']-.08))<1e-12
  assert row['sensory_sample_actual_s']<=row['physical_start_s']+1e-12
  assert row['applied_source_tick']==k-1
  if row['command_actual_application_start_s'] is not None:assert row['command_available_physical_s']<=row['command_actual_application_start_s']+1e-12
 nonzero=np.flatnonzero(np.any(a['ctrl']!=0,axis=1));post=None if not len(nonzero) else float(a['time'][nonzero[0]]-25e-6)
 with (p/'TIMING_CONTACT_COMMAND_CHECK.json').open('x') as h:json.dump(dict(passed=True,contact_maxabs=float(abs(f-a['foot_force']).max()),common_ticks=diff,first_nonzero_application_s=post,post_application_alignment_available=bool(len(nonzero)),actual_max_activation=float(a['act'].max())),h,indent=2)
 print('Delayed interface/contact check passed; first actual nonzero application',post)
if __name__=='__main__':main()
