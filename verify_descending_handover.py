"""Saved-only independent verification of source-identified transient DNp09 input."""
from pathlib import Path
import argparse,json
import numpy as np
ROOTS=[720575941433155799,720575941566493282]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--pulse',action='store_true');a=ap.parse_args();p=Path(a.directory);rec=json.loads((p/'NEURAL_RECORDS.json').read_text());m=json.loads((p/'MAPPING.json').read_text());sp=np.load(p/'NEURAL.npz')['spike_counts'];assert sp.shape==(250,12002)
 # Augmentation appends real identities without changing the original local order.
 old=json.loads((Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')/'MAPPING.json').read_text());assert [r['local_indices'] for r in m['rows']]==[r['local_indices'] for r in old['rows']]
 for r in rec:
  traces=r['descending_command_steps'];assert len(traces)==2
  for x in traces:
   assert x['root_ids']==ROOTS and x['local_indices']==[12000,12001]
   expected=bool(a.pulse and x['relative_s']<.050-1e-12)
   assert (x['amplitude_nA']>0)==expected
   if not expected:assert x['amplitude_nA']==0
  assert r['motor_output_connected'] and r['auxiliary_force_max']==r['external_force_max']==0
 idx=[12000,12001];result=dict(passed=True,network_neurons=12002,source_identified_roots=ROOTS,motor_local_indices_unchanged=True,external_neuronal_pulse=bool(a.pulse),command_zero_after50ms=True,dn_spikes_first50ms=sp[:50,idx].sum(0).tolist(),dn_spikes_after50ms=sp[50:,idx].sum(0).tolist(),motor_events=sum(r['motor_spikes'] for r in rec),new_neural_steps=0,new_physics_steps=0,scope='Validates declared neuronal input and recorded identities; not autonomous walking certification. Delayed synaptic and decoded effects may persist after current withdrawal.')
 out=p/'PARENT_DESCENDING_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
