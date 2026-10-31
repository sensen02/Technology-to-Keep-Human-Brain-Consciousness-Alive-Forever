"""Reuse saved-state comparator for four-node arm; no simulation."""
from pathlib import Path
import json,numpy as np
import compare_required_motor_tier as old
from run_four_motor_tier import O
from run_one_second_motor_pair import dump

def main():
 z=np.load(O/'BODY.npz');idx=np.flatnonzero(z['nonfoot_floor_count']);dump(O/'RESPONSE_NEW_CONTACT.json',dict(first_contact=None if not len(idx) else float(z['time'][idx[0]])))
 import inspect
 src=inspect.getsource(old.main).replace("'required6'","'required4'").replace('Six sourced motor roots','Four sourced motor roots').replace('Commanded affected muscle bundles /9','Commanded tracked muscle bundles /9 (7 membership affected)')
 ns=dict(vars(old));ns['O']=O;exec(compile(src,__file__,'exec'),ns);ns['main']()
 # Separate the four-member effect from the two deliberately excluded entries.
 mapping=json.loads((O/'MAPPING.json').read_text());roots=set(json.loads((O/'PREREGISTRATION.json').read_text())['required']);affected=[r['actuator'] for r in mapping['rows'] if roots & set(r['pool_root_ids'])]
 dump(O/'ACTUAL_AFFECTED_BUNDLES.json',dict(actuators=affected,count=len(affected),note='two missing-cache/version-unconfirmed muscle entries excluded from membership factor; parent report retains them for transparency'))
 a=np.load(old.HISTORY/'normal/INTERVALS.npz');b=np.load(O/'INTERVALS.npz');oldrows=json.loads(Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z/MAPPING.json').read_text())['rows'];checks=[]
 for r0,r1 in zip(oldrows,mapping['rows']):
  if r1['actuator'] not in affected:continue
  i=r1['actuator_index'];checks.append(dict(actuator=r1['actuator'],old_pool_size=len(r0['local_indices']),new_pool_size=len(r1['local_indices']),old_first_command=float(np.flatnonzero(a['commands'][:,i]>0)[0]*.005) if np.any(a['commands'][:,i]>0) else None,new_first_command=float(np.flatnonzero(b['commands'][:,i]>0)[0]*.005) if np.any(b['commands'][:,i]>0) else None))
 dump(O/'DECODER_POOL_SIZE_EFFECT.json',dict(rows=checks,normalization='MotorOutput per-sample mean over present tier pool; adding silent member changes denominator, unchanged algorithm but actual pool mean can decrease',old_spike_prefix_equal_10ms=bool(np.array_equal(a['spikes'][:2],b['spikes'][:2,:12000]))))
 Path('/home/sensen/Desktop/S2_FOUR_MOTOR_TIER.png').write_bytes((O/'REQUIRED_TIER_COMPARISON.png').read_bytes())
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
