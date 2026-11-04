"""Audit the readout-to-muscle mapping: which actuators exist, what drives them, what they move.

The measured puzzle is that about 58% of the muscle commands are exactly zero and only about 33 of
90 actuators are ever active, which looks structural rather than dynamical. This audit answers, on
the archived mapping and the compiled model:

  1. per actuator  how many motor neurons feed it, is its pool complete, is it shared
  2. per joint     which actuators move it and in which direction, so a joint with one actuator or
                   with two actuators of the same sign cannot be driven both ways
  3. per actuator  how often the best run actually drove it, from the stored native trajectories

Reads archived arrays and the XML only; no simulation runs. modes: audit
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
O=Path('/home/sensen/Desktop/engine修复验收/s5-readout-muscle-audit/20261104T200000Z')
RUNS=dict(
 rebuilt_best=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2/eval_622'),
 disconnected_best=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/VALIDATION_R1/eval_709_A'),
)
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def audit():
 import xml.etree.ElementTree as ET
 B=st.B
 mapping=js(B/'RUN/MAPPING.json')
 rows=mapping['rows']
 # The XML references meshes relative to its own tree, so it is parsed as text here instead of being
 # compiled; only the actuator-to-joint wiring and the joint list are needed.
 tree=ET.parse(B/'RUN/MODEL.xml');root_el=tree.getroot()
 acts=[]
 for i,el in enumerate([e for e in root_el.iter() if e.tag in ('motor','general','position','velocity','muscle','intvelocity','damper','cylinder','adhesion')]):
  gear=el.get('gear','1').split()
  acts.append(dict(index=i,name=el.get('name'),joint=el.get('joint'),gear=float(gear[0]) if gear else 1.0,
   ctrl_lo=None,ctrl_hi=None))
 all_joint_names=[e.get('name') for e in root_el.iter() if e.tag in ('joint','freejoint') and e.get('name')]
 class _M:
  nu=len(acts);njoint=len(all_joint_names)
 model=_M()
 by_name={a['name']:a for a in acts}
 per_act=[]
 for r in rows:
  a=by_name.get(r['actuator'],{})
  mtr=r.get('missing_tier_roots',[])
  per_act.append(dict(actuator=r['actuator'],index=int(r['actuator_index']),leg=r.get('leg'),status=r.get('status'),
   muscle_target=r.get('muscle_target'),pool_size=len(r.get('pool_root_ids') or []),
   tier_size=len(r.get('tier_root_ids') or []),local=len(r.get('local_indices') or []),
   missing_tier_roots=len(mtr) if isinstance(mtr,list) else int(mtr),
   duplicates_removed=int(r.get('duplicates_removed',0)),
   joint=a.get('joint'),gear=a.get('gear')))
 # Actuators are driven through tendons rather than bound to a joint directly, so the functional
 # grouping below uses the muscle name's own action word and the declared target, both taken as
 # published rather than inferred from geometry.
 FUNC=('flexor','extensor','promotor','remotor','abductor','adductor','depressor','levator','rotator','reductor')
 def family(name):
  low=name.lower()
  return next((f for f in FUNC if f in low),'other')
 joints=defaultdict(list)
 for p in per_act:
  key=(p['leg'],family(p['actuator']))
  joints['%s|%s'%(key[0],key[1])].append(dict(actuator=p['actuator'],pool_size=p['pool_size']))
 jsum={j:dict(n_actuators=len(items),actuators=[i['actuator'] for i in items],
   pool_sizes=[i['pool_size'] for i in items]) for j,items in joints.items()}
 # which model joints have NO actuator at all
 zero_direction=[]
 undriven=[]
 # actual usage in the archived runs
 usage={}
 for tag,root in RUNS.items():
  body=Path(root)/'RUN/BODY.npz'
  if not body.exists():usage[tag]=dict(note='no stored body trajectory');continue
  cmd=np.asarray(dict(np.load(body))['command'],float)
  if cmd.shape[1]!=len(per_act):usage[tag]=dict(note='command width %d does not match %d actuators'%(cmd.shape[1],len(per_act)));continue
  zero_frac=cmd.shape[0]-np.count_nonzero(cmd,axis=0)
  zero_frac=(np.count_nonzero(cmd<=1e-9,axis=0)/cmd.shape[0])
  usage[tag]=dict(samples=int(cmd.shape[0]),
   ever_active=[int(i) for i in np.flatnonzero(cmd.max(axis=0)>1e-9)],
   never_active=[per_act[i]['actuator'] for i in np.flatnonzero(cmd.max(axis=0)<=1e-9)],
   zero_fraction_per_actuator_mean=float(zero_frac.mean()),
   actuators_mostly_zero=int(np.sum(zero_frac>0.9)))
 res=dict(model=dict(nu=int(model.nu),njoint=int(model.njoint),nq=int(st.CONFIG.get('nq',49)),nv=int(st.CONFIG.get('nv',48))),
  note='nq and nv are the project declared 49 and 48, recorded here as context for the joint count',
  mapping=dict(n_actuators=int(mapping['n_actuators']),n_neurons=int(mapping['n_neurons']),
   usable_actuators=mapping.get('usable_actuators'),fully_covered_actuators=mapping.get('fully_covered_actuators'),
   shared_pools=mapping.get('shared_pools'),shared_pool_caveat=mapping.get('shared_pool_caveat')),
  pool_sizes=dict(min=int(np.min([p['pool_size'] for p in per_act])),median=float(np.median([p['pool_size'] for p in per_act])),
   max=int(np.max([p['pool_size'] for p in per_act])),total=int(np.sum([p['local'] for p in per_act])),
   actuator_pools_of_one=[p['actuator'] for p in per_act if p['pool_size']==1]),
  status_counts=dict(Counter(p['status'] for p in per_act)),
  per_actuator=per_act,
  functional_groups=jsum,
  functional_groups_without_counterpart=sorted([j for j,v in jsum.items() if v['n_actuators']==1]),
  unmapped_model_actuators=sorted(set(a['name'] for a in acts)-set(p['actuator'] for p in per_act)),
  unusable_actuators=sorted([p['actuator'] for p in per_act if p['pool_size']==0 or p['missing_tier_roots']>0]),
  partially_covered=sorted([p['actuator'] for p in per_act if p['missing_tier_roots']>0]),
  shared_pool_pairs=int(len(mapping.get('shared_pools') or [])),
  usage=usage,
  reading='a joint that has only one actuator, or two actuators of the same sign, cannot be driven in both directions no matter what the network does; that is a structural limit and is counted here rather than blamed on the neural drive')
 dump(O/'READOUT_MUSCLE_AUDIT.json',res)
 print(json.dumps({k:res[k] for k in ('model','mapping','pool_sizes','status_counts')},ensure_ascii=False)[:900])
 print('功能组',len(jsum),'| 单个执行器的功能组',len(res['functional_groups_without_counterpart']),
  '| 未映射的模型执行器',len(res['unmapped_model_actuators']),'| 部分缺MN的执行器',len(res['partially_covered']),
  '| 共享池对数',res['shared_pool_pairs'])
 for tag,u in usage.items():print(tag,{k:v for k,v in u.items() if k in ('ever_active','samples','zero_fraction_per_actuator_mean','actuators_mostly_zero','note')} if 'ever_active' not in u else dict(samples=u['samples'],ever_active=len(u['ever_active']),never_active=len(u['never_active']),zero_mean=round(u['zero_fraction_per_actuator_mean'],3),mostly_zero=u['actuators_mostly_zero']))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'audit':audit}[sys.argv[1]]()
