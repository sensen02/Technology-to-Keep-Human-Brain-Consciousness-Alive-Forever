"""Exact source shape adapters: oldflow, newpaired sampler, fixed500lineage."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parent/'20261111T234150Z'
def once(src,old,new):
 if src.count(old)!=1:raise ValueError('frozenadaptershape '+old)
 return src.replace(old,new)
def build(name):
 src=(ORIGINAL/name).read_text()
 if name in ('train.py','verify_saved.py'):
  src=re.sub(r'\b(1001|1000|500)\b',lambda m:{'1001':'501','1000':'500','500':'250'}[m.group(0)],src);src=src.replace('checkpoint_001000.pt','checkpoint_000500.pt')
 if name=='train.py':
  src=once(src,'from data import training_data','from data import training_data,pair_table,expand_pairs')
  src=once(src,'curve=[];schedule=[]','curve=[];schedule=[];pair_schedule=[]')
  src=once(src,'indices=generator.choice(24,4,replace=False);schedule.append(indices.tolist())','pair_ids=generator.choice(12,2,replace=False);indices=expand_pairs(pair_ids);pair_schedule.append(pair_ids.tolist());schedule.append(indices.tolist())')
  src=once(src,'batch_schedule=schedule,config=config','batch_schedule=schedule,pair_schedule=pair_schedule,config=config')
  src=once(src,"schedule=exact(schedule,p['batch_schedule'])","schedule=exact(schedule,p['batch_schedule']),pairs=exact(pair_schedule,p['pair_schedule'])")
  src=once(src,'batch_indices=indices.tolist(),','batch_pair_ids=pair_ids.tolist(),batch_indices=indices.tolist(),')
  src=once(src,"optimizer='freshAdamALL8',","sampler='uniform2of12pairs_expand4_NOshuffle',pair_table=pair_table(),optimizer='freshAdamALL8',")
 elif name=='verify_saved.py':
  src=once(src,'from data import training_data,arrays,split','from data import training_data,arrays,split,pair_table,expand_pairs')
  old="fresh=np.random.Generator(np.random.PCG64(20261111));expected=[fresh.choice(24,4,replace=False).tolist() for _ in range(args.step)];check('actual_uniform_batch_schedule_exact',expected==payload['batch_schedule'])"
  new="fresh=np.random.Generator(np.random.PCG64(20261111));expected_pairs=[fresh.choice(12,2,replace=False).tolist() for _ in range(args.step)];expected=[expand_pairs(ids).tolist() for ids in expected_pairs];check('actual_pair_schedule_exact',expected_pairs==payload['pair_schedule']);check('actual_expanded_batch_schedule_exact',expected==payload['batch_schedule']);check('registered_pair_table_exact',payload['config']['pair_table']==pair_table())"
  src=once(src,old,new)
 elif name=='preflight.py':
  src=once(src,'assert all(checks.values()),checks',"from pair_checks import extra_checks\n checks.update(extra_checks(m))\n assert all(checks.values()),checks")
 elif name=='evaluate.py':
  src=src.replace('checkpoint_001000.pt','checkpoint_000500.pt').replace('VERIFY_001000.json','VERIFY_000500.json').replace('final1000','final500')
  # Descriptive signed statistics do not change anyexistingflags orRMS thresholds.
  src=once(src,"metrics['offset_RMS_offplus0_5to1_0']=rms","metrics['offset_RMS_offplus0_5to1_0']=rms;metrics['offset_signed_mean']=float(np.mean(y[recovery]));metrics['offset_std']=float(np.std(y[recovery]));metrics['offset_abs_peak']=float(np.max(np.abs(y[recovery])))")
  src=once(src,"metrics['noodor_RMS']=float(np.sqrt(np.mean(y*y)))","metrics['noodor_RMS']=float(np.sqrt(np.mean(y*y)));metrics['noodor_signed_mean']=float(np.mean(y));metrics['noodor_std']=float(np.std(y))")
 elif name=='run_candidate.py':
  src=once(src,"('verify_saved.py',['--step','500'],'VERIFY_500'),('verify_saved.py',['--step','1000'],'VERIFY_1000')","('verify_saved.py',['--step','250'],'VERIFY_250'),('verify_saved.py',['--step','500'],'VERIFY_500')")
 elif name!='audit_validation.py':raise ValueError('unknowninheritedadapter '+name)
 return src
def run(name):
 ns={'__name__':'_paired_inherited_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(build(name),str(HERE/name)+'::strictfrozenadapter','exec'),ns);ns['main']()
