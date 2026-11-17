"""Reusefrozenneutralobjective algorithms, bindnewopponentmodelproduction explicitly."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent;OLD=HERE.parents[1]/'landmark_neutral_v6/policy'
def build(name):
 spec=importlib.util.spec_from_file_location('_opponent_previous_adapter',OLD/'adapt_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);src=m.build(name)
 changes=[]
 if name=='preflight.py':changes=[("check('actual672RGB_library',rgb.shape==(672,2,32,32,3))","check('actual728RGB_library',rgb.shape==(728,2,32,32,3))")]
 if name=='train.py':changes=[("status='EXACT_PARENT812f_NEW_NEUTRAL_OBJECTIVE_UNQUALIFIED'","status='EXACT_OTHER12_PARENTbe622_FRESHCONV1_FIXEDOPPONENT_UNQUALIFIED'")]
 for old,new in changes:
  if src.count(old)!=1:raise ValueError('opponentexactadaptershape '+old)
  src=src.replace(old,new)
 return src
def run(name):
 ns={'__name__':'_opponent_actual_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(build(name),str(HERE/name)+'::sameobjective_newencoder','exec'),ns);ns['main']()
