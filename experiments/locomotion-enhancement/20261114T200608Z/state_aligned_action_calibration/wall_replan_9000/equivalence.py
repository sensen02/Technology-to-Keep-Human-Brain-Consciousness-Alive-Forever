"""Stdlib exactscience equivalence; only finite pre-execution wallbudget amended."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent
COPIED=('model.py','common.py','sources.py','data.py','scan.py','utils.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','preflight.py')

def normalized(name,text):
 tree=ast.parse(text)
 if name=='learning.py':
  # Removeonly explicit namespacealias requiredto expose thenewresourcegate.
  tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='_ns' and isinstance(n.targets[0].slice,ast.Constant) and n.targets[0].slice.value=='resource_gate' and isinstance(n.value,ast.Name) and n.value.id=='resource_gate')]
  for n in tree.body:
   if isinstance(n,ast.FunctionDef) and n.name=='resource_gate':
    if len(n.args.defaults)!=1 or not isinstance(n.args.defaults[0],ast.Constant) or n.args.defaults[0].value!=9000.:raise ValueError('Onlyresourcegatewall9000defaultallowed')
    n.args.defaults[0].value=7200.
    for c in ast.walk(n):
     if isinstance(c,ast.Constant) and c.value=='Statealigned wall9000sexceeded,noextension':c.value='Statealigned wall7200sexceeded,noextension'
 elif name=='run.py':
  changes=0
  for n in ast.walk(tree):
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='remaining':
    if not isinstance(n.value,ast.BinOp) or not isinstance(n.value.left,ast.Constant) or n.value.left.value!=9000:raise ValueError('Singlepipelinewall9000only')
    n.value.left.value=7200;changes+=1
   if isinstance(n,ast.Constant) and n.value=='pipelinewall9000expired':n.value='pipelinewall7200expired'
  if changes!=1:raise ValueError('Exactlyonepipelinebudgetchange')
 elif name=='staticcheck.py':
  for n in ast.walk(tree):
   if isinstance(n,ast.Constant) and n.value==9000.:n.value=7200.
   if isinstance(n,ast.Constant) and n.value=='training1800wall9000':n.value='training1800wall7200'
   if isinstance(n,ast.keyword) and n.arg=='training1800wall9000':n.arg='training1800wall7200'
 return ast.dump(tree,include_attributes=False)

def verify_equivalence():
 for name in COPIED:
  if (ROOT/name).read_bytes()!=(ORIGINAL/name).read_bytes():raise ValueError('Scientificbytecopychanged:'+name)
 for name in ('learning.py','run.py','staticcheck.py'):
  original=ast.dump(ast.parse((ORIGINAL/name).read_text()),include_attributes=False)
  if normalized(name,(ROOT/name).read_text())!=original:raise ValueError('OnlyregisteredresourceASTdifferenceallowed:'+name)
 old=json.loads((ORIGINAL/'PREREG.json').read_text());new=json.loads((ROOT/'PREREG.json').read_text())
 amendment=new.pop('wall_replan_provenance');new['resources']['wall_s']=7200
 if new['preflight'].count('<=9000beforetraining')!=1:raise ValueError('ExactlyonePREREGtime-thresholdchange')
 new['preflight']=new['preflight'].replace('<=9000beforetraining','<=7200beforetraining')
 if new!=old or amendment['old_wall_s']!=7200 or amendment['new_wall_s']!=9000:raise ValueError('OnlyPREREGwall/provenancechanged')
 return dict(byteidentical_scientific_files=list(COPIED),resource_AST_inverse_matches=['learning.py','run.py','staticcheck.py'],PREREG_resource_only=True,scientific_equivalence=True,NN_forwards=0,physical_steps=0)
