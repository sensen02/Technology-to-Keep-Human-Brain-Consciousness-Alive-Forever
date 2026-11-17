"""Small stdlib AST binding of frozen direct visual code; no numerical imports."""
from pathlib import Path
import ast,json,hashlib,types
HERE=Path(__file__).resolve().parent
CONFIG_PATH=HERE/'SOURCE_CONFIG.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
CONFIG=read(CONFIG_PATH);ROOT=HERE.parent;BASE=Path(CONFIG['baseline_physics']);EXP=Path(CONFIG['experiment_root']);HEAD_AUDIT_PATH=Path(CONFIG['HEAD_AUDIT']);PROTOCOL_PREREG=Path(CONFIG['protocol_prereg'])
LEGACY=PROTOCOL_PREREG.parent/'visual'
def source_tree(name):
 path=LEGACY/name;require(sha(path)==CONFIG['inherited_visual_sources'][str(path)],'frozen visual source changed '+str(path));return ast.parse(path.read_text()),path

def contract_namespace():
 tree,path=source_tree('contracts.py')
 # Only the inherited protocol-prereg read must point at the original protocol.
 changed=0
 for fn in tree.body:
  if isinstance(fn,ast.FunctionDef) and fn.name=='audited_arrays':
   for node in ast.walk(fn):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prereg' for t in node.targets):
     require(ast.unparse(node.value)=="read(ROOT / 'PREREG.json')",'unexpected protocol read');node.value=ast.Call(func=ast.Name(id='read',ctx=ast.Load()),args=[ast.Name(id='PROTOCOL_PREREG',ctx=ast.Load())],keywords=[]);changed+=1
 require(changed==1,'exactly one inherited protocol read required');ast.fix_missing_locations(tree)
 g={'__name__':'_paired_B_visual_inherited_contract','__file__':str(path),'PROTOCOL_PREREG':PROTOCOL_PREREG}
 exec(compile(tree,str(path)+'::new_B_globals','exec'),g)
 g.update(HERE=HERE,ROOT=ROOT,BASE=BASE,BEHAVIOR_SHA=CONFIG['behavior_sha256'],MOTOR_SHA=CONFIG['motor_sha256'],IDS=g['B_IDS'])
 original=g['audited_arrays'];g['read']=read
 def bound_arrays(condition):
  ledger=verify_scope();data=original(condition)
  require(Path(data['result']['behavior_checkpoint']).resolve()==Path(CONFIG['behavior_checkpoint']),'actual behavior checkpoint path differs')
  data['source_hashes'].update(ledger);return data
 g['audited_arrays']=bound_arrays
 return g

def verify_scope():
 """Runtime input proof; no model import/forward, old guards untouched."""
 require(Path(CONFIG['root'])==ROOT and ROOT.parent.parent==EXP,'new ROOT/experiment directory mismatch')
 require(BASE==EXP.parent/'banc-walking-recovery/20261110T152714Z','physical baseline path mismatch')
 require(HEAD_AUDIT_PATH==EXP/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json','head audit path mismatch')
 b=read(ROOT/'INPUT_BINDINGS.json');f=read(ROOT/'FROZEN.json');pre=read(ROOT/'PREFLIGHT.json')
 require(b['kind']=='CANDIDATE3_PAIRED_BEHAVIOR_PHYSICAL_B6_V1' and b['behavior_format']==CONFIG['behavior_format'],'new behavior admission kind mismatch')
 require(b['behavior_sha256']==CONFIG['behavior_sha256'] and b['motor_sha256']==CONFIG['motor_sha256'],'new policy digest mismatch')
 require(b['behavior_checkpoint']==CONFIG['behavior_checkpoint'] and b['behavior_loader']==CONFIG['behavior_loader'],'new checkpoint/loader path mismatch')
 require(pre['all_passed'] is True and all(v is True for v in pre['checks'].values()) and pre['behavior_checkpoint_sha256']==CONFIG['behavior_sha256'],'new behavior loader preflight invalid')
 codecheck_path=HERE/'CODEONLY_CHECK_FINAL.json';codecheck=read(codecheck_path)
 require(codecheck['all_passed'] is True and all(v is True for v in codecheck['checks'].values()),'visual code-only identity proof missing')
 ledger=dict(f['source_hashes'])
 for source,digest in codecheck['source_hashes'].items():require(sha(source)==digest,'checked visual source changed '+source);ledger[source]=digest
 ledger[str(codecheck_path)]=sha(codecheck_path)
 for path in (ROOT/'INPUT_BINDINGS.json',Path(CONFIG['behavior_loader'])):require(str(path) in ledger and sha(path)==ledger[str(path)],'new frozen admission/loader changed')
 for path,digest in ledger.items():require(sha(path)==digest,'physical B source/input changed '+path)
 ledger.update(CONFIG['inherited_visual_sources'])
 for path,digest in CONFIG['inherited_visual_sources'].items():require(sha(path)==digest,'inherited visual bytes changed')
 for path in (CONFIG_PATH,PROTOCOL_PREREG,HEAD_AUDIT_PATH,ROOT/'FROZEN.json',ROOT/'PREFLIGHT.json',HERE/'adapter.py',HERE/'contracts.py',HERE/'render.py',HERE/'plot_sync.py',HERE/'PREREG.json'):ledger[str(path)]=sha(path)
 return ledger

def visual_entry(script):
 tree,path=source_tree(script)
 if script=='render.py':
  changed=0
  for node in ast.walk(tree):
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='head_audit_path' for t in node.targets):node.value=ast.Name(id='HEAD_AUDIT_PATH',ctx=ast.Load());changed+=1
  require(changed==1,'exactly one head audit path assignment required');ast.fix_missing_locations(tree)
 ns={'__name__':'_new_behavior_visual_'+script.removesuffix('.py'),'__file__':str(HERE/script),'HEAD_AUDIT_PATH':HEAD_AUDIT_PATH}
 exec(compile(tree,str(path)+'::newscope_direct_visual','exec'),ns)
 require(ns['ROOT']==ROOT and ns['HERE']==HERE,'visual ROOT/HERE bound incorrectly')
 if 'BASE' in ns:require(ns['BASE']==BASE,'visual physical baseline bound incorrectly')
 return ns['main']
