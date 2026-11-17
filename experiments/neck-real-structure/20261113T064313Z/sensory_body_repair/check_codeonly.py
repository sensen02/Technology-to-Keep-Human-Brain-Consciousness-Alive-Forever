"""Stdlib exact oldloop/codeobject/numeric/namespace checks; never numerical model or arrays."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent;OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body')
NAMES=('runtime.py','adapter.py','prepare.py','runner.py','audit.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};texts={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('BodyrepairCODEONLY '+k)
 ledger={str(OLD/n):sha(OLD/n) for n in ('PREREG.json','runtime.py','prepare.py','runner.py','audit.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py')}
 for name in NAMES:
  text=(HERE/name).read_text();tree=ast.parse(text);compile(tree,str(HERE/name),'exec');texts[name]=text;check(name+'_ASTcompile',True);check(name+'_publicmain',name in ('runtime.py','adapter.py') or any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in tree.body));check(name+'_no_eager_heavyimports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('torch','numpy','mujoco','physics') for x in n.names) for n in tree.body))
 import runtime,adapter
 pr=json.loads((HERE/'PREREG.json').read_text());oldpr=json.loads((OLD/'PREREG.json').read_text());check('same14conditionorder',list(runtime.IDS)==pr['conditions']==oldpr['conditions']);check('same336state',pr['state_layout']==oldpr['state_layout']==[128,64,64,32,32,16]);check('actualnewROOT_CAL_ODOR',runtime.ROOT==HERE and runtime.CAL==HERE.parent/'fusion_aggregation_repair' and runtime.ODOR==HERE.parent/'odor_memory_repair/TRAIN/checkpoint_000250.pt');check('strictnewformatkind',runtime.FORMAT==pr['calibration_format'] and runtime.KIND==pr['kind'] and runtime.SUMMARY_KIND==pr['summary_kind'] and runtime.ODOR_FORMAT==pr['odor_format']);check('no_odorSHA_guessbeforebinding',runtime.ODOR_SHA is None);check('unknown_bothSHAs_argbindingonly','a.brain_sha256' in texts['prepare.py'] and 'a.odor_sha256' in texts['prepare.py']);check('both_savedproofs_before_bind','completed_proof(ODOR_DIR,250' in texts['prepare.py'] and 'completed_proof(CAL,400' in texts['prepare.py']);check('fusion_uses_exact_newodor_binding','PARENT_ODOR_BINDING.json' in texts['prepare.py'])
 for name in adapter.NAMES:
  original=ast.parse((OLD/name).read_text());reused=adapter.tree(name);check(name+'_allASTnodes_and_numericconstants_exact',ast.dump(original)==ast.dump(reused));ns=adapter.namespace(name);check(name+'_newROOT_globals',ns['main'].__globals__['ROOT']==HERE);check(name+'_original_directentry',ns['main'].__code__.co_filename.endswith('::exactvalidatedoriginaldirectloop'))
 for clsname in ('OdorField','BrainNeckLoop'):
  cls=getattr(runtime,clsname);oldcls=getattr(runtime._PARENT_RUNTIME,clsname)
  for name,fn in cls.__dict__.items():
   if isinstance(fn,type(lambda:None)):check(clsname+'_'+name+'_samecode_newglobals',fn.__code__ is getattr(oldcls,name).__code__ and fn.__globals__['ROOT']==HERE and fn.__globals__['CAL']==runtime.CAL and fn.__globals__['ODOR']==runtime.ODOR)
 check('headproxy_sameaudit',runtime.HEAD_AUDIT==runtime._PARENT_RUNTIME.HEAD_AUDIT);check('ClampFilter_sameexactPEAK',runtime.PEAK_SOURCE==runtime._PARENT_RUNTIME.PEAK_SOURCE and runtime.OdorField.__init__.__code__ is runtime._PARENT_RUNTIME.OdorField.__init__.__code__);cg=runtime.camera_runtime();check('camera_close_capture_newROOT',cg['CameraStream'].close.__globals__['ROOT']==cg['_capture'].__globals__['ROOT']==HERE);check('samevisualneckmotor',runtime.VISION_SHA==oldpr['visual_sha256'] and runtime.BINDING_SHA==oldpr['neck_sha256'] and runtime.MOTOR_SHA==oldpr['motor_sha256']);check('thirty_stages_exact',len(runtime.IDS)*2+2==30)
 check('newodor_op_actual250check',"op['format_version']!=ODOR_FORMAT" in texts['runtime.py'] and "op['step']!=250" in texts['runtime.py']);check('no_newruntimegate_or_reset','turn=' not in texts['runtime.py'] and 'zero_(' not in texts['runtime.py']);check('aliases_restore_allthree',all(x in texts['runtime.py'] for x in ("'model','common','steering_behavior'",'sys.path[:]=paths','sys.modules.pop(k,None)')))
 check('noheavyimports',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('no_actualbodyoutputs',not any((HERE/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','JOBS.json','trials')))
 for p,h in ledger.items():check('oldreadonly:'+Path(p).name,sha(p)==h)
 with (HERE/'OLD_SOURCE_IDENTITIES.json').open('x') as f:json.dump(dict(status='ORIGINAL_MULTISENSORY_SOURCE_UNCHANGED',source_hashes=ledger,all_numerical_directloops_and_codeobjects_exact=True),f,indent=2);f.write('\n')
 hashes={str(HERE/n):sha(HERE/n) for n in NAMES+('PREREG.json','OLD_SOURCE_IDENTITIES.json')};hashes.update(ledger)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,arrays_loaded=0,models_deserialized=0,Core_constructions=0,NN_forward_calls=0,render_calls=0,training_updates=0,physics_steps=0,unknown_brain_and_odor_SHA_not_admitted=True,declared_stages=30),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=0,arrays=0,models=0,Core=0,stages=30)))
if __name__=='__main__':main()
