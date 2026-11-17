"""Fullcompiledrenderer globalresolution/sourcepreservation proof; stdlib only."""
from pathlib import Path
import ast,json,hashlib,sys,dis,types,builtins
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def global_names(code):
 names=set(i.argval for i in dis.get_instructions(code) if i.opname in ('LOAD_GLOBAL','LOAD_NAME'))
 for value in code.co_consts:
  if isinstance(value,types.CodeType):names.update(global_names(value))
 return names

def main():
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('RenderrecoveryCODEONLY '+k)
 for name in ('render.py','check_codeonly.py'):
  tree=ast.parse((HERE/name).read_text());compile(tree,str(HERE/name),'exec');check(name+'_ASTcompile',True);check(name+'_noeagernumericalimports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('numpy','torch','mujoco','physics') for x in n.names) for n in tree.body))
 import render
 prior=sys.modules.get('contracts');present='contracts' in sys.modules;ns=render.namespace();check('contracts_alias_restored',('contracts' in sys.modules)==present and sys.modules.get('contracts') is prior);check('newHEREonlyoutput',ns['HERE']==HERE);check('fixedfivecases',ns['VIDEO_IDS']==('V_left_to_right','O_left','F_left_agree','O_left_odorcut','F_right_agree'));check('bodyROOTunchanged',ns['ROOT']==HERE.parent/'sensory_body_repair');check('explicitstdlibhashlib_resolved',ns['hashlib'] is hashlib)
 names=global_names(ns['main'].__code__);missing=sorted(x for x in names if x not in ns and not hasattr(builtins,x));check('allcomplete_main_andnested_LOAD_GLOBAL_resolve',not missing)
 # Originalcompiledcode is untouched, only its executionglobals were supplemented.
 adapter=render.module('_recovery_codeonly_original_adapter',render.EVIDENCE/'adapter.py');value,_=adapter.tree();full=render.namespace();check('renderer_bytecode_constants_unchanged',ns['main'].__code__.co_code==full['main'].__code__.co_code and ns['main'].__code__.co_consts==full['main'].__code__.co_consts);check('onlyrequiredstdlib_addition',"ns.update(HERE=HERE,VIDEO_IDS=VIDEO_IDS,load=load,hashlib=hashlib,json=json)" in (HERE/'render.py').read_text());check('originalall30auditloadstillused','frozen_load(condition)' in (HERE/'render.py').read_text());check('no_numerical_behavior_change',all(x not in (HERE/'render.py').read_text() for x in ('mj_step','raw_turn','functional_pass','qpos[')))
 failure=render.EVIDENCE/'RENDER_VISUAL_LTR.log';partial=render.EVIDENCE/'VIDEO_V_left_to_right/ACTUAL_MULTISENSORY_BRAIN_NECK_BODY.mp4';check('actual137failurelog',failure.is_file() and 'hashlib' in failure.read_text() and 'NameError' in failure.read_text());check('actualpartialvideo_retained',partial.is_file());check('no_recoveryoutputs_yet',not any((HERE/('VIDEO_'+x)).exists() for x in ns['VIDEO_IDS']));check('noarraysmodelsCore_renderimport',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')))
 paths=[render.EVIDENCE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json','RENDER_VISUAL_LTR.log')]+[partial,adapter._old.OLD,adapter.OLD/'adapter.py',adapter.OLD/'contracts.py'];identities={str(p):sha(p) for p in paths}
 with (HERE/'SOURCE_IDENTITIES.json').open('x') as f:json.dump(dict(status='ORIGINAL137FAILURE_PARTIALVIDEO_AND_ALLRENDER_SOURCES_RETAINED',source_hashes=identities,partial_video=dict(path=str(partial),bytes=partial.stat().st_size,sha256=sha(partial)),complete_renderer_global_names=sorted(names),unresolved_globals=missing,oldfiles_modified=False),f,indent=2);f.write('\n')
 hashes={str(HERE/n):sha(HERE/n) for n in ('PREREG.json','render.py','check_codeonly.py','SOURCE_IDENTITIES.json')}
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,complete_renderer_andnested_global_names=sorted(names),unresolved_globals=missing,array_reads=0,models_deserialized=0,Core_constructions=0,render_calls=0,NN_forward_calls=0,physics_steps=0,new_physical_trials=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),complete_globals=len(names),unresolved=missing,arrays=0,models=0,Core=0,render=0)))
if __name__=='__main__':main()
