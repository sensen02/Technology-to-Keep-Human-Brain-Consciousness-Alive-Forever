"""Stdlib exact-renderer/schema/namespace checks; no arrays,NN,Core,render."""
from pathlib import Path
import ast,json,hashlib,sys
HERE=Path(__file__).resolve().parent
FILES=('contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};texts={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('RepairevidenceCODEONLY '+k)
 for name in FILES:
  texts[name]=(HERE/name).read_text();tree=ast.parse(texts[name]);compile(tree,str(HERE/name),'exec');check(name+'_syntax',True);check(name+'_no_eager_numericalimports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('torch','numpy','mujoco','physics') for x in n.names) for n in tree.body))
 import contracts,adapter
 pr=json.loads((HERE/'PREREG.json').read_text());check('actualnewROOT',contracts.ROOT==HERE.parent/'sensory_body_repair');check('actualnewsummarykind',contracts.SUMMARY_KIND==pr['summary_kind']);check('exactknownnewSHAs',contracts.BRAIN_SHA==pr['weights']['brain'] and contracts.ODOR_SHA==pr['weights']['odor']);check('unchangedvisionneckmotor',contracts.VISION_SHA==pr['weights']['vision'] and contracts.NECK_SHA==pr['weights']['neck'] and contracts.MOTOR_SHA==pr['weights']['motor']);check('fixedfourcases_sameold',list(contracts.VIDEO_IDS)==pr['videos'] and contracts.VIDEO_IDS==contracts._parent.VIDEO_IDS and len(contracts.IDS)==14)
 source=ast.parse((contracts.OLD/'contracts.py').read_text());original=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='admission');updated=ast.parse(ast.unparse(contracts._fn))
 class Reverse(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant('SOURCE_NECK_MULTISENSORY_BODY_V1'),n) if n.value=='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1' else n
 restored=Reverse().visit(updated);check('integritygates_originalAST_exceptonecasekind',ast.dump(restored.body[0])==ast.dump(original));check('all30stages_no_gate_relaxation','len(jobs)==30' in ast.unparse(restored).replace(' ',''));check('newcaseonlyone_literal',contracts._change.changes==1)
 value,changes=adapter.tree();parentvalue,parentchanges=adapter._old.tree();check('actualfinalrenderer_everyASTnode_exact',ast.dump(value)==ast.dump(parentvalue) and changes==parentchanges);ns=adapter.namespace('render.py');check('renderer_newROOT_andidentities',ns['ROOT']==contracts.ROOT and ns['HERE']==HERE and ns['BRAIN_SHA']==contracts.BRAIN_SHA and ns['ODOR_SHA']==contracts.ODOR_SHA);check('renderer_main_realcore',ns['main'].__code__.co_filename.endswith('::sameverified_multisensorycore_newnamespace'))
 pns=adapter.namespace('plot_sync.py');check('originalplot_entry_newnamespace',pns['ROOT']==contracts.ROOT and pns['main'].__code__.co_filename.endswith('::sameverified_multisensorycore_newnamespace'));check('contract_load_newglobalcontext',contracts._base_load.__globals__['ROOT']==contracts.ROOT and contracts._base_load.__globals__['admission'] is contracts.admission);check('sourceledger_includes_oldcore_andnewcore',all(p.is_file() for p in contracts.own_sources() if p.name!='CODEONLY_CHECK.json'))
 check('strictnewstep250400formatchecks',all(x in texts['contracts.py'] for x in ('ODOR_NEURAL_OFFSET_REPAIR_V1','SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1',"b['odor_final_step']==250","b['brain_final_step']==400")));check('actualweightbytehashes','sha(b[key])==h' in texts['contracts.py']);check('noaddedfunctionalpass_admission',"require(summary['visual_normal6']['all_functional_pass']" not in texts['contracts.py']);check('JSONonlyactualcomparison','SENSORY_BODY_SUMMARY.json' in texts['contracts.py'] and "['autonomous_cycles']['complete_per_leg']" in texts['plot_sync.py']);check('failure_originalslip_display_inherited','Original slip' in ast.unparse(value) and 'NO RELEASED RGB' in ast.unparse(value));check('same336actualmetadata','neural_state_total=336' in ast.unparse(value));check('nobodyortrainingexecution',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('noactualoutputsyet',not any((HERE/n).exists() for n in ('SYNCHRONIZED_EVIDENCE',)+tuple('VIDEO_'+n for n in contracts.VIDEO_IDS)))
 hashes={str(HERE/n):sha(HERE/n) for n in FILES+('PREREG.json',)};hashes.update({str(contracts.OLD/n):sha(contracts.OLD/n) for n in ('contracts.py','adapter.py','render.py','plot_sync.py')});hashes[str(adapter._old.OLD)]=sha(adapter._old.OLD)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,actual_renderer_AST_and_numericconstants_unchanged=True,actual_old_integrity_AST_reversible_casekind_only=True,arrays_loaded=0,models_deserialized=0,NN_forward_calls=0,Core_constructions=0,render_calls=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),arrays=0,models=0,Core=0,render=0)))
if __name__=='__main__':main()
