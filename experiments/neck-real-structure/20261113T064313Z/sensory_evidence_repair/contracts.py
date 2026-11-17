"""Exact verified14case evidence gates, new repair identities/namespace, no heavy imports."""
from pathlib import Path
import importlib.util,sys,types,ast
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'sensory_body_repair';OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_evidence')
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
_parent=module('_sensoryrepair_evidence_parent_contract',OLD/'contracts.py');g=dict(_parent.__dict__);g.update(HERE=HERE,ROOT=ROOT,OLD=OLD,SUMMARY_KIND='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_SUMMARY_V1',BRAIN_SHA='92dc0d243dad42b1df284ab797fa3dcaa13ca2d9ebfbb4cb434cebba7c9fe490',ODOR_SHA='e21dfb68e38760a6c74a871cfa3686cb79d1512b1ca60b5432397a456e473bfc')
for name,value in list(g.items()):
 if isinstance(value,types.FunctionType):g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
for name in ('IDS','VIDEO_IDS','PROJECT','VIS','BASE','VISION_SHA','NECK_SHA','MOTOR_SHA'):globals()[name]=g[name]
for name in ('read','sha','require','fresh','unchanged','arrays'):globals()[name]=g[name]
BRAIN_SHA=g['BRAIN_SHA'];ODOR_SHA=g['ODOR_SHA'];SUMMARY_KIND=g['SUMMARY_KIND']
# Only the exact source case-kind literal changes; reverse-substitution must restore originalAST.
_tree=ast.parse((OLD/'contracts.py').read_text());_original=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='admission')
class RepairKind(ast.NodeTransformer):
 def __init__(self):self.changes=0
 def visit_Constant(self,node):
  if node.value=='SOURCE_NECK_MULTISENSORY_BODY_V1':self.changes+=1;return ast.copy_location(ast.Constant('SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1'),node)
  return node
_change=RepairKind();_fn=_change.visit(_original)
if _change.changes!=1:raise ValueError('Exactlyoneactualcasekindchange')
_module=ast.fix_missing_locations(ast.Module(body=[_fn],type_ignores=[]));exec(compile(_module,str(HERE/'contracts.py')+'::same_integrity_newcasekind','exec'),g);_base_admission=g['admission'];_base_load=g['load']
def own_sources():
 return [HERE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[OLD/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','CODEONLY_CHECK.json')]+[_parent.OLD/'render.py',_parent.OLD/'contracts.py',VIS/'camera.py',ROOT/'runtime.py',ROOT/'runner.py',ROOT/'audit.py',ROOT/'summary.py',ROOT/'review.py']
def admission():
 summary,review,ledger=_base_admission();b=read(ROOT/'INPUT_BINDINGS.json');require(b['kind']=='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1' and b['summary_kind']==SUMMARY_KIND and b['brain_format']=='SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1' and b['brain_final_step']==400 and b['odor_format']=='ODOR_NEURAL_OFFSET_REPAIR_V1' and b['odor_final_step']==250,'Actualnewodor250/fusion400formats');require(b['brain_sha256']==BRAIN_SHA and b['odor_sha256']==ODOR_SHA,'ActualbothrepairSHAidentity')
 for key,h in (('brain_checkpoint',BRAIN_SHA),('odor_checkpoint',ODOR_SHA)):require(sha(b[key])==h,'Actualnewweightbytes '+key);ledger[b[key]]=h
 return summary,review,ledger
g.update(own_sources=own_sources,admission=admission)
def load(condition):
 case=_base_load(condition);case['candidate']='newodor250_newfusion400_repair';return case

def old_summary():
 parentbody=OLD.parent/'sensory_body';s=read(parentbody/'SENSORY_BODY_SUMMARY.json');v=read(parentbody/'PARENT_READY_REVIEW.json');require(s['kind']=='SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1' and s['all14_integrity_verified'] and v['integrity_all_pass'] and all(v['checks'].values()) and v['summary_sha256']==sha(parentbody/'SENSORY_BODY_SUMMARY.json'),'Actualoldfailed14casemetadata');unchanged(s['source_hashes']);ledger=dict(s['source_hashes']);ledger[str(parentbody/'SENSORY_BODY_SUMMARY.json')]=sha(parentbody/'SENSORY_BODY_SUMMARY.json');ledger[str(parentbody/'PARENT_READY_REVIEW.json')]=sha(parentbody/'PARENT_READY_REVIEW.json')
 for name in ('O_left_odorcut','O_noodor','F_left_agree'):
  f=parentbody/'trials'/name;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'),'Actualoldindependentaudit '+name);ledger[str(f/'RESULT.json')]=sha(f/'RESULT.json');ledger[str(f/'VERIFY.json')]=sha(f/'VERIFY.json')
 return s,ledger
