"""Complete new visual namespace over one immutable v1 reader; no heavy imports here."""
from pathlib import Path
import importlib.util,sys,ast
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');ROOT=SCOPE/'body_pilot_v2';CAL=SCOPE/'feedback_aggregation_v2'
OLD_VISUAL=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_visual';OLD_BODY=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_pilot'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
_base=module('_body_visual_v2_original_contract',OLD_VISUAL/'contracts.py');_g=_base.__dict__
BRAIN_SHA='415304b7027dda17f359350052934670c1aa2bd44a0025b1dd0089d549997b27';NECK_SHA=_base.NECK_SHA;MOTOR_SHA=_base.MOTOR_SHA;BASE=_base.BASE;IDS=_base.IDS;ORIGINAL_RENDER=_base.ORIGINAL_RENDER
_g.update(HERE=HERE,SCOPE=SCOPE,ROOT=ROOT,PROJECT=PROJECT,BRAIN_SHA=BRAIN_SHA,NECK_SHA=NECK_SHA,MOTOR_SHA=MOTOR_SHA)
for _name in ('read','sha','require','fresh','unchanged','arrays'):globals()[_name]=_g[_name]
class _ActualNewSummary(ast.NodeTransformer):
 def __init__(self):self.kind_changes=0;self.check_changes=0
 def visit_Constant(self,node):
  if node.value=='ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY':self.kind_changes+=1;return ast.copy_location(ast.Constant('SOURCE_NECK_AGGREGATION_SIX_CASE_BODY_SUMMARY_V2'),node)
  if node.value=='CODEONLY_CHECK.json':self.check_changes+=1;return ast.copy_location(ast.Constant('CODEONLY_CHECK_RECOVERY_V1.json'),node)
  return node
_old_admission_tree=next(n for n in ast.parse((OLD_VISUAL/'contracts.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='admission');_fix=_ActualNewSummary();_new_admission_tree=_fix.visit(_old_admission_tree)
if (_fix.kind_changes,_fix.check_changes)!=(1,1):raise ValueError('Exactly one actualsummary kind and one distinctrecoverycheck path expected')
_tree=ast.Module(body=[_new_admission_tree],type_ignores=[]);ast.fix_missing_locations(_tree);exec(compile(_tree,str(HERE/'contracts.py')+'::actualnewsummarykind_only','exec'),_g)
_base_admission=_g['admission'];_base_load=_g['load']
def own_sources():
 return [HERE/n for n in ('PREREG.json','AMENDMENT_PREREG.json','AMENDMENT_INITIAL_IDENTITY.json','CODEONLY_CHECK_INITIAL.json','SCHEMA_RECOVERY_PREREG.json','SCHEMA_RECOVERY_EVIDENCE.json','PLOT.log','PLOT_FAILED_109.log','CODEONLY_CHECK_BEFORE_SCHEMA_RECOVERY.json','contracts.py','adapter.py','render.py','plot_sync.py','codecheck.py','CODEONLY_CHECK.json','check_recovery_v1.py','CODEONLY_CHECK_RECOVERY_V1.json')]+ [OLD_VISUAL/n for n in ('contracts.py','render.py','plot_sync.py')]+[ORIGINAL_RENDER]
def admission():
 amendment=read(HERE/'AMENDMENT_PREREG.json');require(sha(HERE/'CODEONLY_CHECK_INITIAL.json')==amendment['initial_codecheck_sha256'],'initialpre-rendercheckarchiveSHA');summary,ledger=_base_admission();binding=read(ROOT/'INPUT_BINDINGS.json');require(binding['kind']=='SOURCE_NECK_FEEDBACK_AGGREGATION_BODY_PILOT_V2' and binding['brain_final_step']==400 and binding['brain_format']=='SOURCE_NECK_FEEDBACK_AGGREGATION_V2','actualnewcandidatefinal400admission');require(binding['brain_sha256']==BRAIN_SHA and binding['neck_sha256']==NECK_SHA and binding['motor_sha256']==MOTOR_SHA,'strictactualweight/neck/motorbinding')
 require(Path(binding['brain_checkpoint']).resolve()==CAL/'TRAIN/checkpoint_000400.pt' and sha(binding['brain_checkpoint'])==BRAIN_SHA,'actualnew400checkpointbytes')
 for condition in IDS:require(read(ROOT/'trials'/condition/'RESULT.json')['kind']=='SOURCE_NECK_FEEDBACK_AGGREGATION_BODY_PILOT_V2','newactualcasekind')
 offline=read(CAL/'EVALUATION/RESULT.json');audit=read(CAL/'EVALUATION/VERIFY.json');saved=read(CAL/'TRAIN/VERIFY_000400.json');require(offline['checkpoint_sha256']==audit['checkpoint_sha256']==saved['sha256']==BRAIN_SHA and audit['integrity_all_pass'] and all(audit['checks'].values()) and saved['all_passed'] and all(saved['checks'].values()),'final400independentoffline/storageevidence');require(audit['result_sha256']==sha(CAL/'EVALUATION/RESULT.json'),'offlineauditresultsha')
 for proof in (offline,audit,saved):
  for key in ('source_hashes','data_hashes'):unchanged(proof[key]);ledger.update(proof[key])
 for p in [CAL/'EVALUATION/RESULT.json',CAL/'EVALUATION/VERIFY.json',CAL/'TRAIN/VERIFY_000400.json',CAL/'TRAIN/checkpoint_000400.pt']:ledger[str(p)]=sha(p)
 return summary,ledger
_g.update(own_sources=own_sources,admission=admission)
def load(condition):
 case=_base_load(condition);case['candidate']='feedback_aggregation_v2_final400';return case

def load_v1_summary():
 # Only metadata for actualv1-v2comparison; no oldarrays,NN or hidden model reexecution.
 summary=read(OLD_BODY/'BODY_SUMMARY.json');require(summary['kind']=='ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY' and summary['all6_integrity_verified'] and summary['brain_sha256']=='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e','actualoldv1summary');unchanged(summary['source_hashes']);hashes=dict(summary['source_hashes']);hashes[str(OLD_BODY/'BODY_SUMMARY.json')]=sha(OLD_BODY/'BODY_SUMMARY.json')
 for condition in ('P_empty','P_left_pulse','P_right_pulse'):
  folder=OLD_BODY/'trials'/condition;a=read(folder/'VERIFY.json');r=read(folder/'RESULT.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'actualoldv1auditrequired');unchanged(r['source_hashes']);hashes.update(r['source_hashes'])
 return summary,hashes
