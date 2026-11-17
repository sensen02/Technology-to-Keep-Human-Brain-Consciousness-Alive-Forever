"""Exactnewopponentactualbody reader; stdlibonlyuntilexplicitload."""
from pathlib import Path
import ast,json,hashlib,types
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE/'body';VIS=SCOPE.parent;POLICY=SCOPE/'policy';ORIGINAL=VIS/'visual_body_v2/visual'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd';VISUAL_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
NORMAL=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');CONTROLS=('V_left_occluded','V_flip_fixedframe');IDS=NORMAL+CONTROLS
class NewSummary(ast.NodeTransformer):
 def __init__(self):self.changed=0
 def visit_Constant(self,node):
  if node.value=='VISUAL_BODY_OCCLUSION_FIXEDFRAME_V2':self.changed+=1;return ast.copy_location(ast.Constant('VISUAL_OPPONENT_FAR_BODY_V7'),node)
  return node
def inherited():
 tree=ast.parse((ORIGINAL/'contracts.py').read_text());change=NewSummary();tree=change.visit(tree);ast.fix_missing_locations(tree)
 if change.changed!=1:raise ValueError('exactoldsummarycontractshape')
 ns={'__name__':'_opponent_savedactual_contracts','__file__':str(ORIGINAL/'contracts.py')};exec(compile(tree,str(HERE/'contracts.py')+'::explicitopponentroots','exec'),ns);ns.update(HERE=HERE,ROOT=ROOT,VIS=VIS,VISUAL_SHA=VISUAL_SHA,MOTOR_SHA=MOTOR_SHA,BASE=BASE);return ns
_g=inherited()
for _k in ('read','sha','require','fresh','unchanged','arrays'):
 globals()[_k]=_g[_k]
_base_load=_g['load']
def own_sources():
 return [HERE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[ORIGINAL/n for n in ('contracts.py','render.py','plot_sync.py')]
def load(condition):
 checked=read(HERE/'CODEONLY_CHECK.json');require(checked['all_passed'] is True and all(checked['checks'].values()),'visualsourcecodecheckrequired');unchanged(checked['source_hashes'])
 case=_base_load(condition);binding=read(ROOT/'INPUT_BINDINGS.json');offline=POLICY/'VALIDATION/RESULT.json';auditpath=POLICY/'VALIDATION/VERIFY.json';saved=POLICY/'TRAIN/VERIFY_001000.json';r=read(offline);a=read(auditpath);s=read(saved)
 require(binding['visual_format']=='VISION_OPPONENT_POLICY_V7','newopponentloaderbinding');require(r['checkpoint_sha256']==a['checkpoint_sha256']==s['sha256']==VISUAL_SHA and r['all_targets_pass'] is True and a['all_targets_pass'] is True,'newfinal1000allofflinePASSrequired');require(a['integrity_all_pass'] is True and all(v is True for v in a['checks'].values()) and a['result_sha256']==sha(offline) and s['all_passed'] is True and all(v is True for v in s['checks'].values()),'newindependentoffline/savedproof')
 required=[row for row in r['rows'] if row['functional_required']];require(len(required)==72 and all(row['all_targets_pass'] is True and all(v is True for v in row['flags'].values()) for row in required),'all72actualconfigurationgatesrequired');identity=r['identity'];require(identity['format']=='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7' and identity['shared_eye_CNN']==[1,8,8] and identity['receptor_trainable'] is False and identity['fixed_receptor']=='Conv1x1RGBweights[1,-.5,-.5]bias0thenReLU','fixedengineeringneuralopponentidentity')
 ledger=case['source_hashes']
 for proof in (r,a,s):unchanged(proof['source_hashes']);ledger.update(proof['source_hashes'])
 for p in own_sources()+[offline,auditpath,saved,POLICY/'common.py',POLICY/'model.py',POLICY/'PREREG.json',ROOT/'PREREG_PROPOSAL.json']:
  require(p.is_file(),'requiredvisualsource '+str(p));ledger[str(p)]=sha(p)
 case['opponent_identity']=identity;case['source_hashes']=ledger;return case
