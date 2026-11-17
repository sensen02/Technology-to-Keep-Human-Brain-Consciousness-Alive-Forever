"""Stdlib namespace/caption/source checks only; no actualtrial arrays/model/Core/render."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 files=('contracts.py','adapter.py','render.py','plot_sync.py','codecheck.py');trees={};checks={}
 for name in files:trees[name]=ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
 import contracts as c
 import adapter as a
 p=json.loads((HERE/'PREREG.json').read_text());checks['explicitnewroots']=str(c.ROOT)==p['body_root'] and str(c.OLD_VISUAL)==p['old_visual_root'];checks['newactual4153identity']=c.BRAIN_SHA==p['brain_sha256'] and c.NECK_SHA==p['neck_sha256'] and c.MOTOR_SHA==p['motor_sha256'];checks['publicnewload_admission']=callable(c.load) and callable(c.admission) and callable(c.load_v1_summary);checks['oldfunctionnamespace_rootrebound']=c._base_load.__globals__['ROOT']==c.ROOT and c._base_admission.__globals__['BRAIN_SHA']==c.BRAIN_SHA;checks['oldsourcecallnewadmission']=c._base_load.__globals__['admission'] is c.admission;checks['oldsourcecallnewledger']=c._base_admission.__globals__['own_sources'] is c.own_sources
 for name in ('render.py','plot_sync.py'):
  ns=a.namespace(name);checks['entry_namespace_'+name]=ns['ROOT']==c.ROOT and ns['HERE']==HERE and ns['load'] is c.load and callable(ns['main']);numbers=lambda tree:[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float)];checks['unchanged_numeric_constants_'+name]=numbers(a.source(name))==numbers(ast.parse((c.OLD_VISUAL/name).read_text()))
 checks['wholeoriginalsource_dependencies_exist']=all((c.OLD_VISUAL/name).is_file() for name in ('contracts.py','render.py','plot_sync.py')) and c.ORIGINAL_RENDER.is_file()
 checks['newbindings400realbytes']=True;b=json.loads((c.ROOT/'INPUT_BINDINGS.json').read_text());checks['newbindings400realbytes']=b['brain_final_step']==400 and b['brain_sha256']==c.BRAIN_SHA and b['brain_format']==p['brain_format'] and sha(b['brain_checkpoint'])==c.BRAIN_SHA
 amendment=json.loads((HERE/'AMENDMENT_PREREG.json').read_text());checks['originaltwo_prereg_preserved']=p['representative_videos']==['P_left_pulse','P_left_DN_cut'];checks['three_video_amendment']=amendment['representative_videos']==['P_empty','P_left_pulse','P_left_DN_cut'] and amendment['new_physical_trials']==0;checks['initial26check_archiveSHA']=sha(HERE/'CODEONLY_CHECK_INITIAL.json')==amendment['initial_codecheck_sha256'];choices=[n for n in ast.walk(a.source('render.py')) if isinstance(n,ast.keyword) and n.arg=='choices'];checks['actualCLI_threecases']=len(choices)==1 and ast.literal_eval(choices[0].value)==('P_empty','P_left_pulse','P_left_DN_cut');checks['sixbody_summarysamekind_explicitnewcasekind']=p['summary_kind']=='ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY' and p['result_kind']=='SOURCE_NECK_FEEDBACK_AGGREGATION_BODY_PILOT_V2';checks['no_newphysics']=p['new_training_updates']==p['new_physics_steps']==0
 checks['noheavyimports_or_actualarrays']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','model','student_wave'))
 plot=(HERE/'plot_sync.py').read_text();contract=(HERE/'contracts.py').read_text();checks['v1v2comparisonthree_actualcases']="selected=('P_empty','P_left_pulse','P_right_pulse')" in plot and 'V1_V2_EMPTY_LEFT_RIGHT_ACTUAL_COMPARISON.png' in plot;checks['comparison_actualfourmetrics']=all(x in plot for x in ('ANN_turn_RMS','recovery_turn_RMS_descriptive','original_slip','final_yaw_deg'));checks['new400offlineproofrequired']='VERIFY_000400.json' in contract and 'SOURCE_NECK_FEEDBACK_AGGREGATION_V2' in contract;checks['nonew_eye_or_targetrender']='capture_rgb' not in (HERE/'adapter.py').read_text() and 'add_decor' not in (HERE/'adapter.py').read_text();checks['newcandidatecaption']='Feedback aggregation v2 | source2 DNp09' in ast.unparse(a.source('render.py'))
 if not all(checks.values()):raise ValueError(str({k:v for k,v in checks.items() if not v}))
 hashes={str(HERE/n):sha(HERE/n) for n in files+('PREREG.json','AMENDMENT_PREREG.json','AMENDMENT_INITIAL_IDENTITY.json','CODEONLY_CHECK_INITIAL.json')};hashes.update({str(c.OLD_VISUAL/n):sha(c.OLD_VISUAL/n) for n in ('contracts.py','render.py','plot_sync.py')})
 active=HERE/'CODEONLY_CHECK.json';initial=HERE/'CODEONLY_CHECK_INITIAL.json'
 if active.exists() and active.read_bytes()!=initial.read_bytes():raise FileExistsError('Only explicitly archived initialcheck may refresh before actualrender')
 if any((HERE/name).exists() for name in ('SYNCHRONIZED_EVIDENCE','VIDEO_P_empty','VIDEO_P_left_pulse','VIDEO_P_left_DN_cut')):raise FileExistsError('No checkrefresh after actualvisualexecution')
 with active.open('w' if active.exists() else 'x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,initial_codecheck_sha256=amendment['initial_codecheck_sha256'],Pempty_CLI_amended_before_render=True,numerical_imports=False,actual_trial_arrays_read=False,Core_constructions=0,model_forward_calls=0,render_calls=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,actualarrays=False)))
if __name__=='__main__':main()
