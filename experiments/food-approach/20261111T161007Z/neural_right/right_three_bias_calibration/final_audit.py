"""Read-only final all4/selection/neutral conjunction and artifact identity audit.
Pure Python, no model/physics libraries, zero integration/optimizer updates.
"""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parent
_cache={}
def digest(path):
 path=Path(path).resolve()
 if path not in _cache:
  with path.open('rb') as f:_cache[path]=hashlib.file_digest(f,'sha256').hexdigest()
 return _cache[path]
def read(path):return json.loads(Path(path).read_text())
def main():
 checks={}
 def check(n,v):
  checks[n]=bool(v)
  if not checks[n]:raise ValueError('final audit failed '+n)
 prep=read(ROOT/'PREPARATION.json');frozen=read(ROOT/'FROZEN.json');proposal=read(ROOT/'PROPOSAL.json');summary=read(ROOT/'FINAL_SUMMARY.json');proof=read(ROOT/'SELECTED.json')
 check('preparation_all36',prep['all_passed'] and all(prep['checks'].values()));check('all4_frozen_before_actual',frozen['all4_hashes_saved_before_any_actual'] and len(frozen['candidates'])==4);check('frozen_hash',digest(ROOT/'FROZEN.json')==prep['frozen_sha256']);check('proposal_hash',digest(ROOT/'PROPOSAL.json')==prep['proposal_sha256']==frozen['proposal_sha256']==proof['proposal_sha256']);check('one_generation',frozen['generations']==1 and summary['extra_generations']==0);check('no_training',summary['optimizer_training_updates']==0)
 for path,h in {**prep['source_hashes'],**prep['input_stream_hashes']}.items():check('preflight_input:'+path,digest(path)==h)
 def actual(c,condition):
  case=ROOT/c['id'];out=case/('EVAL_'+condition);r=read(out/'RESULT.json');a=read(case/('VERIFY_'+condition+'.json'));identity=read(out/'IDENTITY.json');prefix=c['id']+'_'+condition+'_'
  check(prefix+'all_individual_checks',a['integrity_all_pass'] and all(a['checks'].values()));check(prefix+'model_hash',digest(c['checkpoint'])==c['sha256']==r['checkpoint_sha256']==a['checkpoint_sha256']);check(prefix+'condition',r['condition']==condition)
  canonical={k:v for k,v in identity.items() if k!='identity_sha256'};h=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest();check(prefix+'identity_digest',h==identity['identity_sha256']==r['identity_sha256'])
  for path,meta in identity['files'].items():check(prefix+'identity_input:'+path,digest(path)==meta['sha256'])
  for path,h in r['source_hashes'].items():check(prefix+'source:'+path,digest(path)==h)
  for key,filename in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:check(prefix+filename,digest(out/filename)==r[key]['sha256'])
  native=r['native'];flags=dict(original_five=bool(r['autonomous']['all_gates']),all_six=bool(r['autonomous_cycles']['all_legs_completed']),strict_nonfoot_zero=bool(r['autonomous']['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(condition=='neutral' or -math.degrees(r['yaw_change_rad'])>=30.),strict_native_nonfoot_zero=bool(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['extra_force_max']==0.));flags['eligible']=all(flags.values())
  check(prefix+'samewindow_criteria_exact',flags==r['criteria'] and r['engineering_pass']==flags['eligible']==a['engineering_pass']);check(prefix+'native_complete',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
  row=dict(id=c['id'],index=c['index'],delta=c['delta'],checkpoint=c['checkpoint'],sha256=c['sha256'],condition=condition,integrity_all_pass=True,engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],yaw_deg=math.degrees(r['yaw_change_rad']),forward=r['autonomous']['forward_signed_native_per_s'],criteria=r['criteria'],cycles=r['autonomous_cycles']['complete_per_leg'],native_nonfoot_peak=native['post_nonfoot_peak_native'],audit_checks=len(a['checks']),result=str(out/'RESULT.json'),result_sha256=digest(out/'RESULT.json'),audit=str(case/('VERIFY_'+condition+'.json')),audit_sha256=digest(case/('VERIFY_'+condition+'.json')))
  return row
 rows=[actual(c,'right') for c in frozen['candidates']];check('exact_all4_result_binding',rows==summary['results']==proof['all4_results']);check('exact_RIGHT_budget',len(list(ROOT.glob('c*/EVAL_right/RESULT.json')))==summary['actual_RIGHT_rounds']==4)
 passing=[r for r in rows if r['engineering_pass']];selected=None if not passing else min(passing,key=lambda r:(r['slip'],r['index']));check('selection_exact_minimum_slip_index',selected==proof['selected']);check('summary_selection',summary['selected']==(None if selected is None else selected['id']));neutral=None
 if selected is None:check('no_unselected_neutral',summary['neutral_confirmation'] is None and not list(ROOT.glob('c*/EVAL_neutral/RESULT.json')))
 else:
  c=next(c for c in frozen['candidates'] if c['id']==selected['id']);neutral=actual(c,'neutral');check('same_checkpoint_neutral',neutral['sha256']==selected['sha256'] and neutral==summary['neutral_confirmation']);check('only_one_selected_neutral',len(list(ROOT.glob('c*/EVAL_neutral/RESULT.json')))==1)
 check('final_conjunction',summary['selected_all_required_pass']==bool(selected is not None and neutral is not None and neutral['engineering_pass']))
 report=dict(all_passed=all(checks.values()),checks=checks,individual_actual_RIGHT_rounds=4,selected=None if selected is None else selected['id'],selected_all_required_pass=summary['selected_all_required_pass'],results=rows,neutral_confirmation=neutral,selection_proof_sha256=digest(ROOT/'SELECTED.json'),final_summary_sha256=digest(ROOT/'FINAL_SUMMARY.json'),source_sha256=digest(Path(__file__)),physics_steps=0,optimizer_updates=0,model_replays=0,scope='independent final artifact/selection/conjunction audit; each actual neural replay and full endpoint audit separately bound by checked SHA; selection proof additionally bound here')
 out=ROOT/'FINAL_AUDIT.json'
 if out.exists():raise ValueError('do not overwrite final independent audit')
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),selected=report['selected'],selected_all_required_pass=report['selected_all_required_pass'])),flush=True)
if __name__=='__main__':main()
