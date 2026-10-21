"""Bounded frozen-data acceptance verification; no model execution or imports."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
BUNDLES = {
 'fixedinput': ROOT/'outputs/model_branches/p1_fixed_input_clock_replay/20261013_234617_bd6f9aa2',
 'bodyclock': ROOT/'outputs/p2_body_clock/20261013_234422_9814ef63',
 'sampling': ROOT/'outputs/p2_event_sampling/20261014_000748_5a68d98c',
 'state': ROOT/'outputs/p2_state_trace/20261014_002347_4efdb3df',
 'method': ROOT/'outputs/p2_voltage_method_comparison/20261014_003713_eba8a3fc',
 'coupled_history': ROOT/'outputs/model_branches/projected_pose_convergence/20261013_072227',
}
def digest(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out = DESK/'P2-验收矩阵.json'
 report = DESK/'P2-验收矩阵报告.md'
 if out.exists() or report.exists():
  raise FileExistsError('Never overwrite an existing acceptance delivery')
 paths = [p for d in BUNDLES.values() for p in d.iterdir() if p.is_file()]
 contracts = ['AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md','CLOCK_REFINEMENT.zh-CN.md','FIXED_REPLAY_REFINEMENT.zh-CN.md','P2_STATE_TRACE_PROTOCOL.zh-CN.md','P2_VOLTAGE_METHOD_PROTOCOL.zh-CN.md','run_p2_state_trace.py','tools_p2_event_sampling.py','run_p1_fixed_command_body_clock.py','STRICT_ACCEPTANCE_BOUNDARY.zh-CN.md','P2_ACCEPTANCE_MATRIX.zh-CN.md','tools_p2_acceptance_matrix.py']
 paths += [ROOT/p for p in contracts]
 pins = {str(p): digest(p) for p in paths}
 data = {k: json.loads((v/('convergence_results.json' if k=='coupled_history' else 'result.json')).read_text()) for k,v in BUNDLES.items()}
 checks = {}
 for name,j in data.items():
  checks[name+'_stored_checks_all_true'] = all(v is True for v in j.get('checks',{}).values()) if name not in ('sampling','coupled_history') else True
  declared = dict(j.get('artifact_digests',{}))
  for field,fn in [('traces_sha256','traces.npz'),('source_snapshot_sha256','source_snapshot.zip'),('sampling_sha256','sampling.npz'),('plot_sha256','sampling.png')]:
   if field in j: declared[fn]=j[field]
  for fn,h in declared.items(): checks[name+'_artifact_'+fn] = digest(BUNDLES[name]/fn)==h
  if (BUNDLES[name]/'source_snapshot.zip').exists():
   source = j.get('source_hashes',j.get('identity',{}).get('source_hashes',{}))
   with zipfile.ZipFile(BUNDLES[name]/'source_snapshot.zip') as z:
    manifest=json.loads(z.read('manifest.json'))
    members={row['path']:row for row in manifest['files']}
    for p,h in source.items():
     row=members.get(p)
     checks[name+'_zip_'+p] = bool(row and row['sha256']==h and hashlib.sha256(z.read(row['id'])).hexdigest()==h)
 with np.load(BUNDLES['fixedinput']/'traces.npz',allow_pickle=False) as f, np.load(BUNDLES['state']/'traces.npz',allow_pickle=False) as s, np.load(BUNDLES['method']/'traces.npz',allow_pickle=False) as m:
  checks['current_array_exact_all_three'] = np.array_equal(f['input_current_nA'],s['input_current_nA']) and np.array_equal(f['input_current_nA'],m['input_current_nA'])
  checks['tier_array_exact_all_three'] = np.array_equal(f['tier_root_ids'],s['tier_root_ids']) and np.array_equal(f['tier_root_ids'],m['tier_root_ids'])
  suffixes=['spike_counts','applied_controls','decoded_next_controls','motor_rates_hz','event_step','event_neuron','event_time_ms']
  for label in ('base','half','quarter'):
   for suffix in suffixes: checks['BE_prior_'+label+'_'+suffix]=np.array_equal(f[label+'_'+suffix],m['backward_euler_'+label+'_'+suffix])
  for method in ('backward_euler','frozen_exact'):
   for suffix in suffixes: checks[method+'_repeat_'+suffix]=np.array_equal(m[method+'_base_'+suffix],m[method+'_base_repeat_'+suffix])
  for label,dt in [('base',.5),('half',.25),('quarter',.125)]:
   for method in ('backward_euler','frozen_exact'):
    prefix=method+'_'+label+'_'; ids=m[prefix+'event_neuron']; ts=m[prefix+'event_time_ms']; bins=np.searchsorted(np.arange(5,51,5),ts,side='left'); counts=np.zeros((10,12000),dtype=int);np.add.at(counts,(bins,ids),1)
    checks[prefix+'counts_from_events']=np.array_equal(counts,m[prefix+'spike_counts'])
    checks[prefix+'clock']=bool(np.allclose(ts,(m[prefix+'event_step']+1)*dt,atol=1e-12,rtol=0))
    checks[prefix+'command_lag']=np.array_equal(m[prefix+'applied_controls'][1:],m[prefix+'decoded_next_controls'][:-1]) and not np.any(m[prefix+'applied_controls'][0])
 with np.load(BUNDLES['bodyclock']/'traces.npz',allow_pickle=False) as b:
  body=[]
  for key,(a,c) in {'base_vs_half':('base','half'),'half_vs_quarter':('half','quarter'),'base_vs_quarter':('base','quarter')}.items():
   observed=float(np.max(abs(b[a+'_height_model']-b[c+'_height_model']))); stored=data['bodyclock']['comparisons'][key]
   checks['body_'+key+'_stored_height']=observed==stored['max_height_difference_model']
   body.append(dict(pair=key,applicability='diagnostic-not-applicable',verdict='REFERENCE_WITHIN' if observed<=.05 and stored['escape_time_difference_s']<=.001 else 'REFERENCE_EXCEEDED',**stored))
 historical=[]; j=data['coupled_history']
 for key,v in j['comparisons'].items():
  historical.append(dict(pair=key,applicability='applicable',height_verdict='PASS' if v['height_max_abs_delta_mm']<=.05 else 'FAIL',escape_verdict='PASS' if abs(v['escape_delta_s'])<=.001 else 'FAIL',**v))
 for arm in ('normal','sensory_cut','motor_cut'):
  a=j['episodes']['both_half_'+arm];b=j['episodes']['both_quarter_'+arm];h=max(abs(x['height_model']-y['height_model']) for x,y in zip(a['records'],b['records'])); e=abs(a['escape_time_s']-b['escape_time_s'])
  historical.append(dict(pair='both_half_vs_both_quarter_'+arm,applicability='applicable',height_model_mm=h,escape_s=e,height_verdict='PASS' if h<=.05 else 'FAIL',escape_verdict='PASS' if e<=.001 else 'FAIL',execution='historical_saved_records_only'))
 checks['frozen_inputs_unchanged_after_read']=all(digest(p)==h for p,h in pins.items())
 status='PASS' if all(checks.values()) else 'FAIL'
 result=dict(status=status,scope='read-only frozen artifact and numerical summary verification; NOT simulations, biological validation or convergence PASS',stage_S1='PARTIAL_ENGINEERING_VERIFIED_NOT_ACCEPTED',P2_deliverable='DIAGNOSTIC_REPORT_COMPLETE',historical_numerical_reproduction_this_turn=False,new_simulations=0,new_tests=0,checks=checks,source_and_artifact_sha256=pins,body_reference_comparisons=body,historical_applicable_contract_comparisons=historical,method_diagnostic_comparisons=data['method']['comparisons_within_method'],sampling_diagnostic_comparisons=data['sampling']['comparisons'],contract=dict(path=str(ROOT/'AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md'),lines=[11,13],height_model_mm=.05,escape_s=.001,scope='100ms coupled five clocks three arms; no fixed-input command norm'),no_applicable_fixedinput_command_tolerance=True,not_executed=['current-version complete 100ms coupled contract','biological held-out calibration','new standing or walking acceptance'],images_actually_viewed=[str(BUNDLES['sampling']/'sampling.png')],command=str(ROOT/'venv_body/bin/python')+' '+str(Path(__file__).resolve()),exit_code=0 if status=='PASS' else 1)
 report.write_bytes((ROOT/'P2_ACCEPTANCE_MATRIX.zh-CN.md').read_bytes())
 result['report_sha256']=digest(report)
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=status,checks=len(checks),failures=[k for k,v in checks.items() if not v],json=str(out),report=str(report),json_sha256=digest(out),report_sha256=digest(report)),ensure_ascii=False,indent=2))
 return 0 if status=='PASS' else 1
if __name__=='__main__':
 raise SystemExit(main())
