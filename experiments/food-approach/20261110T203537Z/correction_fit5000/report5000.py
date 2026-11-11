"""Summarize settled actual results honestly; no optimization/physical work."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 summary=HERE/'FINAL5000_SUMMARY.json';s=json.loads(summary.read_text())
 if s['status']=='RUNNING':raise ValueError('wait for all authorized actual tasks to settle')
 rows=[];hashes={str(summary):digest(summary)}
 for name in ('neutral','left','right','speed085','speed090'):
  item=s['conditions'].get(name)
  if item is None:rows.append('|'+name+'|NOT_RUN|NOT_RUN|—|—|—|—|');continue
  if not item.get('result'):
   rows.append('|'+name+'|EXECUTION_FAILED|'+str(item.get('integrity_all_pass',False))+'|—|—|—|—|');continue
  path=Path(item['result'])
  if digest(path)!=item['result_sha256']:raise ValueError('result hash drift '+name)
  r=json.loads(path.read_text());a=r['autonomous'];counts=r['autonomous_cycles']['complete_per_leg']
  hashes[str(path)]=digest(path)
  if item.get('audit'):
   audit=Path(item['audit'])
   if digest(audit)!=item['audit_sha256']:raise ValueError('audit hash drift '+name)
   hashes[str(audit)]=digest(audit)
  rows.append('|'+name+'|'+str(item['engineering_pass'])+'|'+str(item['integrity_all_pass'])+'|'+str(a['forward_signed_native_per_s'])+'|'+str(a['slip_ratio_max'])+'|'+str(r['yaw_change_rad'])+'|' +','.join(str(counts[k]) for k in ('LF','LH','LM','RF','RH','RM'))+'|')
 training=json.loads((HERE/'TRAIN/RESULT.json').read_text())
 lines=['# Fixed5000 corrective-label experiment: bounded actual results','',
 'Final checkpoint: `'+training['checkpoint']+'`','',
 'SHA-256: `'+training['sha256']+'`','',
 'Status: **'+s['status']+'**. All five engineering conditions eligible: **'+str(s['all_conditions_eligible'])+'**. All five saved-trajectory integrity: **'+str(s['saved_trajectory_verification'])+'**.','',
 '| Condition | Engineering eligible | Actual saved integrity | Signed forward | Max slip | Yaw rad | Cycles LF,LH,LM,RF,RH,RM |','|---|---|---|---:|---:|---:|---|',*rows,'',
 'The original physical all5 gates, all6 sensed cycles, strict nonfoot0, signed turn≥0.01 and speed≤0.95 qualified matched new neutral are unchanged. Four commands use the exactly registered1s smoothstep ramp. Prior step3000 and ramp3000 failures remain immutable. Integrity and process exit0 establish consistent recorded execution, not passing behavior.','',
 '## Scientific intervention and retained failures','',
 'This is a newly authorized failed-state corrective-label experiment. Both source mixed trajectories remain dataset_safety_admitted=false forever, with recorded whole-mixed slips0.5495288701/0.5519080183. All1920 teacher labels per trajectory were independently replayed bitexactly and root residuals remained below0.02. Valid teacher provenance does not make the observed failed trajectories passing demonstrations, an optimal oracle, or evidence of causal recovery.','',
 'Exact parent3000 model, six Adam moments/counters, RNG and full0–3000 curve were preserved. lr0.0001, architecture, gate/bound1, features and target-position inhibition did not change. Exactly2000 additional updates to global5000. The objective adds exactly two corrective actual trajectories to the four original teacher trajectories, each trajectory mean weighted1/6; corrective total weight1/3. Original four have1920 postprime rows each; corrective two have960 each. Causal frozen base history includes all actual priming rows and source/phase/time are never neural inputs.','',
 'Training elapsed '+str(training['wall_s'])+' seconds;4000 and5000 saved-state audits each passed24 checks including exact resume/parent prefix/Adam/RNG/neutral outputs. Only final5000 was physically evaluated. Teacher-fit improvement alone is not physical success. No budget-matched old-four-only continuation was run, so no isolated causal effect of adding data is claimed.','',
 'No food behavior training, arrival, ingestion or satiety occurred. This is replacementANN engineering on the declared derived full body and finite PD, not original connectome, biology, arbitrary disturbance/duration robustness or full nativeD1–D8 certification. Parent owns any next authorized bounded experiment; this report launches nothing.']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 hashes.update({str(HERE/'FINAL_REPORT.md'):digest(HERE/'FINAL_REPORT.md'),str(HERE/'report5000.py'):digest(__file__),str(HERE/'TRAIN/RESULT.json'):digest(HERE/'TRAIN/RESULT.json')})
 (HERE/'FINAL_REPORT_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(json.dumps(dict(status=s['status'],all_conditions_eligible=s['all_conditions_eligible'],saved_trajectory_verification=s['saved_trajectory_verification'],report=str(HERE/'FINAL_REPORT.md'))))

if __name__=='__main__':main()
