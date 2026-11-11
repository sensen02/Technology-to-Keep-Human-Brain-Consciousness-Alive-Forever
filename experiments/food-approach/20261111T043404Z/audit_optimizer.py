"""Independent JSON/checkpoint/source audit of the finite CEM protocol; no physics."""
from pathlib import Path
import json,hashlib
import numpy as np
from cem import propose,update
ROOT=Path(__file__).resolve().parent

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
summary=json.loads((ROOT/'FINAL_SUMMARY.json').read_text());prep=json.loads((ROOT/'PREPARATION.json').read_text());checks={}
checks['complete_fixed_budget']=summary['status']=='COMPLETE' and summary['generations']==4 and summary['left_actual_rollouts']==16
for p,h in prep['source_hashes'].items():checks['frozen_source:'+p]=digest(p)==h
rng=np.random.default_rng(20261111);mean=np.array([0.,.05,0.]);sigma=np.full(3,.05);allresults=[]
for gen,history in enumerate(summary['history'],1):
 proposal=json.loads((ROOT/f'GEN{gen:02d}_PROPOSAL.json').read_text());frozen=json.loads((ROOT/f'GEN{gen:02d}_FROZEN.json').read_text())
 d,b,a=propose(mean,sigma,rng)
 checks[f'g{gen}_proposal_exact_rng_mean_sigma']=proposal['generation']==gen and proposal['rng_before']==b and proposal['rng_after']==a and np.array_equal(d,proposal['deltas']) and proposal['mean']==mean.tolist() and proposal['sigma']==sigma.tolist()
 scores=[]
 for i,value in enumerate(history['results']):
  result=json.loads(Path(value['result']).read_text());audit=json.loads(Path(value['audit']).read_text());candidate=frozen['candidates'][i]
  checks[f'g{gen}c{i}_actual_artifact_identity']=digest(value['result'])==value['result_sha256'] and digest(value['audit'])==value['audit_sha256'] and digest(value['checkpoint'])==value['sha256'] and candidate['sha256']==value['sha256']
  checks[f'g{gen}c{i}_all_actual_integrity']=audit['integrity_all_pass'] and all(audit['checks'].values()) and len(audit['checks'])==92 and not result['engineering_pass'] and result['criteria']['all_six'] and result['criteria']['strict_nonfoot_zero'] and result['criteria']['turn_sign_magnitude'] and all(result['autonomous']['gates'][k] for k in ('support','nonfoot','com','forward'))
  checks[f'g{gen}c{i}_unmodified_objective']=value['objective']==value['slip']==result['autonomous']['slip_ratio_max'] and value['delta']==d[i].tolist()
  scores.append(value['objective']);allresults.append(value)
 nm,ns,elite=update(mean,sigma,d,scores)
 checks[f'g{gen}_update_exact']=history['elite_indices']==elite and history['mean_after']==nm.tolist() and history['sigma_after']==ns.tolist() and history['rng_after']==rng.bit_generator.state
 mean,sigma=nm,ns
checks['no_selected_case_or_confirmation']=summary['selected'] is None and not summary['selected_all_three_pass'] and summary['confirmations']==[] and not (ROOT/'SELECTED.json').exists()
report=dict(all_passed=all(checks.values()),checks=checks,actual_left_cases=16,best_failed=min(allresults,key=lambda r:r['slip']),physical_integration_steps=0,optimizer_weight_updates=0,scope='Independent exact proposal/update/RNG and verified whole-rollout score audit; no passing motor inferred.')
(ROOT/'OPTIMIZER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),actual_left_cases=16,best_slip=report['best_failed']['slip'])))
if not report['all_passed']:raise ValueError('optimizer audit failed')
