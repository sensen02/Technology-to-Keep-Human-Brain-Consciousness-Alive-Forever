"""New bounded side-only applicability; old C6allstable gate remains unsatisfied."""
from contracts import *
REQUIRED=('C_left_near','C_right_near','C_left_far','C_right_far','C_noodor')
def side_admission():
 sp=C/'C_DEVELOPMENT_SUMMARY.json';summary=read(sp);require(summary['all_integrity_pass'] is True and len(summary['rows'])==6,'all6Cintegrityrequired');hashes={str(sp):sha(sp)};evidence={}
 for row in summary['rows']:
  rp=Path(row['result']);ap=Path(row['audit']);r=read(rp);a=read(ap);require(a['integrity_all_pass'] is True and all(a['checks'].values()) and a['result_sha256']==sha(rp),'Cactualauditbinding');require(r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['behavior_checkpoint_sha256']==a['behavior_checkpoint_sha256']==SENSOR_SHA,'samepolicyC')
  stable=bool(r['functional']['stable_body']);all6=bool(r['autonomous_cycles']['all_legs_completed']);nf=r['native']['post_nonfoot_peak_native']==0.;other4=all(r['autonomous']['gates'][k] for k in ('support','nonfoot','com','forward'))
  evidence[r['condition']]=dict(stable_body=stable,all6=all6,nativeNF0=nf,other4=other4,source_result=str(rp),source_result_sha256=sha(rp),audit=str(ap),audit_sha256=sha(ap))
  if r['condition'] in REQUIRED:require(stable and all6 and nf and other4,'side/noodor Cprerequisitefailed '+r['condition'])
  hashes[str(rp)]=sha(rp);hashes[str(ap)]=sha(ap)
  for p,h in r['source_hashes'].items():require(sha(p)==h,'Csource drift '+p);hashes[p]=h
 require(all(k in evidence for k in REQUIRED),'five requiredside/noodorconditionsmissing')
 for p,h in summary['source_hashes'].items():require(sha(p)==h,'Csummary source drift');hashes[p]=h
 screen=ROOT/'BASELINE_LAYOUT_SCREEN.json';s=read(screen);require(s['before_any_D_actual'] is True and s['proposal_sha256']==sha(ROOT/'PREREG_PROPOSAL.json') and s['side_amendment_sha256']==sha(ROOT/'AMENDMENT_SIDE_ONLY.json') and s['all8_patch_not_naturally_entered'] is True,'parentDsidebaselinegeometrystaticproofrequired')
 hashes[str(screen)]=sha(screen);hashes[str(ROOT/'AMENDMENT_SIDE_ONLY.json')]=sha(ROOT/'AMENDMENT_SIDE_ONLY.json')
 # Explicit original gate and excluded-front evidence, never set originalFalse toTrue.
 proof=dict(scope='SIDE_ONLY_D8_CONFIRMATION',original_C6_allstable=summary['all_stable_body'],original_full_C6_gate_satisfied=bool(summary['all_stable_body']),new_side_only_gate_satisfied=True,front_qualified=bool(evidence['C_front']['stable_body']),front_evidence=evidence['C_front'],required_conditions=list(REQUIRED),required_evidence={k:evidence[k] for k in REQUIRED},not_general_direction_or_platformqualification=True)
 if (ROOT/'SIDE_ADMISSION_PROOF.json').exists():require(read(ROOT/'SIDE_ADMISSION_PROOF.json')==proof,'sideproof drift')
 else:
  with (ROOT/'SIDE_ADMISSION_PROOF.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 hashes[str(ROOT/'SIDE_ADMISSION_PROOF.json')]=sha(ROOT/'SIDE_ADMISSION_PROOF.json');return hashes
