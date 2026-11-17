"""Stdlib+savedarrays matched14pairedreview, no model or physical execution."""
from runtime import *
def main():
 verify();import numpy as np
 summary=read(ROOT/'SENSORY_BODY_SUMMARY.json');checks={};pairs=[]
 for control,normal,rows in [('F_left_DN_cut','F_left_agree',1480),('O_left_odorcut','O_left',1480),('O_left_swap','O_left',1160)]:
  a=arrays(ROOT/'trials'/control/'neural_decisions.npz');b=arrays(ROOT/'trials'/normal/'neural_decisions.npz');prefix=all(np.array_equal(a[k][:rows],b[k][:rows]) for k in ('X238','full_hidden_before','full_hidden_after','actually_applied_action','actor_receptors','held_features','visual_raw_turn','odor_raw_turn'));checks[control+'_prefix_exact']=prefix;pairs.append(dict(control=control,normal=normal,matching_prefix_neural_rows=rows,matching_prefix_allsource_state_actions_exact=prefix))
 checks['14independent_integrity']=summary['all14_integrity_verified'] and summary['unique_physical_trials']==14;checks['strictnewsummarykind']=summary['kind']==SUMMARY_KIND;checks['336ABI']=summary['state_total']==336;checks['nofoodarrival']=not summary['food_arrival_or_parking_claim']
 if not all(checks.values()):raise ValueError('Matched source-intervention review '+str(checks))
 save(ROOT/'PARENT_READY_REVIEW.json',dict(integrity_all_pass=True,checks=checks,paired_checks=pairs,summary_sha256=sha(ROOT/'SENSORY_BODY_SUMMARY.json'),functional_all_PASS_claim=False,original_slip_failures_retained=True,source_kind=KIND,summary_kind=SUMMARY_KIND,state_total=336,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks))))
if __name__=='__main__':main()
