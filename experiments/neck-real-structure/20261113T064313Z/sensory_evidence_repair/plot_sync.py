"""Exact14caseactualplots plus JSON-only old/new diagnostics, no presumed success."""
from adapter import namespace
from contracts import *
def comparison():
 # Only after main actualplot admission; old/new metadata sourceSHA independently checked.
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 new,review,ledger=admission();old,priorledger=old_summary();ledger.update(priorledger);rows=[]
 def case(s,name):return next(x for x in s['rows'] if x['condition']==name)
 specs=[('O_left_odorcut','cut recovery output RMS',lambda s:case(s,'O_left_odorcut')['metrics']['odorcut_recovery_RMS_05to1s_after_cut']),('O_noodor','Original slip (limit .5)',lambda s:case(s,'O_noodor')['original_slip']),('F_left_agree','Legs completed a full cycle (of6)',None),('F_left_agree','Signedbodyeffect2.6 vsOnoodor deg',lambda s:case(s,'F_left_agree')['metrics']['signed_body_effect_at2_6_deg'])]
 # Cycle detail comes from actual RESULT, not a new threshold or summary allboolean surrogate.
 def cycles(root):
  result=read(root/'trials/F_left_agree/RESULT.json');counts=result['autonomous_cycles']['complete_per_leg'];return sum(int(v>=1) for v in counts.values()),counts
 oldcount,oldcounts=cycles(OLD.parent/'sensory_body');newcount,newcounts=cycles(ROOT)
 fig,axes=plt.subplots(1,4,figsize=(17,4.5));names=('old db25/66f','repair newodor/newbrain')
 for i,(condition,title,fn) in enumerate(specs):
  values=[oldcount,newcount] if i==2 else [float(fn(old)),float(fn(new))];axes[i].bar(names,values,color=['#b2182b','#2166ac']);axes[i].set_title(condition+'\n'+title,fontsize=10);axes[i].grid(axis='y',alpha=.2)
  if i==1:axes[i].axhline(.5,color='black',ls='--',label='originalslip.5');axes[i].legend(fontsize=8)
  for k,v in enumerate(values):axes[i].annotate(format(v,'.6g'),(k,v),xytext=(0,5 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
  rows.append(dict(condition=condition,metric=title,old_actual=values[0],new_actual=values[1],no_new_success_threshold=True))
 fig.suptitle('ACTUAL OLD / NEURAL-ODOR-AND-FUSION REPAIR BODY METRICS\nSame14conditions, separate audited physical trials. Actual failures retained, no assumed improvement.',fontsize=12);fig.tight_layout(rect=(0,.04,1,.85));out=HERE/'SYNCHRONIZED_EVIDENCE/OLD_NEW_ACTUAL_ODOR_FUSION_COMPARISON.png';fig.savefig(out,dpi=150);plt.close(fig);unchanged(ledger)
 fresh(out.with_suffix('.json'),dict(status='ACTUAL_METADATA_ONLY_OLD_NEW_COMPARISON',rows=rows,old_Fleft_cycles=oldcounts,new_Fleft_cycles=newcounts,source_hashes=ledger,image=str(out),image_sha256=sha(out),model_forward_calls=0,new_physics_steps=0,parent_actual_review='PENDING'))
def main():
 namespace('plot_sync.py')['main']()
 # Old entry returns without numerical admission ifflag absent; do not bypass it forcomparison.
 import sys
 if '--execute-after-parent-release' in sys.argv:comparison()
if __name__=='__main__':main()
