"""Verifiedsixcase plot namespace plus actualv1/v2 summary comparisons, no models/Core."""
import sys
from adapter import namespace
from contracts import *

def compare_actual_summaries():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 new,hashes=admission();old,oldhashes=load_v1_summary();hashes.update(oldhashes);out=HERE/'SYNCHRONIZED_EVIDENCE';require(out.is_dir(),'newactualsixcaseplotsmustexistfirst');selected=('P_empty','P_left_pulse','P_right_pulse');fields=[('ANN_turn_RMS','ANN turn RMS','ratio'),('recovery_turn_RMS_descriptive','Recorded recovery RMS','ratio'),('original_slip','Original slip (limit .5)','ratio'),('final_yaw_deg','Final actual yaw','deg')];fig,axes=plt.subplots(3,4,figsize=(17,10));rows=[]
 for i,condition in enumerate(selected):
  v1=next(x for x in old['rows'] if x['condition']==condition);v2=next(x for x in new['rows'] if x['condition']==condition);record=dict(condition=condition,v1={},v2={},differences_v2_minus_v1={},v1_original_engineering_pass=v1['original_engineering_pass'],v2_original_engineering_pass=v2['original_engineering_pass'],v1_stable_body=v1['stable_body'],v2_stable_body=v2['stable_body'])
  for j,(key,label,unit) in enumerate(fields):
   a=float(v1[key] if key=='original_slip' else v1['metrics'][key]);b=float(v2[key] if key=='original_slip' else v2['metrics'][key]);record['v1'][key]=a;record['v2'][key]=b;record['differences_v2_minus_v1'][key]=b-a;axis=axes[i,j];axis.bar(['v1 eed1','v2 4153'],[a,b],color=['#b2182b','#2166ac']);axis.axhline(0,color='black',lw=.6);axis.set_title(condition+'\n'+label);axis.set_ylabel(unit);axis.grid(axis='y',alpha=.2)
   if key=='original_slip':axis.axhline(.5,color='black',ls='--',lw=1)
   span=max(abs(a),abs(b),1e-6)
   for k,value in enumerate((a,b)):axis.text(k,value+(.03*span if value>=0 else -.03*span),format(value,'.5f'),ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
   lo=min(0,a,b);hi=max(0,a,b,.5 if key=='original_slip' else 0);padding=max(hi-lo,span)*.18;axis.set_ylim(lo-padding,hi+padding)
  rows.append(record)
 fig.suptitle('ACTUAL V1 / FEEDBACK-AGGREGATION V2 BODY RESULTS\nSeparate completed audited trials under matching case labels; values and retained failures, no presumed improvement',fontsize=14);fig.tight_layout(rect=(0,.05,1,.91));fig.text(.015,.015,'Source 2DN/1AN artificial coupling; OTHER motor keeps basewalking. Recovery metrics descriptive; only dashed slip .5 is original engineering gate.',fontsize=10);image=out/'V1_V2_EMPTY_LEFT_RIGHT_ACTUAL_COMPARISON.png';fig.savefig(image,dpi=150);plt.close(fig);unchanged(hashes)
 fresh(out/'V1_V2_COMPARISON.json',dict(status='ACTUAL_MATCHING_CASE_V1_V2_DESCRIPTIVE_COMPARISON',candidate='feedback_aggregation_v2_final400',rows=rows,image=str(image),image_sha256=sha(image),source_hashes=hashes,new_physical_trials=0,model_forward_calls=0,original_failures_retained=True,all_functional_PASS_claim=False,parent_actual_review='PENDING'));print(str(image))

def main():
 ns=namespace('plot_sync.py');ns['main']()
 if '--execute-after-parent-release' in sys.argv:compare_actual_summaries()
if __name__=='__main__':main()
