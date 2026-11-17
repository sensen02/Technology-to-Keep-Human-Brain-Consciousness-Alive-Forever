"""Actualthreecase yaw/drift/support and event-time cycles; no synthetic pose/physicalsteps."""
import argparse,os
from contracts import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY noarrays/plot');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(k)=='1',k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cases={name:load(name) for name in CASES};ledger={}
 for c in cases.values():ledger.update(c['source_hashes'])
 out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);fig,axes=plt.subplots(3,4,figsize=(18,12),sharex=True)
 for name,ax in zip(CASES,axes):
  c=cases[name];t=c['trace'];e=t['time']-5.8;post=e>=0;d=c['decisions'];ax[0].plot(e,np.degrees(c['yaw']),label='actual yaw fromcandidate start');ax[0].set_ylabel(name);ax[1].plot(e[post],c['drift'][post],label='actual COM drift Lbody');ax[1].axhline(.5,color='gray',ls='--',label='preregistered drift gate .5');ax[2].plot(e,t['loads'].sum(1)/c['result']['weight_native'],label='actual support /weight');ax[2].axhline(.9,color='gray',ls='--',label='support gate .9')
  if name=='stop':ax[3].text(.1,.5,'Cycle counts NA for stop\nSupport verified; learnedSTOP not validated',transform=ax[3].transAxes)
  else:
   for j,leg in enumerate(('LF','LH','LM','RF','RH','RM')):ax[3].step(e,c['cycle_counts'][:,j],where='post',label=leg)
  for item in ax:item.axvspan(-5.8,0,color='gray',alpha=.08);item.axvline(0,color='black',lw=.8);item.grid(alpha=.2);item.legend(fontsize=7,loc='best');item.set_xlabel('Actualcandidate elapsed s (global -5.8)')
  ax[0].set_title(f'FINAL yaw {c["result"]["candidate_metrics"]["yaw_change_deg"]:+.3f} deg');ax[1].set_title(f'FINAL maxdrift {c["result"]["candidate_metrics"]["max_COM_xy_drift_bodylength"]:.6f} Lbody');ax[2].set_title('Native NF=0 / extra force=0, actualfull9.6s');ax[3].set_title('Actualsaved eventtimecyclecounts')
 fig.suptitle('ALGORITHM TEACHER — NOT ANN/NECK PIVOT\nSame-run fixed-shadow stop support and left/right near-in-place pivot; no new motion or physics rendered.',fontsize=14);fig.tight_layout(rect=(0,.035,1,.92));fig.text(.02,.008,'L_body is captured compiledgeometry native scale, absolute SI not established. Originalfive preserved separately; no learnedstop/pivot or biologicalvalidity claim.',fontsize=10);path=out/'ACTUAL_TEACHER_STOP_LEFT_RIGHT_SYNC.png';fig.savefig(path,dpi=150);plt.close(fig);unchanged(ledger);fresh(out/'PLOT_REVIEW.json',dict(kind='ACTUAL_ALGORITHM_TEACHER_STOP_PIVOT_SAVED_PLOTS',image=str(path),image_sha256=sha(path),source_hashes=ledger,physics_steps=0,teacher_forward_calls=0,policy_forward_calls=0,ALGORITHM_TEACHER_ONLY=True,parent_actual_review='PENDING'));print(str(out/'PLOT_REVIEW.json'))
if __name__=='__main__':main()
