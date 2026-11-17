"""Actualsaved14case visual/olfactory/source/body evidence; no NN or physical execution."""
import argparse,os
from contracts import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY noarrays/plot');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cases={name:load(name) for name in IDS};base=cases['O_noodor'];ledger={}
 for c in cases.values():ledger.update(c['source_hashes'])
 out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);fig,axes=plt.subplots(14,4,figsize=(20,39),sharex=True)
 for name,ax in zip(IDS,axes):
  c=cases[name];d=c['decisions'];g=c['geometry'];e=d['time']-5.8;r=c['result'];ax[0].plot(e,d['visual_raw_turn'],label='frozen visual NN');ax[0].plot(e,d['odor_raw_turn'],label='frozen odor NN');ax[0].plot(e,d['raw_turn'],label='DN-delayed neural turn',lw=1);ax[0].set_ylabel(name,fontsize=8);ax[0].set_ylim(-1.1,1.1)
  ax[1].plot(e,d['AN_held'][:,0],label='AN1 receiver');ax[1].plot(e,d['DN_held'][:,0],label='DNp09[0] receiver');ax[1].plot(e,d['DN_held'][:,1],label='DNp09[1] receiver');ax[1].set_ylim(-1.1,1.1)
  ax[2].plot(g['time']-5.8,np.degrees(c['yaw']),label='actual yaw');ax[2].plot(g['time']-5.8,np.degrees(base['yaw']),'--',label='matched O_noodor');ax[2].plot(g['time']-5.8,np.degrees(c['yaw']-base['yaw']),':',label='minus O_noodor');ax[2].set_title('Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+f' slip {r["autonomous"]["slip_ratio_max"]:.5f}; stable={r["functional_stable_body"]}',fontsize=8)
  for side,color in ((0,'#2166ac'),(1,'#b2182b')):ax[3].plot(e,d['receptors_used'][:,side],color=color,label='ENV receptor '+('L' if side==0 else 'R'));ax[3].plot(e,d['actor_receptors'][:,side],'--',color=color,label='ACTOR '+('L' if side==0 else 'R'))
  ax[3].set_title('Dimensionless receptor activity; native field geometry',fontsize=8)
  for item in ax:item.grid(alpha=.2);item.axvline(0,color='black',lw=.7);item.axvspan(-5.8,0,color='gray',alpha=.06)
  if name in ('F_left_DN_cut','O_left_odorcut'):
   for item in ax:item.axvline(1.6,color='red',ls='--',lw=.8)
 for item in axes[0]:item.legend(fontsize=6)
 for item in axes[-1]:item.set_xlabel('Actual ANN elapsed s (global -5.8)')
 fig.suptitle('ACTUAL RGB + HEAD-LOCAL WORLD ODOR -> FUSION BRAIN16 -> REAL-ID NECK -> OTHER MOTOR256 -> FULL BODY\n14 distinct audited complete trials;336 neural memories;CNNcache/filterpending/AN1DN2 separate. Original failures retained.',fontsize=13);fig.tight_layout(rect=(0,.02,1,.975));fig.text(.02,.006,'Visual and olfactory sources are independent world anchors on the same side, NOT one shared object. No food arrival/native neural or multisensory-superiority claim.',fontsize=10);p0=out/'FOURTEEN_CASES_ACTUAL_MULTISENSORY_SOURCE_BODY.png';fig.savefig(p0,dpi=130);plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(15,12));pairs=(('F_left_DN_cut','F_left_agree',1.6),('O_left_odorcut','O_left',1.6),('O_left_swap','O_left',0.))
 for row,(name,normal,when) in enumerate(pairs):
  c=cases[name];n=cases[normal];d=c['decisions'];e=d['time']-5.8;axes[row,0].plot(e,n['decisions']['raw_turn'],label=normal+' decoded');axes[row,0].plot(e,d['raw_turn'],label=name+' decoded');axes[row,0].plot(e,d['odor_raw_turn'],':',label='actual odor NN input');axes[row,1].plot(c['geometry']['time']-5.8,np.degrees(c['yaw']-n['yaw']),label='actual control minus normal yaw')
  for item in axes[row]:item.axvline(when,color='red',ls='--',label='intervention'+str(when)+'s');item.grid(alpha=.2);item.legend(fontsize=8);item.set_xlabel('Actual ANN elapsed s')
 fig.suptitle('MATCHED DN CUT / OLFACTORY ACTOR CUT / RECEPTOR SWAP\nActual state/input/action prefixes independently verified; effects descriptive, no new thresholds.',fontsize=13);fig.tight_layout(rect=(0,0,1,.94));p1=out/'MATCHED_MULTISENSORY_INTERVENTIONS.png';fig.savefig(p1,dpi=150);plt.close(fig);unchanged(ledger)
 fresh(out/'PLOT_REVIEW.json',dict(status='ACTUAL_14CASE_MULTISENSORY_NECK_BODY_PLOTS',source_hashes=ledger,images=[dict(path=str(p),sha256=sha(p)) for p in (p0,p1)],new_physics_steps=0,NN_forward_calls=0,all14_prior_integrity_required=True,same_side_not_same_source=True,food_arrival_or_parking_claim=False,parent_actual_review='PENDING'));print(str(out/'PLOT_REVIEW.json'))
if __name__=='__main__':main()
