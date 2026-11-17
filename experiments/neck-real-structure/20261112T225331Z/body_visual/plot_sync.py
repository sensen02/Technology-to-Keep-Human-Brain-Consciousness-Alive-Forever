"""Parent-only actual six-case body/source signal plots; no model/Core/physics."""
import argparse,os
from contracts import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY no arrays/plots');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cases={name:load(name) for name in IDS};out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);hashes={};fig,axes=plt.subplots(6,3,figsize=(18,21));base=cases['P_empty'];normal=cases['P_left_pulse'];bt=base['trace']['time']-5.8;normalyaw=np.degrees(normal['yaw'])
 for j,name in enumerate(IDS):
  case=cases[name];hashes.update(case['source_hashes']);d=case['decisions'];e=d['time']-5.8;g=case['geometry'];r=case['result'];axes[j,0].plot(e,d['cue'],label='experimental brain-side cue',color='#7b3294');axes[j,0].plot(e,d['AN_held'][:,0],label='actual receiver-held AN1',color='#2166ac');axes[j,0].set_ylabel(name+'\nInstruction / heldAN');axes[j,0].set_ylim(-1.1,1.1)
  axes[j,1].plot(e,d['DN_held'][:,0],label='receiver-held DNp09[0]',alpha=.8);axes[j,1].plot(e,d['DN_held'][:,1],label='receiver-held DNp09[1]',alpha=.8);axes[j,1].plot(e,d['raw_turn'],label='learned decoder turn',color='black',lw=1.2);axes[j,1].set_ylabel('Actual DN2 / learnedturn');axes[j,1].set_ylim(-1.1,1.1)
  yaw=np.degrees(case['yaw']);emptyyaw=np.degrees(base['yaw']);axes[j,2].plot(bt,yaw,label='actual body yaw',color='#2166ac');axes[j,2].plot(bt,emptyyaw,label='matched P_empty',color='#d95f02',ls='--');axes[j,2].plot(bt,yaw-emptyyaw,label='actual minus P_empty',color='#1b9e77',ls=':')
  if name in ('P_left_DN_cut','P_left_AN_cut','P_left_cuezero'):axes[j,2].plot(bt,normalyaw,label='matched normal P_left_pulse',color='gray',alpha=.65,ls='-.')
  axes[j,2].set_ylabel('Yaw from ANN handover (deg)');axes[j,2].set_title('Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+' | slip '+format(r['autonomous']['slip_ratio_max'],'.6f')+' | body-stable '+str(r['functional_stable_body']))
  for axis in axes[j]:
   axis.axvspan(-5.8,0,color='gray',alpha=.08);axis.axvline(0,color='black',lw=.7);axis.axvline(.6,color='#7b3294',lw=.6,ls=':');axis.axvline(3.6,color='#7b3294',lw=.6,ls=':');axis.grid(alpha=.2)
   if name in ('P_left_DN_cut','P_left_AN_cut'):axis.axvline(1.6,color='#b2182b',lw=1,ls='--')
  if j==0:
   for axis in axes[j]:axis.legend(fontsize=7,loc='best')
  if name=='P_left_DN_cut':axes[j,2].legend(fontsize=7,loc='best')
 for axis in axes[-1]:axis.set_xlabel('Actual ANN elapsed seconds (global time - 5.8); full settle/warmup retained')
 fig.suptitle('ACTUAL ARTIFICIAL BRAIN16 -> SOURCE-IDENTIFIED NECK PORTS -> OTHER MOTOR256 -> FULL BODY\n6 separate physical trials; real1ms queue / 2ms release / 5ms neural consumption; no visual or odor input',fontsize=14);fig.tight_layout(rect=(0,.025,1,.95));fig.text(.015,.006,'Cuts ANN1.6/global7.4 are engineering source silence + receiver0, not forced turn0; low-level basewalking belongs OTHER. No whole-neck/native-function claim.',fontsize=10);path=out/'SIX_CASES_SOURCE_BRAIN_BODY_SYNC.png';fig.savefig(path,dpi=150);plt.close(fig)
 # Explicit matched differences do not imply all sensors necessary or new independent repeats.
 fig,axes=plt.subplots(2,2,figsize=(14,9))
 for j,(name,label) in enumerate((('P_left_DN_cut','DN cut'),('P_left_AN_cut','AN cut'))):
  n=normal['decisions'];c=cases[name];d=c['decisions'];e=d['time']-5.8;axes[j,0].plot(e,n['raw_turn'],label='normal learnedturn');axes[j,0].plot(e,d['raw_turn'],label=label+' learnedturn');axes[j,0].set_ylabel(label+' / turn');axes[j,1].plot(bt,np.degrees(c['yaw']-normal['yaw']),label='cut minus matched normal actualyaw');axes[j,1].set_ylabel('Actual body paired yaw delta (deg)')
  for axis in axes[j]:axis.axvline(1.6,color='red',ls='--',label='engineering cut1.6s');axis.axvline(0,color='black',lw=.7);axis.grid(alpha=.2);axis.legend(fontsize=8);axis.set_xlabel('Actual ANN elapsed seconds')
 fig.suptitle('MATCHED ACTUAL PHYSICAL COMMUNICATION INTERVENTIONS\nConditional on this bounded scene/initial state; original slip FAIL and low-level basewalking retained');fig.tight_layout(rect=(0,0,1,.92));paired=out/'MATCHED_DN_AN_CUT_COMPARISONS.png';fig.savefig(paired,dpi=150);plt.close(fig)
 unchanged(hashes);fresh(out/'PLOT_REVIEW.json',dict(status='ACTUAL_SAVED_SIX_CASE_SIGNALS_AND_TRAJECTORIES_PLOTTED',source_hashes=hashes,plots=[dict(path=str(path),sha256=sha(path)),dict(path=str(paired),sha256=sha(paired))],neural_records_each=3080,body_records_each=15400,raw_eye_images_used=False,target_geometry_added=False,model_forward_calls=0,Core_constructions=0,physics_steps=0,parent_actual_review='PENDING'))
 print(str(path));print(str(paired))
if __name__=='__main__':main()
