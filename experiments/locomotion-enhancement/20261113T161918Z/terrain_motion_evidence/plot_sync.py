"""Actualthree terrain baseline comparisons and realregisteredcontactcoverage; no neuralexecution."""
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
 cases={x:load(x) for x in CASES};ledger={}
 for c in cases.values():ledger.update(c['source_hashes'])
 out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);fig,axes=plt.subplots(3,4,figsize=(18,12),sharex=True)
 for name,ax in zip(CASES,axes):
  c=cases[name];r=c['result'];t=c['trace'];tt=c['terrain'];e=t['time']-5.8;ax[0].plot(e,t['com'][:,0],label='Actual COM worldX native');ax[0].plot(e,t['com'][:,1],label='Actual COM worldY native');ax[0].set_ylabel(name,fontsize=8);ax[1].plot(e,tt['loads_normal'].sum(1)/r['terrain_summary']['weight_native'],label='Normal loads /weight');ax[1].plot(e,tt['com_relative_height'],label='COM above terrain ray native');ax[2].plot(e,tt['pad_ground_height'].max((1,2)),label='MAX actual padgroundheight');ax[2].plot(e,tt['registered_contact_count'],label='Actual1ms contacts (solverrecords)',alpha=.6)
  d=c['decisions'];ax[3].plot(d['time']-5.8,d['raw_turn'],label='Decoded actualneural turn');ax[3].plot(d['time']-5.8,d['visual_raw_turn'],label='ActualterrainRGB visual u');ax[3].plot(d['time']-5.8,d['odor_raw_turn'],label='NaturalodorNN u');ax[3].set_title(f'Original slip {r["autonomous"]["slip_ratio_max"]:.6f}; rough '+('PASS' if r['terrain_summary']['rough_terrain_pass'] else 'FAIL'),fontsize=9)
  for item in ax:item.axvline(0,color='black',lw=.7);item.axvspan(-5.8,0,color='gray',alpha=.06);item.grid(alpha=.2);item.legend(fontsize=7);item.set_xlabel('ActualANN elapsed s (global -5.8)')
 fig.suptitle('ACTUAL DERIVED TERRAIN XML + UNCHANGED336-SOURCE NEURAL CONTROLLER\nTwo archivedcontrols and one repairedstart-area roughtrial, no repeatedcontrolphysics or newmotorclaim.',fontsize=13);fig.tight_layout(rect=(0,.03,1,.93));fig.text(.02,.008,'Terrain heights .02/.04 native, not exaggerated. Onlyblock00/01 contacted, notwholecoursecompletion. Warmguard failure and roughslipFAIL retained.',fontsize=10);path=out/'ACTUAL_THREE_TERRAIN_NEURAL_BASELINES.png';fig.savefig(path,dpi=150);plt.close(fig);unchanged(ledger);fresh(out/'PLOT_REVIEW.json',dict(kind='ACTUAL_TERRAIN_NEURAL_BASELINE_COMPARISON',source_hashes=ledger,image=str(path),image_sha256=sha(path),new_physics_steps=0,NN_forward_calls=0,no_wholecourseclaim=True,rough_slip_fail_retained=True,parent_actual_review='PENDING'));print(str(out/'PLOT_REVIEW.json'))
if __name__=='__main__':main()
