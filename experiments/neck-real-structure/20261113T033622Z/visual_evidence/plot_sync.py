"""Actual saved ten-case visual/sourceport/body timing and paired intervention figures."""
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
 cases={name:load(name) for name in IDS};empty=cases['V_empty'];ledger={}
 for c in cases.values():ledger.update(c['source_hashes'])
 out=HERE/'SYNCHRONIZED_EVIDENCE';out.mkdir(exist_ok=False);fig,axes=plt.subplots(10,4,figsize=(20,30),sharex=True)
 for name,ax in zip(IDS,axes):
  c=cases[name];d=c['decisions'];e=d['time']-5.8;g=c['geometry'];yawtime=g['time']-5.8;r=c['result'];ax[0].plot(e,d['visual_raw_turn'],label='frozen actual RGB visual raw',lw=1);ax[0].plot(e,d['raw_turn'],label='DN-delayed neural decoded turn',lw=1);ax[0].set_ylabel(name,fontsize=9);ax[0].set_ylim(-1.1,1.1)
  ax[1].plot(e,d['AN_held'][:,0],label='receiver AN1');ax[1].plot(e,d['DN_held'][:,0],label='receiver DNp09[0]');ax[1].plot(e,d['DN_held'][:,1],label='receiver DNp09[1]');ax[1].set_ylim(-1.1,1.1)
  ax[2].plot(yawtime,np.degrees(c['yaw']),label='actual body yaw');ax[2].plot(yawtime,np.degrees(empty['yaw']),'--',label='matched V_empty');ax[2].plot(yawtime,np.degrees(c['yaw']-empty['yaw']),':',label='minus V_empty');ax[2].set_title('Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+f' slip {r["autonomous"]["slip_ratio_max"]:.6f} | stable={r["functional_stable_body"]}',fontsize=8)
  valid=d['image_valid'];ax[3].plot(e[valid],d['raw_released_image_capture_id'][valid],label='raw released capture ID');ax[3].plot(e[valid],d['image_capture_id'][valid],'--',label='actor used capture ID');ax[3].set_title('20ms exposure/+20ms release;5ms decision',fontsize=8)
  for item in ax:item.axvline(0,color='black',lw=.7);item.grid(alpha=.2);item.axvspan(-5.8,0,color='gray',alpha=.07)
  if name in ('V_left_DN_cut','V_left_AN_cut'):
   for item in ax:item.axvline(1.6,color='red',ls='--',lw=.8)
  if name=='V_flip_fixedframe':
   for item in ax:item.axvline(4.8,color='purple',ls='--',lw=.8)
 for item in axes[0]:item.legend(fontsize=6)
 for item in axes[-1]:item.set_xlabel('Actual ANN elapsed seconds (global time - 5.8)')
 fig.suptitle('ACTUAL RGB -> FROZEN VISION32 -> ARTIFICIAL BRAIN16 -> REAL-ID NECK PORTS -> OTHER MOTOR256 -> FULL BODY\n10 distinct audited physical trials;304 memories;CNNcache16/AN1/DN2 separate. Original slips and visual failures retained.',fontsize=13);fig.tight_layout(rect=(0,.025,1,.965));fig.text(.02,.008,'Port activity/delay are engineering approximations over real anatomical identities, not native membrane/retina/electrode validation. Low-level walking belongs OTHER.',fontsize=10);p0=out/'TEN_CASES_ACTUAL_RGB_NECK_BODY_SYNC.png';fig.savefig(p0,dpi=150);plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(15,12))
 normal=cases['V_left_pulse'];nd=normal['decisions'];ne=nd['time']-5.8
 for row,name in enumerate(('V_left_DN_cut','V_left_AN_cut')):
  c=cases[name];d=c['decisions'];axes[row,0].plot(ne,nd['raw_turn'],label='normal decoded turn');axes[row,0].plot(ne,d['raw_turn'],label=name+' decoded turn');axes[row,0].plot(ne,d['visual_raw_turn'],':',label='actual visual input continues');axes[row,1].plot(c['geometry']['time']-5.8,np.degrees(c['yaw']-normal['yaw']),label='actual cut minus normal yaw')
  for item in axes[row]:item.axvline(1.6,color='red',ls='--',label='engineering cut1.6s')
 normal=cases['V_left_to_right'];fixed=cases['V_flip_fixedframe'];e=normal['decisions']['time']-5.8;axes[2,0].plot(e,normal['decisions']['raw_turn'],label='normal switch decoded turn');axes[2,0].plot(e,fixed['decisions']['raw_turn'],label='fixed actor frame decoded turn');axes[2,0].plot(e,fixed['decisions']['visual_raw_turn'],':',label='fixedframe frozenvision output');axes[2,1].plot(normal['geometry']['time']-5.8,np.degrees(normal['yaw']-fixed['yaw']),label='normal minus fixedimage actual yaw')
 for item in axes[2]:item.axvline(4.82,color='purple',ls='--',label='new image released4.82s')
 for item in axes.flat:item.grid(alpha=.2);item.legend(fontsize=8);item.set_xlabel('Actual ANN elapsed s')
 fig.suptitle('MATCHED ACTUAL SOURCE-CUT AND FROZEN-IMAGE INTERVENTIONS\nPrefixes and actual source timings verified; effect size descriptive, no new acceptance threshold.',fontsize=13);fig.tight_layout(rect=(0,0,1,.94));p1=out/'SOURCE_CUT_AND_FIXED_IMAGE_ACTUAL_COMPARISONS.png';fig.savefig(p1,dpi=150);plt.close(fig)
 unchanged(ledger);fresh(out/'PLOT_REVIEW.json',dict(status='ACTUAL_TENCASE_RGB_NEURAL_PORT_BODY_PLOTS',source_hashes=ledger,images=[dict(path=str(p),sha256=sha(p)) for p in (p0,p1)],new_physics_steps=0,NN_forward_calls=0,all10_prior_integrity_required=True,parent_actual_review='PENDING'));print(str(out/'PLOT_REVIEW.json'))
if __name__=='__main__':main()
