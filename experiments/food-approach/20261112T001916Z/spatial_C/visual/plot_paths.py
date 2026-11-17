"""Spatial C saved path/distance/receptor development figure. Default CODE_ONLY."""
import argparse,os
from contracts_visual import HERE,IDS,arrays,summary,require,sha,save,unchanged

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY spatial figure not executed');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Circle
 allrows={name:summary(arrays(name)) for name in IDS};baseline=allrows['C_noodor'];bp=baseline['geometry']['sensor_midpoint_xyz'][5799:,:2];hashes={str(HERE/'plot_paths.py'):sha(HERE/'plot_paths.py')};rows=[]
 out=HERE/'C6_PATHS';out.mkdir(exist_ok=False);fig,axes=plt.subplots(6,3,figsize=(18,24))
 for i,name in enumerate(IDS):
  data=allrows[name];g=data['geometry'];d=data['decisions'];r=data['result'];m=r['spatial_metrics'];hashes.update(data['hashes']);t=g['time'][5799:]-5.8;xy=g['sensor_midpoint_xyz'][5799:,:2];center=data['source'];radius=data['radius'];bd=np.linalg.norm(bp-center,axis=1)
  require(np.array_equal(g['time'],baseline['geometry']['time']),'newCnoodor timestamps differ')
  left,mid,right=axes[i];left.plot(xy[:,0],xy[:,1],label='actual head-proxy path',color='#2563eb',lw=1.6);left.plot(bp[:,0],bp[:,1],label='new C_noodor path',color='#737373',ls='--',lw=1.)
  left.add_patch(Circle(center,radius,fill=False,color='#16a34a',lw=1.4));left.scatter([center[0]],[center[1]],marker='+',color='#16a34a',label='fixed source / patch (no-odor: amplitude0)');left.scatter(xy[[0,-1],0],xy[[0,-1],1],s=22,color=['#000000','#dc2626'],label='start / end')
  left.set_aspect('equal',adjustable='datalim');left.set(xlabel='world X (native)',ylabel='world Y (native)');left.grid(alpha=.2)
  mid.plot(t,g['center_distance'][5799:],label='center distance',color='#2563eb');mid.plot(t,g['edge_distance'][5799:],label='edge distance',color='#16a34a',lw=1.);mid.plot(t,bd,label='matched no-odor center distance',color='#737373',ls='--');mid.axhline(radius,color='#16a34a',ls=':',label='patch radius')
  inside=g['inside_patch'][5799:].astype(bool);mid.fill_between(t,0,1,where=inside,transform=mid.get_xaxis_transform(),color='#16a34a',alpha=.09)
  changes=np.diff(np.r_[False,inside].astype(int));entries=t[np.flatnonzero(changes==1)];exits=t[np.flatnonzero(changes==-1)]
  for eventtime in entries:mid.axvline(eventtime,color='#16a34a',lw=.6,alpha=.7)
  for eventtime in exits:mid.axvline(eventtime,color='#dc2626',lw=.6,alpha=.7)
  if m['confirmed_elapsed_s'] is not None:mid.axvline(m['confirmed_elapsed_s'],color='#9333ea',ls='--',lw=1.,label='endpoint proxy confirmed')
  mid.set(xlabel='ANN elapsed (s)',ylabel='distance (native)');mid.grid(alpha=.2)
  q=d['time']-5.8;right.plot(q,d['receptors'][:,0],label='causal z left',color='#2563eb',lw=1.);right.plot(q,d['receptors'][:,1],label='causal z right',color='#dc2626',lw=1.)
  turnaxis=right.twinx();turnaxis.plot(q,d['raw_turn'],label='raw behavior turn',color='#9333ea',alpha=.8,lw=1.);turnaxis.plot(q,d['command'][:,1],label='applied turn',color='#ea580c',ls=':',lw=.8);turnaxis.set_ylim(-1.05,1.05);turnaxis.set_ylabel('neural turn')
  right.axvspan(-5.8,0,color='#64748b',alpha=.06);right.axvline(0,color='black',ls='--',lw=.7);right.set(xlabel='ANN elapsed (PRE-ANN <0)',ylabel='receptor response');right.grid(alpha=.2)
  for ax in (mid,right):ax.axvline(.6,color='#a16207',ls=':',lw=.8);ax.axvline(.8,color='#a16207',ls=':',lw=.8)
  row=data['summary_row'];left.set_title(name+' | DEVELOPMENT',fontsize=10);mid.set_title('Endpoint arrival '+str(m['arrived_endpoint_proxy'])+' | pass-through '+str(m['overshoot_or_pass_through']),fontsize=10);right.set_title('Stable '+str(row['stable_body'])+' | Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+f' | slip {r["autonomous"]["slip_ratio_max"]:.4f}',fontsize=9)
  if i==0:
   left.legend(fontsize=7,loc='best');mid.legend(fontsize=7,loc='best');handles,labels=right.get_legend_handles_labels();h2,l2=turnaxis.get_legend_handles_labels();right.legend(handles+h2,labels+l2,fontsize=7,loc='best')
  rows.append(dict(condition=name,spatial_metrics=m,summary_metrics=row['development_metrics'],actual_entry_endpoints_s=entries.tolist(),actual_exit_endpoints_s=exits.tolist(),source_center_xy=center.tolist(),radius=radius,engineering_pass=r['engineering_pass'],stable_body=row['stable_body'],matched_noodor_relative_to_current_source=True))
 fig.suptitle('SPATIAL_GAUSSIAN / C6 DEVELOPMENT — fixed world sources, actual head-proxy feedback\nEndpoint entry/dwell/pass-through shown separately; no parking/ingestion or D-heldout qualification; strict engineering FAIL retained',fontsize=14)
 fig.tight_layout(rect=(0,.01,1,.965));image=out/'SPATIAL_C6_PATHS_DISTANCES.png';fig.savefig(image,dpi=150);plt.close(fig);unchanged(hashes)
 save(out/'PLOT.json',dict(status='SAVED_SPATIAL_C6_FIGURE_REVIEW_PENDING',plot=str(image),plot_sha256=sha(image),source_hashes=hashes,rows=rows,baseline='new C_noodor actual path',source_fixed=True,model_forward_calls=0,Core_constructions=0,physics_steps=0,parking=False,ingestion=False,continuous_inside_proven=False,visual_review='PENDING_PARENT_OPENING'))
 print(str(out/'PLOT.json'))
if __name__=='__main__':main()
