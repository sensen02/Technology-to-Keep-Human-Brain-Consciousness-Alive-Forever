"""Publication-independent sensory diagnostic artifact, no physical images or claims."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,digest,save_json,guard


def main():
 guard();r=json.loads((HERE/'EVALUATION/RESULT.json').read_text());meta=json.loads((HERE/'DATASET.json').read_text())
 with np.load(HERE/'EVALUATION/PREDICTIONS.npz',allow_pickle=False) as z:time=z['time_s'].copy();x=z['receptors'].copy();pred=z['learned_turn'].copy();target=z['teacher_turn'].copy();swap=z['swapped_turn'].copy()
 rows=r['heldout_episodes'];configs=meta['splits']['heldout']['episodes_config']
 def index(angle,strength=1.5,onset=.35,seed=20261120):return next(i for i,c in enumerate(configs) if c['angle']==angle and c['amplitude']==strength and c['onset_s']==onset and c['seed']==seed)
 plus=index(.12);minus=index(-.12);zero=next(i for i,c in enumerate(configs) if c['no_food']);late=index(.12,onset=.55)
 fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
 fig.suptitle('SENSOR RESPONSE ONLY — NO PHYSICAL APPROACH\nFixed1000 sensory neural pretraining: direction PASS; heldout MSE / no-food RMS FAIL',fontsize=17)
 ax=axes[0,0];g=meta['geometry'];sensors=np.asarray(g['sensor_xy']);mid=np.asarray(g['sampling_midpoint_xy']);ax.scatter(sensors[:,0],sensors[:,1],c=['tab:blue','tab:orange'],s=80,label='Audited static head sensor proxies');ax.scatter(*mid,c='black',marker='x')
 for angle in [-.12,-.06,.06,.12]:
  i=index(angle);center=np.asarray(configs[i]['source_center_xy']);ax.scatter(*center,marker='*',s=160);ax.annotate(f'{angle:+.2f} rad',center,xytext=(4,4),textcoords='offset points');ax.plot([mid[0],center[0]],[mid[1],center[1]],color='gray',alpha=.4)
 ax.set_aspect('equal');ax.set_title('Heldout source layout; static geometry only');ax.set_xlabel('Model X');ax.set_ylabel('Model Y');ax.grid(alpha=.2)
 ax=axes[0,1];ax.plot(time,x[plus,:,0],label='Left receptor');ax.plot(time,x[plus,:,1],label='Right receptor');ax.axvline(.35,color='gray',ls=':',label='Environment onset');ax.set_title('Causal local receptor inputs; +0.12 rad');ax.set_xlabel('Sensory time s');ax.legend();ax.grid(alpha=.2)
 ax=axes[0,2]
 for i,label,color in [(plus,'+0.12 rad','tab:blue'),(minus,'−0.12 rad','tab:orange')]:
  ax.plot(time,pred[i],color=color,label='Neural '+label);ax.plot(time,target[i],color=color,ls='--',label='Engineering target '+label)
 ax.axhline(0,color='black',lw=.6);ax.set_title('Learned turn vs target on heldout angles');ax.set_xlabel('Sensory time s');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axes[1,0];ax.plot(time,pred[zero],label='Neural no-food control');ax.plot(time,target[zero],'k--',label='Target 0');ax.axhline(.02,color='red',ls=':',label='RMS gate is whole control, not this line');ax.set_title(f'No-food drift: RMS={r["metrics"]["zerofood_RMS"]:.4f} > .02 FAIL');ax.set_xlabel('Sensory time s');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axes[1,1];ax.plot(time,pred[plus],label='Normal inputs (+.12)');ax.plot(time,swap[plus],label='Swapped local inputs');ax.plot(time,target[plus],'k--',label='Original target');ax.plot(time,pred[late],ls=':',label='Heldout later onset .55s');ax.set_title('Sensory dependence and unseen onset');ax.set_xlabel('Sensory time s');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axes[1,2]
 for strength,marker in [(.75,'o'),(1.5,'s')]:
  selected=[row for row in rows if row['amplitude']==strength and row['onset_s']==.35 and row['seed']==20261120]
  selected.sort(key=lambda row:row['angle']);a=np.asarray([row['angle'] for row in selected]);b=[row['mean_turn_last200ms'] for row in selected];c=[row['label_mean_last200ms'] for row in selected]
  ax.plot(a,b,marker=marker,label=f'Neural amplitude {strength}');ax.plot(a,c,'--',alpha=.6,label=f'Target amplitude {strength}')
 ax.axhline(0,color='black',lw=.6);ax.set_xlabel('Heldout bearing rad — teacher/plot only');ax.set_ylabel('Last 200ms mean turn');ax.set_title('Dose and mirrored angle response');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.text(.5,-.018,f'44 train / 36 heldout synthetic static-head episodes | sign {r["metrics"]["qualified_sign_fraction"]:.4f} >= .90 PASS | MSE {r["metrics"]["heldout_MSE"]:.6f} > .02 FAIL | zero physical steps / no motor model',ha='center',fontsize=11)
 path=HERE/'SENSORY_PRETRAIN_DIAGNOSTIC.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
 dest=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮4-气味神经反应预训练-非物理趋近.png');dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 shutil.copy2(path,dest)
 if digest(path)!=digest(dest):raise ValueError('desktop image mismatch')
 save_json(HERE/'DELIVERY.json',dict(project_image=str(path),desktop_image=str(dest),sha256=digest(path),physical_steps=0,source_hashes={str(HERE/n):digest(HERE/n) for n in ('plot_results.py','EVALUATION/RESULT.json','EVALUATION/PREDICTIONS.npz','DATASET.json')},visual_review='pending actual read_image by agent'))
 guard();print(json.dumps(dict(image=str(path),desktop=str(dest),sha256=digest(path))))
if __name__=='__main__':main()
