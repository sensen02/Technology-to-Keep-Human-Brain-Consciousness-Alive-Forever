"""Fresh-test sensory response chart, clearly separated from physical approach."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,digest,save_json,guard

def main():
 guard();r=json.loads((HERE/'EVALUATION/RESULT.json').read_text());meta=json.loads((HERE/'FRESH_TEST.json').read_text());conf=meta['episodes_config']
 with np.load(HERE/'EVALUATION/PREDICTIONS.npz',allow_pickle=False) as z:t=z['time_s'].copy();x=z['receptors'].copy();y=z['teacher_turn'].copy();pred=z['learned_turn'].copy();swap=z['swapped_turn'].copy()
 def ix(a,s=1.25,o=.30,seed=20261124):return next(i for i,c in enumerate(conf) if c['angle']==a and c['amplitude']==s and c['onset_s']==o and c['seed']==seed)
 plus,minus,late=ix(.10),ix(-.10),ix(.10,o=.60);zero=next(i for i,c in enumerate(conf) if c['no_food']);center=ix(0.)
 fig,axs=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
 tag='ALL FOUR SENSORY TARGETS PASS' if r['all_predeclared_targets_pass'] else 'SENSORY TARGETS FAIL — SEE INDIVIDUAL GATES'
 fig.suptitle('SENSOR RESPONSE ONLY — NO PHYSICAL APPROACH\nBalanced correction final3000 / fresh unseen angle-dose-onset-noise test / '+tag,fontsize=15)
 ax=axs[0,0];g=meta['geometry'];p=np.asarray(g['sensor_xy']);mid=np.asarray(g['sampling_midpoint_xy']);ax.scatter(p[:,0],p[:,1],s=70,c=['tab:blue','tab:orange']);ax.scatter(*mid,marker='x',c='black')
 for a in [-.10,-.05,.05,.10]:
  c=np.asarray(conf[ix(a)]['source_center_xy']);ax.scatter(*c,marker='*',s=130);ax.plot([mid[0],c[0]],[mid[1],c[1]],c='gray',alpha=.4);ax.annotate(f'{a:+.2f}',c,xytext=(5,4),textcoords='offset points')
 ax.set_xlim(mid[0]-.1,mid[0]+.8);ax.set_ylim(p[:,1].min()-.1,p[:,1].max()+.1);ax.set_aspect('equal');ax.set_title('Static audited sensor proxies / fresh sources');ax.set_xlabel('Model X');ax.set_ylabel('Model Y');ax.grid(alpha=.2)
 ax=axs[0,1];ax.plot(t,x[plus,:,0],label='Left receptor');ax.plot(t,x[plus,:,1],label='Right receptor');ax.axvline(.30,c='gray',ls=':');ax.set_title('Only two local receptor values enter network');ax.set_xlabel('Sensory time s');ax.legend();ax.grid(alpha=.2)
 ax=axs[0,2]
 for i,label,c in [(plus,'+.10 rad','tab:blue'),(minus,'−.10 rad','tab:orange')]:ax.plot(t,pred[i],c=c,label='Neural '+label);ax.plot(t,y[i],c=c,ls='--',label='Target '+label)
 ax.axhline(0,c='black',lw=.5);ax.set_title('Heldout neural direction / declared target');ax.set_xlabel('Sensory time s');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axs[1,0]
 for i,cfg in enumerate(conf):
  if cfg['no_food']:ax.plot(t,pred[i],alpha=.8,label=f'Noise seed {cfg["seed"]}, onset metadata {cfg["onset_s"]}' )
 ax.plot(t,np.zeros_like(t),'k--',label='Target0');ax.axhline(.02,c='red',ls=':');ax.axhline(-.02,c='red',ls=':');ax.set_title(f'No-food RMS {r["metrics"]["no_food_RMS"]:.5f} (gate .02)');ax.set_xlabel('Sensory time s');ax.legend(fontsize=6);ax.grid(alpha=.2)
 ax=axs[1,1];ax.plot(t,pred[plus],label='Normal +.10');ax.plot(t,swap[plus],label='Swapped local inputs');ax.plot(t,pred[late],ls=':',label='Fresh later onset .60');ax.plot(t,pred[center],ls='--',label='Centered odor target0');ax.set_title('Swap, unseen onset and center controls');ax.set_xlabel('Sensory time s');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axs[1,2];names=['All-test MSE','Food-only MSE','No-food RMS'];vals=[r['metrics']['fresh_all_MSE'],r['metrics']['fresh_food_only_MSE'],r['metrics']['no_food_RMS']];ax.bar(names,vals,color=['tab:blue','tab:green','tab:orange']);ax.axhline(.02,c='red',ls='--',label='All three must <=.02');ax.set_ylim(0,max(.023,max(vals)*1.25));ax.set_title(f'Sign fraction {r["metrics"]["qualified_sign_fraction"]:.5f} (gate >=.90)');ax.legend(fontsize=8)
 for j,v in enumerate(vals):ax.text(j,v,f'{v:.6f}',ha='center',va='bottom',fontsize=9)
 fig.text(.5,-.015,'44 pre-frozen fresh test episodes | old test is development history | deployment_qualified=false | no motor model / zero Core or physical steps',ha='center',fontsize=10)
 path=HERE/'BALANCED_SENSOR_DIAGNOSTIC.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
 dest=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮4-平衡气味神经反应-纯感觉新测试.png')
 if dest.exists():raise FileExistsError(dest)
 shutil.copy2(path,dest)
 if digest(path)!=digest(dest):raise ValueError('desktop mismatch')
 save_json(HERE/'DELIVERY.json',dict(project_image=str(path),desktop_image=str(dest),sha256=digest(path),physical_steps=0,source_hashes={str(HERE/n):digest(HERE/n) for n in ('plot_results.py','EVALUATION/RESULT.json','EVALUATION/PREDICTIONS.npz','FRESH_TEST.json')},visual_review='pending actual read_image'))
 guard();print(json.dumps(dict(image=str(path),desktop=str(dest),sha256=digest(path))))
if __name__=='__main__':main()
