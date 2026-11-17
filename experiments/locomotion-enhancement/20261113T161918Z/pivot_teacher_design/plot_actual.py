"""Plot already independently audited teacher trajectories; no physics/model imports."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,3,figsize=(14,10));sources={str(Path(__file__)):sha(__file__)}
 for i,name in enumerate(('stop','left_pivot','right_pivot')):
  d=ROOT/'trials'/name;r=json.loads((d/'RESULT.json').read_text());v=json.loads((d/'VERIFY.json').read_text());assert v['integrity_all_pass'] and all(v['checks'].values()) and sha(d/'RESULT.json')==v['result_sha256'];p=d/'trace_1ms.npz';g=d/'geometry_1ms.npz';assert sha(p)==r['trace']['sha256'] and sha(g)==r['geometry']['sha256']
  for f in (p,g,d/'RESULT.json',d/'VERIFY.json'):sources[str(f)]=sha(f)
  with np.load(p,allow_pickle=False) as z:times=z['time'].copy();com=z['com'].copy();loads=z['loads'].copy();nf=z['nonfoot'].copy()
  with np.load(g,allow_pickle=False) as z:gt=z['time'].copy();yaw=z['thorax_yaw'].copy()
  use=times>5.8+1e-8;gu=gt>5.8+1e-8;boundary=r['candidate_start'];length=r['body_length']['length_native'];t=times[use]-5.8;yd=np.degrees(np.unwrap(np.r_[boundary['yaw'],yaw[gu]])[1:]-boundary['yaw']);delta=com[use,:2]-np.asarray(boundary['com'])[:2]
  axes[i,0].plot(gt[gu]-5.8,yd,label='actual body yaw');axes[i,0].set_ylabel(name+'\nyaw (deg)');axes[i,0].axhline(0,color='k',lw=.5);axes[i,0].legend(fontsize=8)
  axes[i,1].plot(t,com[use,2],label='COM native');axes[i,1].axhline(1.2,color='orange',ls='--',label='COM gate1.2');axes[i,1].axhline(.8,color='red',ls=':',label='early-stop COM .8');axes[i,1].set_ylabel('COM/native length');axes[i,1].legend(fontsize=8)
  axes[i,2].plot(t,np.linalg.norm(delta,axis=1)/length,label='COM drift/body length');axes[i,2].plot(t,loads[use].sum(1)/r['weight_native'],label='foot load/bodyweight');axes[i,2].plot(t,nf[use]/r['weight_native'],label='nonfoot load/bodyweight');axes[i,2].legend(fontsize=8);axes[i,2].set_ylabel('normalized displacement/load')
  for ax in axes[i]:ax.set_xlabel('seconds after teacher switch');ax.grid(alpha=.2)
 fig.suptitle('FIRST STOP / PIVOT TEACHER CANDIDATE: ALL THREE EARLY-STOPPED\nActual saved full-body physics; engineering teacher only, no ANN/pivot success; no trajectory modification',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));out=ROOT/'ACTUAL_TEACHER_FAILURES.png';fig.savefig(out,dpi=140);plt.close(fig)
 for p,h in sources.items():assert sha(p)==h
 with (ROOT/'PLOT_VERIFICATION.json').open('x') as f:json.dump(dict(source_hashes=sources,image=str(out),image_sha256=sha(out),all_three_independent_audits_pass=True,actual_physics_not_modified=True,new_physics_steps=0,neural_forwards=0),f,indent=2);f.write('\n')
 print(out)
if __name__=='__main__':main()
