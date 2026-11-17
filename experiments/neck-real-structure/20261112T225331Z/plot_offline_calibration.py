"""Saved offline calibration visual evidence; no network, training or physical integration."""
from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rp=ROOT/'neural_calibration/EVALUATION/RESULT.json';vp=rp.with_name('VERIFY.json');r=json.loads(rp.read_text());v=json.loads(vp.read_text());assert v['integrity_all_pass'] and v['result_sha256']==sha(rp);fig,axes=plt.subplots(6,3,figsize=(16,17),sharex=True);hashes={str(rp):sha(rp),str(vp):sha(vp),str(Path(__file__)):sha(__file__)}
 for row,ax in zip(r['rows'],axes):
  p=Path(row['artifact']);assert sha(p)==r['output_hashes'][str(p)];hashes[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:turn=z['turn'][:,0,0];target=z['teacher'][:,0,0];cue=z['cue'][:,0,0];dn=z['DN'][:,0];an=z['AN'][:,0,0];h=z['brain_hidden'][:,0]
  t=np.arange(1920)*.005;ax[0].plot(t,target,label='Archived teacher turn',lw=1.3);ax[0].plot(t,turn,label='ANN brain -> DN delay -> neural decoder',lw=1.1);ax[0].plot(t,cue,':',label='Explicit instruction (brain side)',alpha=.6);ax[0].set_title(row['condition']+' | '+row['split']+' | RMSE '+format(row['metrics']['turn_RMSE'],'.4f'));ax[0].set_ylim(-1.1,1.1)
  ax[1].plot(t,dn[:,0],label='DNp09 source1');ax[1].plot(t,dn[:,1],label='DNp09 source2');ax[1].plot(t,an,':',label='Engineered AN encoding');ax[1].set_ylim(-1.1,1.1);ax[1].set_title('Source-ordered activity; dimensionless engineering units')
  ax[2].plot(t,np.linalg.norm(h,axis=1),label='Artificial brain16 state norm');ax[2].set_title('Neural memory, continuous through instruction changes')
  for a in ax:a.grid(alpha=.2)
 for a in axes[0]:a.legend(fontsize=7)
 for a in axes[-1]:a.set_xlabel('Sequence time (s), 1920 x 5ms')
 fig.suptitle('ACTUAL FIXED200 OFFLINE ARTIFICIAL-BRAIN CALIBRATION\nSource-grounded neck IDs; engineered activity/delay/adapters. Archived body inputs, NOT new walking or biological tuning.',fontsize=14);fig.tight_layout(rect=(0,0,1,.95));out=ROOT/'OFFLINE_ARTIFICIAL_BRAIN_CALIBRATION.png';fig.savefig(out,dpi=140);plt.close(fig)
 for p,digest in hashes.items():assert sha(p)==digest
 with (ROOT/'OFFLINE_PLOT_REVIEW.json').open('x') as f:json.dump(dict(image=str(out),image_sha256=sha(out),source_hashes=hashes,actual_saved_neural_outputs=True,training_updates=0,new_physics_steps=0,parent_actual_view='PENDING'),f,indent=2);f.write('\n')
 print(str(out))
if __name__=='__main__':main()
