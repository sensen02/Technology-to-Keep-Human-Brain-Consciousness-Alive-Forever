from pathlib import Path
import sys,json,numpy as np,torch
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'20261110T235830Z/train_two_only_control'
sys.path.insert(0,str(OLD))
from model_io import load_checked,PARENT,sha256
path=OLD/'TRAIN/checkpoint_001500.pt'
model,p=load_checked(path,sha256(path));torch.set_num_threads(1)
with np.load(HERE/'right_yaw_m060/SAMPLES_5MS.npz',allow_pickle=False) as z:x=torch.tensor(z['X238'],dtype=torch.float64);y=z['Y42'].copy();u=torch.tensor(z['command2'],dtype=torch.float64)
with torch.no_grad():
 state=model.base.initial_state(1);base=[]
 for row in x:
  mean,state=model.base.step(row[None],state);base.append(mean[0].numpy().copy())
 base=np.stack(base)
target=np.arctanh(np.clip(y,-1+1e-12,1-1e-12));required=(target-base)[960:];a=y[960:]
r=dict(teacher_action_abs_max=float(np.abs(a).max()),action_clipped_boundary_count=int(np.sum(np.abs(a)>=1)),required_latent_abs_max=float(np.abs(required).max()),required_latent_abs_quantiles=np.quantile(np.abs(required),[.5,.9,.95,.99,1]).tolist(),old_max_correction=float(.5*np.tanh(1)),fraction_exceeds_old_bound=float(np.mean(np.abs(required)>.5*np.tanh(1))),old_bound_action_MSE_floor=float(np.mean((np.tanh(base[960:]+np.clip(required,-.5*np.tanh(1),.5*np.tanh(1)))-a)**2)),scope='same teacher states; no physical or training; clipped atanh endpoints not exact latent requirement')
(HERE/'RIGHT_CAPACITY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
