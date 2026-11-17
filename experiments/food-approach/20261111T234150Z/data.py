"""Long causal static receptor sequences; environment targets never policy inputs."""
from common import *
MOTIFS=('noodor','equal','left_pulse','right_pulse','left_to_right','right_to_left')
def smooth(x):
 x=np.clip(x,0.,1.);return x*x*(3.-2*x)
def levels(amplitude):
 a=read(GEOMETRY);pos=np.asarray(a['head']['xpos']);rot=np.asarray(a['head']['xmat']);local=np.asarray(a['sensor_offsets']['local_xyz']);points=(pos+local@rot.T)[:,:2];mid=points.mean(0);ex=rot[:2,0].copy();ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0]]);center=mid+.6*ex+.6*np.tan(.12)*ey
 return amplitude*np.exp(-.5*np.sum(((points-center)/.5)**2,axis=1))
def episode(motif,cfg):
 t=np.arange(1920)*.005;left=levels(cfg['amplitude']);right=left[::-1].copy();rise=smooth((t-cfg['on_s'])/.2);decline=1-smooth((t-cfg['off_s'])/.2);flip=smooth((t-cfg['switch_s'])/.2)
 if motif=='noodor':C=np.zeros((1920,2));factor=np.zeros(1920)
 elif motif=='equal':C=np.repeat(np.full((1,2),left.mean()),1920,axis=0)*(rise*decline)[:,None];factor=np.zeros(1920)
 elif motif in ('left_pulse','right_pulse'):
  sign=1. if motif=='left_pulse' else -1.;C=(left if sign>0 else right)[None]*(rise*decline)[:,None];factor=.8*sign*rise*decline
 else:
  sign=1. if motif=='left_to_right' else -1.;pair0=left if sign>0 else right;pair1=right if sign>0 else left;C=((1-flip[:,None])*pair0[None]+flip[:,None]*pair1[None])*rise[:,None];factor=.8*sign*(1-2*flip)*rise
 state=np.zeros(2);pending=np.zeros(2);x=[];y=[];alpha=-np.expm1(-.005/.05)
 for k in range(1920):
  current=pending.copy();x.append(current);y.append(factor[k]*np.clip(current.mean()/.15,0,1));state=state+alpha*(C[k]/(1+C[k])-state);pending=state.copy()
 target=np.asarray(y);neutral=(np.abs(target)<.05).astype(float);on=1-neutral
 weights=np.ones(1920)/1920
 if neutral.sum()>0 and on.sum()>0:weights=.5*neutral/neutral.sum()+.5*on/on.sum()
 return np.asarray(x),target,weights,C,dict(motif=motif,**cfg,noise_std=0.,seed_note='Recorded design identity; deterministic zero-noise transduction consumes no RNG')
def split(which):
 pr=read(HERE/'PREREG.json');rows=[episode(m,c) for m in MOTIFS for c in pr['data'][which+'_variants']]
 return dict(receptors=np.stack([r[0] for r in rows],axis=1),targets=np.stack([r[1] for r in rows],axis=1),weights=np.stack([r[2] for r in rows],axis=1),concentrations=np.stack([r[3] for r in rows],axis=1),time=np.arange(1920)*.005),[r[4] for r in rows]
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def training_data():
 a=arrays(HERE/'train.npz');return {k:torch.from_numpy(v) for k,v in a.items() if k in ('receptors','targets','weights')}
