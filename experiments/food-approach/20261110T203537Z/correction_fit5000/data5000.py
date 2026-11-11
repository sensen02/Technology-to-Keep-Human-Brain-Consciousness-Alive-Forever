"""Explicit unsafe-state/valid-label corrective data, never safe-demo admission."""
from pathlib import Path
import sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import sha256,verify_hashes
from fast_data import prepare as original_prepare,cached,neutral,metrics
PARENT_SHA='17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8'


def prepare(model):
 episodes,hashes=original_prepare(model);admissions=[]
 manifest=ROOT/'recovery_data/ARTIFACT_SHA256.json'
 if sha256(manifest)!='46e44c767cf9a3ad8d00a340072921307efe815940063dfcce9893776e95f1a2':raise ValueError('final corrective evidence manifest drift')
 hashes[str(manifest)]=sha256(manifest)
 for direction in ('left','right'):
  folder=ROOT/'recovery_data'/('COLLECT_'+direction)
  rp=folder/'RESULT.json';ap=folder/'VERIFIED_SAMPLES.json';lp=folder/'LABEL_REPLAY.json';sp=folder/'SAMPLES_5MS.npz';tp=folder/'trace_1ms.npz';ip=folder/'IDENTITY.json'
  r=json.loads(rp.read_text());a=json.loads(ap.read_text());label=json.loads(lp.read_text());identity=json.loads(ip.read_text())
  if not(a['integrity_all_pass'] is True and all(a['checks'].values()) and a['dataset_safety_admitted'] is False and r['dataset_safety_admitted'] is False):raise ValueError('requires verified explicitly FAILED observation trajectory '+direction)
  if not(r['complete'] is True and r['early_stop'] is None and r['physical_steps']==848000 and r['checkpoint_sha256']==PARENT_SHA and a['checkpoint_sha256']==PARENT_SHA and Path(a['run']).resolve()==folder):raise ValueError('incomplete/wrong corrective collection')
  c=r['collection']
  if not(c['sample_count']==4800 and c['nonfoot_force_max_native']==0 and c['foot_support_mean']>=.9 and c['com_height_min']>=1.2 and c['slip_ratio_max']>.5 and c['gates']['slip'] is False):raise ValueError('documented failure/safety envelope changed')
  for path,key in ((rp,'result_sha256'),(sp,'samples_sha256'),(tp,'trace_sha256')):
   if sha256(path)!=a[key]:raise ValueError('independent artifact digest mismatch '+str(path))
  if not(label['condition']==direction and label['all_teacher_labels_bitexact'] is True and label['max_label_error']==0 and label['root_balance_residuals_bitexact'] is True and label['root_balance_max']<=.02 and label['root_balance_above_point02_count']==0 and label['label_rows']==1920 and label['physical_integration_steps']==0):raise ValueError('teacher label replay not valid')
  verify_hashes(label['source_hashes'])
  # Captured identity independently binds all exact physical sources and assets.
  sys.path.insert(0,str(ROOT.parent.parent/'banc-walking-recovery/20261110T152714Z'))
  from identity import verify
  verify(identity)
  for path in (rp,ap,lp,sp,tp,ip):hashes[str(path)]=sha256(path)
  hashes.update(label['source_hashes'])
  with np.load(sp,allow_pickle=False) as z:
   x=torch.tensor(z['X238'].copy(),dtype=torch.float64);u=torch.tensor(z['command2'].copy(),dtype=torch.float64);y=torch.tensor(z['teacher_action'].copy(),dtype=torch.float64)
   sources=z['source'].copy();actual_time=z['time'].copy();residual=z['root_balance_residual'].copy();before=z['hidden_before'].copy();after=z['hidden_after'].copy()
  if not(x.shape==(1920,238) and u.shape==(1920,2) and y.shape==(1920,42) and sources.shape==(1920,) and np.all(sources[:960]==2) and np.sum(sources[960:]==3)==120 and np.sum(sources[960:]==5)==840 and torch.equal(u[:960],torch.zeros_like(u[:960]))):raise ValueError('actual corrective row/source contract')
  if not(np.max(abs(actual_time-(1+np.arange(1920)*.005)))<1e-8 and np.max(residual)<=.02 and torch.isfinite(x).all() and torch.isfinite(u).all() and torch.isfinite(y).all() and y.abs().max()<=1):raise ValueError('corrective finite/timing/label bounds')
  state=model.initial_state(1);means=[];states=[]
  with torch.no_grad():
   for row in x:
    mean,state=model.base.step(row[None],state);means.append(mean[0]);states.append(state[0])
   mean=torch.stack(means);hidden=torch.stack(states);features,gate=model.features(x,hidden,u)
  if not(np.array_equal(hidden.numpy(),after) and np.array_equal(before[0],np.zeros(128)) and np.array_equal(before[1:],after[:-1])):raise ValueError('causal mixed hidden state mismatch')
  episodes.append(dict(name='corrective_'+direction,x=x,y=y,u=u,mean=mean,base_hidden=hidden,features=features,gate=gate))
  admissions.append(dict(direction=direction,role='FAILED_OBSERVATIONS_VALID_TEACHER_LABELS_ONLY',original_dataset_safety_admitted=False,whole_mixed_slip=float(c['slip_ratio_max']),label_rows=1920,postprime_training_rows=960,teacher_root_residual_max=label['root_balance_max'],samples_sha256=sha256(sp),actual_audit_sha256=sha256(ap),label_audit_sha256=sha256(lp),passing_demonstration=False))
 if len(episodes)!=6:raise ValueError('exactly six trajectories required')
 return episodes,hashes,admissions


def objective(model,episodes):
 # Per-trajectory means (1920 vs960 postrows) all weight1/6, NOT row-count pooled.
 return sum((cached(model,e)[960:].tanh()-e['y'][960:]).square().mean() for e in episodes)/6
