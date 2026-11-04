"""One shot probe: append the missing upstream partners and see if the silent cells wake.

No training. This builds the tier once with the declared append of the silent motor cells' missing
upstream partners, runs the current best vector once with the spike recorder attached, and reports how
many of the forty two silent motor cells now fire, together with the objective and the active fraction.

If the cells stay silent even with their partners present, augmentation is not the answer either and
that is worth knowing before any retraining is paid for.

modes: run
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-tier-augmentation-probe/20261107T200000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-synthetic-readout-cont/20261107T120000Z/VERIFY_A/SCORE.json')
CAUSE=Path('/home/sensen/Desktop/engine修复验收/s5-silent-motoneuron-cause/20261107T020000Z/SILENT_CAUSE.json')
def _clean(o):
 import math
 if isinstance(o,dict):return {k:_clean(x) for k,x in o.items()}
 if isinstance(o,(list,tuple)):return [_clean(x) for x in o]
 if isinstance(o,float):return None if (math.isnan(o) or math.isinf(o)) else o
 if isinstance(o,np.ndarray):return _clean(o.tolist())
 if hasattr(o,'item'):return _clean(o.item())
 return o
def dump(p,v):Path(p).write_text(json.dumps(_clean(v),ensure_ascii=False,indent=2)+'\n')

def missing_partners(tier_ids):
 z=np.load('data/flywire/banc_connectome.npz',allow_pickle=True)
 pre=z['pre'];post=z['post'];syn=z['syn'].astype(float);roots=z['root_ids']
 ridx={int(r):i for i,r in enumerate(roots)}
 cause=json.loads(CAUSE.read_text())
 silent=[x['cell'] for x in cause['per_cell'] if x['spikes']==0]
 silent_roots={int(tier_ids[k]) for k in silent}
 sel=[ridx[r] for r in silent_roots if r in ridx]
 m=(syn>0)&np.isin(post,np.asarray(sel,dtype=np.int64))
 u,inv=np.unique(pre[m],return_inverse=True);w=np.zeros(u.size);np.add.at(w,inv,syn[m])
 ts=set(int(x) for x in tier_ids)
 out=[(int(roots[i]),float(ww)) for i,ww in zip(u,w) if int(roots[i]) not in ts and ww>=10.0]
 out.sort(key=lambda x:-x[1])
 return [g for g,_ in out],out

def run():
 O.mkdir(parents=True,exist_ok=True)
 import dataclasses,importlib
 os.environ['PCR_OUT']=str(O);os.environ['PCR_SYNTH']='1'
 import train_per_cell_readout as t
 importlib.reload(t)
 t.make_set()
 import engine.embodied.adapters as A
 from engine.embodied.adapters import TierConfig
 base=A.build_neural_tier(dataclasses.replace(TierConfig(),sign_fill='bristle_only'))
 gidx=np.asarray(base['tier']['global_index'],dtype=np.int64)
 roots_all=np.load('data/flywire/banc_connectome.npz',allow_pickle=True)['root_ids']
 tier_ids=roots_all[gidx]        # global_index -> root id, the two are not interchangeable
 print('基线 tier',tier_ids.size)
 extra,ranked=missing_partners(tier_ids)
 print('待追加 %d 个'%len(extra))
 dump(O/'APPENDED_ROOT_IDS.json',dict(n=len(extra),root_ids=extra,
  top=[dict(root_id=g,synapses=w) for g,w in ranked[:20]],
  rule='upstream partners of the silent motor cells in the full BANC graph that are absent from the 12000 tier and carry at least ten synapses'))
 # The two frozen artifacts that carry a size must be rebuilt at the augmented size.  The rows and
 # every local index stay identical, because appending never renumbers 0..11999.
 roots_all_np=np.load('data/flywire/banc_connectome.npz',allow_pickle=True)['root_ids']
 aug_n=int(gidx.size+len(extra))
 # NEVER write through a symlink into the sealed package.  SET/RUN is a symlink to the frozen run
 # directory, so first replace it with a real directory whose entries are symlinks, then write the
 # patched MAPPING as a real file inside it.
 import shutil as _sh
 run_dir=t.SET/'RUN'
 if run_dir.is_symlink():
  target=run_dir.resolve()
  run_dir.unlink()
  run_dir.mkdir()
  for e in target.iterdir():
   (run_dir/e.name).symlink_to(e)
 mp=json.loads((run_dir/'MAPPING.json').read_text())
 mp['n_neurons']=aug_n
 (run_dir/'MAPPING.json').unlink(missing_ok=True)
 (run_dir/'MAPPING.json').write_text(json.dumps(mp,ensure_ascii=False))
 frozen=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
 import hashlib as _h
 assert _h.sha256(frozen.read_bytes()).hexdigest().startswith('5774b6f2'), 'sealed MAPPING was touched'
 ip=t.SET/'IDENTITIES.npz'
 if ip.is_symlink():
  src=ip.resolve();ip.unlink();_sh.copyfile(src,ip)
 ident=np.load(ip,allow_pickle=False)
 extra_g=np.asarray([int(np.flatnonzero(roots_all_np==int(r))[0]) for r in extra],dtype=np.int64)
 order=np.argsort(extra_g)                      # the engine sorts appended cells by global index
 extra=np.asarray(extra,dtype=np.int64)[order];extra_g=extra_g[order]
 new_roots=np.concatenate([ident['root_ids'],extra])
 new_gidx=np.concatenate([ident['global_index'],extra_g])
 np.savez_compressed(str(ip),root_ids=new_roots,global_index=new_gidx)
 print('构件已按 %d 重建：MAPPING.n_neurons=%d，IDENTITIES=%d'%(aug_n,mp['n_neurons'],new_roots.size))
 orig=A.build_neural_tier
 def patched(config=None,*a,**k):
  return orig(dataclasses.replace(config or TierConfig(),sign_fill='bristle_only',
   extra_root_ids=tuple(extra)),*a,**k)
 A.build_neural_tier=patched
 from engine.embodied import motor_output as mo
 rec=[]
 o2=mo.MotorOutput.consume
 def patched2(self,spikes,**kw):
  try:rec.append(np.asarray(spikes,dtype=np.int8).sum(axis=0))
  except Exception:pass
  return o2(self,spikes,**kw)
 mo.MotorOutput.consume=patched2
 vec=json.loads(BEST.read_text())['vec']
 folder=O/'AUGMENTED'
 if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
 try:r=t.eval_one((0,vec,folder))
 finally:
  mo.MotorOutput.consume=o2;A.build_neural_tier=orig
 cause=json.loads(CAUSE.read_text())
 silent=[x['cell'] for x in cause['per_cell'] if x['spikes']==0]
 tot=np.array(rec).sum(axis=0) if rec else np.zeros(1)
 awake=int(sum(1 for k in silent if k<tot.size and tot[k]>0))
 out=dict(appended=len(extra),windows=len(rec),cells=int(tot.size),
  silent_total=len(silent),silent_cells_awakened=awake,
  active_fraction=float((tot>0).mean()),J=float(r['J']),parts=r.get('parts'),
  baseline=dict(J=3.402395,silent_cells_awakened=0,active_fraction=0.703),
  reading='if the appended partners wake the silent cells the objective should also move, because those cells belong to the two legs that carried no drive')
 dump(O/'PROBE.json',out)
 print('追加 %d 个细胞后：tier=%s 静默唤醒 %d/%d 全网络活跃 %.3f J=%.4f'%(
  len(extra),tot.size,awake,len(silent),out['active_fraction'],out['J']))
if __name__=='__main__':
 {'run':run}[sys.argv[1]]()
