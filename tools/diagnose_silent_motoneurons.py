"""Why do eighteen motor neurons with intact partners never fire?

The readout cannot help the silent pools because any gain maps zero to zero, so the cause has to be
found one layer upstream. This records the spikes of all cells during the release window, then for
every motor neuron asks three questions:

  drive      how much excitatory and inhibitory input it receives from cells that ARE active
  partners   whether its own presynaptic partners fire at all
  threshold  what fraction of the cells in the tier are active at all, as a context number

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
O=Path('/home/sensen/Desktop/engine修复验收/s5-silent-motoneuron-cause/20261107T020000Z')
from scipy.sparse import load_npz
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-per-cell-readout-long/20261106T000000Z/EVALS/eval_294/SCORE.json')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if hasattr(x,'item') else str(x)))+'\n')

def run():
 import importlib
 os.environ['PCR_OUT']=str(O)
 import train_per_cell_readout as t
 importlib.reload(t)
 t.make_set()
 from engine.embodied import motor_output as mo
 recorded=[]
 orig=mo.MotorOutput.consume
 def patched(self,spikes,**kw):
  try:recorded.append(np.asarray(spikes,dtype=np.int8).sum(axis=0))
  except Exception:recorded.append(None)
  return orig(self,spikes,**kw)
 mo.MotorOutput.consume=patched
 vec=json.loads(BEST.read_text())['vec']
 folder=O/'RECORD_RUN'
 try:
  r=t.eval_one((0,vec,folder))
 finally:
  mo.MotorOutput.consume=orig
 if (O/'CELL_SPIKES.npz').exists():
  counts=np.load(O/'CELL_SPIKES.npz')['per_window'];r={'J':float('nan')}
 else:
  counts=np.array([x for x in recorded if x is not None])
 print('windows',counts.shape,'window_count',len(recorded),'J',r.get('J'))
 np.savez_compressed(O/'CELL_SPIKES.npz',per_window=counts,total=counts.sum(axis=0))
 # 结构
 We=load_npz(REBUILD/'REBUILT_We.npz').tocsr();Wi=load_npz(REBUILD/'REBUILT_Wi.npz').tocsr()
 man=json.loads((t.SET/'RUN/MAPPING.json').read_text())['rows']
 tot=counts.sum(axis=0)
 motor={}
 for i,row in enumerate(man):
  for k in (row.get('local_indices') or []):motor.setdefault(int(k),[]).append(row.get('actuator'))
 res=[]
 for k,acts in motor.items():
  active=np.flatnonzero(tot>0).astype(int)          # sparse matrices need integer indices
  # convention verified here: row = postsynaptic, so a cell's inputs are its ROW
  row_e=We[k,:].tocsr();row_i=Wi[k,:].tocsr()
  pe=set(row_e.indices.tolist());pi=set(row_i.indices.tolist())
  e_in=float(row_e[:,active].sum() if row_e.shape[1] else 0.0)
  i_in=float(row_i[:,active].sum() if row_i.shape[1] else 0.0)
  e_in=float(np.asarray(row_e.toarray()).ravel()[active].sum())
  i_in=float(np.asarray(row_i.toarray()).ravel()[active].sum())
  pres=np.zeros(12000,bool);pres[list(pe|pi)]=True
  n_e=len(pe);n_i=len(pi)
  pres_active=float((tot[pres]>0).mean()) if pres.any() else 0.0
  res.append(dict(cell=int(k),spikes=int(tot[k]),actuators=acts,
   exc_drive_from_active=e_in,inh_drive_from_active=i_in,
   n_exc_partners=n_e,n_inh_partners=n_i,partners=int(pres.sum()),
   partner_active_fraction=pres_active,
   exc_to_inh_ratio=float(n_e/max(n_i,1))))
 silent=[x for x in res if x['spikes']==0];active=[x for x in res if x['spikes']>0]
 def agg(xs):return dict(n=len(xs),mean_exc=float(np.mean([x['exc_drive_from_active'] for x in xs])),
   mean_inh=float(np.mean([x['inh_drive_from_active'] for x in xs])),
   mean_exc_partners=float(np.mean([x['n_exc_partners'] for x in xs])),
   mean_inh_partners=float(np.mean([x['n_inh_partners'] for x in xs])),
   mean_partner_active=float(np.mean([x['partner_active_fraction'] for x in xs])),
   mean_partners=float(np.mean([x['partners'] for x in xs])))
 out=dict(windows=int(counts.shape[0]),cells=int(counts.shape[1]),
  active_cells_total=int((tot>0).sum()),fraction_cells_active=float((tot>0).mean()),
  motor_cells=len(res),silent=len(silent),active=len(active),
  aggregate=dict(silent=agg(silent),active=agg(active)),
  per_cell=res,
  reading='if the silent group shows a similar mean partner_active fraction to the active group, their partners do fire and the cause is inhibition or threshold; if it is near zero, their input is upstream silent')
 dump(O/'SILENT_CAUSE.json',out)
 print('活跃细胞 %d / %d（%.1f%%）'%(out['active_cells_total'],out['cells'],100*out['fraction_cells_active']))
 for k in ('silent','active'):
  a=out['aggregate'][k]
  print('%-7s n=%-3d 兴奋伙伴%.1f 抑制伙伴%.1f 活跃输入的兴奋%.4f 抑制%.4f 伙伴活跃率%.3f'%(
   k,a['n'],a['mean_exc_partners'],a['mean_inh_partners'],a['mean_exc'],a['mean_inh'],a['mean_partner_active']))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
