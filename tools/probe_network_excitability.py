"""Sweep the global sensory input gain and watch the silent motor cells.

The silence was shown to be a property of the recurrent dynamics rather than of missing input, so the
cheapest remaining lever is the network's overall operating point. This does NOT train anything: it
scales all sensory channels by one multiplier, runs the best vector once per level, and reports three
things so the trade off is visible:

    silent_cells_awakened   how many of the forty two silent motor cells now fire
    active_fraction         how much of the whole tier is active, as an over excitation check
    J                       the objective, so a gain that only inflates activity is visible as such

modes: run / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-excitability-sweep/20261107T060000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-per-cell-readout-long/20261106T000000Z/EVALS/eval_294/SCORE.json')
CAUSE=Path('/home/sensen/Desktop/engine修复验收/s5-silent-motoneuron-cause/20261107T020000Z/SILENT_CAUSE.json')
LEVELS=(0.25,0.5,1.0,2.0,4.0,8.0)
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if hasattr(x,'item') else str(x)))+'\n')

def one(mult):
 import importlib
 os.environ['PCR_OUT']=str(O)
 import train_per_cell_readout as t
 import train_stage_a_pool_modes as st
 importlib.reload(t)
 importlib.reload(st)
 from engine.embodied import motor_output as mo
 from engine.embodied.sensory_channels import CHANNEL_SPEC
 rec=[]
 orig=mo.MotorOutput.consume
 def patched(self,spikes,**kw):
  try:rec.append(np.asarray(spikes,dtype=np.int8).sum(axis=0))
  except Exception:pass
  return orig(self,spikes,**kw)
 mo.MotorOutput.consume=patched
 t.make_set()
 st.CONFIG['neural_sensory_channel_weights']={c:float(mult) for c in CHANNEL_SPEC}
 vec=json.loads(BEST.read_text())['vec']
 folder=O/('GAIN_%05d'%round(mult*100))
 if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
 try:r=t.eval_one((0,vec,folder))
 finally:
  mo.MotorOutput.consume=orig;st.CONFIG.pop('neural_sensory_channel_weights',None)
 if not rec:return dict(gain=mult,J=None,error='no spikes recorded')
 tot=np.array(rec).sum(axis=0)
 cause=json.loads(CAUSE.read_text())
 silent=[x['cell'] for x in cause['per_cell'] if x['spikes']==0]
 awake=int(sum(1 for k in silent if tot[k]>0))
 return dict(gain=mult,J=float(r['J']),silent_total=len(silent),silent_cells_awakened=awake,
  active_fraction=float((tot>0).mean()),active_cells=int((tot>0).sum()),windows=len(rec),
  parts=r.get('parts'))

def run():
 O.mkdir(parents=True,exist_ok=True)
 dump(O/'PREREGISTRATION.json',dict(
  question='does raising the global sensory input gain wake the silent motor cells, and at what cost in objective',
  method='scale all seven sensory channels by one multiplier, run the best vector once per level with the spike recorder attached, no training',
  levels=list(LEVELS),
  why='the silence was shown to be emergent recurrent dynamics rather than missing input, so the cheapest untried lever is the network operating point',
  acceptance='reported as it comes out, including if the silent cells stay silent at every level'))
 out=[]
 for m in LEVELS:
  try:x=one(m)
  except SystemExit as e:print('skip',m,e);continue
  out.append(x)
  print('增益 %-5.2f J=%-8s 唤醒静默 %s/%s 全网络活跃 %.3f'%(m,
   ('%.4f'%x['J']) if x.get('J') is not None else 'na',
   x.get('silent_cells_awakened','na'),x.get('silent_total','na'),x.get('active_fraction',0) or 0),flush=True)
  dump(O/'SWEEP.json',dict(levels=out))
if __name__=='__main__':
 {'run':run}[sys.argv[1]]()
