"""Direct simulation load test: how much extra weight can the assisted body carry?

No linear program, no hand-derived contact algebra: the assist controller from the archived run is
used unchanged, and a downward force is added at the thorax before every physics step. The test asks
only whether the body keeps following the reference stride and keeps its feet on the ground.

The assist phase is run without the neural runtime, so what this measures is the capacity of the body
plus the reference controller. It is NOT a neural result and is labelled as such.

modes: prereg / sweep / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import run_physical_walk_prewarm as rw
O=Path('/home/sensen/Desktop/engine修复验收/s5-sim-load-capacity/20261105T200000Z')
LEVELS=(0.0,0.5,1.0,2.0,3.0,5.0,8.0)
DURATION=1.10
CLIP=float(os.environ.get('LOAD_TEST_CLIP','50.0'))
KP=float(os.environ.get('LOAD_TEST_KP','100.0'))
TAGPREFIX=os.environ.get('LOAD_TEST_PREFIX','LOAD')
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def run_level(folder,mult):
 import mujoco
 W=24.446354755878005
 real_step=mujoco.mj_step
 state={'apply':True}          # the patch is installed only for this run, so it is always armed
 def patched(m,d,*a,**k):
  if state['apply'] and mult>0:
   tb=int(m.body('Thorax').id)
   d.xfrc_applied[tb,2]-=mult*W
  return real_step(m,d,*a,**k)
 mujoco.mj_step=patched
 try:
  rw.run(folder,duration=DURATION,kp=KP,clip=CLIP,dt=1.25e-5)
 finally:
  mujoco.mj_step=real_step

def measure(folder):
 import numpy as np
 z=np.load(folder/'BODY.npz',allow_pickle=False)
 t=np.asarray(z['time'],float);com=np.asarray(z['com'],float)
 foot=np.asarray(z['foot_force'],float)[:,:,2]
 W=24.446354755878005
 m=t>=0.30
 return dict(com_mean=float(com[m,2].mean()),com_min=float(com[m,2].min()),com_final=float(com[-1,2]),
  foot_support_mean=float(foot[m].sum(1).mean()),
  feet_down_mean=float((foot[m]>1e-9).sum(1).mean()),
  feet_down_final=float((foot[-1]>1e-9).sum()),
  time_span=[float(t[0]),float(t[-1])])

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 dump(O/'PREREGISTRATION.json',dict(
  question='how much extra downward load can the assisted body carry before it stops following the reference stride',
  method='archived assist controller unchanged; a downward force of mult times body weight is added at the thorax before every physics step',
  levels=list(LEVELS),duration_s=DURATION,
  failure_criterion='mean centre of mass height over the measured window below 90 percent of the zero load mean, or fewer than three feet on the ground at the end',
  scope='capacity of body plus reference controller; the neural runtime is NOT involved and no autonomy claim follows'))

def sweep():
 base=None
 for mult in LEVELS:
  tag='%s_%03d'%(TAGPREFIX,round(mult*100))
  folder=O/tag
  if (folder/'BODY.npz').exists():
   pass
  else:
   if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
   try:run_level(folder,mult)
   except BaseException:print(tag,'FAILED',traceback.format_exc()[-300:],flush=True);continue
  mm=measure(folder)
  if base is None or mult==0.0:base=mm
  mm['expected_ground_force']=24.446354755878005*(1.0+mult)
  mm['load_actually_applied']=bool(abs(mm['foot_support_mean']-mm['expected_ground_force'])<0.05*mm['expected_ground_force'])
  rel=mm['com_mean']/base['com_mean']
  mm.update(tag=tag,load_multiple=mult,com_ratio=rel,
   verdict=('invalid' if not mm['load_actually_applied'] else
    ('holds' if (rel>=0.90 and mm['feet_down_final']>=3) else 'fails')))
  print('%-12s 载荷 %.1f 倍 | 地面力 %.2f (应 %.2f) 施加=%s | 质心比 %.3f | 脚数 %.0f → %s'%(
   tag,mult,mm['foot_support_mean'],mm['expected_ground_force'],mm['load_actually_applied'],rel,mm['feet_down_final'],mm['verdict']),flush=True)
  dump(O/f'{tag}/LOAD_RESULT.json',mm)
  dump(O/('SWEEP_%s.json'%TAGPREFIX),dict(clip=CLIP,kp=KP,levels=[js(O/f'{TAGPREFIX}_{round(x*100):03d}/LOAD_RESULT.json') for x in LEVELS if (O/f'{TAGPREFIX}_{round(x*100):03d}/LOAD_RESULT.json').exists()]))

def report():
 rows=[js(O/f'LOAD_{round(x*100):03d}/LOAD_RESULT.json') for x in LEVELS if (O/f'LOAD_{round(x*100):03d}/LOAD_RESULT.json').exists()]
 holds=[r for r in rows if r['verdict']=='holds']
 invalid=[r for r in rows if r['verdict']=='invalid']
 cap=max((r['load_multiple'] for r in holds),default=None)
 dump(O/'SUMMARY.json',dict(rows=rows,capacity_multiple=cap,
  reading='capacity is the largest tested load at which the body still follows the stride with at least three feet down; the criterion is in the preregistration',
  scope='body plus reference assist controller, no neural involvement'))
 for r in rows:print('  %.1f 倍 → %s (质心比 %.3f)'%(r['load_multiple'],r['verdict'],r['com_ratio']))
 print('承载上限(测得的档位):',cap)
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'report':report}[sys.argv[1]]()
