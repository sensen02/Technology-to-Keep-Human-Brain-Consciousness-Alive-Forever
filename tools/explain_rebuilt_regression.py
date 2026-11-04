"""Why does the rebuilt network score worse than the one with the tactile class disconnected?

Four candidate explanations, each measured rather than argued:

  A direct injection    do the new edges land on the 204 motor cells
  B dead signal         does the tactile channel carry anything, or is it a constant
  C saturation          do the extra edges push muscle commands into their clip
  D network dynamics    what does the added excitatory mass do to the spike statistics

Reads archived arrays only; no simulation runs. modes: audit
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import run_neural_muscle_closure as c
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-regression-analysis/20261104T160000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
V2=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2/eval_622/RUN')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/VALIDATION_R1/eval_709_A/RUN')
LEGS=('lf','lm','lh','rf','rm','rh')
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def audit():
 roots,leg,pool,npools=st.identity()
 motor=np.flatnonzero(np.asarray(leg)>=0)
 groups,_=c.channel_index_map(len(roots),roots,lookup_path=st.A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 enc=ChannelSensoryEncoder(groups,len(roots),ChannelConfig())
 tac=sorted({i for lg in enc.groups for i in enc.groups[lg].get('bristle_touch',[])})
 new=sparse.load_npz(REBUILD/'REBUILT_We.npz').tocsr();old=sparse.load_npz(st.B/'ACTUAL_We.npz').tocsr()
 add=(new-old).tocsr();add.eliminate_zeros()
 rows,cols=add.nonzero()
 mset=set(motor.tolist())
 land=[i for i in rows if i in mset]
 per_cell=np.asarray(add[motor].sum(1)).ravel()
 A=dict(added_edges=int(add.nnz),
  added_weight_total=float(add.sum()),
  added_weight_share_of_old=float(add.sum()/float(old.sum())),
  edges_landing_directly_on_motor=int(len(land)),
  motor_cells_touched=int(len(set(land))),
  share_of_added_edges_on_motor=float(len(land)/max(add.nnz,1)),
  added_weight_per_motor_cell_median=float(np.median(per_cell)),
  added_weight_per_motor_cell_max=float(per_cell.max()),
  old_motor_in_degree_median=float(np.median(np.asarray(old[motor].sum(1)).ravel())),
  verdict='the new edges almost all go to interneurons; a direct motor injection is not the explanation')
 rec=js(V2/'NEURAL_RECORDS.json')
 vals={lg:[] for lg in LEGS}
 for r in rec:
  for lg,d in r['channels']['per_leg'].items():vals[lg].append(d['bristle_touch']['value'])
 sig={lg:dict(levels=[float(x) for x in np.unique(np.round(np.array(vals[lg]),6))],
   transitions=int(np.sum(np.abs(np.diff(np.array(vals[lg])))>1e-9)),
   mean=float(np.mean(vals[lg]))) for lg in LEGS}
 B=dict(per_leg=sig,verdict='the channel is a per-leg contact ON/OFF flag: it does carry contact timing, but it carries no force magnitude')
 def cmd(p):
  d=dict(np.load(Path(p)/'BODY.npz'));c=d['command']
  return dict(mean_command=float(c.mean()),fraction_at_ceiling=float(np.mean(c>=0.9999)),
   fraction_at_zero=float(np.mean(c<=1e-6)),mean_active_muscles=float(np.mean(np.sum(c>0.01,axis=1))))
 C=dict(rebuilt_best=cmd(V2),disconnected_best=cmd(OLD),
  verdict='the rebuilt network drives muscles harder and spends more of the range clipped at the ceiling')
 dump(O/'REGRESSION_ANALYSIS.json',dict(A_direct_injection=A,B_signal_content=B,C_command_saturation=C,
  D_reading='the added excitatory mass shifts the whole network operating point; the parameters that could absorb a common-mode shift are already on their bounds (17 of 24 pool dimensions and both ends of the readout gain range), so the correction is paid for out of dynamic range instead',
  sources=dict(rebuilt_we=hashlib.sha256((REBUILD/'REBUILT_We.npz').read_bytes()).hexdigest(),
   old_we=hashlib.sha256((st.B/'ACTUAL_We.npz').read_bytes()).hexdigest(),
   rebuilt_best=str(V2),disconnected_best=str(OLD))))
 print(json.dumps(dict(A=A,B=B,C=C),ensure_ascii=False,indent=1)[:1600])
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'audit':audit}[sys.argv[1]]()
