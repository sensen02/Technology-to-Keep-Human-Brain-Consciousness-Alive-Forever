"""Reuse the frozen four-seed protocol with five preregistered wider initial guesses.
These factors change initial joint angles, not constraints on the final footprint.
No claim of biological abduction direction follows from native joint names.
"""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-abduction-seeds/20261101T201309Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wider-stance/20261101T205257Z')
FACTORS=(.9,1.1,1.25,1.4,1.6)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')

def main():
    assert not OUT.exists(), 'Unique output only; do not restart'
    OUT.mkdir(parents=True)
    table=json.loads((OLD/'SEED_TABLE.json').read_text())
    base=np.array(table['archived_hinge_values']);lo=np.array(table['joint_lower']);hi=np.array(table['joint_upper'])
    names=table['hinge_names'];idx=[names.index(n) for n in table['contract_order']]
    rows=[]
    for k in FACTORS:
        raw=base.copy();raw[idx]*=k;proj=np.clip(raw,lo,hi)
        changes=[dict(variable=int(i),joint=names[i],raw=float(raw[i]),projected=float(proj[i]),delta=float(proj[i]-raw[i]),bound='lower' if raw[i]<lo[i] else 'upper') for i in np.flatnonzero(raw!=proj)]
        row=dict(table['seeds'][0]);row.update(retained_fraction_k=k,raw_contracted_42=raw,projected_42=proj,x0_109=np.r_[proj,table['archived_root_z'],np.full(66,.1)],projections=changes,n_projections=len(changes),total_projection_abs=float(abs(proj-raw).sum()),sum_abs_abduction_roll_yaw_x0=float(abs(proj[idx]).sum()),contracted_values_after_factor={names[i]:float(raw[i]) for i in idx})
        rows.append(row)
    table.update(seeds=rows,contraction_factors=list(FACTORS),n_projections_total=sum(r['n_projections'] for r in rows))
    dump(OUT/'SEED_TABLE.json',table)
    prereg=json.loads((OLD/'PREREGISTRATION.json').read_text())
    # Keep objective, gates, solver, selection and implementation identical.
    prereg.update(task='Five wider native-angle initial guesses; final stance unconstrained by k',question='Does increasing the initial coxa roll/yaw multiplier improve the selected local-search outcomes?',seed_table_sha256=sha(OUT/'SEED_TABLE.json'),factors=list(FACTORS),limitations_correction='Near bounds is not a KKT binding-constraint proof. Scaling native roll and yaw is not a measured anatomical stance-width change. This is a multistart test, not a fixed-footprint causal experiment.',source_wrapper_sha256=sha(Path(__file__)),source_protocol_sha256=sha(HERE/'search_native_equilibrium_abduction_seeds.py'),wider_classification='Support directional trend only if h/v for k>=1.1 is monotonically nonincreasing and minimum<0.15; otherwise not supported. Actual full gate determines any static candidate.',legacy_classification_warning='Reused engine emits its old contraction classifier; that classifier is NOT_APPLICABLE to this experiment and is not interpreted.')
    dump(OUT/'PREREGISTRATION.json',prereg)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    spec=importlib.util.spec_from_file_location('seed_protocol',HERE/'search_native_equilibrium_abduction_seeds.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.O=OUT;module.FACTORS=FACTORS
    code=module.main()
    assert not code, 'Protocol execution failed; preserve outputs'
    print('WIDER_RUN_COMPLETE',OUT)

if __name__=='__main__':main()
