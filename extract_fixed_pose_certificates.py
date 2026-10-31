"""S171 correct separator orientation from saved duals only; no optimization rerun."""
from pathlib import Path
import argparse,json
import numpy as np
pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;rows=[]
for k in [1,2,3]:
 with np.load(p/f'CASE_{k}.npz') as z:
  A=z['A'];b=z['b'];dual=z['dual_marginals'];y=dual[:48]-dual[48:];gap=float(b@y-np.maximum(A.T@y,0).sum());assert gap>1e-8
  np.savez_compressed(p/f'CERTIFICATE_{k}.npz',y=y,separator_gap=gap)
  rows.append(dict(condition=k,status='INFEASIBLE_CERTIFICATE_PENDING_INDEPENDENT_CHECK',gap=gap,orientation='upper dual minus lower dual',source=f'CASE_{k}.npz',lp_rerun=False))
(p/'SUMMARY_FINAL.json').write_text(json.dumps(dict(cases=rows,correction='Initial saved separator orientation was reversed; original CASE/SUMMARY preserved, certificates algebraically reoriented from saved duals, no LP rerun',calls=dict(original_phaseI_LPs=3,additional_optimization=0,model_load=0,forward=0,step=0,neural=0)),indent=2));print(json.dumps(rows))
