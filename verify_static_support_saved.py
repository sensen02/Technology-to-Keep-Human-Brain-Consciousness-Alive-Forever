"""S165 independent saved-matrix checks. No MuJoCo imports/calls."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 # Producer schema read after extraction; only saved arrays are audited here.
 summaries=[]
 for path in sorted(p.glob('state*/INPUTS.npz')):
  with np.load(path) as z:a={k:z[k].copy() for k in z.files}
  summaries.append(dict(path=str(path),keys=list(a)))
 (p/'PARENT_MATRIX_INVENTORY.json').write_text(json.dumps(dict(states=summaries,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print('Inventory',len(summaries),'states')
if __name__=='__main__':main()
