"""Neutral offline target loader. No models, controllers or training invoked."""
from pathlib import Path
import json
import numpy as np
DEFAULT=Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z')

def load_targets(root=DEFAULT, source='nmf2'):
    """Return array dictionary and scientific metadata; caller must apply masks.

    nmf2: xyz_source in UNKNOWN source units; geometric interior radians.
    caltech: derived angle_rad, masks/animal split; upstream test quarantined.
    Never infer root or contact from either target.
    """
    root=Path(root)
    if source=='nmf2':
        fn=root/'NMF2_CONTINUOUS.npz';meta=root/'NMF2_FIELDS_AND_TRANSFORMS.json'
    elif source=='caltech':
        fn=root/'caltech_angles/train_neutral.npz';meta=root/'caltech_angles/METADATA.json'
    else: raise ValueError('source must be nmf2 or caltech')
    with np.load(fn,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert all(not x.dtype.hasobject for x in arrays.values())
    return arrays,json.loads(meta.read_text())

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--source',choices=['nmf2','caltech'],default='nmf2');a=ap.parse_args()
    arrays,meta=load_targets(a.root,a.source)
    print(json.dumps(dict(source=a.source,fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},training_executed=False,root_and_contact_ground_truth_available=False),indent=2))
