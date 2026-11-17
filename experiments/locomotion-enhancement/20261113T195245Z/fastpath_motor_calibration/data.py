"""One explicit global binding of immutable actual8 cache implementation."""
from pathlib import Path
import ast
import sys
import types
import sources

PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration')
namespace={'__name__':'_fastpath_exact_actual8_data','__file__':str(PARENT/'data.py')}
exec(compile(ast.parse((PARENT/'data.py').read_text()),str(PARENT/'data.py')+'::same_actual8_numerical_code','exec'),namespace)
# Original fromsources resolves this newdirectory on freshparent entry.
namespace.update(ROOT=sources.ROOT,read=sources.read,verify=sources.verify)
prepare=namespace['prepare']
production_sequence=namespace['production_sequence']


def dev_sources(binding):
    import numpy as np
    import torch
    result=[]
    for entry in binding['dev_entries']:
        with np.load(entry['path'],allow_pickle=False) as archive:
            x=archive['X238'].copy();u=archive['command'].copy()
        if x.shape!=(3080,238) or u.shape!=(3080,2):raise ValueError('UnlabeledDEVactualshape')
        result.append(dict(entry=entry,warm=torch.tensor(x[200:1160],dtype=torch.float64),x=torch.tensor(x[1160:],dtype=torch.float64),u=torch.tensor(u[1160:],dtype=torch.float64),labels_available=False))
    return result
