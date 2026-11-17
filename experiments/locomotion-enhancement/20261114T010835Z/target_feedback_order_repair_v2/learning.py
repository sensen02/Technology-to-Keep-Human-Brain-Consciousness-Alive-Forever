"""Original objective/masks/normalizations/resource limits, new component dispatch.
No backend import or execution at module import; old loss source stays immutable.
"""
from pathlib import Path
import ast
ORIGINAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration')
_ns={'__name__':'_orderrepair_exact_original_learning','__file__':str(ORIGINAL/'learning.py')}
exec(compile(ast.parse((ORIGINAL/'learning.py').read_text()),str(ORIGINAL/'learning.py')+'::UNCHANGED_OBJECTIVE_HELPERS','exec'),_ns)
for _name in ('cpu1','rng_snapshot','rng_equal','rng_same','restore_rng','adapter_anchor','resource_gate','time_weights','objective','metrics'):
    globals()[_name]=_ns[_name]
_original_slice=_ns['slice_data']

def slice_data(data,lo,hi,T=None):
    result=_original_slice(data,lo,hi,T=T)
    for name in ('basehead','right','left'):
        result[name]=data[name][:T,lo:hi] if T is not None else data[name][:,lo:hi]
    return result


def forward(scan,data,forced=False):
    forced_targets=data['actual_pre_targets'] if forced else None
    return scan(data['physical'],data['oldafter'],data['commands'],
                data['basehead'],data['right'],data['left'],data['gain'],
                data['initial_target'],data['q0'],data['halfspan'],
                data['bounds'],data['max_rate'],forced_targets)

# Original metrics function retains the exact same loss/statistical code but must
# resolve repaired forward and explicit three-component slicing in its globals.
_ns.update(slice_data=slice_data,forward=forward)
