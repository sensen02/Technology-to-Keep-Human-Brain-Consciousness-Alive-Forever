"""S109 offline composite; distinct from frozen s96 formal-only proxy."""
import numpy as np

EARLY_LOAD_SCALE = .8
EARLY_LIMIT_SCALE = .001


def score(arrays, weight_native, scales, objective):
    """Frozen formal proxy + two equally weighted startup mean-square penalties."""
    if arrays['qpos'].shape != (4001,49):
        raise ValueError('complete 400ms free-body trajectory required')
    if not np.isclose(weight_native, objective.WEIGHT_NATIVE,rtol=0,atol=1e-12):
        raise ValueError('frozen native weight differs')
    old = objective.proxy(arrays, scales)
    load = np.asarray(arrays['foot_force'],float)[:101,:,2].sum(1)/weight_native
    limit = np.asarray(arrays['limit_excess'],float)[:101]
    early_load = float(np.trapezoid((np.maximum(.8-load,0)/EARLY_LOAD_SCALE)**2,dx=.0001)/.01)
    early_limit = float(np.trapezoid((np.maximum(limit-1e-6,0)/EARLY_LIMIT_SCALE)**2,dx=.0001)/.01)
    total = float(old['total']+early_load+early_limit)
    if not np.isfinite([total,old['total'],early_load,early_limit]).all():
        raise ValueError('nonfinite objective')
    return dict(total=total,formal_proxy=float(old['total']),early_load=early_load,early_limit=early_limit)
