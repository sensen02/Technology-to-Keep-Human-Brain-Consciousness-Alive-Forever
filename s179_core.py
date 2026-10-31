"""S179 semantics core (engine-free): row/DOF identity, root-failure classes, best selection."""
import numpy as np


class IdentityError(AssertionError):
    pass


def build_row_dof_map(dof_names, first_row_dof=6):
    """42 balance rows <-> real DOFs, built from the model's own dof names."""
    names = list(dof_names)
    m = []
    for row, dof in enumerate(range(first_row_dof, first_row_dof+42)):
        m.append(dict(joint_row=row, dof=dof, dof_name=names[dof], leg=leg_of_name(names[dof])))
    assert [v['dof'] for v in m] == list(range(first_row_dof, first_row_dof+42))
    assert len({v['dof_name'] for v in m}) == 42
    return m


def leg_of_name(name):
    """Accept both 'joint_RMCoxa_yaw' and 'rm' style names."""
    n = str(name)
    if n.startswith('joint_'):
        n = n[len('joint_'):]
    low = n.lower()
    if low in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'):
        return low
    for pref, leg in (('LF', 'lf'), ('LM', 'lm'), ('LH', 'lh'), ('RF', 'rf'), ('RM', 'rm'), ('RH', 'rh')):
        if n.startswith(pref):
            return leg
    return 'unlabeled'


def validate_identity(row_map, rows, dofs, context=''):
    """Any table/ledger entry must pass this; mismatched row<->dof is a hard error."""
    if len(rows) != len(dofs):
        raise IdentityError(f'{context}: rows/dofs length mismatch {len(rows)} vs {len(dofs)}')
    by_dof = {v['dof']: v for v in row_map}
    for r, d in zip(rows, dofs):
        if int(d) not in by_dof:
            raise IdentityError(f'{context}: dof {d} not in row map')
        if by_dof[int(d)]['joint_row'] != int(r):
            raise IdentityError(f'{context}: row {r} maps to dof {by_dof[int(d)]["joint_row"]+6}, not {d}')
    return True


def row_of_dof(row_map, dof):
    for v in row_map:
        if v['dof'] == int(dof):
            return v['joint_row']
    raise IdentityError(f'dof {dof} unknown')


CLASSES = ('ROOT_BALANCED', 'ROOT_DIVERGE', 'ROOT_CONTACT_SET_CHANGE', 'ROOT_RESIDUAL_FLOOR')


def classify_root(history_norms, root_norm, tol=1e-8, contact_changed=False, nonfinite=False):
    """Split the old single 'not converged' bucket into three physically distinct classes."""
    if root_norm <= tol:
        return 'ROOT_BALANCED'
    if nonfinite or not np.all(np.isfinite(history_norms)):
        return 'ROOT_DIVERGE'
    h = [float(x) for x in history_norms]
    if len(h) >= 2 and max(h[1:]) > h[0]*(1+1e-12):
        return 'ROOT_DIVERGE'
    if contact_changed:
        return 'ROOT_CONTACT_SET_CHANGE'
    return 'ROOT_RESIDUAL_FLOOR'


def select_best(candidates):
    """Frozen rule: rank ONLY inside {root_ok and geom_ok}; never across classes."""
    valid = [c for c in candidates if c.get('root_ok') and c.get('geom_ok')]
    if not valid:
        return dict(selection='NO_ROOT_BALANCED_GEOM_VALID_POINT', best=None, n_valid=0)
    best = min(valid, key=lambda c: c['minimax'])
    return dict(selection='BEST_WITHIN_ROOT_BALANCED_GEOM_VALID', best=best, n_valid=len(valid))
