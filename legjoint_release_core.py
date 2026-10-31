"""Engine-free core for S178: per-row interval/sign-flip helpers and block hill-climb.

Mandate: acceptance uses only the current iterate's 42-row minimax; the logging best is
bookkeeping and must never be consulted by the accept test. Root balance is an inner
gate (root_ok) and geometry is a constraint, never folded into a scalar score.
"""
import json
from pathlib import Path
import numpy as np


class BudgetStop(Exception):
    pass


class DefectStop(Exception):
    pass


def row_intervals(A, target):
    """Per-row achievable box interval [sum(min), sum(max)] versus target."""
    lo = np.minimum(A, 0.).sum(axis=1)
    hi = np.maximum(A, 0.).sum(axis=1)
    viol = np.maximum.reduce([lo - target, np.zeros_like(lo), target - hi])
    reach = (lo - 1e-9 <= target) & (target <= hi + 1e-9)
    return lo, hi, viol, reach


def sign_flip_estimate(col_values, joint_angles, base_index):
    """Estimate joint angle needed to flip the dominant muscle column's sign.

    col_values and joint_angles are the 1-D scan samples along one joint. Uses the two
    scan samples adjacent to the release point as the derivative probes (no extra
    forwards), then extrapolates |m| / |dm/dq| clipped to the frozen range.
    """
    m0 = float(col_values[base_index])
    idx = [i for i in (base_index - 1, base_index + 1) if 0 <= i < len(col_values)]
    slopes = []
    for i in idx:
        dq = float(joint_angles[i] - joint_angles[base_index])
        if abs(dq) > 1e-12:
            slopes.append((float(col_values[i]) - m0) / dq)
    if not slopes:
        return dict(slope=None, angle_needed=None, flippable=False, reason='no usable probe')
    slope = float(np.mean(slopes))
    if abs(slope) < 1e-12:
        return dict(slope=0., angle_needed=None, flippable=False, reason='zero derivative')
    angle_needed = float(abs(m0)/abs(slope))
    span = float(joint_angles[-1] - joint_angles[0])
    return dict(slope=slope, angle_needed=angle_needed, span_available=span,
                flippable=bool(angle_needed <= span), reason='extrapolated')


class BlockHillClimb:
    def __init__(self, evaluate, lower, upper, blocks, cap, reserved=0, output=None,
                 step0=0.25, step_floor=1/64, name=''):
        self.evaluate = evaluate
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.blocks = list(blocks)
        self.cap = int(cap)
        self.reserved = int(reserved)
        self.output = Path(output) if output else None
        self.step0 = float(step0)
        self.step_floor = float(step_floor)
        self.name = name
        self.ledger = dict(candidate_evaluations=0, forward=0, cache_hits=0, root_newton_iterations=0,
                           root_jacobians=0, fd_probes=0, inner_lp=0, accepted_moves=0,
                           contact_set_changes=0, rejected_geometry=0, rejected_root=0, sweeps=0)
        self.cache = {}
        self.best = None
        self.events = []
        self.history = []

    def _jsonable(self, v):
        if isinstance(v, dict): return {str(k): self._jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [self._jsonable(x) for x in v]
        if isinstance(v, np.generic): v = v.item()
        if isinstance(v, float) and not np.isfinite(v): return None
        if isinstance(v, (int, float, bool, str)) or v is None: return v
        return str(v)

    def _dump(self):
        if self.output:
            self.output.write_text(json.dumps(self._jsonable(dict(
                name=self.name, ledger=self.ledger,
                best=None if self.best is None else self.best['summary'],
                events=self.events, history=self.history)), indent=2, allow_nan=False))

    def _record(self, **e):
        e['event_id'] = len(self.events)+1
        self.events.append(e)
        self._dump()

    def remaining(self):
        return self.cap - self.reserved - self.ledger['forward']

    def evaluate_q(self, q_leg, role, note=''):
        key = np.asarray(q_leg, float).tobytes()
        if key in self.cache:
            self.ledger['cache_hits'] += 1
            return self.cache[key]
        rec = self.evaluate(np.asarray(q_leg, float), self)
        self.cache[key] = rec
        self.ledger['candidate_evaluations'] += 1
        self._record(role=role, note=note, forward=self.ledger['forward'],
                     root_ok=bool(rec['root_ok']), root_norm=float(rec['root_norm']),
                     minimax=float(rec['minimax']), geom_ok=bool(rec['geom_ok']),
                     contact_signature=str(rec.get('contact_signature')))
        if self.best is None or rec['minimax'] < self.best['minimax']:
            self.best = dict(rec)
            self.best['summary'] = self._jsonable({k: v for k, v in rec.items() if k not in ('residual6', 'rows', 'row_violation')})
            self._dump()
        return rec

    def run(self, q0):
        q = np.asarray(q0, float).copy()
        cur = self.evaluate_q(q, 'initial', 'S177 root-balanced baseline')
        cur_ok = bool(cur['root_ok'] and cur['geom_ok'])
        self.history.append(dict(iteration=0, role='initial', minimax=cur['minimax'], root_norm=cur['root_norm'],
                                 root_ok=cur['root_ok'], geom_ok=cur['geom_ok'],
                                 unreachable=cur['unreachable'], q=q.tolist()))
        step = self.step0
        stop = 'STEP_FLOOR'
        sweeps = 0
        while step >= self.step_floor:
            sweeps += 1
            self.ledger['sweeps'] = sweeps
            improved = False
            for block in self.blocks:
                for direction in (+1, -1):
                    need = 8 + 7*1
                    if self.remaining() < need:
                        stop = 'BUDGET_BEFORE_BLOCK'
                        break
                    trial = q.copy()
                    trial[block] = np.clip(trial[block] + direction*step*(self.upper[block]-self.lower[block]), self.lower[block], self.upper[block])
                    if np.allclose(trial[block], q[block], rtol=0, atol=1e-15):
                        continue
                    rec = self.evaluate_q(trial, 'block_trial', f'sweep{sweeps}_block{block[0]}_step{step:+.4f}')
                    if not rec['root_ok']:
                        self.ledger['rejected_root'] += 1
                        self._record(role='REJECT_ROOT_NOT_BALANCED', minimax=float(rec['minimax']), root_norm=float(rec['root_norm']))
                        continue
                    if not rec['geom_ok']:
                        self.ledger['rejected_geometry'] += 1
                        self._record(role='REJECT_GEOMETRY', minimax=float(rec['minimax']))
                        continue
                    if rec.get('contact_signature') != cur.get('contact_signature'):
                        self.ledger['contact_set_changes'] += 1
                    if rec['minimax'] < cur['minimax'] - 1e-15:
                        q = trial; cur = rec; cur_ok = True; improved = True
                        self.ledger['accepted_moves'] += 1
                        self.history.append(dict(iteration=len(self.history), role='accepted', minimax=cur['minimax'],
                                                 root_norm=cur['root_norm'], root_ok=cur['root_ok'], geom_ok=cur['geom_ok'],
                                                 unreachable=cur['unreachable'], q=q.tolist(), step=step))
                        self._dump()
                        break
                if stop == 'BUDGET_BEFORE_BLOCK':
                    break
            if stop == 'BUDGET_BEFORE_BLOCK':
                break
            if not improved:
                step /= 2
        if self.ledger['accepted_moves'] == 0 and stop == 'STEP_FLOOR':
            stop = 'ZERO_ACCEPT_ALL_SWEEPS'
        return dict(q=q.tolist(), final=cur, stop=stop, sweeps=sweeps, ledger=self.ledger, history=self.history)
