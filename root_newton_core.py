"""Engine-free Gauss-Newton core for the S177 six-row root wrench solve.

Design constraints fixed by the owner's S177 mandate:
  * acceptance decisions use the CURRENT iterate's residual only; the logging layer
    must never mutate the quantity that the acceptance test compares against
    (that mutation was the S176 defect);
  * global best is bookkeeping only and is never consulted by the accept test;
  * root (6 rows) and joint (42 rows) residuals are never merged into one number;
  * every forward, cache hit, jacobian, FD probe and line-search trial is counted
    separately, and an incomplete Jacobian is refused before any probe is spent.
"""
from pathlib import Path
import json
import numpy as np


class Budget(Exception):
    pass


class RefusedJacobian(Exception):
    pass


class RootNewton:
    def __init__(self, callback, lower, upper, cap=600, output=None, fd_step=1e-4,
                 max_iter=60, tol=1e-8, max_halvings=5, name='', dim=None):
        self.callback = callback
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.dim = int(dim) if dim is not None else int(self.lower.size)
        assert self.lower.shape == (self.dim,) and self.upper.shape == (self.dim,), (self.lower.shape, self.dim)
        assert np.all(self.lower < self.upper)
        self.cap = int(cap)
        self.output = Path(output) if output else None
        self.fd_step = float(fd_step)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_halvings = int(max_halvings)
        self.name = name
        self.ledger = dict(model_load=0, forward=0, cache_hits=0, newton_iterations=0, jacobians=0,
                           fd_probes=0, line_search_trials=0, line_search_halvings=0,
                           pose_evaluations=0, accepted_iterations=0, contact_set_changes=0)
        self.cache = {}
        self.events = []
        self.best = None          # bookkeeping only; never used by the accept test
        self.history = []

    # ---------- bookkeeping ----------
    @staticmethod
    def _jsonable(v):
        if isinstance(v, dict): return {str(k): RootNewton._jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [RootNewton._jsonable(x) for x in v]
        if isinstance(v, np.generic): v = v.item()
        if isinstance(v, float) and not np.isfinite(v): return None
        if isinstance(v, (int, float, bool, str)) or v is None: return v
        return str(v)

    def _dump(self):
        if self.output:
            payload = self._jsonable(dict(name=self.name, ledger=self.ledger,
                                          best=self.best['summary'] if self.best else None,
                                          events=self.events, history=self.history))
            self.output.write_text(json.dumps(payload, indent=2, allow_nan=False))

    def _record(self, **event):
        self.events.append(dict(event_id=len(self.events)+1, **event))
        self._dump()

    # ---------- one physics evaluation ----------
    def evaluate(self, p, role, iteration=None, note='', base_iterate=None):
        key = np.asarray(p, dtype=np.float64).tobytes()
        if key in self.cache:
            self.ledger['cache_hits'] += 1
            rec = self.cache[key]
            self._record(role=role, iteration=iteration, note=note, cache_hit=True,
                         forward=self.ledger['forward'], p=p.tolist())
            return rec
        if self.ledger['forward'] + 1 > self.cap:
            self._record(role='BUDGET_REFUSED', iteration=iteration, note=note,
                         forward=self.ledger['forward'])
            raise Budget('forward cap reached before evaluation')
        rec = self.callback(np.asarray(p, dtype=float))
        self.ledger['forward'] += 1
        self.ledger['pose_evaluations'] += 1
        rec = dict(rec)
        rec['p'] = np.asarray(p, dtype=float).copy()
        rec['norm_inf'] = float(abs(rec['residual6']).max())
        self.cache[key] = rec
        self._record(role=role, iteration=iteration, note=note, cache_hit=False,
                     forward=self.ledger['forward'], p=p.tolist(), norm_inf=rec['norm_inf'],
                     contact_signature=str(rec.get('contact_signature')), base_iterate=base_iterate)
        if self.best is None or rec['norm_inf'] < self.best['norm_inf']:
            self.best = dict(rec)
            self.best['summary'] = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                    for k, v in rec.items() if k != 'contact_sets'}
            self._dump()
        return rec

    # ---------- 6x6 finite-difference Jacobian, atomic ----------
    def jacobian(self, p, iteration, base_record=None):
        if self.ledger['forward'] + self.dim > self.cap:
            self._record(role='JACOBIAN_REFUSED_BUDGET', iteration=iteration,
                         forward=self.ledger['forward'], required=self.dim,
                         remaining=self.cap - self.ledger['forward'])
            raise RefusedJacobian('insufficient forward budget for a complete 6-column Jacobian')
        base = base_record if base_record is not None else self.evaluate(p, 'jacobian_base', iteration)
        cols = []
        steps = []
        for j in range(self.dim):
            h = self.fd_step
            y = np.asarray(p, dtype=float).copy()
            # choose an INWARD direction: at/near a frozen bound the step must point inside
            if y[j] + h <= self.upper[j]:
                pass
            elif y[j] - h >= self.lower[j]:
                h = -h
            else:
                raise RefusedJacobian(f'parameter {j} has no inward step inside frozen bounds')
            y[j] += h
            if not (self.lower[j] <= y[j] <= self.upper[j]):
                raise RefusedJacobian(f'parameter {j} cannot be perturbed inside frozen bounds')
            rec = self.evaluate(y, 'difference_probe', iteration, note=f'param{j}', base_iterate=True)
            self.ledger['fd_probes'] += 1
            cols.append((rec['residual6'] - base['residual6']) / h)
            steps.append(float(h))
        J = np.column_stack(cols)
        assert J.shape[1] == self.dim and J.shape[0] == len(base['residual6'])
        self.ledger['jacobians'] += 1
        return J, base, steps

    # ---------- solve ----------
    def solve(self, p0):
        p_cur = np.asarray(p0, dtype=float).copy()
        cur = self.evaluate(p_cur, 'initial', 0)
        cur_norm = cur['norm_inf']
        self.history.append(dict(iteration=0, role='initial', norm_inf=cur_norm, residual6=cur['residual6'].tolist(), p=p_cur.tolist()))
        stop = 'MAX_ITER'
        for iteration in range(1, self.max_iter + 1):
            if cur_norm <= self.tol:
                stop = 'TOLERANCE_REACHED'
                break
            try:
                J, base, steps = self.jacobian(p_cur, iteration, base_record=cur)
            except RefusedJacobian as exc:
                stop = f'JACOBIAN_REFUSED:{exc}'
                break
            try:
                sv = np.linalg.svd(J, compute_uv=False)
            except np.linalg.LinAlgError:
                sv = np.full(6, np.nan)
            smax = float(np.max(sv)) if np.all(np.isfinite(sv)) else float('nan')
            rank = int(np.sum(sv > max(smax, 0.)*1e-10)) if smax and np.isfinite(smax) else 0
            cond = float(smax/sv[-1]) if np.isfinite(sv[-1]) and sv[-1] > 0 else float('inf')
            r6 = base['residual6']
            try:
                P = J @ np.linalg.pinv(J)
                controllable = P @ r6
                uncontrollable = r6 - controllable
            except np.linalg.LinAlgError:
                controllable = np.full(6, np.nan); uncontrollable = np.full(6, np.nan)
            self.ledger['newton_iterations'] += 1
            self._record(role='JACOBIAN_RANK', iteration=iteration, singular_values=sv.tolist(), rank=rank, cond=cond,
                         controllable_inf=float(np.max(abs(controllable))) if np.all(np.isfinite(controllable)) else None,
                         uncontrollable_inf=float(np.max(abs(uncontrollable))) if np.all(np.isfinite(uncontrollable)) else None,
                         uncontrollable_vector=uncontrollable.tolist())
            try:
                delta = -np.linalg.lstsq(J, base['residual6'], rcond=None)[0]
            except np.linalg.LinAlgError:
                stop = 'LINALG_ERROR'
                break
            if not np.all(np.isfinite(delta)):
                stop = 'NONFINITE_STEP'
                break
            accepted = False
            trial_norm = None
            for halving in range(self.max_halvings + 1):
                scale = 0.5 ** halving
                p_try = np.clip(p_cur + scale * delta, self.lower, self.upper)
                clipped = bool(np.any(p_try == self.lower) or np.any(p_try == self.upper))
                self.ledger['line_search_trials'] += 1
                if halving:
                    self.ledger['line_search_halvings'] += 1
                try:
                    rec = self.evaluate(p_try, 'line_search_trial', iteration,
                                        note=f'scale={scale:g}' + ('_BOUND_CLIPPED' if clipped else ''))
                except Budget:
                    stop = 'BUDGET_STOP_IN_LINE_SEARCH'
                    break
                trial_norm = rec['norm_inf']
                changed = rec.get('contact_signature') != cur.get('contact_signature')
                if changed:
                    self.ledger['contact_set_changes'] += 1
                self._record(role='LINE_SEARCH_DECISION', iteration=iteration, scale=scale,
                             previous_norm=cur_norm, trial_norm=trial_norm,
                             contact_set_changed=bool(changed), clipped=clipped,
                             accepted=bool(trial_norm < cur_norm))
                if trial_norm < cur_norm:
                    accepted = True
                    self.ledger['accepted_iterations'] += 1
                    p_cur, cur, cur_norm = p_try, rec, trial_norm
                    self.history.append(dict(iteration=iteration, role='accepted', p=p_cur.tolist(), residual6=cur['residual6'].tolist(), norm_inf=cur_norm,
                                             scale=scale, cond=cond, rank=rank, singular_values=sv.tolist(),
                                             uncontrollable_inf=float(np.max(abs(uncontrollable))),
                                             jacobian_step=steps,
                                             delta_norm=float(abs(delta).max()),
                                             contact_signature=str(rec.get('contact_signature'))))
                    break
            if stop.startswith('BUDGET_STOP'):
                break
            if not accepted:
                self.history.append(dict(iteration=iteration, role='no_improvement', norm_inf=cur_norm,
                                         cond=cond, best_trial=trial_norm))
                stop = 'NO_MONOTONE_IMPROVEMENT'
                break
        if cur_norm <= self.tol:
            stop = 'TOLERANCE_REACHED'
        try:
            Jf, basef, _ = self.jacobian(p_cur, 999, base_record=cur)
            Pf = Jf @ np.linalg.pinv(Jf)
            contr = Pf @ cur['residual6']; uncont = cur['residual6'] - contr
            final_decomp = dict(singular_values=np.linalg.svd(Jf, compute_uv=False).tolist(),
                                rank=int(np.sum(np.linalg.svd(Jf, compute_uv=False) > max(np.linalg.svd(Jf, compute_uv=False).max(), 0.)*1e-10)),
                                controllable=contr.tolist(), uncontrollable=uncont.tolist(),
                                controllable_inf=float(np.max(abs(contr))), uncontrollable_inf=float(np.max(abs(uncont))))
        except (RefusedJacobian, Budget):
            final_decomp = None
        return dict(p=p_cur.tolist(), residual6=cur['residual6'].tolist(), norm_inf=cur_norm,
                    stop=stop, ledger=self.ledger, record=cur, history=self.history, final_decomposition=final_decomp)
