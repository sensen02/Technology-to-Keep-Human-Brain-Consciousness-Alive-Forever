"""Episode-only body-return selection. Never imported by runtime motor decoding.
Engineering optimization, not biological calibration; no synthetic action targets.
"""
import numpy as np


def body_return(samples, weight, initial_height, *, dt=.0001, duration=.55):
    """Full fixed-window right-endpoint quadrature, including all collapsed states."""
    s = np.asarray(samples, dtype=float)
    count = round(duration / dt)
    if (s.shape != (count, 5) or not np.isfinite(s).all() or
            not np.isfinite(weight) or weight <= 0 or
            not np.isfinite(initial_height) or initial_height <= 0 or
            not np.allclose(s[:, 0], np.arange(1, count+1)*dt, rtol=0, atol=1e-10) or
            np.any(s[:, 1:3] < 0)):
        raise ValueError('complete finite fixed-window body evidence required')
    foot = np.clip(s[:, 1]/weight, 0, 1)
    height = np.clip(s[:, 3]/initial_height, 0, 1)
    upright = np.clip(1-s[:, 4]/.35, 0, 1)
    nonfoot = s[:, 2]
    positive = foot*height*upright
    penalty = 2*(nonfoot > 0) + .1*np.clip(nonfoot, 0, 10)
    return dict(score=float(np.mean(positive-penalty)),
                capped_support_height_upright=float(positive.mean()),
                nonfoot_duration_s=float(np.count_nonzero(nonfoot)*dt),
                nonfoot_count_integral=float(nonfoot.sum()*dt),
                interval_count=count, duration_s=duration)


class AntitheticBodySearch:
    """Six externally capped pairs; select best actually evaluated feasible probe.

    Body returns enter only after completed episodes. Directions use a private RNG;
    runtime neural RNG, simulation state and motor decoding are untouched.
    """
    def __init__(self, weights, *, seed=20261018, radius=.025, nonfoot_tradeoff='warn'):
        self.center = np.asarray(weights, dtype=float).copy()
        if self.center.shape != (66, 5) or not np.isfinite(self.center).all() or np.any((self.center < 0) | (self.center > 4)):
            raise ValueError('330 feasible FIR weights required')
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('positive finite probe radius required')
        if nonfoot_tradeoff not in ('warn', 'stop'):
            raise ValueError('nonfoot_tradeoff must be warn or stop')
        self.nonfoot_tradeoff = nonfoot_tradeoff
        self.rng = np.random.default_rng(seed)
        self.radius = radius
        self.best = None
        self.best_label = None
        self.round = 0

    def probes(self):
        direction = self.rng.choice([-1., 1.], size=self.center.shape)
        return direction, np.clip(self.center+self.radius*direction, 0, 4), np.clip(self.center-self.radius*direction, 0, 4)

    def update(self, candidates):
        """Candidates ordered plus then minus; exact tie retains earlier best.
        Record higher-score/nonfoot tradeoffs separately from the unchanged return.
        Default warning is owner-authorized; explicit stop preserves legacy policy.
        """
        if len(candidates) != 2:
            raise ValueError('one completed antithetic pair required')
        winner = max(candidates, key=lambda c: c['return']['score'])
        comparisons = [c['return'] for c in candidates if c is not winner]
        if self.best is not None:
            comparisons.append(self.best)
        warnings = []
        for old in comparisons:
            worsened = [k for k in ('nonfoot_duration_s', 'nonfoot_count_integral')
                        if winner['return'][k] > old[k]+1e-12]
            if winner['return']['score'] > old['score'] + 1e-12 and worsened:
                warnings.append(dict(kind='higher_return_worse_nonfoot',
                                     winner_label=winner['label'], worsened_metrics=worsened,
                                     winner_return=dict(winner['return']), comparison_return=dict(old)))
        if warnings and self.nonfoot_tradeoff == 'stop':
            raise RuntimeError('metric-hacking pause: higher return with worse nonfoot contact')
        if self.best is None or winner['return']['score'] > self.best['score']:
            self.best = dict(winner['return'])
            self.best_label = winner['label']
            self.center = np.asarray(winner['weights']).copy()
        self.round += 1
        return dict(round=self.round, selected_label=self.best_label, selected_return=self.best,
                    weights=self.center.tolist(), nonfoot_tradeoff_policy=self.nonfoot_tradeoff,
                    warnings=warnings)
