"""Sensed curriculum cycle proxy, NOT original walking proof or an action generator.

Canonical sample: time [s], loads [native force], padz [native length],
relativefoot [6,3 native length in thorax frame]. Order LF LH LM RF RH RM.
Alternatively loads_W supplies force/bodyweight directly. Geometry is copied.

Each actual 1ms transition exposes potential and local shape gamma*new-old.
PARENT 5ms CONTRACT: use gamma_policy*potential_after-potential_before ONCE;
ignore local shape rewards when doing this. Set potential_after=0 on terminal
(including deadline). Add completion_reward separately: the parent may sum the
.10 event scores within its 5ms block as block-lumped discrete bonuses (not a
physical reward rate); this convention is separate from exact potential shaping. finalize zeros pending
potential, reports previous_potential; its shape_reward is a correction only for
clients summing local shape: add it to the SAME LAST 1ms reward. Never combine
this correction with the parent's zero-terminal-potential calculation.

With local aggregation use sum(gamma**j*r_j), then gamma**n bootstrap. Raw sums
are not exactly equivalent. For gamma=1 closed failed attempts telescope to0;
for gamma<1 the discounted telescope is exact and pending progress cannot farm.
No dwell/phase state generates actions. Thresholds are curriculum proxies only.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

LEG_ORDER = ('LF', 'LH', 'LM', 'RF', 'RH', 'RM')

@dataclass(frozen=True)
class CycleConfig:
    weight: float = 24.446354755878005
    dt: float = .001
    gamma: float = math.exp(-.001)
    loaded: float = .1
    unloaded: float = .05
    ground: float = .004
    clearance: float = .015
    advance: float = .03
    loaded_dwell: float = .010
    unload_dwell: float = .005
    timeout: float = .8
    phi_cap: float = .05
    completion_bonus: float = .10

class CycleTracker:
    def __init__(self, config=None):
        self.config = config if isinstance(config, CycleConfig) else CycleConfig(**(config or {}))
        c = self.config
        vals = asdict(c)
        if any(not math.isfinite(float(v)) for v in vals.values()):
            raise ValueError('config must be finite')
        if not (c.weight > 0 and c.dt > 0 and 0 < c.gamma <= 1 and
                0 <= c.unloaded < c.loaded and c.clearance > c.ground and
                c.clearance > 0 and c.advance > 0 and c.loaded_dwell > 0 and
                c.unload_dwell > 0 and c.timeout > c.unload_dwell and
                0 < c.phi_cap <= c.completion_bonus and c.completion_bonus > 0):
            raise ValueError('invalid cycle config')
        self.reset()

    def _leg(self):
        return dict(armed=False, baseline=0., load_sum=0., load_samples=0,
                    load_dwell=0., low_dwell=0., active=False,
                    start=None, x0=None, peak_lift=0., peak_advance=0.,
                    reload_dwell=0., unloaded_duration=0., phi=0.)

    def reset(self, start_time=0.):
        if not math.isfinite(float(start_time)):
            raise ValueError('nonfinite start_time')
        self.time = float(start_time)
        self.legs = [self._leg() for _ in LEG_ORDER]
        self.potential = 0.
        self._last = None
        self._finalized = False
        self.counts = dict(completed=0, aborted=0, timeout=0, recontact=0)
        return self

    def _result(self, shape=0., bonus=0., events=None, duplicate=False, **extra):
        return dict(shape_reward=float(shape), completion_reward=float(bonus),
                    total_reward=float(shape+bonus), potential=float(self.potential),
                    events=events or [], diagnostics=dict(
                        time=self.time, gamma=self.config.gamma, dt=self.config.dt,
                        per_leg_potential=[float(s['phi']) for s in self.legs],
                        active=[s['active'] for s in self.legs],
                        counts=self.counts.copy(), duplicate=duplicate,
                        label='curriculum proxy; not frozen walking proof', **extra))

    def _sample(self, sample):
        t = float(sample['time'])
        normalized = 'loads_W' in sample
        loads = np.array(sample['loads_W'] if normalized else sample['loads'], dtype=float, copy=True)
        z = np.array(sample['padz'], dtype=float, copy=True)
        foot = np.array(sample.get('relativefoot', sample.get('foot')), dtype=float, copy=True)
        if (loads.shape != (6,) or z.shape != (6,) or foot.shape != (6,3)
            or not math.isfinite(t) or not all(np.isfinite(a).all() for a in (loads,z,foot))
            or (loads < 0).any()):
            raise ValueError('finite time/loads6>=0/padz6/relativefoot6x3 required')
        if not normalized:
            loads /= self.config.weight
        return t, loads, z, foot

    def update(self, sample):
        t, loads, z, foot = self._sample(sample)
        if self._finalized:
            raise RuntimeError('reset required after finalize')
        dt = t-self.time
        if abs(dt) < 1e-10:
            if self._last is not None:
                if not all(np.array_equal(a,b) for a,b in zip(self._last,(loads,z,foot))):
                    raise ValueError('conflicting sample at repeated time')
                return self._result(duplicate=True)
            self._last = (loads,z,foot)
            return self._result()
        if not math.isclose(dt, self.config.dt, rel_tol=0., abs_tol=1e-9):
            raise ValueError('update requires exactly next 1ms/config.dt sample')
        c = self.config
        old = self.potential
        events = []
        bonus = 0.
        for i,s in enumerate(self.legs):
            loaded = loads[i] >= c.loaded and z[i] <= c.ground
            low = loads[i] <= c.unloaded
            if not s['active']:
                if not s['armed']:
                    s['load_dwell'] = s['load_dwell']+dt if loaded else 0.
                    if loaded:
                        s['load_sum'] += float(loads[i]); s['load_samples'] += 1
                    else:
                        s['load_sum'] = 0.; s['load_samples'] = 0
                    if s['load_dwell']+1e-10 >= c.loaded_dwell:
                        s['armed'] = True
                        s['baseline'] = max(c.loaded, s['load_sum']/s['load_samples'])
                    continue
                # Frozen measured loaded baseline: smooth reward before unload threshold.
                # This potential disappears again if load returns; it is not an event.
                s['phi'] = .2*c.phi_cap*float(np.clip(
                    (s['baseline']-loads[i])/(s['baseline']-c.unloaded), 0., 1.))
                if low:
                    if s['low_dwell'] == 0:
                        s['start'] = t-dt
                        s['x0'] = float(foot[i,0])
                    s['low_dwell'] += dt
                    s['peak_lift'] = max(s['peak_lift'], float(z[i]))
                    if z[i] > c.clearance:
                        s['peak_advance'] = max(s['peak_advance'], float(foot[i,0])-s['x0'])
                    if s['low_dwell']+1e-10 >= c.unload_dwell:
                        s['active'] = True
                        s['unloaded_duration'] = s['low_dwell']
                else:
                    s['low_dwell'] = 0.
                    s['peak_lift'] = 0.; s['peak_advance'] = 0.
                if not s['active']:
                    continue
            elif low:
                s['unloaded_duration'] += dt
            age = t-s['start']
            if low:
                s['peak_lift'] = max(s['peak_lift'], float(z[i]))
                # Forward displacement counts ONLY while truly unloaded AND clear.
                if z[i] > c.clearance:
                    s['peak_advance'] = max(s['peak_advance'], float(foot[i,0])-s['x0'])
            qualified = s['peak_lift'] > c.clearance and s['peak_advance']+1e-12 >= c.advance
            reason = None
            if age > c.timeout+1e-10:
                reason = 'timeout'
            elif loaded and not qualified:
                reason = 'recontact'
            if reason:
                events.append(dict(type='abort', leg=LEG_ORDER[i], time=t,
                                   start_time=s['start'], reason=reason,
                                   lift=s['peak_lift'], advance=s['peak_advance'],
                                   unloaded_duration=s['unloaded_duration'], duration=age))
                self.counts['aborted'] += 1
                self.counts[reason] += 1
                self.legs[i] = self._leg()
                continue
            s['reload_dwell'] = s['reload_dwell']+dt if loaded else 0.
            if qualified and s['reload_dwell']+1e-10 >= c.loaded_dwell:
                events.append(dict(type='complete', leg=LEG_ORDER[i], time=t,
                                   start_time=s['start'], lift=s['peak_lift'],
                                   advance=s['peak_advance'], reload_dwell=s['reload_dwell'],
                                   initial_loaded_dwell=c.loaded_dwell,
                                   unload_dwell=s['low_dwell'],
                                   unloaded_duration=s['unloaded_duration'], duration=age,
                                   bonus=c.completion_bonus))
                self.counts['completed'] += 1
                bonus += c.completion_bonus
                self.legs[i] = self._leg()
                # Completed grounded dwell can arm the NEXT actual unload;
                # no reward from holding or leg-specific refractory scheduling.
                self.legs[i]['armed'] = True
                self.legs[i]['load_dwell'] = s['reload_dwell']
                self.legs[i]['baseline'] = max(c.loaded, float(loads[i]))
                continue
            height_progress = np.clip(float(z[i])/c.clearance, 0., 1.) if low else 0.
            advance_progress = (np.clip((float(foot[i,0])-s['x0'])/c.advance,0.,1.)
                                if low and z[i] > c.clearance else 0.)
            # Current physical progress can reverse; no positive repeated milestone payout.
            s['phi'] = c.phi_cap*(.2 + .4*height_progress + .4*advance_progress)
            if qualified and loaded:
                s['phi'] = c.phi_cap*(.8+.2*np.clip(s['reload_dwell']/c.loaded_dwell,0.,1.))
        self.time = t
        self._last = (loads,z,foot)
        self.potential = float(sum(s['phi'] for s in self.legs))
        return self._result(c.gamma*self.potential-old, bonus, events)

    def measure(self, core):
        """Call once after every ACTUAL 1ms step; never advances/writes physics."""
        m = core.measure()
        com = np.array(core.d.subtree_com[core.th],float,copy=True)
        return self.update(dict(time=float(core.d.time), loads=np.array(m['loads'],copy=True),
                                padz=np.array(m['padz'],copy=True),
                                relativefoot=np.array(m['foot'],copy=True), com=com))

    def finalize(self, terminal=True):
        """Zero terminal potential; parent reads previous_potential for accounting.

        For local 1ms rewards only, add returned shape to last transition. Parent
        boundary-potential users ignore shape and use after=0. Idempotent.
        """
        if not terminal:
            raise ValueError('finalize is only for actual terminal/deadline')
        if self._finalized:
            return self._result(previous_potential=0., already_finalized=True)
        prev = self.potential
        events = [dict(type='abort', reason='terminal', leg=LEG_ORDER[i], time=self.time,
                       start_time=s['start'], lift=s['peak_lift'], advance=s['peak_advance'],
                       unloaded_duration=s['unloaded_duration'])
                  for i,s in enumerate(self.legs) if s['active']]
        self.counts['aborted'] += len(events)
        self.legs = [self._leg() for _ in LEG_ORDER]
        self.potential = 0.
        self._finalized = True
        return self._result(-self.config.gamma*prev, events=events,
                            previous_potential=prev, already_finalized=False)
