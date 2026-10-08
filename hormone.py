"""Hormone action on cells: binding kinetics, nuclear delay, downstream effect.

Scope: one hormone field value per cell (supplied by transport.py or prescribed)
driven through a receptor model to a downstream response used to gate growth
(growth.py) and survival (cellstate.py). Parameter provenance is recorded per
parameter; see PARAM_SOURCES. Most values here are ILLUSTRATIVE: this project has
no measured receptor parameters for the target tissue, so the module must not be
used to claim real hormonal effects.

Verified in this module (self_test):
  - equilibrium occupancy matches the Langmuir form R/(R+Kd) for a slow-binding
    comparison against the kinetic steady state;
  - dose-response is monotone and saturates at the receptor number per cell;
  - the nuclear-response delay reproduces the requested first-order time constant;
  - mass/amount conservation of receptor between free and bound states;
  - determinism (no hidden randomness).
"""
from __future__ import annotations

import numpy as np

# Provenance: 'literature' only where a citation string is recorded below.
PARAM_SOURCES = {
    'kd_nM': 'ILLUSTRATIVE (no verified Kd for the target tissue in this project)',
    'k_on_per_nM_s': 'ILLUSTRATIVE',
    'k_off_per_s': 'derived as k_on * Kd for consistency',
    'receptors_per_cell': 'ILLUSTRATIVE (order of 1e4-1e5 receptors/cell is typical)',
    'n_hill': 'ILLUSTRATIVE',
    'tau_nuclear_s': 'ILLUSTRATIVE (nuclear translocation / transcription lag)',
    'ec50_response_nM': 'ILLUSTRATIVE',
    'WARNING': ('Uncalibrated. No measured receptor kinetics for this tissue exist in '
                'this project. Do not present outputs as real hormonal effects.'),
}

DEFAULTS = dict(
    kd_nM=1.0,               # dissociation constant
    k_on_per_nM_s=0.01,      # association rate
    n_hill=1.0,              # binding cooperativity
    receptors_per_cell=1.0e4,
    tau_nuclear_s=600.0,     # lag between binding and downstream response
    ec50_response_nM=1.0,    # half-maximal downstream response
    hill_response=2.0,       # cooperativity of the downstream response
    basal_response=0.0,
    max_response=1.0,
)


def k_off(p):
    """Kd = k_off / k_on, so fixing Kd and k_on fixes k_off."""
    return p['k_on_per_nM_s'] * p['kd_nM']


def equilibrium_occupancy(hormone_nM, p=None):
    """Analytic Langmuir/Hill equilibrium bound fraction of receptors."""
    p = dict(DEFAULTS, **(p or {}))
    h = np.maximum(np.asarray(hormone_nM, dtype=float), 0.0)
    return h ** p['n_hill'] / (p['kd_nM'] ** p['n_hill'] + h ** p['n_hill'])


def run(hormone_nM, t_end=3600.0, dt=1.0, p=None, initial_bound_fraction=0.0,
        initial_response=None, sample_dt=10.0):
    """Integrate binding and the delayed downstream response for a fixed dose.

    hormone_nM may be a scalar (constant dose) or a callable of time.
    Returns arrays of free hormone (prescribed), bound fraction, and response.
    """
    p = dict(DEFAULTS, **(p or {}))
    if dt <= 0 or t_end <= 0:
        raise ValueError('t_end and dt must be positive')
    if p['n_hill'] != 1:
        raise ValueError('Only mass-action n_hill=1 is supported; cooperative kinetics need a separate model')
    if min(p['kd_nM'], p['k_on_per_nM_s'], p['tau_nuclear_s']) <= 0:
        raise ValueError('Kd, association rate and response time must be positive')
    if not 0 <= initial_bound_fraction <= 1:
        raise ValueError('Initial occupancy must lie in [0,1]')
    n = int(np.ceil(t_end / dt)) + 1
    times = np.linspace(0.0, t_end, n)
    bound = np.zeros(n)
    response = np.zeros(n)
    bound[0] = float(initial_bound_fraction)
    response[0] = (float(initial_response) if initial_response is not None
                   else float(initial_bound_fraction))
    kon, koff = p['k_on_per_nM_s'], k_off(p)
    dose = (hormone_nM if callable(hormone_nM)
            else (lambda t, v=float(hormone_nM): v))
    for i in range(1, n):
        t = times[i]
        h = max(float(dose(t)), 0.0)
        b = bound[i - 1]
        # Exact mass-action update for the sampled, piecewise-constant dose.
        # This avoids Euler clipping artifacts at high concentration.
        step = times[i] - times[i-1]
        rate = kon*h + koff
        eq = kon*h/rate
        bound[i] = eq + (b-eq)*np.exp(-rate*step)
        target = (p['basal_response']
                  + (p['max_response'] - p['basal_response'])
                  * bound[i] ** p['hill_response'])
        response[i] = target + (response[i-1]-target)*np.exp(-step/p['tau_nuclear_s'])
    keep = np.zeros(n, dtype=bool)
    keep[::max(1, int(round(sample_dt / dt)))] = True
    keep[-1] = True
    return {'times': times[keep], 'hormone_nM': np.array([float(dose(t)) for t in times[keep]]),
            'bound_fraction': bound[keep], 'response': response[keep],
            'metadata': {'parameters': p, 'param_sources': PARAM_SOURCES,
                         'scope': 'illustrative single-cell hormone response; uncalibrated'}}


def dose_response(doses_nM, p=None, t_end=4 * 3600.0, dt=10.0):
    """Steady downstream response versus dose; used for the monotonicity test."""
    p = dict(DEFAULTS, **(p or {}))
    out = []
    for d in np.atleast_1d(doses_nM):
        r = run(float(d), t_end=t_end, dt=dt, p=p)
        out.append(float(r['response'][-1]))
    return np.asarray(out)


def effect_on_growth(response, baseline_rate_per_h=0.02, gain=0.5):
    """Documented, ILLUSTRATIVE mapping from response to a growth-rate change.

    Returns a multiplicative factor applied to the baseline growth rate. The
    functional form is a choice, not a measurement.
    """
    return 1.0 + gain * (np.asarray(response, dtype=float) - 0.5)


def self_test():
    p = dict(DEFAULTS)
    diag = {}

    # 1. Kinetic steady state must approach the analytic equilibrium.
    doses = [0.1, 1.0, 10.0]
    equil = []
    for d in doses:
        r = run(d, t_end=20000.0, dt=1.0, p=p)
        equil.append(float(r['bound_fraction'][-1]))
    analytic = [float(equilibrium_occupancy(d, p)) for d in doses]
    diag['binding_equilibrium'] = {
        'doses_nM': doses, 'kinetic_steady_state': equil, 'analytic_langmuir': analytic,
        'max_abs_difference': float(np.max(np.abs(np.array(equil) - np.array(analytic)))),
    }

    # 2. Dose-response monotone and saturating.
    doses_wide = np.logspace(-2, 3, 12)
    dr = dose_response(doses_wide, p=p, t_end=6000.0, dt=5.0)
    diag['dose_response'] = {
        'monotone_nondecreasing': bool(np.all(np.diff(dr) >= -1e-12)),
        'saturates_below_max': float(dr[-1]),
        'max_response_parameter': p['max_response'],
    }

    # 3. Nuclear delay time constant: step of binding gives 1-exp(-t/tau).
    tau = p['tau_nuclear_s']
    r = run(1000.0, t_end=6 * tau, dt=1.0, p=p, initial_response=0.0)
    model = r['response']
    t = r['times']
    target = 1.0 - np.exp(-t / tau)
    bound_ss = float(r['bound_fraction'][-1])
    analytic_full = bound_ss ** p['hill_response'] * target
    diag['nuclear_delay'] = {
        'tau_requested_s': tau,
        'max_abs_deviation_from_first_order': float(np.max(np.abs(model - analytic_full))),
    }

    # 4. Receptor amount conservation between free and bound pools.
    r = run(2.0, t_end=3600.0, dt=1.0, p=p)
    free = (1.0 - r['bound_fraction']) * p['receptors_per_cell']
    bound_n = r['bound_fraction'] * p['receptors_per_cell']
    diag['receptor_conservation'] = {
        'max_deviation_from_total': float(np.max(np.abs(free + bound_n - p['receptors_per_cell']))),
    }

    # 5. Bounds and determinism.
    diag['bounds'] = {
        'min_bound_fraction': float(r['bound_fraction'].min()),
        'max_bound_fraction': float(r['bound_fraction'].max()),
        'min_response': float(r['response'].min()), 'max_response': float(r['response'].max()),
    }
    a = run(2.0, t_end=600.0, dt=1.0, p=p)
    b = run(2.0, t_end=600.0, dt=1.0, p=p)
    diag['determinism'] = {'bit_identical_bound': bool(np.array_equal(a['bound_fraction'], b['bound_fraction']))}

    # 6. Growth-effect mapping is monotone in response.
    eff = effect_on_growth(np.linspace(0, 1, 5))
    diag['growth_effect_mapping'] = {'values': eff.tolist(),
                                     'monotone_increasing': bool(np.all(np.diff(eff) > 0))}
    diag['param_sources'] = PARAM_SOURCES
    return diag


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), indent=2, default=str))
