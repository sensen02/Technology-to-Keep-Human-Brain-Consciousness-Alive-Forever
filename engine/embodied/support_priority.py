"""Offline support-first evaluation. No controller, pose updates or muscle commands.
A clean-foot-support horizon is an experiment requirement, not universal animal gait.
"""
import numpy as np


def evaluate_support(times_s,foot_fz,nonfoot_positive_fz,com_height,*,weight,h0,horizon_s,
                     force_epsilon_weight=1e-8,early_tau_s=.02,minimum_mean_foot_load=.8,
                     minimum_height_fraction=.8,assistance=None):
    """Endpoint bins from t=0; samples must completely cover the declared horizon.
    Any positive nonfoot load above a declared numerical deadband makes clean horizon
    infeasible. Average motion/pose improvements cannot cancel that violation.
    """
    t=np.asarray(times_s,dtype=float);foot=np.asarray(foot_fz,dtype=float);nf=np.asarray(nonfoot_positive_fz,dtype=float);height=np.asarray(com_height,dtype=float)
    if t.ndim!=1 or len(t)<1 or foot.shape!=(len(t),6) or nf.shape!=t.shape or height.shape!=t.shape:raise ValueError('Expected t endpoints, six signed foot forces, nonfoot positive sum and COMheight')
    if not all(np.isfinite(a).all() for a in [t,foot,nf,height]) or not all(np.isfinite(v) for v in [weight,h0,horizon_s,force_epsilon_weight,early_tau_s]):raise ValueError('Finite evidence required')
    if weight<=0 or h0<=0 or horizon_s<=0 or force_epsilon_weight<0 or early_tau_s<=0:raise ValueError('Positive physical references/horizon and nonnegative force deadband required')
    if not all(np.isfinite(v) and v>0 for v in [minimum_mean_foot_load,minimum_height_fraction]):raise ValueError('Explicit positive foot-load/height engineering gates required')
    dt=np.diff(np.r_[0.,t])
    if (dt<=0).any() or abs(float(t[-1])-horizon_s)>max(1e-10,horizon_s*1e-8):raise ValueError('Strictly increasing endpoints must cover the complete declared horizon')
    if (nf<0).any():raise ValueError('Nonfoot input is sum of positive contact forces, never signed aggregate')
    if assistance is not None:
        aux=np.asarray(assistance,dtype=float)
        if aux.shape[0]!=len(t) or not np.isfinite(aux).all():raise ValueError('One finite assistance row per endpoint required')
        if np.any(aux!=0):raise ValueError('Undeclared assistance disqualifies neural-only evaluation')
    hit=nf/weight>force_epsilon_weight;indices=np.flatnonzero(hit);first=None if not len(indices) else float(t[indices[0]])
    # Survived duration is interval start before first failed endpoint, a lower bound.
    clean_duration=horizon_s if first is None else float(t[indices[0]]-dt[indices[0]])
    w=dt*np.exp(-t/early_tau_s);deficit=(foot.sum(1)/weight-1)**2;nonfoot=nf/weight;fall=((height-h0)/h0)**2
    meanload=float(np.dot(dt,foot.sum(1)/weight)/horizon_s);minheight=float(height.min()/h0)
    gates=dict(no_positive_nonfoot=first is None,mean_foot_load=meanload>=minimum_mean_foot_load,minimum_height=minheight>=minimum_height_fraction)
    return dict(clean_horizon_feasible=all(gates.values()),feasibility_gates=gates,first_nonfoot_s=first,event_right_censored=first is None,
                observed_clean_lower_bound_s=float(clean_duration),declared_horizon_s=float(horizon_s),
                nonfoot_force_epsilon_weight=float(force_epsilon_weight),early_weight_tau_s=float(early_tau_s),positive_nonfoot_endpoint_count=int(hit.sum()),
                positive_nonfoot_time_s=float(dt[hit].sum()),
                early_weighted_support_cost=float(np.dot(w,deficit+nonfoot+fall)/w.sum()),
                early_weighted_foot_cost=float(np.dot(w,deficit)/w.sum()),
                early_weighted_nonfoot_cost=float(np.dot(w,nonfoot)/w.sum()),
                early_weighted_height_cost=float(np.dot(w,fall)/w.sum()),
                mean_foot_load=meanload,min_height_fraction=minheight,minimum_mean_foot_load_gate=float(minimum_mean_foot_load),minimum_height_fraction_gate=float(minimum_height_fraction))


def select_support_first(records):
    """Return admitted choice only among declared clean-horizon feasible candidates.
    If none qualify, provide a labelled diagnostic leader, never an admitted policy.
    Records carry label/support/motion_loss, all on one identical horizon.
    """
    if not records:raise ValueError('At least one completed candidate required')
    horizon=None;labels=set();common_settings=None
    for record in records:
        if record['label'] in labels:raise ValueError('Candidate labels must be unique')
        labels.add(record['label']);s=record['support']
        if not np.isfinite(record['motion_loss']) or record['motion_loss']<0:raise ValueError('Finite nonnegative motion loss required')
        if horizon is None:horizon=s['declared_horizon_s']
        if s['declared_horizon_s']!=horizon:raise ValueError('Do not rank different observation horizons')
        settings=tuple(s[k] for k in ['nonfoot_force_epsilon_weight','early_weight_tau_s','minimum_mean_foot_load_gate','minimum_height_fraction_gate'])
        if common_settings is None:common_settings=settings
        if settings!=common_settings:raise ValueError('Do not rank different numerical or engineering gates')
    feasible=[r for r in records if r['support']['clean_horizon_feasible']]
    if feasible:
        best=min(feasible,key=lambda r:(r['motion_loss'],r['support']['early_weighted_support_cost']))
        return dict(status='CLEAN_HORIZON_CANDIDATE_ONLY',admitted_label=best['label'],diagnostic_leader=None,feasible_candidates=len(feasible),walking_certified=False)
    # Time before loss of clean support is primary; motion cannot compensate.
    best=min(records,key=lambda r:(-r['support']['observed_clean_lower_bound_s'],r['support']['early_weighted_support_cost'],r['motion_loss']))
    return dict(status='NO_CLEAN_HORIZON_CANDIDATE',admitted_label=None,diagnostic_leader=best['label'],feasible_candidates=0,walking_certified=False)
