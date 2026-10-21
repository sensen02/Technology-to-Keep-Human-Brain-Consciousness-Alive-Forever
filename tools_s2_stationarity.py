"""Pure dense-array stationarity arithmetic. No simulation or solver PASS threshold."""
import numpy as np

UNITS = ('translation/model_force', 'rotation/model_torque')


def real_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(name+' must be finite real numeric')
    return a.astype(float, copy=True)


def summarize(error, denominator):
    valid = denominator != 0
    peak = float(np.max(np.abs(error))) if error.size else None
    rms = None if peak is None else 0.0 if peak == 0 else peak*float(np.sqrt(np.mean((error/peak)**2)))
    return dict(count=int(error.size), absmax=peak,
                rms=rms,
                relative_undefined=int(np.sum(~valid)),
                relative_absmax=float(np.max(np.abs(error[valid])/denominator[valid])) if np.any(valid) else None)


def analyze(M, J, qacc, qacc_smooth, qfrc_smooth, qfrc_constraint, force, dof_units):
    M,J,qacc,qacc_smooth,qfrc_smooth,qfrc_constraint,force = [real_array(v,k) for v,k in zip(
        (M,J,qacc,qacc_smooth,qfrc_smooth,qfrc_constraint,force),
        ('M','J','qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','force'))]
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[0] != M.shape[1]:
        raise ValueError('M shape')
    nv=M.shape[0]
    if not np.allclose(M,M.T,rtol=1e-12,atol=1e-14): raise ValueError('M asymmetric')
    try: np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc: raise ValueError('M not positive definite') from exc
    if J.ndim!=2 or J.shape[1]!=nv or force.shape!=(J.shape[0],): raise ValueError('J/force shape')
    if any(v.shape!=(nv,) for v in (qacc,qacc_smooth,qfrc_smooth,qfrc_constraint)): raise ValueError('DOF vector shape')
    labels=np.asarray(dof_units)
    if labels.shape!=(nv,) or not all(v in UNITS for v in labels): raise ValueError('DOF units')
    # Overflow must not silently create an apparently valid report.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            Ma=M@qacc; Ms=M@qacc_smooth; mapped=J.T@force
            mapping=qfrc_constraint-mapped
            direct=Ma-qfrc_smooth-qfrc_constraint
            rows=Ma-qfrc_smooth-mapped
            smooth=Ms-qfrc_smooth
            closure=rows-direct-mapping
            denominators=dict(g_rows=np.maximum.reduce([np.abs(Ma),np.abs(qfrc_smooth),np.abs(mapped)]),
                              g_direct=np.maximum.reduce([np.abs(Ma),np.abs(qfrc_smooth),np.abs(qfrc_constraint)]),
                              mapping_error=np.maximum(np.abs(qfrc_constraint),np.abs(mapped)),
                              smooth_error=np.maximum(np.abs(Ms),np.abs(qfrc_smooth)))
            arrays=dict(Ma=Ma,Mqacc_smooth=Ms,mapped=mapped,g_rows=rows,g_direct=direct,mapping_error=mapping,smooth_error=smooth,closure_error=closure)
            if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('nonfinite arithmetic')
            groups={}
            for label in UNITS:
                mask=labels==label
                groups[label]={k:summarize(arrays[k][mask],den[mask]) for k,den in denominators.items()}
        except FloatingPointError as exc: raise ValueError('arithmetic overflow') from exc
    return dict(policy='REPORT_ONLY',arrays=arrays,denominators=denominators,groups=groups,
                limitations='not full KKT, static equilibrium, convergence, biological validation or behavior')
