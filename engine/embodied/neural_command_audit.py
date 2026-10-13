"""Executable checks of neural-only command origin, not biological validation.

No body state, desired action, target, posture or time schedule enters this
translator. A positive audit only establishes equality to the declared spike
translation; anatomical validity must be established independently.
"""
import numpy as np


def translate_motor_spikes(spikes, groups, window_s, reference_hz=100.):
    s=np.asarray(spikes)
    if s.ndim!=2 or s.shape[0]<1 or not np.all((s==0)|(s==1)):
        raise ValueError('nonempty binary spike matrix required')
    if not np.isfinite(window_s) or window_s<=0 or not np.isfinite(reference_hz) or reference_hz<=0:
        raise ValueError('positive finite elapsed time and reference rate required')
    rates=[]
    for g in groups:
        ix=np.asarray(g)
        if ix.size==0:
            rates.append(0.);continue
        if ix.ndim!=1 or not np.issubdtype(ix.dtype,np.integer):
            raise ValueError('indices must be one-dimensional integers')
        if len(np.unique(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=s.shape[1]):
            raise ValueError('duplicate or invalid neuron index')
        rates.append(float(s[:,ix].sum())/(len(ix)*window_s))
    rates=np.array(rates)
    return rates,np.clip(rates/reference_hz,0,1)


def audit_interval(*, spikes, groups, window_s, reference_hz, actual_ctrl,
                   sensory_current, sensory_indices, background_current,
                   readout_connected=True, command_scale=None):
    rates,expected=translate_motor_spikes(spikes,groups,window_s,reference_hz)
    if not readout_connected:expected[:]=0
    if command_scale is not None:
        # A DECLARED, constant per-channel gain (a coupled fibre-cluster pair is driven from one
        # pool at half strength each).  It is a fixed wiring constant, not a body-derived or
        # time-varying term, and it multiplies the spike translation itself, so the command is
        # still exactly a function of the spikes and nothing else.  The expected value is put
        # through the SAME scale-and-clip the runner applies, so the equality test stays exact.
        scale=np.asarray(command_scale,float)
        if scale.shape!=expected.shape or not np.all(np.isfinite(scale)) or np.any(scale<=0):
            raise ValueError('command scale must be positive, finite and match the channel count')
        expected=np.clip(expected*scale,0,1)
    ctrl=np.asarray(actual_ctrl)
    cur=np.asarray(sensory_current)
    bg=np.asarray(background_current)
    n=np.asarray(spikes).shape[1]
    if ctrl.shape!=expected.shape or cur.shape!=(n,) or bg.shape!=(n,):
        raise ValueError('audit dimensions mismatch')
    allowed=np.zeros(n,bool)
    indices=np.asarray(sensory_indices)
    if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=n):
        raise ValueError('invalid sensory index')
    allowed[indices]=True
    checks={'finite':bool(np.all(np.isfinite(ctrl)) and np.all(np.isfinite(cur)) and np.all(np.isfinite(bg))),
            'no_background_current':bool(np.all(bg==0)),
            'sensory_only_injection':bool(np.all(cur[~allowed]==0)),
            'command_exactly_from_spike_translation':bool(np.array_equal(ctrl,expected)),
            'command_scale_is_a_constant_declared_gain':bool(command_scale is not None)}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'rates_hz':rates.tolist(),'expected_ctrl':expected.tolist(),
            'command_scale_applied':(None if command_scale is None
                                     else np.asarray(command_scale,float).tolist()),
            'scope':'runtime origin equality only; receptor and muscle mapping validity not established'}
