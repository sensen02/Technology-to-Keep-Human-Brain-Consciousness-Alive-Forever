"""Canonical circuit restoration and MN-only same-leg low-rank engineering actor.

No physical engine, optimizer, teacher, target, time or body input is imported.
BASIS and state paths default relative to this file, independent of cwd.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
CORE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
CORE_SHA256='c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
IDENTITY_FIELDS=('root_ids','cache_indices','allowed_sensory_local','source_sensory_local','excluded_ambiguous_sensory_local','descending_local','ascending_local','mn_local','mn_leg','strict_source_edge_rows','_sensory_mask','_descending_mask')
FLOAT_FIELDS=('v','ge','gi','g_adapt','refractory_until_ms')
ARRAY_FIELDS=FLOAT_FIELDS+('spike_count','_ring','last_spikes')
LEG_ORDER=('LF','LH','LM','RF','RH','RM')


def load(path):
    """Load a non-pickle NPZ into detached arrays (zero simulation steps)."""
    with np.load(Path(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def parameter_contract(c):
    return json.dumps({'point':asdict(c.net.p),'adaptation':asdict(c.net.adaptation_params),'dt_ms':c.net.dt,'delay_steps':c.net.delay_steps,'filter_tau_ms':c.filter_tau_ms,'total_conductance_uS':c.total_conductance_uS},sort_keys=True,separators=(',',':'))


def snapshot(c):
    """Return detached complete neural state and exact identity/configuration."""
    result={k:getattr(c.net,k).copy() for k in ARRAY_FIELDS}
    result.update({k:getattr(c,k).copy() for k in IDENTITY_FIELDS})
    result.update(filtered_rates_hz=c.filtered_rates_hz.copy(),_ring_i=np.array(c.net._ring_i,dtype=np.int64),steps=np.array(c.net.steps,dtype=np.int64),t_ms=np.array(c.net.t_ms,dtype=float),neural_module_sha256=np.array(CORE_SHA256),parameter_contract_json=np.array(parameter_contract(c)))
    return result


def new_circuit():
    if hashlib.sha256(CORE.read_bytes()).hexdigest()!=CORE_SHA256:raise ValueError('frozen neural source changed')
    key='_foundation_round10_core'
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,CORE);m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m)
    return sys.modules[key].LocalCircuit()


def restore_circuit(c,path=None):
    """Transactionally validate, then restore ALL neural state; no reset/stepping.

    Identity/topology remains owned by c; archive identities must match exactly.
    A corrupted archive never partially mutates c.
    """
    a=load(HERE/'NEURAL_START.npz' if path is None else path)
    required=set(ARRAY_FIELDS+IDENTITY_FIELDS+('filtered_rates_hz','_ring_i','steps','t_ms','neural_module_sha256','parameter_contract_json'))
    if not required.issubset(a):raise ValueError('incomplete neural snapshot')
    for k in IDENTITY_FIELDS:
        expected=getattr(c,k)
        if a[k].shape!=expected.shape or a[k].dtype!=expected.dtype or not np.array_equal(a[k],expected):raise ValueError('snapshot identity mismatch: '+k)
    if a['neural_module_sha256'].shape!=() or str(a['neural_module_sha256'])!=CORE_SHA256:raise ValueError('snapshot core hash mismatch')
    if a['parameter_contract_json'].shape!=() or str(a['parameter_contract_json'])!=parameter_contract(c):raise ValueError('snapshot parameter contract mismatch')
    if hashlib.sha256(CORE.read_bytes()).hexdigest()!=CORE_SHA256:raise ValueError('frozen neural source changed')
    for k in FLOAT_FIELDS+('filtered_rates_hz',):
        if a[k].shape!=(20000,) or a[k].dtype!=np.dtype('float64') or not np.isfinite(a[k]).all():raise ValueError('invalid finite float state: '+k)
    for k in ('ge','gi','g_adapt','filtered_rates_hz','refractory_until_ms'):
        if np.any(a[k]<0):raise ValueError('negative state: '+k)
    if np.any(a['v'] < -80-1e-10) or np.any(a['v'] > -50+1e-10):raise ValueError('voltage outside declared domain')
    if a['spike_count'].shape!=(20000,) or a['spike_count'].dtype!=np.dtype('int64') or np.any(a['spike_count']<0):raise ValueError('invalid spike counts')
    ring_shape=(c.net.delay_steps+1,20000)
    if a['_ring'].shape!=ring_shape or a['_ring'].dtype!=np.dtype(bool) or a['last_spikes'].shape!=(20000,) or a['last_spikes'].dtype!=np.dtype(bool):raise ValueError('invalid delay/spike arrays')
    for k in ('_ring_i','steps'):
        if a[k].shape!=() or a[k].dtype!=np.dtype('int64') or int(a[k])<0:raise ValueError('invalid integer clock: '+k)
    steps,ri=int(a['steps']),int(a['_ring_i'])
    if a['t_ms'].shape!=() or a['t_ms'].dtype!=np.dtype('float64') or not np.isfinite(a['t_ms']) or float(a['t_ms'])!=steps*c.net.dt:raise ValueError('inconsistent neural clock')
    if ri>=len(a['_ring']) or ri!=steps%len(a['_ring']):raise ValueError('inconsistent ring index')
    if not np.array_equal(a['last_spikes'],a['_ring'][(ri-1)%len(a['_ring'])]):raise ValueError('last spike does not match ring')
    if np.any(a['refractory_until_ms']>float(a['t_ms'])+c.net.p.refractory_ms+1e-10) or np.any(a['spike_count']>steps):raise ValueError('inconsistent spike/refractory clock')
    for actual,original in ((c.net.We,c._We),(c.net.Wi,c._Wi)):
        if actual.shape!=original.shape or (actual!=original).nnz:raise ValueError('circuit connectivity has been altered')
    # All validation precedes mutation; arrays are detached, never archive views.
    for k in ARRAY_FIELDS:setattr(c.net,k,a[k])
    c.net._ring_i=ri;c.net.steps=steps;c.net.t_ms=float(a['t_ms'])
    c.filtered_rates_hz=a['filtered_rates_hz']
    return c


def load_circuit(path=None):
    """Construct the frozen core and restore canonical start (no neural steps)."""
    return restore_circuit(new_circuit(),path)


def load_basis(path=None):
    a=load(HERE/'BASE.npz' if path is None else path)
    if a['basis'].shape!=(323,24) or not np.isfinite(a['basis']).all():raise ValueError('invalid basis')
    if a['mn_local'].shape!=(323,) or a['mn_root_ids'].shape!=(323,) or a['mn_leg'].shape!=(323,) or a['joint_names'].shape!=(42,):raise ValueError('invalid basis identities')
    if tuple(a['leg_order'])!=LEG_ORDER:raise ValueError('basis leg order mismatch')
    if not np.array_equal(a['joint_leg'],np.repeat(np.array(LEG_ORDER),7)):raise ValueError('joint leg order mismatch')
    for i,leg in enumerate(LEG_ORDER):
        mask=a['mn_leg']==leg.lower()
        if not np.any(mask) or np.any(a['basis'][~mask,4*i:4*i+4]!=0):raise ValueError('cross-leg basis loading')
    return a


class Actor:
    """Actor(theta168): bounded engineering MN-only translation, no bias/history.

    theta is flat168 or(6,4,7), finite and in[-.25,.25]. Effective sameleg
    W323x42 is clipped[-1,1]. action(MN323)=30*tanh((MNHz/100)@W).
    """
    def __init__(self,theta=None,*,basis_path=None):
        self.metadata=load_basis(basis_path)
        raw=np.zeros(168) if theta is None else np.asarray(theta)
        if raw.shape not in ((168,),(6,4,7)) or raw.dtype.kind not in 'iuf':raise ValueError('theta must be real168 or6x4x7')
        coefficients=np.array(raw,dtype=float,copy=True).reshape(6,4,7)
        if not np.isfinite(coefficients).all() or np.any(np.abs(coefficients)>.25):raise ValueError('theta outside finite[-.25,.25]')
        weights=np.zeros((323,42))
        for i,leg in enumerate(LEG_ORDER):
            ni=np.flatnonzero(self.metadata['mn_leg']==leg.lower());ji=np.flatnonzero(self.metadata['joint_leg']==leg)
            weights[np.ix_(ni,ji)]=np.clip(self.metadata['basis'][ni,4*i:4*i+4]@coefficients[i],-1.,1.)
        self.theta=coefficients.reshape(168);self.theta.flags.writeable=False
        self.weights=weights;self.weights.flags.writeable=False
        self.joint_names=self.metadata['joint_names'].copy();self.mn_local=self.metadata['mn_local'].copy();self.mn_root_ids=self.metadata['mn_root_ids'].copy();self.mn_leg=self.metadata['mn_leg'].copy()

    def action(self,mn323):
        raw=np.asarray(mn323)
        if raw.shape!=(323,) or raw.dtype.kind not in 'iuf':raise ValueError('MN input must be real shape323')
        x=np.asarray(raw,dtype=float)
        if not np.isfinite(x).all() or np.any(x<0):raise ValueError('MN rates must be finite nonnegative Hz')
        return 30.*np.tanh((x/100.)@self.weights)
