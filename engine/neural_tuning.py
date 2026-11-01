"""Opt-in finite-parameter conductance experiments with frozen roots and E/I masks.
No optimizer, policy, input generator or automatic production configuration change.
Bounds are declared engineering ranges, not measured fly physiology.
"""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
from scipy import sparse
from .neural_cond import ConductanceNetwork,ConductanceParams

STATE_FIELDS=('_ring','v','ge','gi','adaptation','refractory_until_ms','spike_count')

@dataclass(frozen=True)
class TuningParameters:
    exc_gain: float=1.
    inh_gain: float=1.
    threshold_shift_mV: float=0.

class FrozenConnectomeTuning:
    """Create independent parameter probes; never mutate the source network/arrays."""
    def __init__(self,root_ids,We,Wi,*,params=None,dt_ms=.5,threshold_offset_mV=None,
                 gain_bounds=(.25,4.),threshold_bounds_mV=(-5.,5.)):
        roots=np.asarray(root_ids)
        if roots.ndim!=1 or roots.dtype.kind not in 'iu' or len(roots)==0 or len(set(map(int,roots)))!=len(roots):
            raise ValueError('Unique nonempty integer root_ids required')
        self.root_ids=roots.astype(np.int64,copy=True)
        if np.any(self.root_ids<=0):raise ValueError('Positive representable root IDs required')
        self.root_ids.flags.writeable=False
        self.params=params or ConductanceParams();self.params.validate();self.dt_ms=float(dt_ms)
        if not np.isfinite(self.dt_ms) or self.dt_ms<=0:raise ValueError('Positive finite dt required')
        self.gain_bounds=tuple(map(float,gain_bounds));self.threshold_bounds=tuple(map(float,threshold_bounds_mV))
        for bounds in (self.gain_bounds,self.threshold_bounds):
            if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:raise ValueError('Invalid bounds')
        if self.gain_bounds[0]<=0:raise ValueError('Positive gains preserve existing conductance support')
        if not self.gain_bounds[0]<=1<=self.gain_bounds[1] or not self.threshold_bounds[0]<=0<=self.threshold_bounds[1]:
            raise ValueError('Bounds must include baseline')
        n=len(roots);self.offset=np.zeros(n) if threshold_offset_mV is None else np.asarray(threshold_offset_mV,dtype=float).copy()
        if self.offset.shape!=(n,) or not np.isfinite(self.offset).all():raise ValueError('Invalid threshold offsets')
        self.offset.flags.writeable=False;self._base=[]
        for matrix in (We,Wi):
            w=sparse.csr_matrix(matrix,dtype=float,copy=True);w.sum_duplicates();w.sort_indices()
            if w.shape!=(n,n) or np.any(~np.isfinite(w.data)) or np.any(w.data<=0):raise ValueError('Expected finite positive stored conductance entries')
            self._base.append(w)
        self._identity=self._fingerprint()

    def _fingerprint(self):
        h=hashlib.sha256(self.root_ids.tobytes())
        h.update(self.offset.tobytes());h.update(json.dumps(asdict(self.params),sort_keys=True).encode());h.update(repr(self.dt_ms).encode())
        for w in self._base:
            for a in (w.indptr,w.indices,w.data):h.update(a.tobytes())
        return h.hexdigest()

    def snapshot(self,network):
        if network.n!=len(self.root_ids) or network.dt!=self.dt_ms or asdict(network.p)!=asdict(self.params):raise ValueError('Network configuration mismatch')
        if network.voltage_method!='backward_euler':raise ValueError('Snapshot requires declared backward Euler source')
        if not np.array_equal(network.threshold_offset_mV,self.offset):raise ValueError('Snapshot requires baseline threshold configuration')
        for actual,base in zip((network.We,network.Wi),self._base):
            if not all(np.array_equal(a,b) for a,b in [(actual.indptr,base.indptr),(actual.indices,base.indices),(actual.data,base.data)]):raise ValueError('Snapshot requires baseline weights')
        state=dict(identity=self._identity,root_ids=self.root_ids.copy(),steps=int(network.steps),t_ms=float(network.t_ms),ring_i=int(network._ring_i))
        for name in STATE_FIELDS:state[name]=getattr(network,name).copy()
        return state

    def spawn(self,tuning=None,*,snapshot=None):
        p=tuning or TuningParameters()
        if not isinstance(p,TuningParameters):raise ValueError('Use explicit TuningParameters')
        vals=np.array([p.exc_gain,p.inh_gain,p.threshold_shift_mV],dtype=float)
        if not np.isfinite(vals).all():raise ValueError('Nonfinite tuning value')
        if not all(self.gain_bounds[0]<=g<=self.gain_bounds[1] for g in vals[:2]):raise ValueError('Gain outside declared bounds')
        if not self.threshold_bounds[0]<=vals[2]<=self.threshold_bounds[1]:raise ValueError('Threshold shift outside declared bounds')
        if self._fingerprint()!=self._identity:raise ValueError('Source identity mutated')
        matrices=[]
        for base,gain in zip(self._base,vals[:2]):
            w=base.copy();w.data*=gain;matrices.append(w)
        net=ConductanceNetwork(len(self.root_ids),self.params,dt_ms=self.dt_ms,voltage_method='backward_euler',threshold_offset_mV=self.offset+vals[2])
        net.set_connectivity(*matrices)
        for actual,base in zip((net.We,net.Wi),self._base):
            if not np.array_equal(actual.indptr,base.indptr) or not np.array_equal(actual.indices,base.indices):raise ValueError('Conductance mask changed')
        if snapshot is not None:
            if snapshot['identity']!=self._identity or not np.array_equal(snapshot['root_ids'],self.root_ids):raise ValueError('Snapshot identity/root order mismatch')
            steps=snapshot['steps'];t=snapshot['t_ms'];ri=snapshot['ring_i']
            if int(steps)!=steps or steps<0 or not np.isfinite(t) or abs(t-steps*self.dt_ms)>1e-9 or ri!=int(steps)%len(net._ring):raise ValueError('Invalid snapshot clock/ring')
            for name in STATE_FIELDS:
                a=np.asarray(snapshot[name]);target=getattr(net,name)
                if a.shape!=target.shape or a.dtype!=target.dtype or not np.isfinite(a).all():raise ValueError('Snapshot array mismatch: '+name)
                target[:]=a
            net.steps=int(steps);net.t_ms=float(t);net._ring_i=int(ri)
        return net

    def contract(self):
        return dict(identity=self._identity,neurons=len(self.root_ids),nonzero_exc=self._base[0].nnz,nonzero_inh=self._base[1].nnz,
                    gain_bounds=self.gain_bounds,threshold_bounds_mV=self.threshold_bounds,baseline=asdict(TuningParameters()),
                    unchanged='root order, sparse E/I support, intrinsic parameters, delays and step rule',
                    not_provided='optimizer/autograd/training objective/maze sensory interface or runtime deployment',
                    scope='Engineering finite-parameter probes, not a calibrated physiological model')
