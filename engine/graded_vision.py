"""Continuous graded photoreceptor release with an explicit chloride contract.

This adapter reserves the network's entire inhibitory conductance state for
histamine; it is NOT a general independent-channel recurrent integration API.
The benchmark uses only the actual BANC sensory projection and an otherwise
uncoupled neuron state vector. PhotoReceptor parameters and conductance scale
are illustrative, not measured biological calibration.

Low-level step() advances photo adaptation and returns conductance increments;
it does not advance a network. apply_to() validates the reserved-channel
contract and advances the network exactly once. No photoreceptor spikes are
created by this interface. External callers must not separately reinject its
increments or reuse the reserved inhibitory state for another transmitter.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .receptors import PhotoReceptor

class GradedVision:
    def __init__(self, n_neurons, visual_indices, pre, post, synapses, pure_histamine,
                 dt_ms=0.2, scale_uS_per_synapse=1e-5, e_hist_mV=-70.0,
                 tau_release_ms=8.0):
        if not np.isscalar(n_neurons) or not np.isfinite(n_neurons) or int(n_neurons) != n_neurons:
            raise ValueError('neuron count must be a finite integer')
        self.n=int(n_neurons)
        self.visual=np.asarray(visual_indices)
        self.dt=float(dt_ms); self.scale=float(scale_uS_per_synapse)
        self.e_hist=float(e_hist_mV); self.tau=float(tau_release_ms)
        if self.n < 1 or not np.isfinite(self.dt) or self.dt <= 0 or not np.isfinite(self.scale) or self.scale < 0 or not np.isfinite(self.e_hist) or not np.isfinite(self.tau) or self.tau <= 0: raise ValueError('invalid interface parameters')
        if self.visual.ndim != 1 or not np.issubdtype(self.visual.dtype,np.integer) or np.any(self.visual<0) or np.any(self.visual>=self.n) or np.unique(self.visual).size != self.visual.size: raise ValueError('visual indices must be unique in-bounds integers')
        pre=np.asarray(pre); post=np.asarray(post); syn=np.asarray(synapses); pure=np.asarray(pure_histamine)
        if pre.ndim != 1 or post.ndim != 1 or syn.ndim != 1 or pure.ndim != 1 or not (pre.shape==post.shape==syn.shape==pure.shape): raise ValueError('edge shape mismatch')
        if not np.issubdtype(pre.dtype,np.integer) or not np.issubdtype(post.dtype,np.integer): raise ValueError('edge indices must be integers')
        if np.any(pre<0) or np.any(pre>=self.n) or np.any(post<0) or np.any(post>=self.n): raise ValueError('edge index out of bounds')
        if pure.dtype != np.dtype(bool): raise ValueError('pure_histamine must be boolean')
        syn=syn.astype(float,copy=False)
        if np.any(~np.isfinite(syn)) or np.any(syn<0): raise ValueError('synapses must be finite and nonnegative')
        keep=pure; self.pre=pre[keep]; self.post=post[keep]
        self.W=sparse.csr_matrix((syn[keep]*self.scale,(self.post,self.pre)),shape=(self.n,self.n))
        self.Wvisual=self.W[:,self.visual].tocsr()
        self.receptor=PhotoReceptor(self.visual.size,gain_mV=1.0,dt_ms=self.dt)
        self.last_activation=np.zeros(self.visual.size); self.last_target_uS=np.zeros(self.n); self.last_increment_uS=np.zeros(self.n)
        self.total_edges=int(keep.sum())
    def step(self, light, dt_ms=None):
        dt=self.dt if dt_ms is None else float(dt_ms)
        if not np.isfinite(dt) or dt<=0: raise ValueError('dt must be positive finite')
        arr=np.asarray(light,float)
        if np.any(~np.isfinite(arr)) or np.any(arr<0): raise ValueError('light must be finite and nonnegative')
        arr=np.broadcast_to(arr,(self.visual.size,))
        # Exact relaxation preserves adaptation bounds even for dt > tau_a.
        p=self.receptor.p
        target_a=arr/(arr+p['i50'])
        self.receptor.A += (-np.expm1(-dt/p['tau_a_ms']))*(target_a-self.receptor.A)
        a=self.receptor.instantaneous(arr); self.receptor.state=a
        self.last_activation=a
        target=np.asarray(self.Wvisual @ a).ravel(); target=np.maximum(target,0); self.last_target_uS=target
        inc=target*(-np.expm1(-dt/self.tau)); self.last_increment_uS=inc
        return inc.copy()
    def apply_to(self, net, light, dt_ms=None):
        """Apply one step, requiring the network's inhibitory channel to match E_hist/tau."""
        if not hasattr(net,'p') or not np.isclose(net.p.inh_reversal_mV,self.e_hist) or not np.isclose(net.p.inh_tau_ms,self.tau):
            raise ValueError('network inhibitory reversal/tau do not match graded histamine contract')
        if net.n != self.n or not np.isclose(net.dt,self.dt if dt_ms is None else float(dt_ms)): raise ValueError('network size/dt mismatch')
        if net.Wi.nnz or net.We[:,self.visual].nnz:
            raise ValueError('histamine channel must be dedicated; visual spike projections forbidden')
        inc=self.step(light,dt_ms); net.step(inh_events_uS=inc); return inc
    def describe(self):
        return {'visual_receptors':int(self.visual.size),'pure_histamine_edges':int(self.Wvisual.nnz),'reachable_posts':int(np.count_nonzero(np.diff(self.Wvisual.indptr))),'scale_uS_per_synapse':self.scale,'E_hist_mV':self.e_hist,'release_tau_ms':self.tau,'parameters_status':'illustrative hypothesis; no biological calibration'}

def pure_histamine_mask(verified, pre):
    v=np.asarray(verified); p=np.asarray(pre)
    if p.ndim!=1 or not np.issubdtype(p.dtype,np.integer) or np.any(p<0) or np.any(p>=len(v)): raise ValueError('invalid annotation indices')
    return v[p] == 'histamine'
