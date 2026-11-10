"""Causal same-leg MN readout; NumPy+stdlib only, no simulation or training.

observe(current START-bin MNfilteredHz), then action(). History is oldest50ms at
5ms bins. All-zero history gives exact zero action; zero current alone may not.
"""
from pathlib import Path
import csv
import json
import numpy as np

PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
IDENTITY=PROJECT/'experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MOTOR_READOUT_PORTS.csv'
INTERFACE=PROJECT/'experiments/flybody-teacher/20261107T180844Z/mapping/STUDENT_INTERFACE.json'


def canonical_identity():
    with IDENTITY.open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['readout_eligible_single_leg']=='True']
    roots=np.array([int(r['root_id_string']) for r in rows],dtype=np.int64)
    local=np.array([int(r['local_index_0based']) for r in rows],dtype=np.int64)
    legs=np.array([r['nerve_matched_legs'] for r in rows])
    names=np.asarray(json.loads(INTERFACE.read_text())['output']['ordered_joint_names'])
    return roots,local,legs,names


class Readout:
    """Load a frozen model, validate identities and maintain causal MN history."""
    def __init__(self,model_path):
        self.model_path=Path(model_path).resolve()
        with np.load(self.model_path,allow_pickle=False) as archive:
            model={k:archive[k] for k in archive.files}
        required=('weights','mn_root_ids','mn_local','mn_leg','joint_names','lags_bins','feature_hz_scale','rate_cap','bias','bin_ms')
        if any(k not in model for k in required):raise ValueError('Missing required readout model metadata')
        roots,local,legs,names=canonical_identity()
        for key,expected in (('mn_root_ids',roots),('mn_local',local),('mn_leg',legs),('joint_names',names)):
            if not np.array_equal(model[key],expected):raise ValueError('Frozen source identity mismatch: '+key)
        lags=model['lags_bins']
        if lags.dtype.kind not in 'iu' or lags.ndim!=1 or lags.tolist() not in ([0],[0,1,4,10]):
            raise ValueError('Only preregistered lags [0] or [0,1,4,10] are admitted')
        if float(model['feature_hz_scale'])!=100 or float(model['rate_cap'])!=30 or float(model['bin_ms'])!=5:
            raise ValueError('Readout scales or bin duration differ from contract')
        w=np.asarray(model['weights'],dtype=float)
        if w.shape!=(323*len(lags),42) or not np.isfinite(w).all() or np.any(np.abs(w)>1):
            raise ValueError('Weights must be finite, correctly shaped and within [-1,1]')
        if model['bias'].shape!=(42,) or np.any(model['bias']!=0):raise ValueError('Bias must be exactly zero')
        jointlegs=np.array([name.removeprefix('joint_')[:2].lower() for name in names])
        featurelegs=np.tile(legs,len(lags))
        if np.any(w[featurelegs[:,None]!=jointlegs[None,:]]!=0):raise ValueError('Cross-leg readout weight is nonzero')
        self.weights=w.copy();self.weights.flags.writeable=False
        self.lags_bins=lags.astype(np.int64,copy=True);self.lags_bins.flags.writeable=False
        self.mn_root_ids=roots;self.mn_local=local;self.mn_leg=legs;self.joint_names=names
        for value in (self.mn_root_ids,self.mn_local,self.mn_leg,self.joint_names):value.flags.writeable=False
        self.feature_hz_scale=100.;self.rate_cap=30.;self.bin_ms=5.
        self.history_rates_hz=np.zeros((int(lags.max())+1,323),dtype=np.float64)
        self.observations_seen=0

    def observe(self,mn_rates323):
        raw=np.asarray(mn_rates323)
        if raw.shape!=(323,) or raw.dtype.kind not in 'fiu':raise ValueError('MN rates must be a real numeric (323,) vector')
        rates=np.asarray(raw,dtype=np.float64)
        if not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('MN rates must be finite and nonnegative')
        self.history_rates_hz[1:]=self.history_rates_hz[:-1].copy()
        self.history_rates_hz[0]=rates
        self.observations_seen+=1
        return self

    def feature_vector(self):
        return self.history_rates_hz[self.lags_bins].reshape(-1)/self.feature_hz_scale

    def action(self):
        # Fixed scale rather than an inferred clock; no teacher or physical input.
        return self.rate_cap*np.tanh(self.feature_vector()@self.weights)

    def state_dict(self):
        return {'history_rates_hz':self.history_rates_hz.copy(),
                'observations_seen':np.asarray(self.observations_seen,dtype=np.int64),
                'lags_bins':self.lags_bins.copy(),
                'bin_ms':np.asarray(self.bin_ms)}

    def load_state_dict(self,state):
        h=np.asarray(state['history_rates_hz'])
        n=np.asarray(state['observations_seen'])
        if h.shape!=self.history_rates_hz.shape or h.dtype.kind not in 'fiu' or not np.isfinite(h).all() or np.any(h<0):
            raise ValueError('Invalid predictor history')
        if n.shape!=() or n.dtype.kind not in 'iu' or int(n)<0:raise ValueError('Invalid observation count')
        if not np.array_equal(state['lags_bins'],self.lags_bins) or float(state['bin_ms'])!=self.bin_ms:
            raise ValueError('Predictor state contract mismatch')
        # Initial unused history must be zero; never accept hidden pre-start input.
        if int(n)<len(h) and np.any(h[int(n):]!=0):raise ValueError('Nonzero pre-start padding')
        self.history_rates_hz[:]=h
        self.observations_seen=int(n)
        return self
