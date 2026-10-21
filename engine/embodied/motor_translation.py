"""Finite-memory engineering MN->muscle translation, NOT anatomical weights.
Only binary motor spikes can contribute; no bias, pose input or autonomous state.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .motor_output import MotorOutput


class FiniteMotorTranslator:
    DT = .0005
    BINS = ((0,5),(5,10),(10,20),(20,40),(40,80))
    MAX_WEIGHT = 4.

    def __init__(self, manifest, weights=None):
        validated=MotorOutput(manifest)
        self.n,self.nu=validated.n,validated.nu
        self.pools=sorted({tuple(sorted(g.tolist())) for g in validated.groups if len(g)})
        lookup={p:i for i,p in enumerate(self.pools)}
        self.actuator_pool=np.array([lookup.get(tuple(sorted(g.tolist())),-1) for g in validated.groups])
        identity={'n':self.n,'rows':[{k:r.get(k) for k in ('actuator','actuator_index','local_indices','tier_root_ids','pool_root_ids')} for r in manifest['rows']]}
        self.mapping_digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self._weights=np.tile([.4,.25,.2,.1,.05],(len(self.pools),1)) if weights is None else np.asarray(weights,float).copy()
        if self._weights.shape!=(len(self.pools),5) or not np.isfinite(self._weights).all() or np.any((self._weights<0)|(self._weights>self.MAX_WEIGHT)):
            raise ValueError('weights require shape pools x5, finite and within [0,4]')
        self._weights.flags.writeable=False
        self.reset()

    @property
    def weights(self):
        return self._weights.copy()

    def reset(self):
        self.history=np.zeros((80,len(self.pools)))
        self.elapsed_s=0.;self.interval=-1

    def consume(self, spikes, *, spike_dt_s, start_s, end_s, connected=True):
        sp=np.asarray(spikes)
        if sp.ndim!=2 or sp.shape[1]!=self.n or sp.shape[0]<1 or not np.all((sp==0)|(sp==1)):
            raise ValueError('binary finite time x neuron spikes required')
        if not all(math.isfinite(v) for v in (spike_dt_s,start_s,end_s)) or not math.isclose(spike_dt_s,self.DT,rel_tol=0,abs_tol=1e-15):
            raise ValueError('fixed 0.5ms neural ticks required')
        if not math.isclose(start_s,self.elapsed_s,rel_tol=0,abs_tol=1e-12) or not math.isclose(end_s-start_s,len(sp)*self.DT,rel_tol=0,abs_tol=1e-12):
            raise ValueError('contiguous completed source intervals required')
        counts=np.array([sp[:,p].sum() for p in self.pools],dtype=int)
        if connected:
            activity=np.column_stack([sp[:,p].mean(axis=1)/self.DT/100 for p in self.pools]) if self.pools else np.zeros((len(sp),0))
            self.history=np.concatenate((activity[::-1],self.history),axis=0)[:80]
        else:self.history.fill(0)
        features=np.stack([self.history[lo:hi].mean(axis=0) for lo,hi in self.BINS],axis=1)
        pooled=np.clip((features*self._weights).sum(axis=1),0,1)
        ctrl=np.zeros(self.nu);mapped=self.actuator_pool>=0;ctrl[mapped]=pooled[self.actuator_pool[mapped]]
        self.elapsed_s=float(end_s);self.interval+=1
        return ctrl,{'source_interval':self.interval,'source_start_s':float(start_s),'source_end_s':float(end_s),
            'valid_from_s':float(end_s),'connected':bool(connected),'pool_spike_counts':counts.tolist(),
            'pool_features':features.tolist(),'memory_horizon_s':.04,'mapping_digest':self.mapping_digest,
            'note':'finite causal FIR, no bias; engineering translation not anatomy'}

    def save(self,path):
        Path(path).write_text(json.dumps({'version':1,'mapping_digest':self.mapping_digest,
            'dt_s':self.DT,'bins':self.BINS,'weights':self._weights.tolist()},indent=2))

    @classmethod
    def load(cls,manifest,path):
        data=json.loads(Path(path).read_text());result=cls(manifest,data['weights'])
        if data.get('version')!=1 or data.get('mapping_digest')!=result.mapping_digest or data.get('dt_s')!=cls.DT or data.get('bins')!=[list(b) for b in cls.BINS]:
            raise ValueError('incompatible translator or mapping identity')
        return result
