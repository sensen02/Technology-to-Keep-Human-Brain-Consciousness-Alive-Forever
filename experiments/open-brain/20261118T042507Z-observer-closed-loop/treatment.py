"""Streaming wrapper for the frozen gain observer, with no extra training."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/"20261118T035424Z-adaptive-neck"
sys.path.insert(0,str(PRIOR))
from adaptive_observer import AdaptiveObserver, DT, predict

ARMS=("clean_off","clean_on","drift_off","drift_on")
SEEDS=(11701,11903)
FIT=PRIOR/"OBSERVER/FIT.json"


def load_models():
    saved=json.loads(FIT.read_text())
    return {name:{k:np.asarray(v,dtype=float) if k!="offset" else float(v)
                 for k,v in model.items()} for name,model in saved.items()}


def prescribed_drift(steps):
    return np.interp(np.arange(steps),[0,6100,9100,14000,16200,steps-1],
                     [1.,1.,.66,.66,1.,1.])


class CausalCue:
    def __init__(self):
        self.motor_state=np.zeros((4,2,17))
        self.other_state=np.zeros((4,2))
        self.motor_alpha=-np.expm1(-DT/np.array([.055,.160]))
        self.other_alpha=-np.expm1(-DT/np.array([.035,.105]))
        self.models=load_models()

    def estimate(self):
        motor=self.motor_state.reshape(4,34)
        other=self.other_state
        values={"motor":motor,"other":other,"fused":np.c_[motor,other]}
        return [predict(self.models[name],values[name]) for name in ("motor","other","fused")]

    def update(self, dn_emitted, right_arrived):
        self.motor_state+=self.motor_alpha[None,:,None]*(dn_emitted[:,None,:]/DT-self.motor_state)
        self.other_state+=self.other_alpha*(right_arrived[:,None]/DT-self.other_state)


class AssayTreatment:
    def __init__(self, steps):
        self.cue=CausalCue()
        self.observers=[AdaptiveObserver() for _ in ARMS]
        self.gain=prescribed_drift(steps)
        self.phase=np.zeros(4)

    def step(self, tick, clean_rate):
        raw=clean_rate*np.array([1.,1.,self.gain[tick],self.gain[tick]])
        motor,other,fused=self.cue.estimate()
        reports=[observer.step(float(r),float(m),float(o),float(f),available=True)
                 for observer,r,m,o,f in zip(self.observers,raw,motor,other,fused,strict=True)]
        estimated_gain=np.array([r.gain for r in reports])
        effective=raw.copy()
        effective[[1,3]]=[reports[1].estimate,reports[3].estimate]
        self.phase+=effective*DT
        events=self.phase>=1.
        self.phase-=events
        return raw,effective,estimated_gain,fused,events
