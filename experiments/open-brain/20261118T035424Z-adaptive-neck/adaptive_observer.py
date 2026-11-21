"""Causal, advisory gain estimation. Never emits AN events."""
from dataclasses import dataclass

import numpy as np

DT=.0001


def lagged_rates(events, tau):
    """The returned row t uses only events strictly before t."""
    events=np.asarray(events)
    alpha=-np.expm1(-DT/tau)
    state=np.zeros(events.shape[1])
    result=np.empty(events.shape)
    for tick, row in enumerate(events):
        result[tick]=state
        state+=alpha*(row/DT-state)
    return result


def cues(data, lane):
    movement=data["dn_emitted"][:,lane,:]
    other=data["an_arrived"][:,lane,1:2]
    motor=np.concatenate([lagged_rates(movement,t) for t in (.055,.160)],axis=1)
    sensory=np.concatenate([lagged_rates(other,t) for t in (.035,.105)],axis=1)
    return {"motor":motor,"other":sensory,"fused":np.c_[motor,sensory]}


def fit(features, target):
    center=features.mean(0)
    scale=np.maximum(features.std(0),5.)
    normalized=(features-center)/scale
    samples=len(target)
    gram=normalized.T@normalized/samples
    beta=np.linalg.solve(gram+.1*np.eye(gram.shape[0]),normalized.T@(target-target.mean())/samples)
    return dict(center=center,scale=scale,beta=beta,offset=float(target.mean()))


def predict(model, x):
    return np.clip((x-model["center"])/model["scale"]@model["beta"]+model["offset"],0.,350.)


@dataclass(frozen=True)
class Reading:
    raw: float
    prediction: float
    estimate: float | None
    gain: float
    updated: bool
    cue_agreement: bool
    available: bool
    innovation: float


class AdaptiveObserver:
    """Estimate shrinkage/expansion while preserving the measured sample."""

    def __init__(self):
        self._smooth=-np.expm1(-DT/.07)
        self._learn=-np.expm1(-DT/.16)
        self.reset()

    def reset(self):
        self.gain=1.
        self.observed=0.
        self.expected=0.
        self.persistent=0

    def step(self, raw, motor, other, fused, *, available, other_available=True):
        agreement=other_available and abs(motor-other)<=max(10.,.24*fused)
        innovation=raw-fused
        if not available:
            self.reset()
            return Reading(raw,fused,None,self.gain,False,agreement,False,innovation)
        self.observed+=self._smooth*(raw-self.observed)
        self.expected+=self._smooth*(fused-self.expected)
        ratio=self.observed/max(self.expected,1.)
        evidence=(agreement and self.expected>=35. and .43<=ratio<=1.65
                  and abs(ratio-1.)>=.14)
        self.persistent=self.persistent+1 if evidence else 0
        updated=self.persistent>=1000
        if updated:
            self.gain+=self._learn*(float(np.clip(ratio,.60,1.40))-self.gain)
        elif agreement and .86<ratio<1.14:
            self.gain+=self._learn*(1.-self.gain)
        estimate=raw/self.gain if other_available else raw
        return Reading(raw,fused,estimate,self.gain,updated,agreement,True,innovation)
