"""Clocked Eon and registered neck pathways for the single-body locomotion test."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OPEN=HERE.parent/'20261118T231627Z-open-fruit-loop'
sys.path.insert(0,str(OPEN))
from motor import DT, MotorAdapter, Neck, NeuralInputs
from fastbrain import FastBrain, eon, load_connectome
from cervical import FlightPaths, THRESHOLD

BLOCK=200
TASTED_WINDOW=700
PERIOD=40


class Brain:
    def __init__(self, config):
        torch.set_num_threads(1)
        self.ticks=config['ticks']
        self.enable=config['adapter']
        self.cut=config['cut']
        self.index,_=eon.get_hash_tables(eon.path_comp)
        self.inputs=NeuralInputs(self.index)
        self.neck=Neck(self.index,self.inputs,tactile=True)
        self.flight=FlightPaths()
        self.flight.block[self.flight.columns[:2]]=self.cut
        self.idx=np.array([self.index[root] for root in self.flight.eon_roots])
        self.head=MotorAdapter(json.loads((OPEN/'TRANSLATED_MOTOR.json').read_text()))
        weights=load_connectome(self.index)
        self.brain=FastBrain(weights.T.tocsr(),self.inputs.an_ix)
        scan=json.loads((HERE/'LAUNCH_SCAN.json').read_text())
        selected=next(x for x in scan if x['name']=='mixed8' and x['rate']==180)
        self.drive=np.array([self.index[root] for root in selected['roots']])
        assert not np.intersect1d(self.drive,self.idx).size
        self.seed=config['seed']
        self.draws=self.inputs.masks(self.ticks,self.seed)
        self.high_rng=np.random.default_rng(self.seed+773).random((self.ticks,len(self.drive)),dtype=np.float32)
        self.emitted=np.zeros((self.ticks,4),dtype=bool)
        self.arrivals=np.zeros_like(self.emitted)
        self.levels=np.zeros((self.ticks,4))
        self.ground=np.zeros((self.ticks,2))
        self.pre=np.zeros(self.ticks,dtype=np.uint8)
        self.taste=np.zeros(self.ticks,dtype=np.uint8)
        self.inputs_record=np.zeros((self.ticks//BLOCK,5))
        self.launch=None
        self.descent=None
        self.tick=0
        self.high_count=0
        self.feed_count=0

    def advance(self, obs):
        t=self.tick
        high=float(obs['height_probability']) if self.enable else 0.
        self.high_count=self.high_count+1 if high>=.8 else 0
        rate=180.*high if self.high_count>=2 and not obs['contact'] and self.launch is None else 0.
        if obs['landing_eye'] and self.descent is None:
            self.descent=t
        may_walk=obs['seen'] and self.launch is None and not obs['contact'] and high<.8
        center=float(obs['center'])
        turn=np.clip(.32*center,-.20,.20)
        pfl=self.head.rates(.55,turn,0.) if may_walk else (0.,0.)
        row=self.inputs.rates(pfl,(0.,0.),200. if obs['contact'] else 0.,0.)*DT
        front=np.array(obs['front'],dtype=float)
        end=min(t+BLOCK,self.ticks)
        self.inputs_record[t//BLOCK]=(t*DT,high,rate,int(obs['contact']),int(may_walk))
        for i in range(t,end):
            ground_spikes=self.brain.spikes[self.neck.dn_ix]
            self.ground[i]=self.neck.update(i,ground_spikes,front)
            input_indices=self.inputs.assembled[self.draws[i]<row]
            active=self.drive[self.high_rng[i]<rate*DT]
            self.pre[i]=len(active)
            self.brain.step(np.concatenate((input_indices,active)),self.neck.an_inputs)
            sp=self.brain.spikes[self.idx].copy()
            if self.descent is not None and 0<=i-self.descent<1200 and (i-self.descent)%PERIOD==0:
                sp[2:]=True
            self.emitted[i]=sp
            self.arrivals[i],self.levels[i]=self.flight.step(sp)
            if self.launch is None and self.levels[i,:2].min()>=THRESHOLD:
                self.launch=i
            self.taste[i]=self.brain.spikes[self.inputs.mn9].sum()
            self.feed_count+=int(self.taste[i])
            if i>=TASTED_WINDOW:
                self.feed_count-=int(self.taste[i-TASTED_WINDOW])
        self.tick=end
        return dict(commands=self.levels[t:end:2].tolist(),
                    walk=self.ground[t:end:2].tolist(), feed=bool(self.feed_count>=2))

    def save(self, path):
        for i,delay in enumerate(self.flight.delays):
            expected=np.zeros(self.ticks,dtype=bool)
            if not self.cut or i>=2:
                expected[delay:]=self.emitted[:-delay,i]
            assert np.array_equal(expected,self.arrivals[:,i])
        np.savez_compressed(path,emitted=self.emitted,arrived=self.arrivals,values=self.levels,
                            ground=self.ground,pre=self.pre,mn9=self.taste,head=self.inputs_record)


def main():
    init=json.loads(sys.stdin.readline())
    brain=Brain(init)
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        result=brain.advance(json.loads(line))
        print(json.dumps(result,separators=(',',':')),flush=True)
        if brain.tick==brain.ticks:
            brain.save(init['stream'])
            return


if __name__=='__main__':
    main()
