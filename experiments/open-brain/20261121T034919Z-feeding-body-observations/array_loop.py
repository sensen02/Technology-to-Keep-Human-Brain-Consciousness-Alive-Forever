"""Retained SWC geometry, propagated effects of balanced electrode current, continuous voltage readout."""
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from voltages import r,PARENT
from common import ElectrodeObservation,AxonProbe,FULL
import readback
from omnisim.modules.banc_volume import SectionExposure

class Electrode:
    def __init__(self,n_frames):
        c=np.load(PARENT/'calibration.npz')
        self.basis=c['basis'];self.unit=c['unit'];self.sigma=c['sigma']
        self.threshold=r.read(PARENT/'calibration.json')['threshold']
        width=r.WIDTH
        self.gram=np.zeros((2*width-1,len(self.unit),len(self.unit)),'f')
        for lag in range(width):
            dot=self.unit[:,lag:].reshape(19,-1)@self.unit[:,:width-lag].reshape(19,-1).T
            self.gram[width-1+lag]=dot;self.gram[width-1-lag]=dot.T
        self.pattern=r.patterns();sites=r.contacts('actual');keep=~self.pattern.any(0)
        with np.load(PARENT/'cut_anatomy.npz') as z:
            labels=z['labels'];lo=z['lower']
        count=np.stack([np.bincount(v.ravel(),minlength=53)[1:53] for v in labels.transpose(1,0,2)],axis=1)
        ids=r.geom['cell_index_xyz'];valid=ids[:,1]<labels.shape[1];ids=ids[valid]
        root=labels[tuple(ids.T)].astype(int)-1;good=(root>=0)&(root<52)
        dose=np.zeros_like(count,dtype=float)
        np.add.at(dose,(root[good],ids[good,1]),r.geom['severity'][valid][good])
        profiles={k:SectionExposure(lo[1],nc,ds) for k,nc,ds in zip(r.roots,count,dose)}
        del labels
        cut=[]
        with np.load(PARENT/'extensions.npz') as ext:
            for path in r.paths:
                chains=[ext[k] for k in ext.files if k.startswith(path.root+'_')]
                shared=[]
                for pts in zip(*chains):
                    if not np.allclose(pts,pts[0],atol=1e-13,rtol=0):break
                    shared.append(pts[0])
                xyz=path.points_m
                if len(shared)>1:
                    xyz=np.vstack((np.array(shared)[1:][::-1],xyz)) if xyz[0,1]>xyz[-1,1] else np.vstack((xyz,np.array(shared)[1:]))
                cut.append(replace(path,points_m=xyz))
        self.front=r.FrontEnd(np.sqrt(2*np.pi)*.15e-6)
        sensors=sites[keep]
        self.clock=SimpleNamespace(time_s=0.)
        self.model=ElectrodeObservation(sites,[AxonProbe(p.root,p.points_m,p.speed_m_s,owner=i) for i,p in enumerate(cut)],contact_radius_m=.15e-6,sensor_patches=[],section_exposure=profiles)
        self.model.attach(self.clock,'write')
        self.head={p.root:'a' if p.points_m[0,1]<p.points_m[-1,1] else 'b' for p in cut}
        inds=[r.roots.index(k) for k in r.IO['dn_roots']]
        self.dnw=(r.raw_bank([cut[i] for i in inds],sensors,self.front)@self.basis)*r.geom['route_gain'][inds,None,None].astype('f')
        self.targets=r.IO['an_roots_left_right']
        self.pending=np.zeros((n_frames*50+250,2),bool)
        self.volt=np.zeros((n_frames*200+500,96),'f')
        self.est=np.zeros_like(self.volt)
        rng=np.random.default_rng(4992)
        for i in range(0,len(self.volt),8000):
            n=min(8000,len(self.volt)-i)
            self.volt[i:i+n]=(self.front.noise(n,len(sensors),rng)[0]@self.basis)*1e6
        self.before=[[e['axon'] for e in self.pulse(i,0.,{}) if e['kind']=='axon_fired'] for i in range(4)]
        self.pattern[1]*=1.25
        self.pattern[3]=self.pattern[1]+self.pattern[2]
        self.kernels=[];self.cals=[];self.events=[];self.last={};self.requests=0
        readback.paths=cut
        for i,I in enumerate(self.pattern):
            result=self.pulse(i,0.,{})
            # Waveform engine in microvolts; compute transfer before projection.
            raw=readback.expected_evoked(result,sensors,2200,result)
            active=I!=0
            if active.any():
                field=self.model._lead(sensors,sites[active])@I[active]
                raw[:100]+=field;raw[100:200]-=field
            wave=(self.front.response(raw[None])[0,64:]@self.basis)*1e6
            self.kernels.append(wave)
            # Command-triggered stimulus-only calibration on 64 independent noise windows.
            cal=self.front.noise(len(wave)*64,len(sensors),rng)[0].reshape(64,len(wave),len(sensors)).mean(0)
            self.cals.append(wave+(cal@self.basis)*1e6)
        self.reference=[self.pulse(i,0.,{}) for i in range(4)]
        for i in range(4):
            targets={self.targets[k] for k in range(2) if i & (1<<k)}
            observed={v['axon'] for v in self.reference[i] if v['kind']=='axon_arrived' and v['end']==self.head[v['axon']]}
            assert targets==observed
        self.counts=np.array(c['read_site_ids'])
    def pulse(self,arm,time,last):
        accepted=[]
        self.clock.time_s=time
        for phase in (0,1):
            self.model.events=[]
            self.model.pulse_pattern(self.clock,(-1)**phase*self.pattern[arm],time+phase*.0001)
            fired={e['axon'] for e in self.model.events if e['kind']=='axon_fired' and time+phase*.0001-last.get(e['axon'],-1)>=.002}
            accepted.extend(e for e in self.model.events if e.get('axon') in fired)
            last.update({k:time+phase*.0001 for k in fired})
        return accepted
    def stimulate(self,start,requests,on):
        for ms,arr in enumerate(requests):
            arm=int(arr[0])+2*int(arr[1])
            if not arm:continue
            self.requests+=1
            use=arm if on else 0
            sample=start*200+ms*40
            evt=self.pulse(use,sample/40000,self.last)
            # Registered target sampler enforces refractory. Check cached physical waveform applicability.
            assert [(x['kind'],x['axon'],x.get('end')) for x in evt]==[(x['kind'],x['axon'],x.get('end')) for x in self.reference[use]]
            if use:
                wave=self.kernels[use];self.volt[sample:sample+len(wave)]+=wave
                self.est[sample:sample+len(wave)]+=self.cals[use]
            for e in evt:
                if e['kind']=='axon_arrived' and e['end']==self.head[e['axon']]:
                    self.events.append((sample,use,e['axon'],e['time_s']))
                    if e['axon'] in self.targets:
                        self.pending[int(np.ceil(e['time_s']*10000-1e-8)),self.targets.index(e['axon'])]=True
    def read_dn(self,frame,spikes):
        t=frame*200
        for i,j in zip(*spikes.nonzero()):
            wave=self.dnw[j];start=t+4*i;self.volt[start:start+len(wave)]+=wave
        # Last packet only, with 2 previous packets for the sorter's overlap context.
        end=t+200;start=max(0,end-600);v=self.volt[start:end]-self.est[start:end]
        calls=r.sort(r.scores(v,self.unit),self.gram,self.sigma,self.threshold)[0][-50:,:17]
        return calls.sum(0)
