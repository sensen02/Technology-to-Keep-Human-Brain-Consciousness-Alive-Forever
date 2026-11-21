"""Turn observable footprint patches into per-leg samples and bracketed load episodes."""
from pathlib import Path
import csv
import sys
import numpy as np
from estimate import write_rows
legs='LF LM LH RF RM RH'.split()

def extract(folder):
    rows=list(csv.DictReader((folder/'patches.csv').open()))
    feet=np.load(folder/'observed_feet_mm.npy')
    loads=np.zeros(feet.shape[:2])
    shared=np.zeros_like(loads,dtype=bool)
    centers=np.zeros_like(feet[:,:,:2]);mass=np.zeros_like(loads)
    for r in rows:
        names=r['legs'].split('|');t=int(r['frame'])
        if r['attribution']=='single':
            k=legs.index(names[0]);force=float(r['fz_uN'])
            loads[t,k]+=force
            centers[t,k]+=abs(force)*np.array([float(r['x_mm']),float(r['y_mm'])]);mass[t,k]+=abs(force)
        elif r['attribution']=='shared':
            for name in names:shared[t,legs.index(name)]=True
    loads[shared|~np.isfinite(feet[:,:,0])]=np.nan
    centers/=np.where(mass>0,mass,np.nan)[:,:,None]
    traces=[]
    events=[]
    for k,leg in enumerate(legs):
        state=np.where(~np.isfinite(loads[:,k]),'unknown',np.where(loads[:,k]>=.5,'loaded',
                       np.where(loads[:,k]<=-.5,'tension','below_gate')))
        for t,(fz,xy) in enumerate(zip(loads[:,k],centers[:,k])):
            traces.append(dict(time_ms=(t+1)*2,leg=leg,fz_uN=fz,x_mm=xy[0],y_mm=xy[1],state=state[t]))
        changes=np.flatnonzero(np.r_[True,state[1:]!=state[:-1],True])
        for begin,end in zip(changes,changes[1:]):
            if state[begin]!='loaded' or end-begin<2:continue
            peak=begin+np.argmax(loads[begin:end,k])
            events.append(dict(leg=leg,first_loaded_ms=2*(begin+1),last_loaded_ms=end*2,
                               prior_sample_ms='' if begin==0 else 2*begin,
                               next_sample_ms='' if end==len(state) else 2*(end+1),
                               peak_uN=loads[peak,k],peak_x_mm=centers[peak,k,0],peak_y_mm=centers[peak,k,1]))
    write_rows(folder/'feet.csv',list(traces[0]),sorted(traces,key=lambda r:r['time_ms']))
    write_rows(folder/'load_episodes.csv',['leg','first_loaded_ms','last_loaded_ms','prior_sample_ms','next_sample_ms','peak_uN','peak_x_mm','peak_y_mm'],events)
    return loads,events

if __name__=='__main__':extract(Path(sys.argv[1]))
