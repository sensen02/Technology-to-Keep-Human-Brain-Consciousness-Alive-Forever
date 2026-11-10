"""Integrate clipped first-order neural drive over each timestep (milliseconds)."""
import numpy as np

def step(start, forcing, dt_ms, tau_ms=5.):
 start=np.asarray(start,float);forcing=np.asarray(forcing,float);decay=np.exp(-dt_ms/tau_ms);end=decay*start+(1-decay)*forcing
 # Existing rate recurrence corresponds to constant spike/dt forcing in this bin.
 def primitive(t):return forcing*t+(start-forcing)*tau_ms*(-np.expm1(-t/tau_ms))
 area=primitive(dt_ms);above=(start>=1)&(end>=1);area[above]=dt_ms
 crossing=((start<1)&(end>1))|((start>1)&(end<1))
 for j in np.flatnonzero(crossing):
  t=-tau_ms*np.log((1-forcing[j])/(start[j]-forcing[j]));a=forcing[j]*t+(start[j]-forcing[j])*tau_ms*(-np.expm1(-t/tau_ms))
  area[j]=a+dt_ms-t if start[j]<1 else t+area[j]-a
 return end,area
