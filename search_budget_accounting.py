"""Explicit atomic finite-difference accounting, engine-independent core (S172)."""
from pathlib import Path
import json
import numpy as np
class BudgetExhausted(RuntimeError):pass
class ExplicitJacobian:
 def __init__(self,callback,n,cap=200,initial_forward=1,forward_cost=2,output=None,counters=None,key_function=None):
  self.callback=callback;self.n=n;self.cap=cap;self.used=initial_forward;self.cost=forward_cost;self.output=Path(output) if output else None;self.counters=counters;self.key_function=key_function or (lambda x:np.asarray(x,dtype=np.float64).tobytes());self.cache={};self.events=[];self.bases={};self.jacobians=0;self.inner=0;self.updates=0;self.previous_base=None
 def emit(self,**event):
  self.events.append(dict(event_id=len(self.events)+1,**event))
  if self.output:self.output.write_text(json.dumps(dict(forward=self.used,inner_solves=self.inner,complete_jacobians=self.jacobians,jacobian_base_transitions=self.updates,events=self.events),indent=2))
 def key(self,x):
  x=np.ascontiguousarray(x,dtype=np.float64);assert x.shape==(self.n,) and np.isfinite(x).all();return self.key_function(x)
 def base(self,x):
  k=self.key(x)
  if k not in self.bases:self.bases[k]=len(self.bases)+1
  return self.bases[k]
 def evaluate(self,x,role,base_id,variable=None,perturbation=None):
  x=np.asarray(x,dtype=np.float64);k=self.key(x);hit=k in self.cache
  if hit:self.emit(role=role,base_id=base_id,variable=variable,perturbation=perturbation,cache_hit=True,forward_delta=0,inner_delta=0);return self.cache[k].copy()
  if self.used+self.cost>self.cap:self.emit(role='BUDGET_EXIT_BEFORE_OBJECTIVE',base_id=base_id,remaining=self.cap-self.used);raise BudgetExhausted('No full paired objective budget')
  before=self.counters() if self.counters else (self.used,self.inner)
  success=False
  try:
   value=np.asarray(self.callback(x.copy()),dtype=float);assert value.ndim==1 and np.isfinite(value).all();success=True
  finally:
   after=self.counters() if self.counters else (before[0]+self.cost,before[1]+1);df=after[0]-before[0];di=after[1]-before[1];self.used+=df;self.inner+=di;assert self.used<=self.cap
   self.emit(role=role,base_id=base_id,variable=variable,perturbation=perturbation,cache_hit=False,forward_delta=df,inner_delta=di,success=success,x=x.tolist())
  self.cache[k]=value.copy();return value.copy()
 def fun(self,x):return self.evaluate(x,'objective',self.base(x))
 def schedule(self,x):
  x=np.asarray(x,dtype=float);assert np.all(x>=0) and np.all(x<=1);steps=[]
  for j in range(self.n):
   h=1e-5*abs(x[j]);h=h if h>0 else np.sqrt(np.finfo(float).eps)
   if x[j]+h>1:h=-h
   y=x.copy();y[j]+=h;h=float(y[j]-x[j]);assert h!=0 and 0<=y[j]<=1;steps.append((j,y,h))
  return steps
 def jac(self,x):
  x=np.asarray(x,dtype=float);bid=self.base(x);plan=self.schedule(x);needed={self.key(x)}|{self.key(y) for _,y,_ in plan};miss=sum(k not in self.cache for k in needed);required=miss*self.cost
  if self.used+required>self.cap:
   self.emit(role='BUDGET_EXIT_BEFORE_COMPLETE_JACOBIAN',base_id=bid,required_forward=required,remaining=self.cap-self.used,planned_probes=self.n);raise BudgetExhausted('Insufficient budget for COMPLETE Jacobian; none evaluated')
  self.emit(role='JACOBIAN_BEGIN',base_id=bid,required_forward=required,planned_probes=self.n)
  f=self.evaluate(x,'jacobian_base',bid);cols=[]
  for j,y,h in plan:cols.append((self.evaluate(y,'difference_probe',bid,j,h)-f)/h)
  J=np.column_stack(cols);assert J.shape==(len(f),self.n)
  self.jacobians+=1
  if self.previous_base is not None and self.previous_base!=self.key(x):self.updates+=1
  self.previous_base=self.key(x);self.emit(role='JACOBIAN_COMPLETE',base_id=bid,columns=self.n);return J
