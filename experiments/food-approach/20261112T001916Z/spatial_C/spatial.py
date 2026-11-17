"""Environment-only spatial source/arrival equations; imports numerical libs only at runtime."""
from contracts import ROOT,read
import numpy as np
def layout(name):return next(c for c in read(ROOT/'PREREG_PROPOSAL.json')['layouts'] if c['id']==name)
def anchor_from_head(head_pos,head_rot,local,cfg):
 sites=np.asarray(head_pos)+np.asarray(local)@np.asarray(head_rot).T;mid=sites.mean(0);ex=np.asarray(head_rot)[:2,0].copy();ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0]])
 return mid[:2]+cfg['forward']*ex+cfg['forward']*np.tan(cfg['angle'])*ey

def concentration_at(name,elapsed,sites,center):
 cfg=read(ROOT/'PREREG_PROPOSAL.json')['source'];u=np.clip((float(elapsed)-cfg['onset_ANN_s'])/cfg['rise_s'],0.,1.);on=u*u*(3-2*u);amp=0. if name=='C_noodor' else cfg['amplitude']
 xy=np.asarray(sites)[:,:2];delta=(xy-np.asarray(center))/cfg['sigma'];return amp*on*np.exp(-.5*np.sum(delta*delta,axis=1))
def environment_record(sites,center,valid):
 if not valid:return dict(source_anchor_valid=False,source_center_xy=np.zeros(2),center_distance=0.,edge_distance=0.,inside_patch=False)
 radius=read(ROOT/'PREREG_PROPOSAL.json')['source']['patch_radius'];distance=float(np.linalg.norm(np.asarray(sites).mean(0)[:2]-center));return dict(source_anchor_valid=True,source_center_xy=np.asarray(center).copy(),center_distance=distance,edge_distance=max(0.,distance-radius),inside_patch=bool(distance<=radius))
def arrival_metrics(time,inside,elapsed):
 count=0;start=None;maxspan=0.;confirmed=None;first=None;entries=exits=0;last=False;everentry=False;leftafter=False
 for t,hit,e in zip(time,inside,elapsed):
  hit=bool(hit and e>=.6-1e-8)
  if hit and not last:entries+=1;everentry=True;first=float(e) if first is None else first
  if last and not hit:exits+=1;leftafter=everentry
  if hit:
   if count==0:start=float(t)
   count+=1;span=float(t)-start;maxspan=max(maxspan,span)
   if count>=201 and span>=.2-1e-8 and confirmed is None:confirmed=float(e)
  else:count=0;start=None
  last=hit
 return dict(arrived_endpoint_proxy=confirmed is not None,first_entry_elapsed_s=first,confirmed_elapsed_s=confirmed,max_contiguous_endpoint_span_s=maxspan,entry_count=entries,exit_count=exits,left_after_entry=leftafter,overshoot_or_pass_through=leftafter,required_consecutive_endpoints=201,required_actual_span_s=.2,continuous_inside_not_proven=True,ingestion=False,parking=False)
