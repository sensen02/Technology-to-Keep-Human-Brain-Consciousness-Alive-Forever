"""Scale-free motion reversals, not contact or gait classification."""
import numpy as np
RULES=dict(window_frames=9,window_reason='fixed 11.25ms nominal 800Hz median despike, not fitted to gait',
 persistence_edges=3,min_interval_frames=24,
 interval_reason='30ms nominal separation flags unresolved brief reversals; never deletes candidates',
 deadband_fraction_of_segment_range=1e-6,
 missing='split at every nonfinite sample; never smooth or pair across gap',
 plateau='central sample of zero-slope run, interval endpoints retained; plateau flagged ambiguous',
 boundaries='segment prefixes/suffixes never closed into cycles')
def body_forward(raw,names):
 raw=np.asarray(raw,float);center=raw[:,names.index('Scutellum')]
 x=raw[:,names.index('Antenna_Base')]-center;norm=np.linalg.norm(x,axis=1)
 x=np.divide(x,norm[:,None],out=np.full_like(x,np.nan),where=norm[:,None]>0)
 legs=['T1L','T2L','T3L','T1R','T2R','T3R']
 return np.sum((raw[:,[names.index(k+'_TaTip') for k in legs]]-center[:,None,:])*x[:,None,:],axis=2)
def segments(valid):
 d=np.diff(np.r_[False,valid,False].astype(int));return zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))
def extract_turns(values,source_frames):
 v=np.asarray(values,float);frames=np.asarray(source_frames);events=[];cycles=[];boundaries=[];smooth=np.full(len(v),np.nan)
 if v.ndim!=1 or frames.shape!=v.shape or not np.all(np.diff(frames)==1):raise ValueError('consecutive equal length source frames required')
 for sid,(lo,hi) in enumerate(segments(np.isfinite(v))):
  half=RULES['window_frames']//2
  s=np.array([np.median(v[max(lo,i-half):min(hi,i+half+1)]) for i in range(lo,hi)])
  smooth[lo:hi]=s;tol=np.ptp(v[lo:hi])*RULES['deadband_fraction_of_segment_range'];diff=np.diff(s)
  signs=np.where(diff>tol,1,np.where(diff<-tol,-1,0));nonzero=np.flatnonzero(signs);local=[]
  for p,q in zip(nonzero[:-1],nonzero[1:]):
   if signs[p]==signs[q]:continue
   start=lo+p+1;stop=lo+q;idx=(start+stop)//2;flags=[]
   before=signs[max(0,p-2):p+1];after=signs[q:min(len(signs),q+3)]
   if len(before)<3 or len(after)<3 or not np.all(before==signs[p]) or not np.all(after==signs[q]):flags.append('SHORT_DIRECTION_PERSISTENCE')
   if stop>start:flags.append('PLATEAU_EXTREMUM')
   rawd=np.diff(v[max(lo,idx-half):min(hi,idx+half+1)])
   if np.count_nonzero(np.diff(np.sign(rawd))!=0)>2:flags.append('RAW_DIRECTION_JITTER')
   local.append(dict(segment=sid,index=int(idx),source_frame=int(frames[idx]),nominal_time_s=float(frames[idx]/800),
                     turn_interval_frames=[int(frames[start]),int(frames[stop])],
                     incoming='FORWARD' if signs[p]>0 else 'BACKWARD',outgoing='FORWARD' if signs[q]>0 else 'BACKWARD',
                     smoothed_forward=float(s[idx-lo]),raw_forward=float(v[idx]),ambiguity=flags,
                     previous_interval_frames=None,previous_interval_nominal_s=None))
  for i,e in enumerate(local):
   if i:
    interval=e['source_frame']-local[i-1]['source_frame'];e['previous_interval_frames']=interval;e['previous_interval_nominal_s']=interval/800
    if interval<RULES['min_interval_frames']:
     for candidate in (local[i-1],e):
      if 'SHORT_EVENT_INTERVAL' not in candidate['ambiguity']:candidate['ambiguity'].append('SHORT_EVENT_INTERVAL')
  for i in range(len(local)-2):
   triplet=local[i:i+3]
   if triplet[0]['incoming']!=triplet[2]['incoming']:continue
   flags=sorted(set(f for e in triplet for f in e['ambiguity']))
   cycles.append(dict(segment=sid,start_frame=triplet[0]['source_frame'],end_frame=triplet[2]['source_frame'],
                      interval_frames=triplet[2]['source_frame']-triplet[0]['source_frame'],ambiguity=flags,
                      label='CANDIDATE_CYCLE_NOT_STEP_COUNT'))
  boundaries.append(dict(segment=sid,start_frame=int(frames[lo]),end_frame=int(frames[hi-1]),
                         prefix_to_first_event=None if not local else [int(frames[lo]),local[0]['source_frame']],
                         last_event_to_suffix=None if not local else [local[-1]['source_frame'],int(frames[hi-1])],
                         label='INCOMPLETE_BOUNDARY_INTERVALS_NOT_CYCLES'))
  events.extend(local)
 return dict(events=events,candidate_cycles=cycles,boundaries=boundaries,smoothed=smooth)
