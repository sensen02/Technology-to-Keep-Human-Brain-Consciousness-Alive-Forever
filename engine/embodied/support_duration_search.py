"""Episode-only exact support selection; never a runtime control or reward change."""
import numpy as np
from .body_reward import body_return

ORIGINAL = (.4, .25, .2, .1, .05)
STEPS = ((.2, .125, .1, .05), (.1, .0625, .05, .025))

def support_duration_score(samples, W, h0, dt=6.25e-6, duration=.55):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError('positive finite time grid required')
    if not np.isclose(round(duration/dt)*dt, duration, rtol=0, atol=1e-12):
        raise ValueError('duration must contain whole intervals')
    body_return(samples, W, h0, dt=dt, duration=duration)  # unchanged evidence guard
    a = np.asarray(samples, dtype=float); n = len(a)
    if not np.allclose(a[:,0], np.arange(1,n+1)*dt, rtol=0, atol=1e-11):
        raise ValueError('native endpoint grid required')
    foot, nf, h, tilt = a[:,1]/W, a[:,2], a[:,3]/h0, a[:,4]
    good = (h >= .85) & (tilt <= .35) & (nf == 0)
    def first(bad):
        ix = np.flatnonzero(bad)
        return int(ix[0]+1) if ix.size else None
    fail = first(~good); prefix = n if fail is None else fail-1
    edges = np.diff(np.r_[False,good,False].astype(int))
    starts, ends = np.flatnonzero(edges==1), np.flatnonzero(edges==-1)
    longest = int(max(ends-starts)) if len(starts) else 0
    secondary = float(foot[good].mean()) if good.any() else 0.0
    r = dict(prefix_good_ticks=prefix, goodprefix_ticks=prefix, prefix_good_s=prefix*dt,
        foot_mean_while_good=secondary, secondarymeanfoot=secondary,
        first_failure_tick=fail, first_failure_endpoint_s=fail*dt if fail else None,
        last_good_endpoint_s=prefix*dt, last_good_observed_time_s=float(a[prefix-1,0]) if prefix else 0.,
        first_failure_observed_time_s=float(a[fail-1,0]) if fail else None, censored=fail is None,
        cumulative_good_ticks=int(good.sum()), cumulative_good_s=float(good.sum()*dt),
        longest_good_ticks=longest, longest_good_s=longest*dt, good_segment_count=len(starts),
        support_recovery=bool(fail is not None and np.any(starts>=fail)),
        foot_clipped_integral_while_good_s=float(np.clip(foot[good],0,1).sum()*dt),
        early_0_10ms_height85_fail_count=int(np.sum(h[np.arange(n)*dt < .01]<.85)),
        height_min_ratio=float(h.min()), tilt_max_rad=float(tilt.max()),
        height_at10ms_ratio=float(h[min(n-1,round(.01/dt)-1)]), all_observations_finite=True)
    for key,bad in [('height85',h<.85),('tilt35',tilt>.35),('nonfoot',nf!=0)]:
        tick=first(bad); r['first_'+key+'_failure_tick']=tick
        r['first_'+key+'_failure_s']=tick*dt if tick else None
    return r

def score_tuple(metric):
    return (metric['prefix_good_ticks'], metric['foot_mean_while_good'])

def weight_key(weights):
    a=np.asarray(weights,dtype=float)
    if a.shape!=(66,5) or not np.isfinite(a).all() or np.any((a<0)|(a>4)) or not np.array_equal(a,np.tile(a[0],(66,1))) or a[0,4]!=.05:
        raise ValueError('66 identical feasible rows with fifth fixed .05 required')
    return tuple(a[0].tolist())

class SupportDurationSearch:
    """Two fixed coordinate passes, strict lexicographic incumbent, global cache."""
    def __init__(self, metric, provenance):
        self.center=np.tile(ORIGINAL,(66,1));self.metric=metric;self.label='original_cached_baseline'
        self.cache={weight_key(self.center):dict(label=self.label, metric=metric, provenance=provenance)}
        self.coordinate=0
    def probes(self):
        if self.coordinate>=8: raise ValueError('fixed search exhausted')
        p,i=divmod(self.coordinate,4); step=STEPS[p][i]
        plus=self.center.copy();minus=self.center.copy()
        plus[:,i]=np.clip(plus[:,i]+step,0,4);minus[:,i]=np.clip(minus[:,i]-step,0,4)
        return plus,minus
    def remember(self, weights, label, metric, provenance):
        key=weight_key(weights)
        if key in self.cache: raise ValueError('duplicate must reuse cached measurement')
        self.cache[key]=dict(label=label,metric=metric,provenance=provenance)
    def update(self, plus, minus):
        before=dict(label=self.label,metric=self.metric,theta=self.center[0].tolist())
        winner=None
        for value in (plus,minus):
            candidate=self.cache[weight_key(value)]
            if score_tuple(candidate['metric'])>score_tuple(self.metric) and (winner is None or score_tuple(candidate['metric'])>score_tuple(winner[1]['metric'])):
                winner=(value,candidate)
        if winner is not None:
            self.center=winner[0].copy();self.label=winner[1]['label'];self.metric=winner[1]['metric']
        self.coordinate+=1
        return dict(coordinate=self.coordinate,incumbent_before=before,incumbent_after=dict(label=self.label,metric=self.metric,theta=self.center[0].tolist()),changed=winner is not None)
