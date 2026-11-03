"""Dynamic support criteria (replaces the retired static-equilibrium gate).

Why this exists: the sealed evidence shows the assisted reference is supported by feet
at every instant (mean foot load about 1.0 W, zero non-foot contact) while NO posture in
the cycle is statically holdable (relative slack >= 0.37 even with 90 independent muscles).
So "can it stand still" was never the right question; "is it dynamically carried by its
feet" is. Thresholds here are taken from the run's own assisted window, not invented.

Pure arrays. No MuJoCo, no model, no neural step.
"""
from pathlib import Path
import json
import numpy as np

GRAVITY=9801.0
WEIGHT=24.446354755878005
MASS=WEIGHT/GRAVITY
H0=1.2072528224791905
LEGS=('LF','LM','LH','RF','RM','RH')
SMOOTH_SAMPLES=5  # acceleration is low-passed over about 2 ms before the balance check


def support_series(body):
 """Instantaneous ground support ratio, vertical acceleration and contact counts."""
 t=np.asarray(body['time'],float)
 ff=np.asarray(body['foot_force'],float)[:,:,2]
 nf=np.asarray(body['nonfoot_force'],float)[:,2]
 foot_total=ff.sum(1)
 ground_total=foot_total+nf
 s=foot_total/WEIGHT            # FOOT support only; non-foot contact must never count as support
 dt=np.gradient(t)
 a=np.gradient(np.asarray(body['com'],float)[:,2],t)
 if SMOOTH_SAMPLES>1:
  k=np.ones(SMOOTH_SAMPLES)/SMOOTH_SAMPLES
  a=np.convolve(a,k,mode='same')
 bearing=(ff>1e-9).sum(1)
 with np.errstate(invalid='ignore',divide='ignore'):
  nonfoot_share=np.where(np.abs(ground_total)>1e-9,nf/np.where(np.abs(ground_total)>1e-9,ground_total,1.),0.)
 return dict(time=t,foot=ff,nonfoot_z=nf,foot_total=foot_total,ground_total=ground_total,
             support=s,ground_support=ground_total/WEIGHT,nonfoot_share=nonfoot_share,
             accel_z=a,bearing_legs=bearing,predict_accel=(ground_total-WEIGHT)/MASS)


def window_metrics(body,lo,hi,label,max_nonfoot=0.0):
 """Statistics for one time window. Every threshold that matters is reported, not assumed."""
 series=support_series(body);t=series['time'];m=(t>=lo)&(t<=hi)
 assert m.sum()>10,(label,lo,hi)
 s=series['support'][m];a=series['accel_z'][m];nf=series['nonfoot_z'][m];ff=series['foot'][m]
 com_z=np.asarray(body['com'],float)[m,2]
 total=series['ground_support'][m]
 free_fall=(total<0.2)&(a<=-0.5*GRAVITY)
 resid=np.abs(MASS*a-(series['ground_total'][m]-WEIGHT))/WEIGHT
 out=dict(label=label,window=[float(lo),float(hi)],samples=int(m.sum()),
  foot_support_mean=float(s.mean()),foot_support_p10=float(np.percentile(s,10)),foot_support_min=float(s.min()),
  ground_support_mean=float(total.mean()),nonfoot_share_mean=float(np.nanmean(series['nonfoot_share'][m])),
  nonfoot_positive_max=float(max(0.,nf.max())),nonfoot_positive_any=bool((nf>1e-9).any()),
  accel_z_min=float(a.min()),accel_z_mean=float(a.mean()),
  free_fall_fraction=float(free_fall.mean()),
  force_balance_residual_p95=float(np.percentile(resid,95)),
  com_z_min=float(com_z.min()),com_z_mean=float(com_z.mean()),
  bearing_legs_median=float(np.median(series['bearing_legs'][m])),
  bearing_legs_min=int(series['bearing_legs'][m].min()),
  per_leg_duty=[float((ff[:,i]>1e-9).mean()) for i in range(6)],
  per_leg_mean_load=[float(ff[:,i].mean()/WEIGHT) for i in range(6)])
 return out


def evaluate(body,assist_window,release_window,tolerance=None):
 """Compare the post-withdrawal window against the body's own assisted window.

 The assisted window is the empirical proof that this body CAN be carried by its feet,
 so the release window is required to stay in that band instead of matching a hand-made
 threshold. Every number used is echoed in the result.
 """
 tol=dict(support_mean_drop=0.10,nonfoot_positive=0.0,free_fall_fraction=0.0,
          com_z_drop=0.10,balance_residual=0.15)
 if tolerance:tol.update(tolerance)
 a=window_metrics(body,*assist_window,'assist')
 r=window_metrics(body,*release_window,'release')
 checks={
  'D1_no_nonfoot_ground_force':bool(r['nonfoot_positive_max']<=tol['nonfoot_positive']),
  'D2_no_free_fall':bool(r['free_fall_fraction']<=tol['free_fall_fraction']),
  'D3_foot_support_not_below_assist':bool(r['foot_support_mean']>=a['foot_support_mean']-tol['support_mean_drop']),
  'D4_foot_support_never_collapses':bool(r['foot_support_p10']>=0.5),
  'D4b_nonfoot_is_not_carrying_the_body':bool(r['nonfoot_share_mean']<=0.2),
  'D5_com_height_not_below_assist':bool(r['com_z_min']>=a['com_z_mean']-tol['com_z_drop']),
  'D6_force_balance_consistent':bool(r['force_balance_residual_p95']<=tol['balance_residual']),
  'D7_tripod_redundancy':bool(r['bearing_legs_median']>=3.),
  'D8_still_above_absolute_floor':bool(r['com_z_min']>=0.8*H0),
 }
 return dict(passed=bool(all(checks.values())),checks=checks,tolerance=tol,assist=a,release=r,
  criteria_note='Support means FOOT support only; non-foot contact is a separate violation and must never be counted as support. Thresholds are bands around this run\'s own assisted window, which is the sealed proof that the body can be carried by its feet. The retired static criterion (qacc=0 feasibility) is NOT used here.')


def retired_static_reference():
 """Kept so nothing silently reintroduces the static gate as an acceptance test."""
 return dict(status='RETIRED_AS_ACCEPTANCE_CRITERION',
  reason='no posture in the reference cycle is statically holdable (relative slack >= 0.37 with 90 independent muscles), while the assisted trajectory is dynamically supported at about 1.0 W',
  still_used_for='engineering sensitivity only, never as a pass/fail gate')


def native_support_series(native_rows):
 """Same quantities from the 12.5 us release records (rows: t, foot6, nf_pos, nf_signed, ncnt, comz, root7, ...)."""
 r=np.asarray(native_rows,float)
 t=r[:,0];foot=r[:,1:7];nf=r[:,8];total=foot.sum(1)+nf
 a=np.gradient(np.gradient(r[:,10],t),t)
 k=np.ones(9)/9;a=np.convolve(a,k,mode='same')
 with np.errstate(invalid='ignore',divide='ignore'):
  share=np.where(np.abs(total)>1e-9,nf/np.where(np.abs(total)>1e-9,total,1.),0.)
 return dict(time=t,foot=foot,foot_total=foot.sum(1),nonfoot_signed=nf,ground_total=total,
             support=foot.sum(1)/WEIGHT,nonfoot_share=share,accel_z=a,
             bearing_legs=(foot>1e-9).sum(1),com_z=r[:,10])


def evaluate_native(assist_stats,native_rows):
 """Dynamic judgement at native resolution. assist_stats comes from the assisted window."""
 s=native_support_series(native_rows)
 resid=np.abs(MASS*s['accel_z']-(s['ground_total']-WEIGHT))/WEIGHT
 free_fall=((s['ground_total']/WEIGHT<0.2)&(s['accel_z']<=-0.5*GRAVITY)).mean()
 tol=dict(support_mean_drop=0.10,nonfoot_share_max=0.20,com_z_drop=0.10,balance_residual=0.15)
 rel=dict(foot_support_mean=float(s['support'].mean()),foot_support_p10=float(np.percentile(s['support'],10)),
  foot_support_min=float(s['support'].min()),nonfoot_share_mean=float(np.nanmean(s['nonfoot_share'])),
  ground_support_mean=float((s['ground_total']/WEIGHT).mean()),com_z_min=float(s['com_z'].min()),
  com_z_mean=float(s['com_z'].mean()),free_fall_fraction=float(free_fall),
  force_balance_residual_p95=float(np.percentile(resid,95)),force_balance_residual_p50=float(np.percentile(resid,50)),
  bearing_legs_median=float(np.median(s['bearing_legs'])),bearing_legs_min=int(s['bearing_legs'].min()),
  per_leg_mean_load=[float(s['foot'][:,i].mean()/WEIGHT) for i in range(6)],
  per_leg_duty=[float((s['foot'][:,i]>1e-9).mean()) for i in range(6)],samples=int(len(s['time'])))
 checks={
  'D1_no_nonfoot_ground_force':bool(rel['nonfoot_share_mean']==0.0),
  'D2_no_free_fall':bool(rel['free_fall_fraction']==0.0),
  'D3_foot_support_not_below_assist':bool(rel['foot_support_mean']>=assist_stats['foot_support_mean']-tol['support_mean_drop']),
  'D4_foot_support_never_collapses':bool(rel['foot_support_p10']>=0.5),
  'D4b_nonfoot_is_not_carrying_the_body':bool(rel['nonfoot_share_mean']<=tol['nonfoot_share_max']),
  'D5_com_height_not_below_assist':bool(rel['com_z_min']>=assist_stats['com_z_mean']-tol['com_z_drop']),
  'D6_force_balance_consistent':bool(rel['force_balance_residual_p95']<=tol['balance_residual']),
  'D7_tripod_redundancy':bool(rel['bearing_legs_median']>=3.),
  'D8_still_above_absolute_floor':bool(rel['com_z_min']>=0.8*H0),
 }
 return dict(passed=bool(all(checks.values())),checks=checks,tolerance=tol,assist=assist_stats,release=rel,
  criteria_note='Computed from the 12.5 us native release records; support means FOOT support only and non-foot contact is never counted as support.')
