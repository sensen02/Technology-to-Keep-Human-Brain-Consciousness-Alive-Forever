"""Saved-only full-body vertical momentum budget, no forward/step/neural."""
from pathlib import Path
import json,sys,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_one_second_motor_pair import sha,dump
from run_quarter_second_pair import OUT as BASE
from analyze_early_support_region import O as SUPPORT,LEGS
import run_neural_muscle_closure as c
O=Path('/home/sensen/Desktop/engine修复验收/s2-early-vertical-momentum/20261027T201000Z')

def cumulative(force,t,rule):
 dt=np.diff(t);value=force[:-1] if rule=='left' else force[1:] if rule=='right' else (force[:-1]+force[1:])/2
 return np.r_[0.,np.cumsum(value*dt)]
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),SUPPORT/'SUPPORT_REGION_TIME_SERIES.json',SUPPORT/'PROJECTION_AND_CLASSIFICATION.json',SUPPORT/'MANIFEST_SHA256.json',BASE/'normal/BODY.npz',BASE/'MANIFEST_SHA256.json',Path(c.cp.CANONICAL_XML)]
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='historical0-40ms external vertical impulse ONLY',rules=['fixed intervals0/10/15/21.65/23.15/38.7375ms','gravity foot NONFOOT applied external force and measured COM momentum','left right trapezoid impulses all reported no timestamp shift','do not infer unique mechanism'],primary_quadrature='trapezoid of saved endpoint forces; original forces are post-step recomputed forward, not solver integration force samples',calls='one model metadata load, no forward/step/neural',closure='report residual and left/right discretization differences; no forced equality or new behavioral gate'))
 sm=json.loads((SUPPORT/'MANIFEST_SHA256.json').read_text());bm=json.loads((BASE/'MANIFEST_SHA256.json').read_text());assert sha(SUPPORT/'SUPPORT_REGION_TIME_SERIES.json')==sm['SUPPORT_REGION_TIME_SERIES.json'];assert sha(SUPPORT/'PROJECTION_AND_CLASSIFICATION.json')==sm['PROJECTION_AND_CLASSIFICATION.json'];assert sha(BASE/'normal/BODY.npz')==bm['normal/BODY.npz']
 meta=json.loads((SUPPORT/'PROJECTION_AND_CLASSIFICATION.json').read_text());rows=json.loads((SUPPORT/'SUPPORT_REGION_TIME_SERIES.json').read_text());z=np.load(BASE/'normal/BODY.npz');n=len(rows);t=z['time'][:n];up=np.array(meta['up']);mass=float(meta['total_mass']);gravity=np.array(meta['gravity']);weight=mass*np.linalg.norm(gravity);velocity=z['com_velocity'][:n]@up;momentum=mass*(velocity-velocity[0]);feet=np.array([[np.dot(r['perleg'][leg]['world_force_sum'],up) for leg in LEGS] for r in rows]);nonfoot=np.array([np.dot(r['perleg']['NONFOOT']['world_force_sum'],up) for r in rows]);external=(z['external_force'][:n,:,:3].sum(1))@up
 np.testing.assert_array_equal(velocity,np.array([r['com_vertical_velocity'] for r in rows]));np.testing.assert_allclose(feet.sum(1)+nonfoot,z['ground_force_world'][:n]@up,atol=1e-12,rtol=0)
 m=c.cp.load_model();assert abs(m.body_mass.sum()-mass)<1e-12;np.testing.assert_array_equal(m.opt.gravity,gravity)
 other_meta=dict(body_gravcomp=m.body_gravcomp,viscosity=float(m.opt.viscosity),density=float(m.opt.density),wind=m.opt.wind,neq=int(m.neq),external_applied_max=float(abs(z['external_force'][:n]).max()),gravity=gravity,mass=mass,weight=weight,COM_velocity_source='historical Observer sum(body_mass*jacBodyCom@qvel)/total_mass',saved_timing='initial forward at0; each later sample mj_step then explicit mj_forward then Observer, velocity is completed state; no time shift',internal_forces='muscle and internal joints/contact cancel for whole fly and are not added',qfrc_applied_source='historical episode creates fresh MjData and never writes qfrc_applied; no user generalized force',additional_force_caveat='nonzero gravcomp/fluid/equality would require explicit force accounting')
 assert not m.body_gravcomp.any() and m.opt.viscosity==0 and m.opt.density==0 and m.neq==0 and not z['external_force'][:n].any()
 dump(O/'EXTERNAL_FORCE_AUDIT.json',other_meta)
 forces=dict(gravity=np.full(n,-weight),foot=feet.sum(1),nonfoot=nonfoot,other_applied=external);forces['net']=sum(forces.values());impulses={rule:{k:cumulative(v,t,rule) for k,v in forces.items()} for rule in ['left','right','trapezoid']};residual={rule:momentum-impulses[rule]['net'] for rule in impulses}
 np.savez_compressed(O/'VERTICAL_MOMENTUM_TIME_SERIES.npz',time_s=t,up=up,mass=mass,weight=weight,perleg_vertical_force=feet,perleg_weight_normalized_force=feet/weight,nonfoot_vertical_force=nonfoot,other_applied_vertical_force=external,gravity_vertical_force=forces['gravity'],net_vertical_force=forces['net'],com_vertical_velocity=velocity,measured_momentum_change=momentum,**{rule+'_'+key+'_impulse':value for rule,items in impulses.items() for key,value in items.items()},**{rule+'_momentum_residual':value for rule,value in residual.items()})
 boundaries=[0,800,1200,1732,1852,3099];segments=[]
 for a,b in zip(boundaries[:-1],boundaries[1:]):
  rr=dict(start_index=a,end_index=b,start_s=float(t[a]),end_s=float(t[b]),actual_delta_momentum=float(momentum[b]-momentum[a]),initial_COM_vz=float(velocity[a]),final_COM_vz=float(velocity[b]))
  for rule in impulses:rr[rule]={key:float(v[b]-v[a]) for key,v in impulses[rule].items()};rr[rule]['residual']=rr['actual_delta_momentum']-rr[rule]['net']
  segments.append(rr)
 dump(O/'SEGMENT_BUDGETS.json',segments)
 with (O/'SEGMENT_BUDGETS.csv').open('w') as f:
  fields=['start_ms','end_ms','gravity_impulse','foot_impulse','nonfoot_impulse','other_impulse','net_impulse','actual_delta_momentum','residual'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in segments:w.writerow(dict(start_ms=r['start_s']*1000,end_ms=r['end_s']*1000,gravity_impulse=r['trapezoid']['gravity'],foot_impulse=r['trapezoid']['foot'],nonfoot_impulse=r['trapezoid']['nonfoot'],other_impulse=r['trapezoid']['other_applied'],net_impulse=r['trapezoid']['net'],actual_delta_momentum=r['actual_delta_momentum'],residual=r['trapezoid']['residual']))
 summary=dict(mass=mass,weight=weight,model_units_SI='NOT_ESTABLISHED',checks_resolution_s=.0000125,total_states=n,loss_rh_index=1732,momentum_change_at_rh_loss=float(momentum[1732]),net_trapezoid_impulse_at_rh_loss=float(impulses['trapezoid']['net'][1732]),residual_at_rh_loss=float(residual['trapezoid'][1732]),max_abs_momentum_residual={k:float(abs(v).max()) for k,v in residual.items()},net_impulse_quadrature_spread_max=float(np.max(abs(impulses['left']['net']-impulses['right']['net']))),first_three_segments_measured_delta=[r['actual_delta_momentum'] for r in segments[:3]],closure='DISCRETE_RESIDUAL_REPORTED_NOT_EXACT_CLOSURE',explanation='post-step forward contact forces do not equal original integration substep force evaluations; numerical quadrature and discrete nonlinear momentum update residuals retained, no missing force inferred or fitted')
 dump(O/'SUMMARY.json',summary)
 fig,axs=plt.subplots(4,1,figsize=(16,14));tm=t*1000
 for i,l in enumerate(LEGS):axs[0].plot(tm,feet[:,i]/weight,label=l)
 axs[0].axhline(1,color='k',ls='--',lw=.6);axs[0].set_title('Per-leg WORLD vertical reaction / model weight (not local normal)');axs[0].legend(ncol=6)
 axs[1].plot(tm,forces['foot']/weight,label='six feet');axs[1].plot(tm,nonfoot/weight,label='NONFOOT');axs[1].plot(tm,forces['net']/weight,label='net external');axs[1].axhline(0,color='k',lw=.5);axs[1].set_title('External vertical forces normalized by weight');axs[1].legend()
 axs[2].plot(tm,momentum,label='actual total mass * COM delta vz',lw=2)
 for rule in impulses:axs[2].plot(tm,impulses[rule]['net'],label=rule+' net impulse',ls='--')
 axs[2].set_title('Measured vertical momentum change vs original-time external impulse');axs[2].legend()
 axs[3].plot(tm,velocity,label='COM vertical velocity');ax2=axs[3].twinx();ax2.plot(tm,residual['trapezoid'],color='red',alpha=.6,label='momentum residual');ax2.set_ylabel('measured minus trapezoid impulse, model units');axs[3].legend(loc='upper left');ax2.legend(loc='lower left')
 for ax in axs:
  for b in boundaries[1:]:ax.axvline(tm[b],color='grey',lw=.5,alpha=.5)
  ax.grid(alpha=.2);ax.set_xlabel('saved physical ms')
 fig.suptitle('Historical early external momentum budget / no new trajectory / residuals not hidden');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'VERTICAL_MOMENTUM_BUDGET.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_EARLY_VERTICAL_MOMENTUM.png').write_bytes((O/'VERTICAL_MOMENTUM_BUDGET.png').read_bytes());assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=0,mj_step=0,neural_step=0,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary));print(json.dumps(segments))
if __name__=='__main__':main()
