"""Offline S111 0-2ms six-leg load and root-DOF budget; no physics."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/s111-pool-synergy-response/20261024T054520Z')
ARMS=('baseline','pool3_up','pool15_up','both_up')

def main():
    allrows=[]; summary=[]; hashes={}
    arrays={}
    for arm in ARMS:
        f=OUT/arm/'TRAJECTORY.npz'; hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        arrays[arm]=a;t=a['time']; sel=t<=.002+1e-12; pre=t<.0017-1e-12; idx=np.flatnonzero(sel)
        # Root free-joint z translation is qpos2/qvel2/dof2, per saved XML convention.
        leg=a['foot_force'][:,:,2]; total=leg.sum(1); qz=a['qvel'][:,2]; az=a['qacc'][:,2]
        vertical_components={k:a[k][:,2] for k in ('Ma','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint','qfrc_applied','RHS','residual')}
        # Ma=actuator+passive+constraint+applied-bias, generalized root coordinate only.
        check=vertical_components['Ma']-vertical_components['RHS']
        contact=a['contacts'];first_contact=[float(t[np.flatnonzero(contact[:,j]>0)[0]]*1000) if np.any(contact[:,j]>0) else None for j in range(6)]
        # impulse is generalized root DOF2 impulse, not COM impulse.
        impulse={k:float(np.trapezoid(v[sel],t[sel])) for k,v in vertical_components.items()}
        root_impulse_delta=float(np.trapezoid((vertical_components['RHS']-vertical_components['Ma'])[sel],t[sel]))
        m=dict(arm=arm,window_ms=[0,2],pre_drop_window_ms=[0,1.7],root_coordinate='free joint qpos[2]/qvel[2], generalized DOF2',
          root_qvel_0_2ms_min=float(qz[sel].min()),root_qvel_0_2ms_max=float(qz[sel].max()),root_qacc_0_2ms_min=float(az[sel].min()),root_qacc_0_2ms_max=float(az[sel].max()),
          root_qacc_at_0=float(az[0]),root_qacc_at_1p6ms=float(az[16]),root_qvel_at_1p6ms=float(qz[16]),
          leg_fz_0_2ms_min=leg[sel].min(0).tolist(),leg_fz_0_2ms_max=leg[sel].max(0).tolist(),leg_fz_at_0p1ms=leg[1].tolist(),leg_fz_at_1p6ms=leg[16].tolist(),
          leg_fz_integral_0_1p7ms=np.trapezoid(leg[pre],t[pre],axis=0).tolist(),total_fz_integral_0_1p7ms=float(np.trapezoid(total[pre],t[pre])),
          contact_first_ms=first_contact,contact_legs_at_1p6ms=int((contact[16]>0).sum()),contact_legs_at_1p7ms=int((contact[17]>0).sum()),
          root_generalized_impulse_0_2ms=impulse,root_generalized_balance_residual_max_0_2ms=float(np.max(np.abs(check[sel]))),
          root_generalized_net_rhs_impulse_0_2ms=root_impulse_delta,vertical_foot_contact_force_is_not_root_generalized_force=True,
          missing_for_center_of_mass=['body COM position/velocity/acceleration','all-body external resultant expressed in world z','contact force application point/moment resultant'])
        summary.append(m)
        for j,legname in enumerate(('LF','LH','LM','RF','RH','RM')):
          allrows.append(dict(arm=arm,leg=legname,leg_index=j,force_fz_0p1ms=float(leg[1,j]),force_fz_1p6ms=float(leg[16,j]),force_fz_min_0_2ms=float(leg[sel,j].min()),force_fz_max_0_2ms=float(leg[sel,j].max()),force_fz_impulse_0_1p7ms=float(np.trapezoid(leg[pre,j],t[pre])),contact_at_1p6ms=int(contact[16,j]>0),contact_at_1p7ms=int(contact[17,j]>0)))
    # Pairwise changes, same pool3 background and same time indices.
    pairs=[]
    for left,right in (('baseline','pool15_up'),('pool3_up','both_up')):
      a,b=arrays[left],arrays[right];sel=a['time']<=.002+1e-12;pre=a['time']<.0017-1e-12
      pairs.append(dict(reference=left,intervention=right,leg_fz_delta_0p1ms=(b['foot_force'][1,:,2]-a['foot_force'][1,:,2]).tolist(),leg_fz_delta_1p6ms=(b['foot_force'][16,:,2]-a['foot_force'][16,:,2]).tolist(),leg_fz_impulse_delta_0_1p7ms=(np.trapezoid(b['foot_force'][pre,:,2],a['time'][pre],axis=0)-np.trapezoid(a['foot_force'][pre,:,2],a['time'][pre],axis=0)).tolist(),total_fz_delta_1p6ms=float((b['foot_force'][16,:,2]-a['foot_force'][16,:,2]).sum()),root_qacc_delta_1p6ms=float(b['qacc'][16,2]-a['qacc'][16,2]),root_qvel_delta_1p6ms=float(b['qvel'][16,2]-a['qvel'][16,2]),root_RHS_delta_1p6ms=float(b['RHS'][16,2]-a['RHS'][16,2]),root_Ma_delta_1p6ms=float(b['Ma'][16,2]-a['Ma'][16,2]),interpretation='Observed paired changes; do not allocate joint torque to individual leg support or infer causality from time alignment.'))
    data=dict(passed=True,arms=summary,pairs=pairs,legs_order=['LF','LH','LM','RF','RH','RM'],root_coordinate_note='DOF2 is free-joint generalized z coordinate; it is not whole-body COM acceleration/force.',source_hashes=hashes,new_physics_calls=0,new_neural_ticks=0,conclusion='Early contact load release and insufficient drive establishment remain observationally confounded; saved arrays do not contain sufficient COM resultant data to choose one mechanism.')
    (OUT/'VERTICAL_LOAD_BUDGET.json').write_text(json.dumps(data,indent=2)+'\n')
    with (OUT/'VERTICAL_LEG_LOADS.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    # Figure: six leg lines per arm; total load, root z qvel/qacc; 0-2ms.
    fig,ax=plt.subplots(4,1,figsize=(13,14),sharex=True)
    colors=plt.cm.tab10(np.arange(6))
    for arm in ARMS:
      a=arrays[arm]; sel=a['time']<=.002+1e-12;t=a['time'][sel]*1000
      for j,n in enumerate(('LF','LH','LM','RF','RH','RM')): ax[0].plot(t,a['foot_force'][sel,j,2],color=colors[j],alpha=.75,label=arm+' '+n if arm==ARMS[0] else None)
      ax[1].plot(t,a['foot_force'][sel,:,2].sum(1),label=arm);ax[2].plot(t,a['qvel'][sel,2],label=arm);ax[3].plot(t,a['qacc'][sel,2],label=arm)
    ax[0].set_ylabel('leg Fz native');ax[1].set_ylabel('sum foot Fz');ax[2].set_ylabel('root qvel[2]');ax[3].set_ylabel('root qacc[2]');ax[3].set_xlabel('physical time (ms)')
    for x in ax:x.grid(alpha=.25);x.legend(fontsize=8,ncol=3)
    fig.suptitle('S111 0–2 ms vertical load budget; root DOF is not COM; saved-array offline analysis')
    fig.tight_layout();fig.savefig(OUT/'VERTICAL_LOAD_BUDGET.png',dpi=150);plt.close(fig)
    print('saved',OUT/'VERTICAL_LOAD_BUDGET.json',OUT/'VERTICAL_LEG_LOADS.csv',OUT/'VERTICAL_LOAD_BUDGET.png')
if __name__=='__main__':main()
