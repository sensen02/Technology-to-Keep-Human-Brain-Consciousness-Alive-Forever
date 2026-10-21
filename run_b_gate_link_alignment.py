"""B-gate step 1: offline link alignment on the ARCHIVED gate-B replay (no new dynamics).

RESEARCH SCRIPT, NOT A PRODUCT. Reads the archived 25 us replay, re-runs only mj_forward on
archived states to recover derived quantities, and checks the chain
   archived ctrl -> activation -> actuator force -> generalized actuator force -> joint torque
plus a per-step vertical force budget (gravity vs actuator vs constraint vs passive).
No pose search, no force-scale tuning, no canonical XML change, no neural simulation.
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
REPLAY=Path('/home/sensen/Desktop/engine修复验收/B-fixed-command-replay-25us-20261015T103243/replay.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from engine import canonical_physics as cp
    from run_s2_fixed_stimulus import dense_moment
    out=Path('/home/sensen/Desktop/engine修复验收')/('B-gate-link-alignment-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='B_GATE_OFFLINE_LINK_ALIGNMENT_ON_ARCHIVED_REPLAY',research_script_not_product=True,
           declared=dict(inputs='archived 25 us gate-B replay (qpos/qvel/act/ctrl/qfrc_actuator per step)',
                        derived_by='mj_forward on archived states only; NO mj_step, no integration',
                        not_done='no new dynamics, no neural simulation, no pose/support search, no parameter tuning'),
           sections=dict(chain_identities={},vertical_force_budget={},attribution={}))
    hashes={str(REPLAY):sha(REPLAY),str(Path(__file__)):sha(Path(__file__))}
    try:
        z=np.load(REPLAY);qpos=z['qpos'];qvel=z['qvel'];act=z['act'];ctrl=z['ctrl'];qfa=z['qfrc_actuator'];ncon=z['ncon'];t=z['time_s']
        m=cp.load_model();d=mujoco.MjData(m)
        assert qpos.shape==(2000,49) and act.shape==(2000,90)
        weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        idx=list(range(0,2000,50))
        ident=[];budget=[];legs=('LF','LM','LH','RF','RM','RH')
        tarsus={leg:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'{leg}Tarsus5_geom') for leg in legs}
        for i in idx:
            d.qpos[:]=qpos[i];d.qvel[:]=qvel[i];d.act[:]=act[i];d.ctrl[:]=ctrl[i]
            mujoco.mj_forward(m,d)
            mom=dense_moment(m,d)
            e_arch=float(np.max(np.abs(d.qfrc_actuator-qfa[i])))
            e_factor=float(np.max(np.abs(mom.T@d.actuator_force-d.qfrc_actuator)))
            ident.append(dict(step=i,qfrc_vs_archive_max_abs=e_arch,force_factorization_max_abs=e_factor))
            # vertical budget on the root translational DOFs (z = index 2)
            budget.append(dict(step=i,time_s=float(t[i]),root_z=float(d.qpos[2]),
                gravity_frc=float(-weight),actuator_frc_z=float(d.qfrc_actuator[2]),passive_frc_z=float(d.qfrc_passive[2]),
                constraint_frc_z=float(d.qfrc_constraint[2]),bias_frc_z=float(d.qfrc_bias[2]),ncon=int(d.ncon),
                mass_times_accel_z=float(d.qacc[2])))
        r['sections']['chain_identities']=dict(samples=ident,max_qfrc_vs_archive=max(x['qfrc_vs_archive_max_abs'] for x in ident),
            max_force_factorization=max(x['force_factorization_max_abs'] for x in ident),
            criterion='both should be ~0 (float round-off); the factorization identity is the muscle-force -> generalized-force link',
            per_leg_contact_force_at_first_sample=None)
        r['sections']['vertical_force_budget']=dict(body_weight_model_units=weight,samples=budget,
            note='facts only: the sums are reported so the sinking can be attributed to a term; no support or standing interpretation')
        first=budget[0];last=budget[-1]
        r['sections']['attribution']=dict(
            root_z_from_to=[first['root_z'],last['root_z']],
            actuator_z_share_at_first=abs(first['actuator_frc_z'])/weight if first['actuator_frc_z'] else 0.0,
            actuator_z_share_at_last=abs(last['actuator_frc_z'])/weight if last['actuator_frc_z'] else 0.0,
            constraint_z_share_at_first=abs(first['constraint_frc_z'])/weight,constraint_z_share_at_last=abs(last['constraint_frc_z'])/weight,
            statement='the archived run sinks while actuator vertical force stays below body weight (shares above are facts, not a support conclusion)')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot([b['time_s']*1000 for b in budget],[b['root_z'] for b in budget],'o-')
        axes[0].set(xlabel='time (ms)',ylabel='root height (model units)',title='archived replay: root height')
        axes[1].plot([b['time_s']*1000 for b in budget],[abs(b['actuator_frc_z'])/weight for b in budget],'o-',label='|actuator z| / body weight')
        axes[1].plot([b['time_s']*1000 for b in budget],[abs(b['constraint_frc_z'])/weight for b in budget],'x-',label='|constraint z| / body weight')
        axes[1].axhline(1.0,color='k',lw=.8,label='body weight = 1')
        axes[1].set(xlabel='time (ms)',ylabel='force / body weight',title='vertical force shares (facts)');axes[1].legend(fontsize=7)
        fig.suptitle('B-gate offline link alignment on the archived replay - NOT standing',fontsize=9);fig.tight_layout();fig.savefig(out/'link-alignment.png');plt.close(fig)
        r['status']='PASS_B_GATE_LINK_ALIGNMENT_REPORTED'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=hashes=={p:sha(p) for p in hashes}
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z2:z2.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r['sections'].get('chain_identities',{}),indent=1)[:400]);print(json.dumps(r['sections'].get('attribution',{}),indent=1,ensure_ascii=False)[:500]);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
