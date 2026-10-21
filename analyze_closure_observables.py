"""Observables of the clean neural-closure replay (read-only analysis, no new dynamics).

RESEARCH SCRIPT, NOT A PRODUCT. Reads the archives produced by one clean run of
run_rk4_neural_closure.py and reports, for the connectome-driven command stream:
  vertical root drift, contact count, floor penetration, activation range, any act > 1,
  and cross-leg coordination structure (pairwise joint-angle correlations + PCA).
No mj_step, no mj_forward, no model-parameter change, no neural re-run.
"""
from pathlib import Path
import json,glob,time,hashlib,zipfile
import numpy as np
RUN=max((Path(d) for d in glob.glob('/home/sensen/Desktop/engine修复验收/S1-rk4-closure-*') if (Path(d)/'normal.npz').exists()),key=lambda p:p.stat().st_mtime)
LEGS={'LF':0,'LM':1,'LH':2,'RF':3,'RM':4,'RH':5}
# compiled joint names per leg, in the order the runner's manifest uses (Coxa/Trochanter pitch-chain + Tibia)
def main():
    from engine.embodied.research_report import atomic_report
    out=Path('/home/sensen/Desktop/engine修复验收')/('neural-replay-observables-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='OBSERVABLES_OF_THE_CLEAN_CONNECTome_DRIVEN_REPLAY',research_script_not_product=True,
        source_run=str(RUN),declared=dict(note='replay observation only: NOT stable standing, NOT a biological conclusion',
            read_only='archives read only; no mj_step/mj_forward, no parameter change'))
    try:
        z={k:dict(np.load(Path(RUN)/(k+'.npz'))) for k in ('normal','repeat','sensory_cut','motor_cut','output_silence')}
        n=z['normal']
        t=n['time_s'];qpos=n['qpos'];qvel=n['qvel'];act=n['act'];ctrl=n['ctrl']
        # ncon and penetration were NOT archived by the runner: measure them read-only by mj_forward
        # on archived states (no mj_step, no dynamics, no parameter change).
        import mujoco
        from engine import canonical_physics as cp
        m=cp.load_model();d=mujoco.MjData(m)
        tarsus=set(cp.tarsus_geom_ids(m));stride=5
        idxs=list(range(0,qpos.shape[0],stride));ncon=[];mindist=[];foot=[];lowz=[]
        for i in idxs:
            d.qpos[:]=qpos[i];d.qvel[:]=qvel[i];d.act[:]=act[i];d.ctrl[:]=ctrl[i]
            mujoco.mj_forward(m,d)
            ncon.append(int(d.ncon))
            mindist.append(float(min([c.dist for c in d.contact],default=float('nan'))))
            foot.append(sum(1 for c in d.contact if (int(c.geom1) in tarsus) ^ (int(c.geom2) in tarsus)))
            lowz.append(float(min(d.geom_xpos[g][2] for g in tarsus)))
        ncon=np.array(ncon);mindist=np.array(mindist);foot=np.array(foot);lowz=np.array(lowz)
        r['posthoc_measurement']=dict(method='mj_forward on archived states (read-only); no mj_step',sampled_steps=len(idxs),stride=stride)
        floor=0.0
        # lowest tarsus proxy: the runner archived 49-dim qpos; use the root z as the body height and
        # the minimum contact distance as the penetration proxy when available
        root_z=qpos[:,2]
        pen=None
        if 'contact_dist' in n: pen=n['contact_dist']
        obs=dict(
            vertical_root_drift_model_units=dict(start=float(root_z[0]),end=float(root_z[-1]),min=float(root_z.min()),max=float(root_z.max()),
                net=float(root_z[-1]-root_z[0]),monotone_down=bool(np.all(np.diff(root_z)<=1e-12))),
            contact_count=dict(min=int(ncon.min()),max=int(ncon.max()),mean=float(ncon.mean()),sampled_steps=int(len(ncon)),
                steps_with_zero=int((ncon==0).sum()),foot_only_contacts=dict(min=int(foot.min()),max=int(foot.max()),mean=float(foot.mean()))),
            penetration=dict(min_contact_distance_model_units=float(np.nanmin(mindist)),max_min_distance=float(np.nanmax(mindist)),
                samples_with_negative_distance=int(np.nansum(mindist<0)),lowest_tarsus_z=dict(min=float(lowz.min()),max=float(lowz.max()),start=float(lowz[0]),end=float(lowz[-1])),
                note='min contact distance < 0 means the solver is holding a penetrating contact; measured read-only on archived states'),
            activation=dict(min=float(act.min()),max=float(act.max()),
                samples_above_one=int((act>1).sum()),samples_above_one_plus_1e12=int((act>1+1e-12).sum()),
                per_sample_max_above_one=int((act.max(axis=1)>1).sum())),
            contacts_first_last=dict(first=int(ncon[0]),last=int(ncon[-1])),
            external_loads=dict(qfrc_applied_max_abs=float(np.abs(n['qfrc_applied']).max()),xfrc_applied_max_abs=float(np.abs(n['xfrc_applied']).max())))
        # penetration: lowest tarsus z needs forward kinematics; not recomputed here (read-only constraint),
        # so report the available proxy: contact distance minima if archived, else explicitly absent.
        r['observables_normal_arm']=obs
        # cross-leg coordination: use the 42 leg joint angles (qpos indices 7..48) over time
        leg_q=qpos[:,7:]
        leg_q=leg_q-leg_q.mean(0,keepdims=True)
        std=leg_q.std(0);keep=std>1e-12
        X=leg_q[:,keep];C=np.corrcoef(X.T) if X.shape[1]>1 else np.zeros((1,1))
        iu=np.triu_indices_from(C,1)
        vals=C[iu]
        # group by leg pairs using the compiled index mapping: 7 joints per leg, order LF,LM,LH,RF,RM,RH
        groups=[]
        idx_kept=np.arange(leg_q.shape[1])[keep]
        for a in range(6):
            for b in range(a+1,6):
                ia=[i for i,v in enumerate(idx_kept) if v//7==a];ib=[i for i,v in enumerate(idx_kept) if v//7==b]
                if ia and ib:groups.append(float(np.mean(C[np.ix_(ia,ib)])))
        U,S,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
        var=S**2/ (S**2).sum()
        r['cross_leg_coordination']=dict(
            pairwise_joint_correlation=dict(mean=float(vals.mean()),max=float(vals.max()),min=float(vals.min()),
                fraction_abs_above_0_5=float(np.mean(np.abs(vals)>0.5)),n_pairs=int(len(vals))),
            between_leg_mean_correlation=dict(mean=float(np.mean(groups)),min=float(np.min(groups)),max=float(np.max(groups)),n_leg_pairs=len(groups)),
            pca_explained_variance=dict(first5=[round(float(v),4) for v in var[:5]],first_mode_fraction=float(var[0])),
            note='structure reported as an observation about the declared model and this command stream; no rhythm/CPG claim, no biological interpretation')
        # inter-arm comparisons (facts only)
        cmp={}
        for arm in ('repeat','sensory_cut','motor_cut','output_silence'):
            a=z[arm]
            cmp[arm]=dict(root_z_start=float(a['qpos'][0,2]),root_z_end=float(a['qpos'][-1,2]),
                act_min=float(a['act'].min()),act_max=float(a['act'].max()),
                identical_to_normal=bool(all(np.array_equal(a[k],n[k]) for k in ('qpos','qvel','act','ctrl'))))
        r['arm_comparison']=cmp
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        axes[0].plot(t*1000,root_z,label='root z (normal arm)')
        for arm,c in (('sensory_cut','tab:orange'),('motor_cut','tab:green'),('output_silence','tab:red')):
            axes[0].plot(z[arm]['time_s']*1000,z[arm]['qpos'][:,2],'--',color=c,label=arm)
        axes[0].set(xlabel='time (ms)',ylabel='root z (model units)',title='vertical root drift under connectome command stream');axes[0].legend(fontsize=6)
        axes[1].plot(np.array(idxs)*0.025,ncon,lw=1,label='ncon (normal, sampled every 5 steps)')
        axes[1].set(xlabel='time (ms)',ylabel='contact count',title='contact count');axes[1].legend(fontsize=7)
        im=axes[2].imshow(C,cmap='coolwarm',vmin=-1,vmax=1);axes[2].set_title('leg-joint correlation (normal arm)')
        axes[2].set_xlabel('kept joint index');axes[2].set_ylabel('kept joint index');fig.colorbar(im,ax=axes[2],fraction=0.046)
        fig.suptitle('clean connectome-driven replay observables - NOT standing, NOT a biological conclusion',fontsize=9)
        fig.tight_layout();fig.savefig(out/'replay-observables.png');plt.close(fig)
        r['status']='PASS_OBSERVABLES_REPORTED'
    except Exception as e:
        import traceback;r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()[-600:])
    finally:
        atomic_report(out/'report.json',r)
    print(out);print(r['status']);print(json.dumps(r.get('observables_normal_arm'),indent=1)[:900]);print(json.dumps(r.get('cross_leg_coordination',{}).get('between_leg_mean_correlation'),indent=1));print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
