"""Full-body free dynamics diagnostic; zero neural/active control, not walking.
Each condition starts at the same saved pose. No state writes after initialization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import mujoco as mj
    from run_native_rk4_tick_consumption import MODEL, INITIAL
    from engine.embodied.sensory_channels import LEGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    conditions = [dict(name='baseline', stiffness=1., damping=1., bias=1.),
                  dict(name='no_joint_spring_damping', stiffness=0., damping=0., bias=1.),
                  dict(name='no_muscle_passive_bias', stiffness=1., damping=1., bias=0.),
                  dict(name='damping_x4', stiffness=1., damping=4., bias=1.),
                  dict(name='damping_x16', stiffness=1., damping=16., bias=1.),
                  dict(name='bias_quarter', stiffness=1., damping=1., bias=.25)]
    save('PREREG.json', dict(conditions=conditions, duration_s=.04, dt_s=.0001,
        purpose='S2/S4 integrated causal startup diagnosis before neural training',
        initialization='same saved qpos; fresh MjData, zero qvel/act/ctrl/external loads',
        interventions='static compiled parameter multipliers, no controllers or pose corrections',
        acceptance='descriptive original six physical gates, not training/walking acceptance',
        physical_units='NOT_ESTABLISHED', neural_ticks=0,
        sources={str(x):sha(x) for x in (MODEL, INITIAL, Path(__file__).resolve())}))
    ledger = dict(model_load=0, forward=0, step=0, completed=[], active=None)
    save('CALL_LEDGER.json', ledger)
    results = []
    try:
        for cfg in conditions:
            ledger['active'] = cfg['name']; ledger['model_load'] += 1; save('CALL_LEDGER.json',ledger)
            m = mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq,m.nv,m.nu,m.nbody,m.nmesh)==(49,48,90,73,71)
            m.opt.integrator = mj.mjtIntegrator.mjINT_RK4
            assert m.opt.timestep == .0001
            m.jnt_stiffness[:] *= cfg['stiffness']
            m.dof_damping[:] *= cfg['damping']
            # MuJoCo muscle bias uses its own biasprm; scale positive force only.
            assert np.all(m.actuator_biastype == mj.mjtBias.mjBIAS_MUSCLE)
            assert np.all(m.actuator_biasprm[:,2] > 0)
            m.actuator_biasprm[:,2] *= cfg['bias']
            if cfg['bias'] == 0:
                # force=0 enters automatic negative-force scaling branch, so disable explicitly.
                m.actuator_biastype[:] = mj.mjtBias.mjBIAS_NONE
            with np.load(INITIAL,allow_pickle=False) as z:
                initial = z['new_qpos'].copy()
            d = mj.MjData(m); d.qpos[:] = initial
            assert not any(np.any(x) for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))
            floor, = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)
            geom_leg = {}
            for g in range(m.ngeom):
                gn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
                bn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
                for i,leg in enumerate(LEGS):
                    if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)):
                        geom_leg[g] = i
            limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
            rows = []
            ledger['forward'] += 1; save('CALL_LEDGER.json',ledger); mj.mj_forward(m,d)
            for k in range(401):
                if k:
                    # Reserve actual public calls before execution. No quota enforcement.
                    ledger['step'] += 1; ledger['forward'] += 1
                    save('CALL_LEDGER.json',ledger)
                    mj.mj_step(m,d); mj.mj_forward(m,d)
                assert not any(np.any(x) for x in (d.ctrl,d.act,d.xfrc_applied,d.qfrc_applied))
                assert not any(w.number for w in d.warning)
                assert all(np.isfinite(x).all() for x in (d.qpos,d.qvel,d.qacc))
                contacts = np.zeros(6,dtype=int); force = np.zeros((6,3)); nonfoot = 0.
                for ci,c in enumerate(d.contact):
                    g1,g2 = int(c.geom1),int(c.geom2)
                    if floor not in (g1,g2): continue
                    other = g2 if g1==floor else g1
                    if other not in geom_leg:
                        nonfoot=max(nonfoot,-float(c.dist)); continue
                    li=geom_leg[other]
                    contacts[li] += int(c.dist<0)
                    f=np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                    force[li] += (1 if g1==floor else -1)*c.frame.reshape(3,3).T@f[:3]
                excess=np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
                rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),
                    act=d.act.copy(),ctrl=d.ctrl.copy(),contacts=contacts,foot_force=force,
                    nonfoot_depth=nonfoot,limit_excess=float(excess.max(initial=0)),
                    actuator_force=d.actuator_force.copy(),qfrc_passive=d.qfrc_passive.copy(),
                    qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),
                    warnings=np.asarray([w.number for w in d.warning])))
            a={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
            weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            height=a['qpos'][:,2]/initial[2]
            norms=np.linalg.norm(a['qpos'][:,3:7],axis=1)
            angle=2*np.arccos(np.clip(np.abs(a['qpos'][:,3:7]@initial[3:7])/(norms*np.linalg.norm(initial[3:7])),0,1))
            load=a['foot_force'][:,:,2].sum(1)/weight
            gates=np.column_stack((height>=.85,angle<=.35,(a['contacts']>0).sum(1)>=3,load>=.8,a['nonfoot_depth']<=1e-12,a['limit_excess']<=1e-6))
            good=gates.all(1)
            path=out/(cfg['name']+'.npz'); np.savez_compressed(path,**a,load=load,angle=angle,height=height,gates=gates,
                jnt_stiffness=m.jnt_stiffness,dof_damping=m.dof_damping,actuator_biasprm=m.actuator_biasprm,actuator_biastype=m.actuator_biastype)
            first={key:(float(a['time'][np.flatnonzero(~gates[:,i])[0]]*1000) if (~gates[:,i]).any() else None)
                   for i,key in enumerate(('height','angle','legs','load','nonfoot','limit'))}
            bad=np.flatnonzero(~good)
            result=dict(condition=cfg,trajectory=str(path),sha256=sha(path),weight_native=weight,
                first_failures_ms=first,all_gates_support_ms=float(np.trapezoid(good.astype(float),a['time'])*1000),
                initial_support_prefix_ms=float(a['time'][max(0,int(bad[0])-1)]*1000) if len(bad) else 40.,
                min_height_ratio=float(height.min()),max_angle=float(angle.max()),
                max_nonfoot_depth=float(a['nonfoot_depth'].max()),max_limit_excess=float(a['limit_excess'].max()),
                terminal_load=float(load[-1]),behavior_claim='NONE',neural_control=False)
            results.append(result); save('RESULTS.json',results)
            ledger['completed'].append(cfg['name']); ledger['active']=None; save('CALL_LEDGER.json',ledger)
        assert all(sha(x)==h for x,h in json.loads((out/'PREREG.json').read_text())['sources'].items())
        save('COMPLETE.json',dict(status='COMPLETE',conditions=len(results),physical_units='NOT_ESTABLISHED'))
        print(json.dumps(results,ensure_ascii=False))
    except BaseException:
        save('FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger)); raise


if __name__ == '__main__':
    main()
