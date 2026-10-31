"""Matched neural-only computational initialization then 10ms free full-body loop."""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np
from run_native_rk4_gate_training import sha, atomic_json, atomic_npz, callback_snapshot, require_no_callbacks, raw_observation_record
from run_native_rk4_tick_consumption import MODEL, INITIAL, ENDPOINTS, ROOT

WEIGHTS = Path('/home/sensen/Desktop/engine修复验收/s102-tick-training-second/20261023T224501Z/run/round00_minus/TRAJECTORY.npz')


def main():
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output; out.mkdir(parents=True,exist_ok=False)
    sources=[MODEL,INITIAL,WEIGHTS,ENDPOINTS/'MANIFEST.json',Path(__file__).resolve(),
             ROOT/'engine/neural_cond.py',ROOT/'engine/embodied/motor_translation.py',
             ROOT/'engine/embodied/sensory_channels.py',ROOT/'run_neural_muscle_closure.py',
             ROOT/'engine/embodied/adapters.py',ROOT/'engine/embodied/command_tier.py',
             ROOT/'engine/embodied/motor_endpoints.py',ROOT/'engine/neck_cut_data.py',
             ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    identities={str(s):sha(s) for s in sources}
    atomic_json(out/'PREREG.json',dict(initialization_ms=[0,40],physical_duration_ms=10,dt_ms=.1,neural_dt_ms=.5,
        same_weights=str(WEIGHTS),source_hashes=identities,physical_units='NOT_ESTABLISHED',
        initialization='held actual initial observation; computational only, NOT physical standing or preheating',
        physical_initial='saved qpos once; zero qvel/act/ctrl/external loads; no later pose writes',
        clocks='continuous neural/FIR source clock offset by initialization; generated source END before physical interval START operator split',
        criteria='original six gates; earlier commands alone not success; descriptive matched support and safety'))
    ledger=dict(model_load=0,forward=0,step=0,neural_ticks=0,active=None,completed=[])
    def reserve(**delta):
        for k,v in delta.items(): ledger[k]+=v
        atomic_json(out/'CALL_LEDGER.json',ledger)
    reserve()
    with np.load(WEIGHTS,allow_pickle=False) as z: weights=z['weights'].copy()
    with np.load(INITIAL,allow_pickle=False) as z: q=z['new_qpos'].copy(); sensory_ref=z['sensory_reference_qpos'].copy()
    manifest=json.loads((ENDPOINTS/'MANIFEST.json').read_text())
    built=build_neural_tier(); a=built['data']
    tier=retain_motor_endpoints(a,downstream_union(a,built['global_index'],['720575941433155799','720575941566493282'],hops=2,max_neurons=20000),['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
    roots=a['root_ids'][tier['global_index']]; assert len(roots)==17099
    edges=build_edges(a,tier,built['cut']); groups,_=channel_index_map(len(roots),roots)
    results=[]
    try:
        for pre in (0,80):
            label=f'init{pre//2:02d}ms'; dest=out/label; dest.mkdir(); ledger['active']=label
            reserve(model_load=1); m=mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq,m.nv,m.nu,m.nbody,m.nmesh)==(49,48,90,73,71) and m.opt.timestep==.0001
            m.opt.integrator=mj.mjtIntegrator.mjINT_RK4
            require_no_callbacks(callback_snapshot(mj))
            d=mj.MjData(m); d.qpos[:]=q
            net=ConductanceNetwork(len(roots),dt_ms=.5); net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec=FiniteMotorTranslator(manifest,weights); enc=ChannelSensoryEncoder(groups,len(roots))
            initial_net={k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)}
            atomic_npz(dest/'NEURAL_COLD.npz',**initial_net)
            joints={}; geom_leg={}
            for leg in LEGS:
                js=[j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
                tib=next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
                joints[leg]=dict(q=[int(m.jnt_qposadr[j]) for j in js],tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]))
            for g in range(m.ngeom):
                gn=(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
                bn=(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
                for leg in LEGS:
                    if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)): geom_leg[g]=leg
            floor,=np.flatnonzero(m.geom_type==mj.mjtGeom.mjGEOM_PLANE)
            limited=np.flatnonzero(m.jnt_limited); qa=m.jnt_qposadr[limited]
            reserve(forward=1); mj.mj_forward(m,d)
            initial_observed=observe_quantities(m,d,sensory_ref,joints,geom_leg)
            initial_current,_=enc.encode(initial_observed)
            atomic_json(dest/'INITIAL_OBSERVATION.json',raw_observation_record(initial_observed)[0])
            warmsp=[]; warmf=[]; warmu=[]
            for j in range(pre):
                reserve(neural_ticks=1); sp=net.step(current_nA=initial_current).copy()
                u,info=dec.consume(sp[None,:],spike_dt_s=.0005,start_s=j*.0005,end_s=(j+1)*.0005)
                warmsp.append(sp); warmf.append(info['pool_features']); warmu.append(u)
            release_features=np.stack([dec.history[lo:hi].mean(0) for lo,hi in dec.BINS],axis=1)
            pooled=np.clip((release_features*weights).sum(1),0,1)
            release_ctrl=np.zeros(m.nu); mapped=dec.actuator_pool>=0; release_ctrl[mapped]=pooled[dec.actuator_pool[mapped]]
            atomic_npz(dest/'INITIALIZATION.npz',spikes=np.asarray(warmsp,dtype=bool).reshape(pre,len(roots)),
                features=np.asarray(warmf).reshape(pre,66,5),decoded=np.asarray(warmu).reshape(pre,90),
                current=initial_current,history=dec.history,release_features=release_features,release_decoded=release_ctrl,
                weights=weights,body_qpos=d.qpos,body_qvel=d.qvel,body_act=d.act,body_ctrl=d.ctrl,
                **{k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            assert d.time==0 and np.array_equal(d.qpos,q) and not any(x.any() for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))
            def sample():
                contacts=np.zeros(6,dtype=int); force=np.zeros((6,3)); depth=0.
                for ci,c in enumerate(d.contact):
                    g1,g2=int(c.geom1),int(c.geom2)
                    if floor not in (g1,g2): continue
                    foot=g2 if g1==floor else g1
                    if foot in geom_leg:
                        li=LEGS.index(geom_leg[foot]); contacts[li]+=int(c.dist<0)
                        f=np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                        force[li]+=(1 if g1==floor else -1)*c.frame.reshape(3,3).T@f[:3]
                    else: depth=max(depth,-float(c.dist))
                excess=np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
                return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),
                    contacts=contacts,foot_force=force,nonfoot_depth=depth,limit_excess=float(excess.max(initial=0)),
                    actuator_force=d.actuator_force.copy(),qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),
                    warnings=np.asarray([w.number for w in d.warning]))
            rows=[sample()]; currents=[]; spikes=[]; features=[]; commands=[]; schedule=[]
            for k in range(20):
                if k%10==0:
                    observed=observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current,_=enc.encode(observed)
                    atomic_json(dest/f'OBSERVATION_{k:02d}.json',raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp=net.step(current_nA=current).copy()
                start=(pre+k)*.0005; end=(pre+k+1)*.0005
                u,info=dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
                assert abs(net.t_ms/1000-end)<1e-12 and dec.interval==pre+k
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                schedule.append(dict(tick=k,source_start_s=start,source_end_s=end,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=pre*.0005,consume_id=dec.interval))
                d.ctrl[:]=u
                for sub in range(5):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row=sample(); rows.append(row)
                    assert all(np.isfinite(v).all() for v in row.values()) and not row['warnings'].any()
                    assert not row['qfrc_applied'].any() and not row['xfrc_applied'].any()
            arrays={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
            arrays.update(spikes=np.asarray(spikes),features=np.asarray(features),applied=np.asarray(commands),current=np.asarray(currents),weights=weights)
            atomic_npz(dest/'TRAJECTORY.npz',**arrays); atomic_json(dest/'SCHEDULE.json',schedule)
            weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            record=dict(label=label,initialization_ticks=pre,source_offset_s=pre*.0005,weight_native=weight,
                trajectory_sha256=sha(dest/'TRAJECTORY.npz'),initialization_sha256=sha(dest/'INITIALIZATION.npz'),
                neural_final_s=net.t_ms/1000,FIR_final_s=dec.elapsed_s,physical_final_s=float(d.time),behavior_claim='NONE')
            results.append(record); atomic_json(out/'RESULTS.json',results)
            ledger['completed'].append(label); ledger['active']=None; reserve()
        assert identities=={str(s):sha(s) for s in sources}
        atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',physical_units='NOT_ESTABLISHED'))
        print(json.dumps(results))
    except BaseException:
        atomic_json(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger)); raise


if __name__=='__main__': main()
