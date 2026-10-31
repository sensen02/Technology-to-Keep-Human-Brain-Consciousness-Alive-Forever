"""Offline S110 signed-torque and real FIR reachability screen; no physics."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT=Path('/home/sensen/Desktop/engine修复验收/s110-local-pool-response/20261024T053026Z')


def main():
    records=[]; hashes={}; tables=[]
    def load(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    for arm in ('baseline','down','up'):
        a=load(OUT/arm/'TRAJECTORY.npz'); init=load(OUT/arm/'INITIALIZATION.npz')
        meta_path=OUT/arm/'MOMENT_METADATA.json';meta=json.loads(meta_path.read_text());hashes[str(meta_path)]=hashlib.sha256(meta_path.read_bytes()).hexdigest()
        times=a['time']; pre=np.flatnonzero(times < .0017-1e-12); inclusive=np.flatnonzero(times <= .0017+1e-12)
        assert len(pre)==17 and len(inclusive)==18
        # Physical command starts 0,.5,1,1.5ms; all before first baseline loss.
        features=a['features'][:4]; weights=a['weights']; moment=a['actuator_moment'][pre]; force=a['actuator_force'][pre]
        torque=moment*force[:,:,None]
        for i,name in enumerate(meta['actuator_names']):
            pool=int(a['actuator_pool'][i]); has_target=bool(np.any(np.abs(moment[:,i,8])>1e-12))
            mp=moment[:,i,8]; tp=torque[:,i,8]; fp=force[:,i]
            if pool>=0:
                feat=features[:,pool]; release=init['release_features'][pool]; w=weights[pool]
                adjustable=np.flatnonzero((feat!=0).any(0)&(w>0)&(w<4)).tolist()
                raw=feat@w;ctrl=a['applied'][:4,i]
                # Fixed-feature decoder derivative, NOT a physical incremental force/torque derivative.
                deriv=np.where(((raw>0)&(raw<1))[:,None],feat,0.)
                reachable=bool(adjustable and np.any(deriv[:,adjustable]!=0))
                bins=[dict(bin=b,history_ms=[float(lo*.5),float(hi*.5)],weight=float(w[b]),
                    room_down=float(w[b]),room_up=float(4-w[b]),release_feature=float(release[b]),
                    features_at_0_05_1_15ms=feat[:,b].tolist(),decoder_derivative=deriv[:,b].tolist())
                    for b,(lo,hi) in enumerate(((0,5),(5,10),(10,20),(20,40),(40,80)))]
            else:adjustable=[];bins=[];release=[];raw=np.zeros(4);ctrl=np.zeros(4);reachable=False
            coupled=[]
            for dof in range(48):
                if dof==8 or not np.any(np.abs(moment[:,i,dof])>1e-12):continue
                jid=meta['dof_jntid'][dof]
                coupled.append(dict(dof=dof,joint=meta['joint_names'][jid],moment_min=float(moment[:,i,dof].min()),moment_max=float(moment[:,i,dof].max()),
                    actual_torque_min=float(torque[:,i,dof].min()),actual_torque_max=float(torque[:,i,dof].max()),
                    note='per-DOF only; magnitudes not summed across mixed generalized units'))
            r=dict(arm=arm,index=i,name=name,pool=pool,has_target_moment=has_target,
                positive_moment=bool(np.any(mp>0)),negative_moment=bool(np.any(mp<0)),moment_min=float(mp.min()),moment_max=float(mp.max()),
                actual_force_min=float(fp.min()),actual_force_max=float(fp.max()),actual_torque_min=float(tp.min()),actual_torque_max=float(tp.max()),
                negative_actual_torque=bool(np.any(tp<0)),negative_torque_nonzero_samples=int(np.count_nonzero(tp<0)),
                real_decoder_reachable=reachable,adjustable_bins=adjustable,bins=bins,raw_decoded_before_clip=raw.tolist(),applied_controls=ctrl.tolist(),
                upper_clipped_ticks=int(np.count_nonzero(raw>=1)),lower_clipped_ticks=int(np.count_nonzero(raw<=0)),
                shared_pool_actuators=[meta['actuator_names'][j] for j,p in enumerate(a['actuator_pool']) if pool>=0 and p==pool],
                coupled_joints=coupled,
                reverse_actual_torque_and_decoder_reachable=bool(has_target and np.any(tp<0) and reachable),
                incremental_physics_response='NOT_ESTABLISHED: fixed-state transmission sign + saved force + decoder derivative is not d(force)/d(ctrl) or d(torque)/d(weight)')
            records.append(r)
            r['feature_weight_detail']=json.dumps(bins)
            r['coupling_detail']=json.dumps(coupled)
            tables.append({k:r[k] for k in ('arm','index','name','pool','has_target_moment','moment_min','moment_max','actual_force_min','actual_force_max','actual_torque_min','actual_torque_max','negative_actual_torque','real_decoder_reachable','adjustable_bins','raw_decoded_before_clip','applied_controls','upper_clipped_ticks','shared_pool_actuators','reverse_actual_torque_and_decoder_reachable','feature_weight_detail','coupling_detail')})
    base=[r for r in records if r['arm']=='baseline' and r['has_target_moment']]
    candidates=[r for r in base if r['reverse_actual_torque_and_decoder_reachable']]
    output=dict(window='0 <= physical t <1.7ms; rows0..16, control starts0/.5/1/1.5ms; inclusive1.7ms not used for ranking',
        moment_screen_tolerance_native=1e-12,tolerance_reason='exclude roundoff-level rigid-chain cancellation; raw min/max retained all90; not a biological significance threshold',target_joint='joint_LFCoxa_roll',target_dof=8,positive_direction='towards increasing angle/upper .6236rad',
        target_rows=base,candidates=candidates,all_actuator_rows=records,source_hashes=hashes,
        new_physics_calls=0,new_neural_ticks=0,conclusion='Reverse existing force and tunable decoder access may justify a probe, never compensation success or incremental response certification.')
    (OUT/'REVERSE_REACHABILITY.json').write_text(json.dumps(output,indent=2)+'\n')
    with (OUT/'REVERSE_REACHABILITY.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(tables[0]));wr.writeheader();wr.writerows(tables)
    for r in base:
        print(json.dumps(dict(index=r['index'],name=r['name'],pool=r['pool'],moment=[r['moment_min'],r['moment_max']],torque=[r['actual_torque_min'],r['actual_torque_max']],reachable=r['real_decoder_reachable'],bins=r['adjustable_bins'],candidate=r['reverse_actual_torque_and_decoder_reachable'],coupled=r['coupled_joints'])))
    assert hashes=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}


if __name__=='__main__':main()
