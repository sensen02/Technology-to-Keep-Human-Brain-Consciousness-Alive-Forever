"""Analyze saved divergence order and freeze command counterfactuals."""
from scope_diag import *


def first_diff(a,b,tol=0.):
    difference=np.abs(a.astype(float)-b.astype(float))>tol
    hit=np.flatnonzero(difference.reshape(len(a),-1).any(1))
    return int(hit[0]) if len(hit) else None


def main():
    results=[]
    for seed in SEEDS:
        folder=PREVIOUS/'FORMAL'/f'seed{seed}'
        meta=read(folder/'RESULT.json');verify(meta['source_sha256'])
        with np.load(folder/'NEURAL.npz',allow_pickle=False) as a:neural={k:a[k].copy() for k in a.files}
        traces={m:load_trace(seed,m) for m in ('native_full','reference_boundary','translated')}
        commands={'native':traces['native_full']['command'],'reference':traces['reference_boundary']['command'],
                  'translated':traces['translated']['command']}
        ref,student=commands['reference'],commands['translated']
        ar,ast=ref.mean(1),student.mean(1);tr,ts=(ref[:,1]-ref[:,0])/2,(student[:,1]-student[:,0])/2
        modifications={}
        for name,raw in [('refadvance_studentturn',np.stack((ar-ts,ar+ts),1)),
                          ('studentadvance_refturn',np.stack((ast-tr,ast+tr),1))]:
            clipped=np.clip(raw,-1,1);final=np.where(abs(clipped)>=.035,clipped,0.)
            commands[name]=final
            modifications[name]=dict(clipped_elements=int((clipped!=raw).sum()),deadzone_elements=int((final!=clipped).sum()),
                  maximum_constraint_change=float(abs(final-raw).max()))
        out=HERE/'COMMANDS'/f'seed{seed}';out.mkdir(parents=True)
        pins=sources({folder/'NEURAL.npz',folder/'RESULT.json',*(folder/m/'TRACE.npz' for m in traces)})
        np.savez_compressed(out/'BANK.npz',**commands)
        write(out/'RESULT.json',dict(seed=seed,source_sha256=pins,command_sha256=sha(out/'BANK.npz'),
               command_names=list(commands),counterfactual_constraints=modifications,neural_steps=0,physical_steps=0))
        summaries={}
        for name,c in commands.items():
            advance=c.mean(1);turn=(c[:,1]-c[:,0])/2
            summaries[name]=dict(mean_advance=float(advance[2200:16000].mean()),mean_turn=float(turn[2200:16000].mean()),
                integrated_advance=float(advance.sum()*DT),integrated_signed_turn=float(turn.sum()*DT),
                integrated_absolute_turn=float(abs(turn).sum()*DT))
        ordering={}
        for lane,mode in ((0,'native_full'),(1,'reference_boundary')):
            q=traces[mode];s=traces['translated']
            ordering[mode]=dict(brain_AN_event_tick=first_diff(neural['brain_an_events'][:,lane],neural['brain_an_events'][:,2]),
                DN_event_tick=first_diff(neural['dn_emitted'][:,lane],neural['dn_emitted'][:,2]),
                postneck_command_tick=first_diff(q['command'],s['command'],1e-12),
                motor_target_tick=first_diff(q['control'],s['control'],1e-12),
                physical_qpos_tick=first_diff(q['qpos'],s['qpos'],1e-12))
        results.append(dict(seed=seed,divergence_order=ordering,command_statistics=summaries,hybrid_constraints=modifications,
            source_yaw_degrees={m:read(folder/m/'RESULT.json')['yaw_degrees'] for m in traces}))
        verify(pins)
        print(seed,ordering,summaries['reference'],summaries['translated'],flush=True)
    write(HERE/'SAVED_DIAGNOSTIC.json',dict(rows=results,dt_s=DT,
       timestamp_basis='tick index refers to the same source loop; physical qpos is after that tick physics integration',
       interpretation='Temporal ordering is not a unique mechanism; command/teacher and DN interventions follow.',
       optimizer_updates=0,physical_steps=0,neural_steps=0))


if __name__=='__main__':main()
