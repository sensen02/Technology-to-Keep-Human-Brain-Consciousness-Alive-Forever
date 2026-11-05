"""Read-only trajectory diagnosis; write derived tables only to current package."""
import sys,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LEGS=['LF','LM','LH','RF','RM','RH']
def main():
    root=Path(sys.argv[1]); summaries={}; allrows=[]
    for cond in ['baseline','unmask','unmask_minus2','unmask_minus5']:
        run=root/cond/'EVAL/RUN'
        if not (run/'BODY.npz').exists():continue
        b=dict(np.load(run/'BODY.npz')); n=dict(np.load(run/'NEURAL.npz'))
        records=json.loads((run/'NEURAL_RECORDS.json').read_text());man=json.loads((run/'MAPPING.json').read_text())['rows']
        source=Path('/home/sensen/Desktop/engine修复验收/s5-route-a-substitute-v4/20261108T200000Z')
        synth={x['index'] for x in json.loads((source/'ARM_DEFINITION.json').read_text())['dead_actuators']}
        intervention=json.loads((root/cond/'INTERVENTION.json').read_text());synth-=set(intervention['synthetic_removed'])
        t0=records[0]['absolute_start_s']; masks={'assist':(b['time']>=t0-.2)&(b['time']<t0-1e-8),'release':b['time']>t0+1e-8}
        spikes=n['spike_counts'].sum(axis=0);rates=np.array([r['decoder']['rates_hz'] for r in records])
        rows=[]
        for i,m in enumerate(man):
            cells=m['local_indices'];count=int(spikes[cells].sum()) if cells else 0
            r=dict(condition=cond,index=i,actuator=m['actuator'],leg=m.get('leg'),mapping_status=m['status'],pool_size=len(cells),local_indices=';'.join(map(str,cells)),spikes=count,active_cells=int(np.count_nonzero(spikes[cells])) if cells else 0,native_status='no_mapping' if not cells else ('silent_in_release' if count==0 else 'spiking'),synthetic_override=i in synth,native_filtered_rate_mean_hz=float(rates[:,i].mean()))
            for phase,mask in masks.items():
                for field,label in [('command','command'),('activation','activation'),('muscle_force','force')]:
                    a=b[field][mask,i];r[phase+'_'+label+'_mean']=float(a.mean());r[phase+'_'+label+'_absmax']=float(abs(a).max())
                r[phase+'_command_saturated_fraction']=float((b['command'][mask,i]>=1-1e-9).mean())
            rows.append(r)
        allrows+=rows
        stats=dict(score=json.loads((root/cond/'EVAL/SCORE.json').read_text()),mapped_cells=len({i for m in man for i in m['local_indices']}),silent_mapped_cells=sorted({i for m in man for i in m['local_indices'] if spikes[i]==0}),no_mapping=sum(r['pool_size']==0 for r in rows),silent_actuator_pools=sum(r['native_status']=='silent_in_release' for r in rows),synthetic_count=len(synth),release_aux_max=float(abs(b['assist'][masks['release']]).max()),release_external_max=float(abs(b['external'][masks['release']]).max()),by_leg={})
        stats['score'].pop('vec',None)
        for li,leg in enumerate(LEGS):
            rr=[r for r in rows if r['leg']==leg];mask=masks['release']
            stats['by_leg'][leg]=dict(mapped_spiking_actuators=sum(r['native_status']=='spiking' for r in rr),synthetic_actuators=sum(r['synthetic_override'] for r in rr),mean_foot_force_native=float(b['foot_force'][mask,li,2].mean()),contact_fraction=float(b['foot_contact'][mask,li].mean()),mean_command=float(b['command'][mask][:,[r['index'] for r in rr]].mean()))
        summaries[cond]=stats
    assert len([r for r in allrows if r['condition']=='baseline'])==90
    with (root/'ACTUATOR_DIAGNOSIS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (root/'CHAIN_SUMMARY.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for cond in summaries:
        b=dict(np.load(root/cond/'EVAL/RUN/BODY.npz'));t=b['time'];mask=t>=1.1
        axs[0,0].plot(t[mask]-1.3,b['com'][mask,2],label=cond)
        axs[0,1].plot(t[mask]-1.3,b['foot_force'][mask,:,2].sum(axis=1),label=cond)
        axs[1,0].plot(t[mask]-1.3,b['nonfoot_force'][mask,2],label=cond)
    base=[r for r in allrows if r['condition']=='baseline'];cmd=np.array([r['release_command_mean'] for r in base]).reshape(6,15)
    # The manifest groups actuators by leg but in MANIFEST order (LF,RF,LM,RM,LH,RH), which is NOT the
    # LEGS tuple order. Label rows from the data so the figure cannot disagree with the CSV.
    assert all(len({base[i+k]['leg'] for k in range(15)})==1 for i in range(0,90,15)),'manifest not leg-blocked'
    blocks=[base[i]['leg'] for i in range(0,90,15)]
    im=axs[1,1].imshow(cmd,vmin=0,vmax=1,aspect='auto',cmap='viridis');axs[1,1].set_yticks(range(6),blocks);axs[1,1].set_xlabel('Actuator within leg (manifest order)');fig.colorbar(im,ax=axs[1,1])
    for ax,title in zip([axs[0,0],axs[0,1],axs[1,0]],['COM height (model units)','Foot vertical force (native units)','Nonfoot vertical force (native units)']):
        ax.axvline(0,color='black',linestyle='--');ax.set_title(title);ax.set_xlabel('Time from withdrawal (s)');ax.legend(fontsize=8)
    axs[1,1].set_title('Baseline mean command: 90 actuators')
    fig.suptitle('S5 drive-chain diagnosis — SI NOT_ESTABLISHED');fig.tight_layout();fig.savefig(root/'DRIVE_CHAIN.png',dpi=150);plt.close(fig)
    print(json.dumps({k:{a:v[a] for a in ['no_mapping','silent_actuator_pools','synthetic_count','silent_mapped_cells','score']} for k,v in summaries.items()},ensure_ascii=False))
if __name__=='__main__':main()
