"""Saved-array analysis, event reconstruction and figures; zero simulation steps."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

HERE=Path(__file__).resolve().parent
LEGS=['lf','lm','lh','rf','rm','rh']
MODES=['held_first','recorded100']

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    result=json.loads((HERE/'RESULT.json').read_text());replay=json.loads((HERE/'INDEPENDENT_CHECK.json').read_text())
    assert replay['status']=='PASS'
    a=load(HERE/'INPUTS.npz');initial=load(HERE/'NEURAL_INITIAL.npz');mn=a['all_mn_local']
    pos=np.array([np.flatnonzero(mn==x)[0] for x in a['mn323_local']]);legs=a['mn323_leg']
    traces={x['condition']:load(HERE/'conditions'/x['condition']/'TRACE.npz') for x in result['conditions']}
    analyses=[];reconstruction=[]
    for r in result['conditions']:
        name=r['condition'];t=traces[name];final=load(HERE/'conditions'/name/'FINAL_STATE.npz')
        events=np.unpackbits(t['packed_spikes'],axis=1,count=20000,bitorder='little').astype(bool)
        assert events.shape==(1000,20000)
        counts=initial['spike_count'].copy();filtered=initial['filtered_rates_hz'].copy();gA=initial['g_adapt'].copy();ring=initial['_ring'].copy();ri=int(initial['_ring_i'])
        for i,spiked in enumerate(events):
            counts+=spiked;filtered*=np.exp(-.5/20.);filtered+=(1000./20.)*spiked
            gA*=np.exp(-.5/100.);gA[spiked]+=.002/15.
            ring[ri]=spiked;ri=(ri+1)%5
            assert np.array_equal(t['step_rows'][i,11:13],spiked[[1414,13170]])
            assert np.array_equal(t['step_rows'][i,13:15],counts[[1414,13170]])
            assert np.array_equal(t['step_rows'][i,9:11],filtered[[1414,13170]])
            assert np.array_equal(t['step_rows'][i,7:9],gA[[1414,13170]])
            if i%10==9:
                b=i//10
                assert np.array_equal(t['mn_end_hz'][b],filtered[mn])
                assert np.array_equal(t['mn_end_counts'][b],counts[mn])
        for k,value in [('spike_count',counts),('filtered_rates_hz',filtered),('g_adapt',gA),('_ring',ring),('_ring_i',np.array(ri)),('last_spikes',events[-1])]:assert np.array_equal(final[k],value),k
        assert int(final['steps'])==3000 and float(final['t_ms'])==1500.
        assert np.array_equal(t['mn_start_hz'][0],initial['filtered_rates_hz'][mn])
        assert np.array_equal(t['mn_start_hz'][1:],t['mn_end_hz'][:-1])
        assert np.array_equal(t['mn_start_mV'][1:],t['mn_end_mV'][:-1])
        reconstruction.append({'condition':name,'steps':1000,'all20000_counts_filter_gA_ring_final_exact':True,'all391_MN_bin_rates_counts_exact':True})
        base=traces[r['input_mode']+'_dose000']
        rate=t['mn_end_hz'][:,pos];zero=base['mn_end_hz'][:,pos]
        diff=rate-zero
        perleg={leg:{'n_MN':int((legs==leg).sum()),'mean_filtered_hz':float(rate[:,legs==leg].mean()),
            'delta_mean_filtered_hz_vs_zero':float(diff[:,legs==leg].mean()),
            'spikes':int(events[:,a['mn323_local'][legs==leg]].sum()),
            'delta_spikes_vs_zero':int((t['mn_end_counts'][-1,pos[legs==leg]]-base['mn_end_counts'][-1,pos[legs==leg]]).sum())} for leg in LEGS}
        analyses.append({'condition':name,'input_mode':r['input_mode'],'dose_nA':r['amplitude_nA'],
            'MN323_mean_filtered_hz':float(rate.mean()),'MN323_delta_mean_filtered_hz':float(diff.mean()),
            'MN323_rms_delta_hz':float(np.sqrt(np.mean(diff**2))),
            'MN323_changed_trace_cells':int(np.count_nonzero(np.any(diff!=0,axis=0))),
            'all391_mean_filtered_hz':float(t['mn_end_hz'].mean()),
            'all391_delta_mean_filtered_hz':float((t['mn_end_hz']-base['mn_end_hz']).mean()),'per_leg':perleg})
    dump(HERE/'ANALYSIS.json',{'status':'PASS','neural_steps':0,'physical_steps':0,'sampling':'100 postbin endpoints, 5ms; per-leg only qualified323, other68 unassigned','conditions':analyses,'event_reconstruction':reconstruction})
    fig,ax=plt.subplots(4,2,figsize=(15,14),constrained_layout=True)
    fig.suptitle('Fixed engineering DNp09 pair probe | canonical 1000ms state | no physical steps',fontsize=16)
    colors=['#6c757d','#b66b1a','#1467ad'];time=np.arange(1,101)*5
    matrices=[]
    for mode in MODES:
        t=traces[mode+'_dose050'];z=traces[mode+'_dose000']
        matrices.append(np.array([(t['mn_end_hz'][:,pos[legs==leg]]-z['mn_end_hz'][:,pos[legs==leg]]).mean(axis=1) for leg in LEGS]))
    vmax=max(float(np.abs(x).max()) for x in matrices)
    for col,mode in enumerate(MODES):
        for j,dn in enumerate([1414,13170]):
            for dose,style,alpha in [(0,'--',.65),(50,'-',1.)]:
                t=traces[mode+'_dose'+str(dose).zfill(3)]
                ax[0,col].plot(t['step_rows'][:,0]-1000,t['step_rows'][:,9+j],style,color=['#1467ad','#bd3a32'][j],alpha=alpha,label=f'local{dn} {dose/1000:g}nA')
        ax[0,col].set(title=mode+' sensory input',ylabel='DN filtered rate (Hz)');ax[0,col].legend(ncol=2,fontsize=8)
        for d,color in zip([0,25,50],colors):
            t=traces[mode+'_dose'+str(d).zfill(3)]
            ax[1,col].plot(time,t['mn_end_hz'].mean(axis=1),color=color,label=f'{d/1000:g}nA')
        ax[1,col].set(ylabel='All391 MN mean rate (Hz)');ax[1,col].legend(fontsize=9)
        x=np.arange(6)
        for j,d in enumerate([25,50]):
            vals=next(v for v in analyses if v['condition']==mode+'_dose'+str(d).zfill(3))
            y=[vals['per_leg'][leg]['delta_mean_filtered_hz_vs_zero'] for leg in LEGS]
            ax[2,col].bar(x+(j-.5)*.35,y,width=.35,color=colors[j+1],label=f'{d/1000:g} minus 0nA')
        ax[2,col].set(xticks=x,xticklabels=[x.upper() for x in LEGS],ylabel='Per-leg mean rate delta (Hz)');ax[2,col].axhline(0,color='black',lw=.7);ax[2,col].legend(fontsize=9)
        image=ax[3,col].imshow(matrices[col],origin='lower',aspect='auto',extent=[0,500,-.5,5.5],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        ax[3,col].set(yticks=np.arange(6),yticklabels=[x.upper() for x in LEGS],xlabel='Time since canonical checkpoint (ms)',title='Primary 0.05nA minus matched zero, per-leg mean Hz')
        fig.colorbar(image,ax=ax[3,col],label='Hz')
        for row in [0,1]:ax[row,col].set_xlim(0,500);ax[row,col].grid(alpha=.2)
    fig.savefig(HERE/'DN_MN_RESPONSE.png',dpi=160);plt.close(fig)
    ports=json.loads((HERE/'PORTS.json').read_text());union=ports['directed_reachability']['union_min_hops']
    graph_summary={'source_audit_reused':True,'runtime_graph_verified':True,'candidates':[],
        'union_downstream_targets':union['downstream']['targets'],
        'union_upstream_full_all':union['upstream']['full_graph_all_nodes'],
        'union_upstream_local_selected':union['upstream']['local_graph_selected_nodes'],
        'union_upstream_outside_selected':union['upstream']['full_reachable_outside_selected_positive_hops'],
        'upstream_full_superclasses':union['upstream']['full_reachable_positive_labels']['superclass'],
        'upstream_local_superclasses':union['upstream']['local_reachable_positive_labels']['superclass'],
        'extra_route_intermediates':{k:{x:v[x] for x in ['definition','full_route_intermediate_cells','outside_selected_intermediate_cells']} for k,v in ports['full_graph_extra_intermediates'].items()}}
    for candidate in ports['candidates']:
        graph_summary['candidates'].append({'identity':candidate['identity'],'outgoing':{k:candidate['outgoing'][k]['edge_pairs'] for k in ['full_original','full_strict','selected_strict']},'incoming':{k:candidate['incoming'][k]['edge_pairs'] for k in ['full_original','full_strict','selected_strict']}})
    # Source-backed witnesses: shortest path to lowest-index MN in each leg for
    # each DN. This graph choice does not inspect spike activity or optimize doses.
    local=load(ports['source_hashes']['local_candidate']['path']);mask=local['pure_verified_label_mask'];pre=local['pre_local'][mask];post=local['post_local'][mask]
    graph=csr_matrix((np.ones(len(pre)),(pre,post)),shape=(20000,20000))
    distance,predecessor=shortest_path(graph,directed=True,unweighted=True,indices=[1414,13170],return_predecessors=True,method='D')
    witnesses=[]
    for di,dn in enumerate([1414,13170]):
        for leg in LEGS:
            target=int(a['mn323_local'][legs==leg].min());path=[target]
            while path[-1]!=dn:
                prev=int(predecessor[di,path[-1]]);assert prev>=0;path.append(prev);assert len(path)<=20000
            path.reverse();edges=[]
            for u,v in zip(path[:-1],path[1:]):
                index=np.flatnonzero((pre==u)&(post==v));assert len(index)==1;e=int(index[0]);
                edges.append({'pre_local':u,'post_local':v,'source_edge_row':int(local['strict_source_edge_rows'][e]),'synapse_count':int(local['synapse_counts'][mask][e])})
            assert len(edges)==distance[di,target]
            witnesses.append({'DN_local':dn,'target_leg':leg,'target_local':target,'local_path':path,'root_id_path':[str(int(local['root_ids'][x])) for x in path],'source_edges':edges})
    dump(HERE/'PATH_WITNESSES.json',{'scope':'strict local source paths, not proof of physiological propagation or brain origin','witnesses':witnesses})
    dump(HERE/'GRAPH_BOUNDARIES.json',graph_summary)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    fig.suptitle('Source boundaries: engineering pair input into a local graph, upstream brain omitted',fontsize=15)
    x=np.arange(2);retained=[v['outgoing']['selected_strict'] for v in graph_summary['candidates']];full=[v['outgoing']['full_strict'] for v in graph_summary['candidates']]
    axes[0,0].bar(x,retained,label='Retained strict outgoing pairs',color='#1467ad');axes[0,0].bar(x,np.array(full)-retained,bottom=retained,label='Outside local runtime',color='#c1c1c1')
    axes[0,0].set(xticks=x,xticklabels=['DN local1414','DN local13170'],ylabel='Source edge pairs',title='No new synapses; full source vs20k subset');axes[0,0].legend(fontsize=8)
    for i in range(2):axes[0,0].text(i,full[i]+2,f'{retained[i]} / {full[i]}',ha='center')
    hist=union['downstream']['targets']['exported323']['local_strict_induced'];hops=[hist['hop1'],hist['hop2'],hist['hop3']]
    axes[0,1].bar([1,2,3],hops,color='#41926e');axes[0,1].set(xticks=[1,2,3],xlabel='Minimum directed hops from either DN',ylabel='Exported MN cells',title='323/323 reachable; full/local shortest hops equal')
    for i,n in enumerate(hops):axes[0,1].text(i+1,n+2,str(n),ha='center')
    upstream=union['upstream'];full_anc=upstream['full_graph_all_nodes']['reachable_positive_hops'];local_anc=upstream['local_graph_selected_nodes']['reachable_positive_hops']
    axes[1,0].bar(['Full strict ancestors','Local strict ancestors'],[full_anc,local_anc],color=['#c1c1c1','#1467ad']);axes[1,0].set(ylabel='Positive-hop upstream cells',title='Local upstream paths do not establish brain response')
    for i,n in enumerate([full_anc,local_anc]):axes[1,0].text(i,n+1000,str(n),ha='center')
    axes[1,1].axis('off')
    text='EXTERNAL constant current, 0 / 0.025 / 0.05 nA\n         ↓\nTwo named DNp09/ACh source identities\n         ↓ existing strict local edges\nInterneurons → all391 leg MN (323 assigned to legs)\n\nNo brain dynamics simulated here.\nNo confirmed side/homology/optogenetic equivalence.\nSource annotation ≠ physiological function.\nNo new learning, no physical stepping, no walking claim.\n\nTwelve exact per-leg path witnesses saved with source\nedge rows and integer root IDs.'
    axes[1,1].text(0,.98,text,va='top',fontsize=11,linespacing=1.55)
    fig.savefig(HERE/'GRAPH_BOUNDARIES.png',dpi=160);plt.close(fig)
    result.update(status='COMPLETE_INDEPENDENT_REPLAY_VERIFIED',replay_neural_steps=2000,total_neural_steps=8000,
        independent_replay_status='PASS',historical_zero_reference=replay['historical_zero_reference'],
        all_six_event_reconstruction_passed=True,analysis_sha256=sha(HERE/'ANALYSIS.json'),
        physical_steps=0,training_updates=0,failed_probe_episodes=0,software_exceptions=0)
    dump(HERE/'RESULT.json',result)
    print(json.dumps({'primary_effects':[x for x in analyses if x['dose_nA']==.05],'graph_boundaries':graph_summary,'neural_steps_this_analysis':0},indent=2),flush=True)

if __name__=='__main__':main()
