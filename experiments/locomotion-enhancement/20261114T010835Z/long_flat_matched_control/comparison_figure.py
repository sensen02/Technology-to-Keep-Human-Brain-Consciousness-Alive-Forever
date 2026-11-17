"""Plot already verified saved summary statistics; no model or physics imports."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
LEGS=('LF','LH','LM','RF','RH','RM')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    paths=[ROOT/'SUMMARY.json',SCOPE/'rough_stall_saved_diagnostic/RESULT_SAVED_DIAGNOSTIC.json',SCOPE/'rough_stall_neural_motion_followup/RESULT.json']
    before={str(p):sha(p) for p in paths};flat,rough,neural=map(read,paths)
    extra=[Path(flat['audit_path']),ROOT/'JOBS.json',Path(flat['result_path']),SCOPE/'rough_terrain_long_window_38p4/trials/mild_blocks_seed17_start_repair_v1_long38p4/RESULT.json']
    before.update({str(p):sha(p) for p in extra})
    audit=read(flat['audit_path']);assert audit['integrity_all_pass'] and all(audit['checks'].values())
    jobs=read(ROOT/'JOBS.json');assert len(jobs)==2 and all(x['status']=='FINISHED' and x['exit_code']==0 for x in jobs)
    fw=flat['long_term_diagnostic']['windows'];rw=rough['cases'][0]['windows'];nw=neural['rows'];assert len(fw)==len(rw)==len(nw)==4
    assert all(w['all_six_completed_in_window'] for w in fw)
    case=read(flat['result_path']);geometry=case.get('geometry');geometry_status='NOT_RECORDED_IN_RESULT'
    if isinstance(geometry,dict) and 'path' in geometry:
        assert sha(geometry['path'])==geometry['sha256'];before[geometry['path']]=geometry['sha256'];geometry_status='ACTUAL_RECORDED_ARCHIVE_HASH_VERIFIED'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(2,2,figsize=(13,9),gridspec_kw={'height_ratios':[1,1.18]});labels=['0-9.6','9.6-19.2','19.2-28.8','28.8-38.4'];x=np.arange(4)
    ax[0,0].bar(x-.18,[w['COM_x_displacement'] for w in fw],.36,label='Matched flat',color='#2878B5')
    ax[0,0].bar(x+.18,[w['points']['COM_thorax_subtree']['delta_xyz_native'][0] for w in rw],.36,label='Rough scene',color='#D95319')
    ax[0,0].axhline(0,color='black',lw=.7);ax[0,0].set(xticks=x,xticklabels=labels,ylabel='COM forward displacement (native)',xlabel='Autonomous interval (s)',title='Flat keeps progressing; rough stalls');ax[0,0].legend()
    ax[0,1].plot(x,[w['action_RMS'] for w in fw],'o-',label='Matched flat',color='#2878B5');ax[0,1].plot(x,[w['applied_action_RMS'] for w in nw],'s-',label='Rough scene',color='#D95319');ax[0,1].set(yscale='log',xticks=x,xticklabels=labels,ylabel='Recorded motor action RMS (unitless)',xlabel='Autonomous interval (s)',title='Rough motor output becomes nearly quiet');ax[0,1].legend();ax[0,1].grid(alpha=.25)
    counts=[[w['strict_cycles_completed_by_end_time'][l] for l in LEGS] for w in fw]+[[w['feet'][l]['original_strict_cycle_complete_count'] for l in LEGS] for w in rw]
    ax[1,0].imshow(counts,cmap='Blues',vmin=0,vmax=2,aspect='auto');ax[1,0].set(xticks=range(6),xticklabels=LEGS,yticks=range(8),yticklabels=['Flat '+s for s in labels]+['Rough '+s for s in labels],title='Original strict cycle completions by window')
    for i,row in enumerate(counts):
        for j,n in enumerate(row):ax[1,0].text(j,i,str(n),ha='center',va='center',color='white' if n>1 else 'black')
    f=flat['reference_original_five_gates'];r=read(SCOPE/'rough_terrain_long_window_38p4/trials/mild_blocks_seed17_start_repair_v1_long38p4/RESULT.json')['terrain_summary']['relative_five']
    table=[['Net COM XY displacement',f"{f['net_com_xy_displacement']:.6f}",f"{r['net_com_xy_displacement']:.6f}"],['Net COM XY speed',f"{f['net_com_xy_speed_native_per_s']:.6f}",f"{r['net_com_xy_speed_native_per_s']:.6f}"],['Registered slip ratio',f"{f['slip_ratio_max']:.6f}",f"{r['slip_ratio_max']:.6f}"],['Last-window six-leg cycles','2 / 2 / 2 / 2 / 2 / 2','0 / 0 / 0 / 0 / 0 / 0'],['Obstacle coverage','N/A: no obstacles','Only blocks 00-02']]
    ax[1,1].axis('off');t=ax[1,1].table(cellText=table,colLabels=['Saved statistic','Matched flat','Rough scene'],loc='center',cellLoc='center',colWidths=[.45,.275,.275]);t.auto_set_font_size(False);t.set_fontsize(8.5);t.scale(1,2.05)
    fig.suptitle('Matched 38.4 s autonomous trials | same old336 neural chain and frozen789f motor',fontsize=13)
    fig.text(.5,.025,'Both: 1 s settle + 4.8 s teacher warmup, then no teacher. Only six static blocks removed.\nScene association established here; optical and mechanical effects are not separated. Native lengths are not calibrated SI.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.95]);output=ROOT/'MATCHED_FLAT_VS_ROUGH_38P4.png';assert not output.exists();fig.savefig(output,dpi=150);plt.close(fig)
    assert all(sha(p)==h for p,h in before.items());assert not any(n in sys.modules for n in ('torch','mujoco','physics','flygym'))
    manifest=dict(kind='MATCHED_FLAT_VS_ROUGH_SAVED_STATISTICS_FIGURE',source_hashes=before,script_sha256=sha(__file__),geometry_provenance=geometry_status,figure=str(output),figure_sha256=sha(output),neural_forwards=0,Core=0,physics_steps=0,training_updates=0,causal_scope='Whole scene association only, not separation of optics and mechanics; no new model or biological qualification')
    with (ROOT/'FIGURE_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(output)
if __name__=='__main__':main()
