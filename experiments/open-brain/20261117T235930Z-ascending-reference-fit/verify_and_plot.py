"""Post-save independent evidence and scientific plots, no simulation or learning."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import *
from encoder import load


def exact_npz(left,right):
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        return {k:bool(np.array_equal(a[k],b[k])) for k in a.files}


def main():
    torch.set_num_threads(1)
    fit=read(HERE/'FIT_RESULT.json');verify(fit['source_sha256'])
    binding=read(HERE/'BINDING.json');verify(binding['source_sha256'])
    checkpoints=[]
    for r in fit['log']:
        path=HERE/'FIT'/f"update{r['update']:03d}.pt"
        assert sha(path)==r['sha256']
        state=torch.load(path,weights_only=True,map_location='cpu')
        assert state['update']==r['update']
        assert state['input_manifest_sha256']==sha(HERE/'FIT_INPUTS.json')
        if r['update']:
            assert all(s['step'].item()==r['update'] for s in state['optimizer']['state'].values())
        checkpoints.append(r['update'])
    assert checkpoints==list(range(251))
    evaluations={}
    for case in TRAIN_CASES+HELD_CASES:
        data,meta=load_data(case);r=read(HERE/'EVAL'/case/'RESULT.json');verify(r['source_sha256'])
        assert sha(HERE/'EVAL'/case/'PREDICTIONS.npz')==r['predictions_sha256']
        assert r['data_sha256']==meta['trace_sha256'] and r['checkpoint_sha256']==fit['candidate_sha256']
        evaluations[case]={k:r[k] for k in ('pass_','gates','relative_reduction','nrmse','mse_hz2','initial_mse_hz2')}
    no_held_leak=all((HERE/'FIT_RESULT.json').stat().st_mtime_ns<(HERE/'DATA'/c/'RESULT.json').stat().st_mtime_ns for c in HELD_CASES)
    comparisons=dict(predictions=exact_npz(HERE/'EVAL/held_left/PREDICTIONS.npz',HERE/'REPLAY/held_left/PREDICTIONS.npz'),
        brain=exact_npz(HERE/'BRAIN/BRAIN.npz',HERE/'BRAIN_REPLAY/BRAIN.npz'),
        cut_restore=exact_npz(HERE/'CUT_RESTORE/TRACE.npz',HERE/'CUT_RESTORE_REPLAY/TRACE.npz'))
    brain=read(HERE/'BRAIN/RESULT.json');brain_replay=read(HERE/'BRAIN_REPLAY/RESULT.json')
    cut=read(HERE/'CUT_RESTORE/RESULT.json');cut_replay=read(HERE/'CUT_RESTORE_REPLAY/RESULT.json')
    for folder,result in (('BRAIN',brain),('BRAIN_REPLAY',brain_replay),('CUT_RESTORE',cut),('CUT_RESTORE_REPLAY',cut_replay)):
        verify(result['source_sha256'])
        file='BRAIN.npz' if folder.startswith('BRAIN') else 'TRACE.npz'
        assert sha(HERE/folder/file)==result['trace_sha256']
    gates=dict(checkpoint_ledger=True,held_labels_after_freeze=no_held_leak,
        heldout_reference_agreement=all(evaluations[c]['pass_'] for c in HELD_CASES),
        actual_supervised_gradients=max(r['error'] for r in fit['actual_loss_gradient_checks'])<1e-8,
        actual_adam=fit['first_adam_max_error']<1e-12,checkpoint_resume=fit['checkpoint_resume_exact'],
        independent_arrays=all(all(v.values()) for v in comparisons.values()),
        original_brain_receiving_response=brain['reference_agreement_pass'] and brain_replay['reference_agreement_pass'],
        actual_cut_recovery=cut['pass_'] and cut_replay['pass_'])
    weights,_=load(fit['candidate'])
    state={k:v.tolist() for k,v in weights.state_dict().items()}
    acceptance=dict(status='ENGINEERING_REFERENCE_TRANSLATION_VERIFIED' if all(gates.values()) else 'LIMITED_CANDIDATE_WITH_FAILURES',
       project='P1_open_source_brain',pass_=all(gates.values()),gates=gates,evaluations=evaluations,
       replay_arrays=comparisons,candidate=fit['candidate'],candidate_sha256=fit['candidate_sha256'],parameters=state,
       optimizer_updates=250,qualification_shadow_optimizer_updates=1,reference_lif_ticks=80000,
       new_fixed_neck_ticks=cut['native_neck_ticks']+cut_replay['native_neck_ticks'],
       prior_fixed_neck_ticks_reused=80000,brain_batch_ticks=brain['brain_batch_ticks']+brain_replay['brain_batch_ticks'],
       brain_lane_ticks=brain['brain_lane_ticks']+brain_replay['brain_lane_ticks'],new_physical_steps=0,
       receiver_nrmse=brain['receiver_nrmse'],excluded_feedback_edges=brain['excluded_feedback_edges'],
       actual_dn_events=brain['dn_events_by_lane'],brain_weights_trained=False,
       teacher_kind=binding['teacher_kind'],boundary_scope=binding['boundary'],
       biological_validation='UNKNOWN',autonomous_sensory_behavior='NOT_RUN_NOT_CLAIMED')
    write(HERE/'FINAL_ACCEPTANCE.json',acceptance)
    # Loss and two complete heldout channels: do not hide the onset/offset error.
    fig,axes=plt.subplots(3,1,figsize=(11,9))
    axes[0].semilogy([r['update'] for r in fit['log']],[r['normalized_mse'] for r in fit['log']],color='tab:blue')
    axes[0].set_xlabel('Actual optimizer update');axes[0].set_ylabel('Training normalized MSE');axes[0].grid(alpha=.3)
    with np.load(HERE/'EVAL/held_left/PREDICTIONS.npz',allow_pickle=False) as a:
        for c,ax in enumerate(axes[1:]):
            ax.plot(a['time_s'],a['initial'][:,c],color='.7',label='Initial encoder',lw=.8)
            ax.plot(a['time_s'],a['target'][:,c],color='black',label='Engineering reference',lw=1.2)
            ax.plot(a['time_s'],a['prediction'][:,c],color=('tab:blue','tab:orange')[c],label='Trained encoder',lw=1)
            ax.set_ylabel(('Body-left','Body-right')[c]+' source rate (Hz)');ax.set_xlabel('Physical-record time (s)');ax.grid(alpha=.3);ax.legend()
    fig.suptitle('2-source ascending translation: new engineering reference; frozen body / neck / brain')
    fig.tight_layout();fig.savefig(HERE/'TRANSLATION_LEARNING.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    with np.load(HERE/'BRAIN/BRAIN.npz',allow_pickle=False) as a:
        t=np.arange(len(a['receiver_voltage']))*DT
        for lane,(label,color) in enumerate((('Engineering reference','black'),('Trained','tab:blue'),('Initial','.6'),('Left cut','tab:orange'))):
            axes[0].plot(t,a['receiver_voltage'][:,lane].mean(1),label=label,color=color,lw=1)
    axes[0].set_ylabel('Native receiving-group mean (mV)');axes[0].set_xlabel('Brain time (s)')
    axes[0].set_title('194 native receivers; two externally owned AN sources; 14 incoming edges excluded')
    with np.load(HERE/'CUT_RESTORE/TRACE.npz',allow_pickle=False) as a:
        t=np.arange(len(a['prediction']))*DT
        axes[1].plot(t,a['normal_prediction'][:,0],color='.6',label='Intact body-left source')
        axes[1].plot(t,a['prediction'][:,0],color='tab:blue',label='Actual cut / restore, causal tail retained')
        axes[1].axvspan(.45,.90,color='red',alpha=.1,label='Left axon cut')
    axes[1].set_ylabel('Translated source rate (Hz)');axes[1].set_xlabel('Simulation time (s)')
    for ax in axes:ax.grid(alpha=.3);ax.legend()
    fig.suptitle('Input adaptation only: no DN firing or autonomous behavior in this probe')
    fig.tight_layout();fig.savefig(HERE/'BRAIN_AND_NECK_VERIFICATION.png',dpi=130);plt.close(fig)
    print('final engineering translation gates',gates,flush=True)
    if not all(gates.values()):raise SystemExit(2)


if __name__=='__main__':main()
