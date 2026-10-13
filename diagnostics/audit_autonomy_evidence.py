"""Independently reconstruct stored commands/spike counts; no simulation or tuning.

A passed consistency audit cannot upgrade unvalidated translation into autonomy.
"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/autonomous_final_audit'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    bodypath=ROOT/'outputs/autonomous_sensorimotor_round3/exploratory.json'
    neuralpath=ROOT/'outputs/autonomous_pathways_round4/report.json'
    tracepath=ROOT/'outputs/autonomous_pathways_round4/spike_traces.npz'
    body=json.loads(bodypath.read_text());neural=json.loads(neuralpath.read_text())
    traces=np.load(tracepath,allow_pickle=False)
    checks={};intervals=0
    for name,c in body['cases'].items():
        for row in c['commands']:
            events=np.asarray(row['spikes_sparse_time_neuron'],int).reshape(-1,2)
            counts=np.bincount(events[:,1],minlength=body['tier_config']['max_neurons']) if len(events) else np.zeros(body['tier_config']['max_neurons'])
            expected=[]
            for mapping in body['actuator_mapping']:
                ids=mapping['local']
                rate=float(counts[ids].sum())/(len(ids)*.005) if ids else 0.
                expected.append(min(rate/100.,1.) if name!='motor_readout_cut' else 0.)
            checks.setdefault('reconstructed_all_motor_commands',True)
            checks['reconstructed_all_motor_commands'] &= bool(np.array_equal(expected,row['ctrl']))
            checks.setdefault('unmapped_controls_zero',True)
            checks['unmapped_controls_zero'] &= all(row['ctrl'][i]==0 for i,m in enumerate(body['actuator_mapping']) if not m['local'])
            checks.setdefault('zero_nonsensory_current_reported',True)
            checks['zero_nonsensory_current_reported'] &= row['nonsensory_injected_current_max_nA']==0
            intervals+=1
    checks['180_intervals']=intervals==180
    checks['no_stim_trace_zero']=not np.any(traces['no_stim'])
    checks['counts_recomputed']=all(int(traces[k].sum())==v['network_counts']['total'] for k,v in neural['cases'].items())
    cut=traces['claw_propagation_cut'];outside=np.ones(cut.shape[1],bool)
    outside[neural['cases']['claw_propagation_cut']['stimulated_local_indices']]=False
    checks['cut_spikes_confined_to_stimulated_sensory']=not np.any(cut[:,outside])
    checks['cut_release_silent']=not np.any(cut[100:])
    hashes={}
    for report in (body,neural):
        for p,h in report['source_sha256'].items():
            hashes[p]=Path(p).is_file() and sha(p)==h
    checks['recorded_sources_current']=all(hashes.values())
    checks['zero_background_config']=body['tier_config']['background_sd_nA']==body['tier_config']['background_mean_nA']==neural['config']['background_sd_nA']==neural['config']['background_mean_nA']==0
    out={'audit_status':'PASS' if all(checks.values()) else 'FAIL','goal_status':'NOT_ACHIEVED',
         'checks':checks,'source_checks':hashes,'intervals_reconstructed':intervals,
         'input_sha256':{str(p):sha(p) for p in (bodypath,neuralpath,tracepath)},
         'limitations':['Offline replay consistency cannot prove complete source correctness.',
                        'Sensory currents not reconstructed from calibrated receptors; explorer uses unvalidated body feature broadcast.',
                        'Exact muscle names do not validate anatomical connection or force calibration.',
                        'Body collapses; no autonomous locomotion or visual loop established.']}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'audit.json').write_text(json.dumps(out,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6))
    labels=list(checks);ax.barh(labels,[int(checks[k]) for k in labels],color=['seagreen' if checks[k] else 'crimson' for k in labels])
    ax.set_xlim(0,1.2);ax.set_xticks([0,1],['FAIL','PASS']);ax.invert_yaxis()
    ax.set_title('Independent evidence consistency audit\nAUTONOMY GOAL NOT ACHIEVED — consistency is not biological validation')
    fig.tight_layout();fig.savefig(OUT/'audit.png',dpi=130);plt.close(fig)
    print(json.dumps(out['checks'],indent=2));print(out['audit_status'],out['goal_status'])
    return 0 if out['audit_status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
