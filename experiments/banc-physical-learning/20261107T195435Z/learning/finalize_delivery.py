"""Finalize read-only study summary and mirror source/evidence into both owned roots."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def main():
    result=read(OUT/'STUDY_RESULT.json');verification=read(HERE/'INDEPENDENT_STUDY_VERIFICATION.json')
    trials=[read(OUT/'training'/f'trial_{i:03d}'/'RESULT.json') for i in range(53)]
    selected=read(OUT/'evaluation/selected/RESULT.json');zero=read(OUT/'evaluation/zero/RESULT.json')
    feasible=np.array([r['feasible'] for r in trials]);scores=np.array([r['reward'] for r in trials])
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for eligible,color,label in ((True,'tab:blue','Original support gates passed'),(False,'tab:red','Original support gate failed')):
        mask=feasible==eligible;axes[0].scatter(np.arange(53)[mask],scores[mask],c=color,label=label,s=28)
    axes[0].axhline(scores[0],ls='--',c='black',label='Zero-output baseline reward')
    axes[0].set(xlabel='Training episode in preregistered order',ylabel='Fixed shaping reward',title='53 one-second checkpoint episodes; not a continuous walking trajectory');axes[0].legend(fontsize=9)
    labels=['forward','weighted slip','height excess','nonfoot impulse/W','support deficit','cycle legs']
    keys=['forward_displacement','weighted_slip_distance','height_drop_excess','positive_nonfoot_impulse_overW','support_deficit_integral','completed_cycle_legs']
    x=np.arange(len(keys));axes[1].bar(x-.18,[selected['reward_terms'][k] for k in keys],.36,label='Selected 5s validation')
    axes[1].bar(x+.18,[zero['reward_terms'][k] for k in keys],.36,label='Zero 5s validation')
    axes[1].set_xticks(x,labels);axes[1].set(ylabel='Raw component (mixed declared native units)',title='Separate fixed5s validation; terms are scoring proxies, not walking acceptance');axes[1].legend(fontsize=9)
    fig.suptitle('Physical feedback search | MN-only actor | no biological calibration claim')
    fig.savefig(HERE/'STUDY_COMPARISON.png',dpi=160);plt.close(fig)
    generations='\n'.join('| '+str(g['generation'])+' | '+str(g['feasible'])+' | '+g['incumbent_episode']+' | '+f"{g['incumbent_reward']:.9f}"+' |' for g in result['generations'])
    limits='Selected coefficients remain exactly zero: this search did not improve on the actuator hold baseline.' if result['selected_training_episode']=='trial_000' else 'The selected candidate improved the fixed training score; this alone does not establish neural walking or generalization.'
    report=f'''# Preregistered direct physical feedback learning

Completed the fixed **53 training episodes plus two5s validation episodes** from the same verified actualtime1s checkpoint. These are separate reset episodes, not a continuous walking trajectory. {limits}

The actor uses only323 source-identified MNfiltered firing rates through a frozen same-leg SVD basis and168 bounded coefficients; no bias, no teacher, body/root/time/phase input. Raw body state is used only for privileged training reward and gates. The original complete physical body, native12.5us step, liveforward,1ms PD128/caps,5ms coupling, fixed20k/435126-edge conductance circuit,556encoder andzeroDN remain unchanged. Inner synapses/neural parameters were not optimized.

Baseline training reward: **{result['baseline_training_reward']:.12g}**. Selected training reward: **{result['selected_training_reward']:.12g}**, selected archive `{result['selected_training_episode']}`. Feasible training episodes: **{int(feasible.sum())}/53**, including the originalbaseline and four explicit incumbent reset controls.

| Generation | Feasible candidates /13 | Incumbent archive | Incumbent reward |
|---|---:|---|---:|
{generations}

All48 nonzero candidate policies exercised nonzero actor outputs (rateRMS0.359319–0.613591rad/s). Twelve passed the originalsupportgate, but the bestnonzero reward−1.014442992 remained worse thanzero−0.117040201. Registered stance-air-stance coverage among nonzero trials was0legs in15,1leg in17,2legs in12,3legs in2 and4legs in2; none reached5or6. This is a failed limitedsearch, not a proof that no usefulpolicy exists. Sustained mean rates may cause targetdrift; parent separately analyzes savedDCpower rather than treating that possibility as proven. See FAILURE_INTERPRETATION.json.

The optimizer used exactlyseed20261107, four generations of six antithetic Gaussian pairs plusincumbent, initialstd.01, coeffbounds[-.25,.25], feasible-first up-to4elites, mean=.5old+.5elitemean, std=.7old+.3populationelitestd clipped[.004,.05]. Incumbent changes only for feasible strict reward improvement. All candidate arrays, randomdraws, order, before/aftermeans/std and selection evidence are retained. Incumbent reevaluations are explicitly cross-worker/fullreset reproducibility controls; all four passed, maximum error {max(verification['incumbent_control_max_errors']):.12g}.

## Validation and limitations

| Result | Selected5s | Zero5s |
|---|---:|---:|
| Fixed shaping reward | {selected['reward']:.12g} | {zero['reward']:.12g} |
| Original support gate passed | {selected['physical_gate']['passed']} | {zero['physical_gate']['passed']} |
| Forward COM displacement | {selected['reward_terms']['forward_displacement']:.12g} | {zero['reward_terms']['forward_displacement']:.12g} |
| Loadweighted material slip distance | {selected['reward_terms']['weighted_slip_distance']:.12g} | {zero['reward_terms']['weighted_slip_distance']:.12g} |
| Legs completing cyclecoverage proxy | {selected['reward_terms']['completed_cycle_legs']} | {zero['reward_terms']['completed_cycle_legs']} |

Reward is fixed forward displacement minus slip/lateral/height/nonfoot/supportdeficit/rate penalties plus a limitedstance-air-stance coverage bonus. The bonus is a scoring proxy, not full coordination acceptance. Support gates passing cannot establish neural necessity because zero-rate PDtarget hold may support this body. Absolute SIforce and biological calibration are NOT_ESTABLISHED; no stable walking or complete brain claim is made.

Training slip is approximate5ms endpoint contactmaterial Jacobian sampling over1s. Finalvalidation uses0.5ms samples over5s. Their raw reward magnitudes are not directly comparable across horizon/sampling and do not imply deterioration duringvalidation. Parent saved-array analysis reports median nonzero rateDCpowerfraction0.237922; do not characterize all policies as constantoutput or lacking dynamics. Native20-columncontact/COM data are retained at every12.5us step for **all55episodes**, alongside 1msBODY/COMMANDS,5msPOLICY,ACTOR,INITIAL,completeNEURAL_INITIAL/FINAL and per-episode result/provenance. No failed physical policy trace was silently removed. No softwareerror occurred during the completed study; pretrial fixture corrections are recorded separately.

Each validation starts at actualtime1s using INITIAL.state; COMMANDS timestamps use actual d.time. Parent owns final independent nativephysics replay and video. Release-only data do not contain a newly executed warmup. Parent plans one selected-zero video, with zeroVIDEO_REFERENCE pointing to the identicaltrajectory, avoiding a redundant secondrender. If the parent joins original0..1s warmupBODY to releaseBODY, it must verify the seam and label shared warm-history initialization provenance. This learning agent opened and inspected the actual STUDY_COMPARISON.png; VISUAL_REVIEW records its scope, not a physicalvideo review.

## Verification and actual calls

`{HERE/'verify_study.py'}` independently reconstructs source hashes, canonical physics/neural starts, actorweights/predictions, all53candidate draw/order/thetas, every scoreterm/originalgate, fullreset incumbentcontrols and all generation updates without importing optimizer/runner/scoring or executing physics/neural steps. **{verification['checks_passed']} checks passed**,55episodes checked. It also confirms frozen selectedtheta was used for5svalidation and the second actor was exactlyzero.

Original study actualcalls: **{result['actual_calls']['actual_native_steps']} physicalsteps**, **{result['actual_calls']['actual_neural_steps']} neuralsteps**, **{result['actual_calls']['actual_live_forward_calls']} live/initforward calls**. These are study calls, not independentreplay calls. The saved-data verifier adds0physical/0neural steps and0optimization updates. Actual counters and each trial's elapsedtime are retained inCALL_LEDGER andRESULT. Fourmean/std distribution updates are the learning operations; basis is frozen. This agent does not alter trackedfiles, originalbody/core/encoder/checkpoint orfoundation.

Key files:

- Source preregistration: `{HERE/'PREREG.json'}`.
- Parent review authorization: `{HERE/'START_AUTHORIZATION.json'}`.
- Optimizer and physical runner: `{HERE/'optimize.py'}` and `{HERE/'episode.py'}`.
- Full study evidence: `{OUT/'STUDY_RESULT.json'}` and `{OUT/'CALL_LEDGER.json'}`.
- Selected coefficients: `{OUT/'SELECTED.npz'}`.
- Finalselected native/physics/neural artifacts: `{OUT/'evaluation/selected'}`.
- Finalzero native/physics/neural artifacts: `{OUT/'evaluation/zero'}`.
- Independent verification: `{HERE/'INDEPENDENT_STUDY_VERIFICATION.json'}`.
- Comparison plot: `{OUT/'STUDY_COMPARISON.png'}`.

All episodeartifacts also mirror into `{HERE}` with bytehash verification; sourcecode and reports mirror into `{OUT}`. Delivery verification lists both sides. No extra budget extension, posthoc primary selection, modified gate or reward tuning was performed.
'''
    (HERE/'REPORT.md').write_text(report)
    # Both roots belong to this delegated task. Preserve all original files.
    for path in list(OUT.rglob('*')):
        if path.is_file() and path.name not in ('DELIVERY_VERIFICATION.json',):
            rel=path.relative_to(OUT);dest=HERE/rel
            desktop_owned_data=(rel.parts[0] in ('training','evaluation') or rel.parts[0].startswith('generation_') or rel.name in ('STUDY_RESULT.json','CALL_LEDGER.json','SELECTED.npz'))
            if not dest.exists() or (desktop_owned_data and sha(path)!=sha(dest)):
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    sources=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='DELIVERY_VERIFICATION.json']
    records=[]
    for source in sources:
        relative=source.relative_to(HERE);dest=OUT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha(source)!=sha(dest):shutil.copy2(source,dest)
        sourcehash=sha(source);desthash=sha(dest)
        if sourcehash!=desthash:raise AssertionError(str(relative))
        records.append(dict(relative_path=str(relative),workspace=str(source),desktop=str(dest),bytes=source.stat().st_size,sha256=sourcehash,byte_equal=True))
    delivery=dict(status='PASS_BYTE_IDENTICAL_FULL_STUDY',files=records,files_including_record=len(records)+1,total_bytes_excluding_record=sum(x['bytes'] for x in records),actual_study_physical_steps=result['actual_calls']['actual_native_steps'],actual_study_neural_steps=result['actual_calls']['actual_neural_steps'],record_note='Thisrecord mirrors byteidentically; selfhashnotembeddedrecursively.')
    text=json.dumps(delivery,indent=2,ensure_ascii=False)+'\n'
    for folder in (HERE,OUT):(folder/'DELIVERY_VERIFICATION.json').write_text(text)
    assert sha(HERE/'DELIVERY_VERIFICATION.json')==sha(OUT/'DELIVERY_VERIFICATION.json')
    print(json.dumps({k:delivery[k] for k in ('status','files_including_record','total_bytes_excluding_record','actual_study_physical_steps','actual_study_neural_steps')}))

if __name__=='__main__':main()
