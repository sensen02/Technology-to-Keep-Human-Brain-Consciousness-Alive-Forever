"""Independent final evidence conjunction. No integration, training, or evidence edits."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'20261110T152714Z'
sys.path.insert(0,str(TRAIN))
from identity import verify,verify_runtime,sha256
from physics import Core


def load(path):return json.loads(Path(path).read_text())
def need(ok,message):
    if not ok:raise ValueError(message)


def main():
    audit_path=HERE/'NATIVE_ROLE_2500_AUDIT.json';audit=load(audit_path)
    need(audit['status']=='PASS' and not audit['failures_or_unknowns'],'native audit not passed')
    for path,digest in audit['provenance']['all_input_barrier_hashes'].items():need(sha256(path)==digest,'audited input changed before final closure')
    for key in ('frozen_evaluator','measurement_freeze','wrapper'):
        row=audit['provenance'][key];need(sha256(row['path'])==row['sha256'],'audit implementation changed')
    need(audit['physical_metric_conjunction'] and audit['engineering_native_all_pass'] and audit['six_cycles_pass'],'native metric conjunction failed')
    need(set(audit['engineering_native'])=={'support','nonfoot','com','forward','slip'},'incomplete engineering gates')
    need(all(x['status']=='PASS' for x in audit['predeclared_measurement_criteria'].values()),'physical criterion missing')
    need(audit['native_phase_sample_counts']==dict(settle=80000,auxiliary=384000,autonomous=768000),'full native phase counts absent')
    need(audit['native_diagnostics']['all_actual_extra_forces_zero'],'actual extra force')
    need(audit['neural_synchronization']['all_checkpoint_outputs_bitexact'] and audit['neural_synchronization']['hchain_exact'],'neural closure failed')
    dep=audit['dependency_evidence']
    need(dep['matched_zero_control']['prefix_all_exact'] and dep['offline_saved_hidden_sensory_cuts']['any_command_dependence'],'dependency evidence failed')
    # Do not rewrite the original frozen evaluator's historical uncertainty.
    need(audit['frozen_legacy_report']['criteria']['D7']['status']=='UNKNOWN','legacy D7 was rewritten')
    native=HERE/'NATIVE_ROLE_2500';candidate=TRAIN/'BC_GRU_ROLE_3000/checkpoint_002500.pt'
    need(audit['physical_identity']['checkpoint_sha256']==sha256(candidate),'selected checkpoint changed')
    core=Core(adhesion=False);perturbs=[]
    for seed in (20261111,20261112):
        run=HERE/f'PERTURB_ROLE_2500_SEED_{seed}';result=load(run/'RESULT.json');identity=load(run/'IDENTITY.json')
        verify(identity);verify_runtime(identity,core)
        need(result['seed']==seed and identity['config']['cli']['seed']==seed,'perturb seed mismatch')
        need(Path(identity['config']['cli']['checkpoint']).resolve()==candidate,'perturb checkpoint mismatch')
        need(result['control']=='student' and not result['sampled'],'perturb not deterministic ANN')
        need(result['transition']['all_exact'] and result['transition']['no_reset'],'perturb handover reset')
        need(result['autonomous']['all_gates'] and result['autonomous_cycles']['all_legs_completed'],'perturb did not pass every gate/cycle')
        need(set(result['autonomous']['gates'])=={'support','nonfoot','com','forward','slip'} and all(result['autonomous']['gates'].values()),'perturb gate coverage')
        pulse=result['perturbation']
        need(pulse['seed']==seed and pulse['duration_ms']==20 and pulse['normal_std']==.02 and not pulse['state_writes'] and not pulse['external_forces'],'perturbation contract changed')
        expected=np.clip(np.random.default_rng(seed).normal(0,.02,42),-1,1)
        need(np.array_equal(np.asarray(pulse['action']),expected),'pulse does not match seed')
        for key,name in [('trace','trace_1ms.npz'),('decisions','decisions_5ms.npz')]:need(sha256(run/name)==result[key]['sha256'],'perturb output hash mismatch')
        with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
            expected_source=np.r_[np.ones(20),np.zeros(1000),np.full(4800,2),np.full(9600,3)]
            need(np.array_equal(z['control_source'],expected_source),'perturb complete phase blocks missing')
            action=z['actual_requested_action'];need(np.array_equal(action[:20],np.tile(expected,(20,1))) and np.all(action[20:1020]==0),'pulse/settle actual commands differ')
            need(all(np.max(np.abs(z[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')),'perturb extra force')
        perturbs.append(dict(seed=seed,run=str(run),result_sha256=sha256(run/'RESULT.json'),slip=result['autonomous']['slip_ratio_max'],forward=result['autonomous']['forward_signed_native_per_s'],COMmin=result['autonomous']['com_height_min'],gates=result['autonomous']['gates'],cycles=result['autonomous_cycles']['complete_per_leg'],identity_verified=True))
    video_dir=HERE/'VIDEO_NATIVE_ROLE_2500';video=load(video_dir/'VIDEO.json')
    need(video['source_run']==str(native) and video['frames']==video['decoded_frames']==770,'video same-run/frame count mismatch')
    need(video['mesh_geoms']==71 and video['all_mesh_opaque'] and video['physics_steps']==0 and video['all_qpos_qvel_exact'],'video/body/pose claim mismatch')
    need(video['source_unchanged'] and video['compiled_arrays_unchanged'] and video['native_rows']==1232000,'video original run identity mismatch')
    need(isinstance(video['visual_review'],dict) and video['visual_review'].get('parent_opened_images') is True,'parent visual review not recorded')
    need(video['visual_review'].get('continuous_handover_and_full_body_visible') is True,'visual review failed')
    need(sha256(video['video'])==video['sha256'],'video changed')
    for path,digest in video['source_hashes'].items():need(sha256(path)==digest,'video source changed')
    for row in video['keyframes']:need(sha256(row['path'])==row['sha256'],'decoded frame changed')
    ledger=load(HERE/'SOURCE_LEDGER.json')
    for path,digest in ledger['source_hashes'].items():need(sha256(path)==digest,'confirmation source changed')
    need(float(core.d.time)==0.,'final checker unexpectedly integrated physics')
    output=dict(status='PASS',scope='authorized substitute-ANN / derived complete body / finite incremental PD bounded walking implementation',selected_checkpoint=str(candidate),checkpoint_sha256=sha256(candidate),native_audit=str(audit_path),native_audit_sha256=sha256(audit_path),native_engineering=audit['engineering_native'],physical_criteria=audit['predeclared_measurement_criteria'],legacy_D7=audit['frozen_legacy_report']['criteria']['D7'],perturbations=perturbs,video=str(video['video']),video_sha256=video['sha256'],video_evidence=str(video_dir/'VIDEO.json'),video_evidence_sha256=sha256(video_dir/'VIDEO.json'),visual_review=video['visual_review'],neural_synchronization=audit['neural_synchronization'],dependency_scope='matched actual zero-output physical prefix; current offline saved-hidden sensory cuts; old300 physical sensory intervention is historical lineage evidence, not a BC2500 physical cut',test_duration=dict(settle_s=1.,teacher_s=4.8,autonomous_s=9.6,perturb_seeds=[20261111,20261112]),qualifications=['D7 legacy UNKNOWN retained; conservative >=.1W median4 supplies predeclared sufficient evidence for reported lower positive carrying thresholds','D6 primary continuous causal1ms native velocity-difference method, signed original p95 and raw/absolute/.25/.5/1/2ms sensitivities retained','two pulse-before-settle initial conditions do not establish arbitrary-disturbance or arbitrary-duration robustness'],not_established=['original connectome capability','frozen original-body certification','independent SI or biological validity','arbitrary-duration robustness'],physical_steps_in_final_checker=0,all_confirmation_sources_unchanged=True,checker=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)))
    path=HERE/'FINAL_ACCEPTANCE.json'
    with path.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',scope=output['scope'],out=str(path),perturbation_slips=[r['slip'] for r in perturbs],native_slip=audit['engineering_native']['slip']['value'],video=str(video['video']))),flush=True)

if __name__=='__main__':main()
