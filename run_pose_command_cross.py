"""2x2 pose/constant-command factorial replay. Mechanical research, not neural control."""
import json
import time
from pathlib import Path
import numpy as np
import run_dynamic_support_audit as dyn
s=dyn.s; cp=dyn.cp
OUT=s.HERE/'outputs/workspace_recheck/pose_command_cross.json'


def contrasts(values):
    # positive outcome means greater drop/downward speed, not improvement
    return dict(pose_effect_at_command_A=values[1][0]-values[0][0],
                pose_effect_at_command_B=values[1][1]-values[0][1],
                command_effect_at_pose_A=values[0][1]-values[0][0],
                command_effect_at_pose_B=values[1][1]-values[1][0],
                interaction=values[1][1]-values[1][0]-values[0][1]+values[0][0])


def main(report):
    start=time.monotonic(); before=cp.sha256_file(cp.CANONICAL_XML)
    assert before==cp.CANONICAL_XML_SHA256
    source=json.loads(dyn.SOURCE.read_text()); prior_path=s.HERE/'outputs/workspace_recheck/dynamic_support_audit.json'
    prior=json.loads(prior_path.read_text()); rows=dyn.select_rows(source)
    assert rows[0]['label']=='baseline_exact_source'
    assert rows[1]['label']=='root_height_-0.010mm'
    model=cp.load_model(); covered,_,_=s.coverage(model)
    excluded=np.setdiff1d(np.arange(model.nu),covered)
    result=[]
    for i,pose in enumerate(rows):
        for j,command in enumerate(rows):
            a=np.array(command['activation']); np.testing.assert_array_equal(a[excluded],model.actuator_ctrlrange[excluded,0])
            tr=dyn.integrate_observed(model,pose['qpos'],a,start+80)
            ref=s.integrate(model,pose['qpos'],a)
            identity=all(dyn.bit_identical(tr[k],ref[k]) for k in ('final_qpos','final_activation')) and all(dyn.bit_identical(tr['trace'][k],ref['trace'][k]) for k in ref['trace'])
            assert identity
            assert tr['finite'] and tr['command_unchanged'] and not tr['warning_count'] and not tr['automatic_reset_detected']
            assert all(c['within_tolerance'] for sample in tr['samples'] for c in [*sample['checks'].values(),*sample['muscles']['checks'].values()])
            diagonal=None
            if i==j:
                old=next(t for t in prior['trajectories'] if t['label']==pose['label'])
                diagonal=all(dyn.bit_identical(tr['trace'][k],old['trace'][k]) for k in tr['trace']) and dyn.bit_identical(tr['final_qpos'],old['final_qpos']) and dyn.bit_identical(tr['final_activation'],old['final_activation'])
                assert diagonal
            samples=tr['samples']; first,last=samples[0],samples[-1]
            measures=dict(root_drop_mm=first['root_z_mm']-last['root_z_mm'],COM_drop_mm=first['COM_position_mm'][2]-last['COM_position_mm'][2],final_COM_downward_speed_mm_s=-last['COM_velocity_mm_s'][2],initial_support=first['support_over_weight'],final_support=last['support_over_weight'])
            result.append(dict(pose_index=i,command_index=j,label=f'P{"AB"[i]}_C{"AB"[j]}',pose_label=pose['label'],command_label=command['label'],initial_qpos=pose['qpos'],initial_activation=a.tolist(),observation_identity=identity,prior_diagonal_identity=diagonal,metrics=measures,**tr))
            print(result[-1]['label'],measures,flush=True)
    effects={}
    for metric in result[0]['metrics']:
        values=[[next(r['metrics'][metric] for r in result if r['pose_index']==i and r['command_index']==j) for j in range(2)] for i in range(2)]
        effects[metric]=dict(matrix=values,**contrasts(values))
    after=cp.sha256_file(cp.CANONICAL_XML); assert before==after
    guard=report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):guard['previous_soft_limit_mb']='unlimited'
    out=dict(status='VERIFIED_FACTORIAL_REPLAY',trajectories=result,effects=effects,method='2 saved poses x 2 saved constant commands, 50ms native step, initial act=ctrl once; all diagnostic calculations scratch-only',provenance=dict(xml_before=before,xml_after=after,source=str(dyn.SOURCE),source_hash=cp.sha256_file(dyn.SOURCE),prior=str(prior_path),prior_hash=cp.sha256_file(prior_path),code_hash=cp.sha256_file(Path(__file__))),wall_s=time.monotonic()-start,memory_guard=guard,limits=['Deterministic two-level local contrasts, not statistical/general biological causation','Crossed commands were optimized elsewhere, not reoptimized per crossed state','Initial activation changes with command: command factor includes activation initialization','No neural controller or standing success demonstrated'])
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n'); print(OUT)

if __name__=='__main__':
    with s.mg.guarded_run('pose_command_cross',**s.mg.BUDGETS['neural_muscle_closure']) as report:main(report)
