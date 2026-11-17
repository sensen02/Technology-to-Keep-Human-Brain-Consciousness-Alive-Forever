"""Report predeclared functional motion from completed body JSON only; never edit results."""
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parent
BODY=ROOT/'state_aligned_motor_body'
CASES=('walk','stop','left_pivot','right_pivot')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--report-completed-body',action='store_true');a=p.parse_args()
    if not a.report_completed_body:p.error('Requires actual completed body results')
    output=ROOT/'FUNCTIONAL_MOTION_REPORT.json'
    if output.exists():raise FileExistsError('No overwrite or after-result threshold changes')
    prereg=read(ROOT/'FUNCTIONAL_MOTION_REPORT_PREREG.json');jobs=read(BODY/'JOBS.json');summary=read(BODY/'SUMMARY.json');binding=read(BODY/'INPUT_BINDINGS.json')
    if len(jobs)!=9 or any(j.get('status')!='COMPLETED' or j.get('exit_code')!=0 for j in jobs):raise ValueError('All original nine physical/audit stages must finish')
    ledger={str(x):sha(x) for x in (ROOT/'FUNCTIONAL_MOTION_REPORT_PREREG.json',Path(__file__),BODY/'JOBS.json',BODY/'SUMMARY.json',BODY/'INPUT_BINDINGS.json')};rows=[];total=0
    for name in CASES:
        folder=BODY/'trials'/name;rpath=folder/'RESULT.json';vpath=folder/'VERIFY.json';r=read(rpath);v=read(vpath)
        if not v.get('integrity_all_pass') or not v.get('checks') or not all(x is True for x in v['checks'].values()):raise ValueError('Actual neural replay and saved metrics must pass')
        if r['checkpoint_sha256']!=binding['checkpoint_sha256'] or r['training_updates']!=0 or r['teacher_calls_after_handover']!=0:raise ValueError('Same saved model, zero training and completely withdrawn teacher required')
        metrics=r['candidate_metrics'];g=metrics['gates']
        required=prereg['walk_functional_report']['required_original_gate_names'] if name=='walk' else list(g)
        if any(k not in g or type(g[k]) is not bool for k in required):raise ValueError('Missing or changed declared task gate')
        functional=all(g[k] for k in required)
        if name!='walk' and functional!=metrics['task_pass_candidate']:raise ValueError('Stop or pivot criteria must remain identical')
        rows.append(dict(condition=name,original_task_pass=metrics['task_pass_candidate'],secondary_functional_motion_pass=functional,required_original_gates=required,failed_required_gates=[k for k in required if not g[k]],legacy_slip_gate=g.get('legacy_slip'),original_five=r['original_five'],actual_candidate_metrics=metrics,early_stop=r['early_stop'],native_steps=r['actual_native_steps'],meaning='Secondary predeclared report; original result and historical FAIL remain untouched.'))
        total+=r['actual_native_steps'];ledger.update({str(rpath):sha(rpath),str(vpath):sha(vpath)})
    if total!=summary['actual_new_native_steps']:raise ValueError('Same four physical windows required')
    if any(sha(path)!=digest for path,digest in ledger.items()):raise ValueError('Original evidence changed during reporting')
    result=dict(kind=prereg['kind'],checkpoint_sha256=binding['checkpoint_sha256'],rows=rows,combined_component_functional_motion_pass=all(r['secondary_functional_motion_pass'] for r in rows),original_all_four_pass=all(r['original_task_pass'] for r in rows),original_files_unchanged=True,source_and_proof_hashes=ledger,training_updates=0,neural_forwards=0,Core=0,physical_steps=0,full_brain_neck_deployment_qualified=False,rough_terrain_traversal_qualified=False)
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(output)
if __name__=='__main__':main()
