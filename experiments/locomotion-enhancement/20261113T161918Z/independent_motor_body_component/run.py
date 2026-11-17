"""Parent finite4component bodies+4NNsaved audits+summary, no automatic extension."""
import argparse
import datetime
import subprocess
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-fourcases',action='store_true');a=p.parse_args()
    if not a.execute_parent_fourcases:p.error('Parent explicit4component release required')
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Actual same-source1Core0integration preflightrequired')
    path=ROOT/'JOBS.json'
    if path.exists():raise FileExistsError('No overwrite orautomaticretry')
    jobs=[]
    def journal():
        temp=path.with_suffix('.tmp')
        with temp.open('x') as stream:json.dump(jobs,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    journal()
    for condition in CONDITIONS:
        for stage,script,flag in (('PHYSICS','runner.py','--execute-parent-physical'),('FRESH_NN_SAVED_AUDIT','audit.py','--execute-parent-saved-audit')):
            verify(sources);command=[sys.executable,'-B',str(ROOT/script),'--condition',condition,flag]
            row=dict(condition=condition,stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(condition+'_'+stage+'.log')));jobs.append(row);journal();print(json.dumps(row),flush=True)
            with open(row['log'],'x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
            row.update(status='COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal()
            if result.returncode!=0:raise RuntimeError('Finitecomponent stagefailed; noretry/extension: '+condition+' '+stage)
    row=dict(stage='SUMMARY',status='RUNNING',start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());jobs.append(row);journal()
    rows=[];steps=0
    for condition in CONDITIONS:
        folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
        if not audit['integrity_all_pass'] or audit['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actual freshNNsaved auditrequired')
        steps+=r['native']['actual_native_steps'];rows.append(dict(condition=condition,command=CONDITIONS[condition],native_steps=r['native']['actual_native_steps'],candidate_metrics=r['candidate_metrics'],original_five=r['original_five'],early_stop=r['early_stop'],result=str(folder/'RESULT.json'),result_sha256=sha(folder/'RESULT.json'),audit=str(folder/'VERIFY.json'),audit_sha256=sha(folder/'VERIFY.json')))
    if steps>4928000:raise ValueError('Finitecomponent4928000native budgetexceeded')
    verify(sources)
    save(ROOT/'SUMMARY.json',dict(kind=FORMAT+'_SUMMARY',actual_new_native_steps=steps,max_native_budget=4928000,rows=rows,checkpoint_sha256=read(ROOT/'INPUT_BINDINGS.json')['checkpoint_sha256'],scope='ExplicitbodyNNcomponent experiments only; no new brain-realneck deployment or complete3partgoalqualification',new_neck_deployed=False,source_hashes=sources))
    row.update(status='COMPLETED',exit_code=0,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary_sha256=sha(ROOT/'SUMMARY.json'));journal();print('FOUR_COMPONENT_CASES_SAVED_AND_AUDITED',flush=True)


if __name__=='__main__':main()
