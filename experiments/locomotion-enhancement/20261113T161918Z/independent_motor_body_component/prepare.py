"""Parent metadata-only actualfinal600 qualification binding; unknownSHA rejected."""
import argparse
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--bind-completed-final600',action='store_true');p.add_argument('--sha256',required=True);a=p.parse_args()
    if not a.bind_completed_final600:p.error('Actual final600 qualification binding required')
    if len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256):raise ValueError('Actual64hexSHA required, unknownrejected')
    target=CAL/'TRAIN/checkpoint_000600.pt';trained=read(CAL/'TRAIN/RESULT.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');audit=read(CAL/'EVALUATION_VERIFY.json');jobs=read(CAL/'JOBS.json')
    if trained['updates']!=600 or trained['checkpoint']!=str(target) or trained['sha256']!=a.sha256 or sha(target)!=a.sha256:raise ValueError('Actual final600 SHA/steps required')
    if not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint_sha256']!=a.sha256 or stage['new_neck_deployed'] is not False or stage['body_behavior_pass'] is not None:raise ValueError('Actual pipeline3stageNNqualificationonly')
    if len(jobs)!=3 or any(j.get('exit_code')!=0 or j['status']!='COMPLETED' for j in jobs):raise ValueError('All3actualstages required')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or evaluation['checkpoint_sha256']!=a.sha256 or evaluation['step']!=600:raise ValueError('Actual independentfinalmetric proofrequired')
    proofs=[CAL/'TRAIN/RESULT.json',CAL/'STAGE_VERIFICATION.json',CAL/'EVALUATION.json',CAL/'EVALUATION_VERIFY.json',CAL/'JOBS.json',target]
    for step in (0,300,600):
        path=CAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path)
        if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['step']!=step or sha(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('Actualsaved0/300/600freshproduction qualificationrequired')
        proofs.extend((path,Path(v['checkpoint'])))
    if sha(evaluation['arrays']['path'])!=evaluation['arrays']['sha256']:raise ValueError('Actual independentmetric artifact drift')
    proofs.append(Path(evaluation['arrays']['path']))
    verify(trained['source_hashes']);verify(trained['data_hashes'])
    source={str(ROOT/n):sha(ROOT/n) for n in FILES}
    source.update(trained['source_hashes'])
    save(ROOT/'INPUT_BINDINGS.json',dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=a.sha256,motor_format='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1',final_step=600,state_dims=[128,64,64,96],source_hashes=source,data_hashes=trained['data_hashes'],proof_hashes={str(q):sha(q) for q in proofs},unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False))
    print(json.dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=a.sha256,body_behavior_pass=None)))


if __name__=='__main__':main()
