"""Stdlib actual completed fastpath qualification; no model or array reads."""
import argparse
from common import ROOT,CAL,FORMAT,FILES,read,save,sha,verify
from adapter import OLD,MOTOR_FORMAT


def passed(value):
    return value.get('integrity_all_pass') is True and bool(value.get('checks')) and all(v is True for v in value['checks'].values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-completed-final600',action='store_true');parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if not args.bind_completed_final600:parser.error('Completed actual final600 qualification required')
    digest=args.sha256
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Actual lowercase64hex SHA required')
    if (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('No overwrite/rebinding')
    code=read(ROOT/'CODECHECK.json');verify(code['source_hashes'])
    if code['status']!='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN' or not all(code['checks'].values()):raise ValueError('Frozen READYSTD required')
    old=read(ROOT/'OLD_SOURCE_SNAPSHOT.json');verify(old['source_hashes'])
    cal_binding=read(CAL/'INPUT_BINDINGS.json');verify(cal_binding['source_hashes']);verify(cal_binding['data_hashes'])
    trained=read(CAL/'TRAIN/RESULT.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');audit=read(CAL/'EVALUATION_VERIFY.json');jobs=read(CAL/'JOBS.json');pre=read(CAL/'PREFLIGHT.json')
    target=CAL/'TRAIN/checkpoint_000600.pt'
    if trained['updates']!=600 or trained['checkpoint']!=str(target) or trained['sha256']!=digest or sha(target)!=digest:raise ValueError('Actual TRAIN600/PT SHA required')
    expected=('TRAIN_FIXED600_AND_3FRESH_SAVED_VERIFICATIONS','FINAL600_EVALUATION','INDEPENDENT_SAVED_METRIC_AUDIT')
    if len(jobs)!=3 or tuple(j['stage'] for j in jobs)!=expected or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs):raise ValueError('Actual exact3mainstagesexit0 required')
    if not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint']!=str(target) or stage['checkpoint_sha256']!=digest or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or stage['new_neck_deployed'] is not False or stage['body_behavior_pass'] is not None:raise ValueError('Completed stage qualification only')
    if not passed(audit) or audit['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or evaluation['checkpoint_sha256']!=digest or evaluation['step']!=600:raise ValueError('Actual independent final evaluation proof required')
    if not pre['all_passed'] or pre['source_hashes']!=cal_binding['source_hashes']:raise ValueError('Actual same-source calibration preflight required')
    proofs=[CAL/n for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','TRAIN/RESULT.json','STAGE_VERIFICATION.json','EVALUATION.json','EVALUATION_VERIFY.json','JOBS.json','DEV_EVALUATION.json')]+[target,ROOT/'CODECHECK.json']
    for step in (0,300,600):
        path=CAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);checkpoint=CAL/'TRAIN'/f'checkpoint_{step:06d}.pt'
        devpath=CAL/'TRAIN'/f'DEV_VERIFY_{step:06d}.json';dev=read(devpath)
        actual=sha(checkpoint)
        if not passed(v) or v['step']!=step or v['checkpoint']!=str(checkpoint) or v['checkpoint_sha256']!=actual:raise ValueError('Actual fresh saved0/300/600 proof required')
        if not passed(dev) or dev['step']!=step or dev['checkpoint_sha256']!=actual or dev['physical_steps']!=0 or dev['teacher_labels_available'] is not False or len(dev['rows'])!=4:raise ValueError('Actual four unlabeled DEV fresh proof required')
        verify(v['source_hashes']);verify(v['data_hashes']);proofs.extend((path,devpath,checkpoint))
    dev=read(CAL/'DEV_EVALUATION.json')
    if dev['checkpoint_sha256']!=digest or len(dev['rows'])!=4 or dev['teacher_labels_available'] is not False:raise ValueError('Actual final unlabeled DEV evaluation required')
    arrays=evaluation['arrays']
    if sha(arrays['path'])!=arrays['sha256']:raise ValueError('Actual evaluation arrays artifact drift')
    proofs.append(__import__('pathlib').Path(arrays['path']))
    verify(trained['source_hashes']);verify(trained['data_hashes'])
    source={**old['source_hashes'],**cal_binding['source_hashes'],**trained['source_hashes'],**code['source_hashes']}
    for name in FILES:source[str(ROOT/name)]=sha(ROOT/name)
    data={**cal_binding['data_hashes'],**trained['data_hashes']}
    before={str(path):sha(path) for path in proofs}
    verify(source);verify(data);verify(before)
    binding=dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=digest,motor_format=MOTOR_FORMAT,final_step=600,state_dims=[128,64,64,96],source_hashes=source,data_hashes=data,proof_hashes=before,unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False,scope='BodyANNcomponent_NOTnewbrainneckdeployment')
    verify(source);verify(data);verify(before)
    save(ROOT/'INPUT_BINDINGS.json',binding)
    print(__import__('json').dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=digest,body_behavior_pass=None)))


if __name__=='__main__':main()
