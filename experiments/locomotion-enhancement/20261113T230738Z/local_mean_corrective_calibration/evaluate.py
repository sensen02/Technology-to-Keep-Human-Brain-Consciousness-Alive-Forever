"""Final12 maskedfit +8unlabeledDEV descriptors, no body/teacherquery inference."""
import argparse
import sys
from sources import ROOT,read,save,sha,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-evaluation',action='store_true');a=p.parse_args()
    if not a.execute_parent_evaluation:p.error('Parentfinal600evaluationrequired')
    from learning import cpu1,metrics
    cpu1()
    import torch,numpy as np
    import common,model
    from data import prepare,dev_sources,production_sequence
    tr=read(ROOT/'TRAIN/RESULT.json')
    if tr['updates']!=600 or tr['checkpoint']!=str(ROOT/'TRAIN/checkpoint_000600.pt') or sha(tr['checkpoint'])!=tr['sha256']:raise ValueError('Actualfinal600only')
    for step in (0,300,600):
        proof=read(ROOT/'TRAIN'/f'VERIFY_{step:06d}.json')
        if not proof['integrity_all_pass'] or sha(proof['checkpoint'])!=proof['checkpoint_sha256']:raise ValueError('Actualall3freshsavedverifyrequired')
    m,payload=common.load_checkpoint(tr['checkpoint'],tr['sha256'],final=True);data=prepare(m);scan=model.make_scan(m);rows=metrics(scan,data)
    with torch.no_grad():
        residual,h=scan(data['features']);prediction=(data['oldmean']+residual).tanh();lo=(data['oldmean']-6.).tanh();hi=(data['oldmean']+6.).tanh();nearest=data['y'].clamp(min=lo,max=hi);floor=(nearest-data['y']).square()
        path=ROOT/'EVALUATION_ARRAYS.npz';np.savez_compressed(path,actual_target=data['y'].numpy().copy(),old_mean=data['oldmean'].numpy().copy(),new_residual=residual.numpy().copy(),new_action=prediction.numpy().copy(),command=data['u'].numpy().copy(),valid=data['valid'].numpy().copy(),floor_squared=floor.numpy().copy())
        dev=[]
        for raw in dev_sources(read(ROOT/'INPUT_BINDINGS.json')):
            mean,hidden=production_sequence(m,raw);dev.append(dict(id=raw['entry']['id'],actual_suffix_rows=len(raw['x']),labels_available=False,training_input=False,teacher_MSE=None,action_RMS=float(mean.tanh().square().mean().sqrt()),action_absmax=float(mean.tanh().abs().max()),new96_absmax=float(hidden[:,256:].abs().max()),scope='ConditionaloutputsonfailedactualX, notnewbodytrajectory'))
    verify(payload['config']['source_hashes']);verify(payload['config']['data_hashes'])
    save(ROOT/'EVALUATION.json',dict(kind='LOCALMEAN_CORRECTIVE_FINAL600_MASKED12_EVALUATION_V1',step=600,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],rows=rows,DEV=dev,arrays=dict(path=str(path),sha256=sha(path)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,behavior_pass=None,new_neck_deployed=False,invalid_rows_preserved_not_physical_pass=True))


if __name__=='__main__':main()
