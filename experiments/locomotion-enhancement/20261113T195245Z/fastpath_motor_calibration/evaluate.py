"""Final8fit and separately fourunlabeledDEV outputs, no corrective targets."""
from adapter import run
from sources import ROOT,read,save,verify


def main():
    run('evaluate.py')
    import torch
    import common
    from data import dev_sources,production_sequence
    trained=read(ROOT/'TRAIN/RESULT.json');m,payload=common.load_checkpoint(trained['checkpoint'],trained['sha256'],final=True)
    rows=[]
    with torch.no_grad():
        for raw in dev_sources(read(ROOT/'INPUT_BINDINGS.json')):
            mean,state=production_sequence(m,raw)
            rows.append(dict(id=raw['entry']['id'],labels_available=False,training_input=False,action_RMS=float(mean.tanh().square().mean().sqrt()),action_absmax=float(mean.tanh().abs().max()),new96_absmax=float(state[:,256:].abs().max()),teacher_MSE=None,scope='Fixedfailedobservations outputs only, no newtrajectory or teacherstatequery'))
    save(ROOT/'DEV_EVALUATION.json',dict(kind='FASTPATH_FOUR_UNLABELED_FAILEDOBS_DEV_V1',rows=rows,checkpoint_sha256=trained['sha256'],physical_steps=0,teacher_labels_available=False))


if __name__=='__main__':main()
