"""Select among fixed checkpoints using only complete development episodes."""
import argparse
import json
from pathlib import Path
from context import HERE,first,hash_file
from tasks import DEVELOP


def select(branch,epochs,prefix,out):
    loss={}
    for line in (branch/'JOURNAL.jsonl').read_text().splitlines():
        row=json.loads(line)
        loss[row['epoch']]=row['development']['rmse_rad']
    entries=[]
    for epoch in epochs:
        root=HERE/(prefix+str(epoch))
        checkpoint=branch/f'epoch{epoch:03d}.pt'
        cases={}
        for task in DEVELOP:
            directory=root/task.name
            record=json.loads((directory/'RESULT.json').read_text())
            assert record['weight_sha256']==hash_file(checkpoint)
            assert record['trace_sha256']==hash_file(directory/'TRACE.npz')
            cases[task.name]=dict(pass_=record['pass_'],gates=record['gates'])
        entries.append(dict(path=str(checkpoint.resolve()),sha256=hash_file(checkpoint),epoch=epoch,
                       pass_count=sum(c['pass_'] for c in cases.values()),dev_rmse_rad=loss[epoch],cases=cases))
    best=min(entries,key=lambda row:(-row['pass_count'],row['dev_rmse_rad'],row['epoch']))
    first.write(out,dict(chosen=best,candidates=entries,complete=best['pass_count']==len(DEVELOP),
              rule='all physical development gates first, then development joint RMSE, then earliest epoch'))
    print('Selected',best['epoch'],best['pass_count'],'/',len(DEVELOP),'rmse',best['dev_rmse_rad'],flush=True)
    for entry in entries:
        print(entry['epoch'],entry['pass_count'],[name for name,c in entry['cases'].items() if not c['pass_']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',type=Path,default=HERE/'FIT')
    parser.add_argument('--epochs',type=int,nargs='+',default=[10,20,30])
    parser.add_argument('--prefix',default='DEV')
    parser.add_argument('--out',type=Path,default=HERE/'SELECTION.json')
    a=parser.parse_args()
    select(a.branch,a.epochs,a.prefix,a.out)
