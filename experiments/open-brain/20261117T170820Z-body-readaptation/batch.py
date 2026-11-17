"""Run only declared full trajectories and retain the exit status of each worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from context import HERE
from tasks import TRAIN,DEVELOP,CORRECT,EXTRA


def execute(split,mode,out,checkpoint,jobs):
    tasks={'train':TRAIN,'dev':DEVELOP,'correct':CORRECT,'extra':EXTRA}[split]
    out=out.resolve()
    out.mkdir()
    def worker(task):
        cmd=[str(HERE/'drive.sh'),'collect.py',mode,task.name,str(out/task.name)]
        if checkpoint:cmd+=['--checkpoint',str(checkpoint.resolve())]
        with (out/f'{task.name}.log').open('x') as stream:
            code=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
        (out/f'{task.name}.exit').write_text(f'{code}\n')
        print(task.name,code,flush=True)
        return code
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return int(any(tuple(pool.map(worker,tasks))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('train','dev','correct','extra'))
    parser.add_argument('mode',choices=('teacher','correct','evaluate'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--jobs',type=int,default=8)
    args=parser.parse_args()
    raise SystemExit(execute(args.split,args.mode,args.out,args.checkpoint,args.jobs))
