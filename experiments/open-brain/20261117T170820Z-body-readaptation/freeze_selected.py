"""Freeze the learned candidate and evaluation scripts before held inference."""
import json
from pathlib import Path
from context import HERE,first,hash_file


def main():
    selected=json.loads((HERE/'SELECTION.json').read_text())
    assert selected['complete']
    weight=Path(selected['chosen']['path'])
    assert selected['chosen']['sha256']==hash_file(weight)
    sources=[HERE/name for name in ('SELECTION.json','evaluate.py','collect.py','audit.py','batch.py',
             'regress.sh','tasks.py','TASKS.json','context.py','PROTOCOL.zh-CN.md','TRAINING_INPUTS.json')]
    sources += [weight,weight.parent/'GRADIENT.json',weight.parent/'ACTUAL_PARAMETER_UPDATES.json',
                weight.parent/'SAVE_RELOAD.json',weight.parent/'JOURNAL.jsonl',weight.parent/'INPUTS.json']
    first.write(HERE/'EVALUATION_INPUTS.json',dict(sha256={str(file.resolve()):hash_file(file) for file in sources},
                  checkpoint=str(weight.resolve()),weight_sha256=hash_file(weight)))
    print(str(weight.resolve()),flush=True)


if __name__=='__main__':main()
