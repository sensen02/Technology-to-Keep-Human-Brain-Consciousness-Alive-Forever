"""Fit one feed-forward motor policy to the shared local physical target."""
import argparse
import json
from pathlib import Path

import torch

from expansion import CORRECT, DEVELOP, FIRST, HERE, TRAIN, load_policy, sha
from cases import CORRECT as FIRST_CORRECT, TRAIN as FIRST_TRAIN
from fit import collect, decompose, save

ANCHOR = FIRST/'BC2_NO_SETPOINT/epoch078.pt'


def training_data(correction):
    specs = [(FIRST/'TEACHER', FIRST_TRAIN, True), (FIRST/'DAG1',FIRST_CORRECT, False),
             (HERE/'TEACHER', TRAIN, True)]
    if correction:
        specs.append((HERE/'CORRECT', CORRECT, False))
    arrays, sources = [], []
    for directory, cases, qualify in specs:
        batch = collect([directory/case.name for case in cases], qualify)
        arrays.append(batch[:3])
        sources.extend(batch[3])
    x, y, w = (torch.cat(parts) for parts in zip(*arrays, strict=True))
    return x, y, w, sources


def train(out, start, correction):
    torch.set_num_threads(1)
    torch.manual_seed(7127)
    generator = torch.Generator().manual_seed(5153)
    x, y, w, sources = training_data(correction)
    dev = collect([HERE/'TEACHER'/case.name for case in DEVELOP], True)
    epochs = 30 if correction else 45
    model = load_policy(start)
    model.train()
    model.target_bound.copy_(torch.maximum(model.target_bound, (1.15*y.abs().amax(0)).clamp_min(.06)))
    optimizer = torch.optim.AdamW(model.parameters(),lr=4e-4 if correction else 7e-4,weight_decay=1e-5)
    out.mkdir()
    inputs = dict(train=sources, dev=dev[3], n_train=len(x), n_dev=len(dev[0]), epochs=epochs,
                  previous=sha(start), previous_path=str(start), optimizer_restored=False,
                  objective='weighted normalized 42-joint square error, current observations only')
    (out/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
    save(out, 'start.pt', model, optimizer, generator, 0, inputs)
    with (out/'JOURNAL.jsonl').open('x',buffering=1) as log:
        for epoch in range(1, epochs+1):
            loss_sum = 0.
            for idx in torch.randperm(len(x), generator=generator).split(768):
                error = (model(x[idx])-y[idx]) / model.target_scale
                loss = (error.square().mean(1)*w[idx]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.,error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach())*len(idx)
            with torch.no_grad():
                scores = decompose(model,dev)
            path, digest = save(out,f'epoch{epoch:03d}.pt',model,optimizer,generator,epoch,inputs)
            log.write(json.dumps(dict(epoch=epoch,path=path.name,sha256=digest,
                                      train_loss=loss_sum/len(x),dev=scores))+'\n')
            if epoch%5 == 0:
                print(epoch, scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--start',type=Path,default=ANCHOR)
    parser.add_argument('--correction',action='store_true')
    args = parser.parse_args()
    train(args.out,args.start,args.correction)
