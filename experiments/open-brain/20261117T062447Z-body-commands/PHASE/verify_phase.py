"""Independently verify every held run and surgical pair for the selected phase model."""
import argparse
from pathlib import Path

import torch

from phase import HERE, study
from verify_commands import narrow, paired_speed, record, summarize, verify_brake, verify_cut, verify_replays
from cases import HELD as OLD_HELD


def verify(checkpoint):
    digest = study.sha(checkpoint)
    names = [case.name for case in (*study.HELD, study.CUT)]
    old_names = [case.name for case in OLD_HELD]
    new = summarize(HERE/'HELD',names)
    old = summarize(HERE/'OLD',old_names)
    replay = verify_replays(HERE/'HELD',HERE/'REPLAY',names,digest)
    old_replay = verify_replays(HERE/'OLD',HERE/'OLD_REPLAY',old_names,digest)
    speed = paired_speed(new,'held_{}_{}')
    cut = verify_cut(HERE/'HELD')
    brake = {name:verify_brake(HERE/'HELD',name) for name in names if 'brake' in name}
    families = {family:all(new[f'held_{family}_{i}']['pass_'] for i in range(2))
                for family in study.FAMILIES}
    ckpt = torch.load(checkpoint,weights_only=True,map_location='cpu')
    study.write(HERE/'FINAL.json',dict(
        checkpoint=str(checkpoint),sha256=digest,epoch=ckpt['epoch'], mode=new[names[0]]['mode'],
        new=narrow(new),old=narrow(old),old_replay=old_replay,replay=replay,speed=speed,cut=cut,brake=brake,
        families=families,native_validation_steps=2*study.TICKS*(len(names)+len(old_names)),
        all_pass=all(families.values()) and all(r['pass_'] for r in old.values()) and
                 all(v['pass_'] for v in speed) and all(cut['gates'].values()) and
                 all(min(b.values())>0 for b in brake.values())))
    print('families', families, 'old', {name:r['pass_'] for name,r in old.items()}, 'cut', cut,
          'speed', speed, 'brake', brake, 'independent matches', len(replay)+len(old_replay))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    verify(args.checkpoint.resolve())
