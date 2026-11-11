"""Strict deployment wrapper added without changing active training source identities."""
from pathlib import Path
import torch
from conditional_motor import load_motor,sha256,BASE_SHA256


def load_checked_motor(path, expected_checkpoint_sha256):
    path=Path(path).resolve()
    if sha256(path)!=expected_checkpoint_sha256:
        raise ValueError('selected checkpoint digest mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['identity']['baseline_sha256']!=BASE_SHA256:
        raise ValueError('baseline ancestry mismatch')
    config=payload['config']
    for group in ('source_hashes','data_hashes'):
        if not config.get(group):raise ValueError('missing immutable training identity '+group)
        for filename,digest in config[group].items():
            if not Path(filename).is_absolute() or sha256(filename)!=digest:
                raise ValueError('checkpoint '+group+' drift: '+filename)
    if config.get('resume') and sha256(config['resume'])!=config['resume_sha256']:
        raise ValueError('parent training checkpoint changed')
    model,loaded=load_motor(path)
    model.eval()
    return model,loaded
