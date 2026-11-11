"""Pure-data draft contract; no Torch/MuJoCo imports."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

LAYOUT_KEYS={'layout_id','offset_forward','offset_lateral','patch_radius','sigma','amplitude','noise_std','onset_elapsed_s','rise_s','arrival_dwell_s','seed'}


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def layout_from(path):
    p=Path(path)
    if not p.is_absolute():raise ValueError('layout path must be absolute')
    d=json.loads(p.read_text());validate_layout(d);return d


def validate_layout(d):
    if set(d)!=LAYOUT_KEYS:raise ValueError('layout fields must match exact draft schema')
    if not isinstance(d['layout_id'],str) or not d['layout_id']:raise ValueError('layout_id required')
    for key in LAYOUT_KEYS-{'layout_id','seed'}:
        if isinstance(d[key],bool) or not isinstance(d[key],(int,float)) or not math.isfinite(d[key]):raise ValueError('finite numeric '+key+' required')
    for key in ('patch_radius','sigma','amplitude','rise_s','arrival_dwell_s'):
        if d[key]<=0:raise ValueError(key+' must be positive')
    if d['noise_std']<0 or not 0<=d['onset_elapsed_s']<9.6 or d['arrival_dwell_s']>9.6:raise ValueError('invalid timing/noise')
    if isinstance(d['seed'],bool) or not isinstance(d['seed'],int) or d['seed']<0:raise ValueError('nonnegative integer seed required')
    if math.hypot(d['offset_forward'],d['offset_lateral'])<=d['patch_radius']:raise ValueError('initial sampling midpoint must lie outside food patch')
    return d


def source_center(midpoint_xy,head_rotation,layout):
    validate_layout(layout)
    midpoint=np.asarray(midpoint_xy,dtype=np.float64);rotation=np.asarray(head_rotation,dtype=np.float64)
    if midpoint.shape!=(2,) or rotation.shape!=(3,3) or not np.isfinite(midpoint).all() or not np.isfinite(rotation).all():raise ValueError('finite actual handover geometry required')
    forward=rotation[:2,0].copy();norm=np.linalg.norm(forward)
    if norm<1e-8:raise ValueError('head forward has no horizontal direction')
    forward/=norm;left=np.array([-forward[1],forward[0]])
    return midpoint+layout['offset_forward']*forward+layout['offset_lateral']*left


def qualification(report,motor_path,motor_sha):
    # Deliberately strict canonical schema; producer must explicitly bind selected artifact.
    return bool(report.get('all_conditions_eligible') is True and report.get('checkpoint_sha256')==motor_sha
        and report.get('checkpoint')==str(Path(motor_path).resolve()))


class Arrival:
    """Count actual consecutive1ms endpoints; exclude pre-onset arrival, no round-up."""
    def __init__(self,dwell_s):self.required=float(dwell_s);self.count=0;self.max_count=0;self.first_confirmed=None
    def tick(self,elapsed_s,inside,onset_s):
        self.count=self.count+1 if inside and elapsed_s>=onset_s else 0
        self.max_count=max(self.max_count,self.count)
        if self.first_confirmed is None and self.count*.001>=self.required:self.first_confirmed=float(elapsed_s)
    def report(self):
        return dict(arrived=self.first_confirmed is not None,first_confirmed_elapsed_s=self.first_confirmed,max_contiguous_inside_s=self.max_count*.001,required_dwell_s=self.required,ingestion=False)
