"""Offline optical estimate from registered RGB terrain views and tracked landmark centres."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

RADIUS = .066  # mm, externally specified marker diameter is 0.132 mm
PATCH = np.linspace(-.23, .23, 15)
DX, DY = (a.ravel() for a in np.meshgrid(PATCH, PATCH))


def match(images, projections, x, y, heights, slope=(0., 0.)):
    """Normalized texture agreement of hypothetical local planes across the four views."""
    zz = heights[:, None] + slope[0]*DX + slope[1]*DY
    shape = zz.shape
    points = np.stack(np.broadcast_arrays((x+DX)/1000, (y+DY)/1000, zz/1000, 1), axis=-1)
    patches = []
    for img, p in zip(images, projections):
        uvw = points @ p.T
        uv = uvw[..., :2]/uvw[..., 2:]
        # P locates pixel centres at half-integer image coordinates.
        u, v = uv[..., 0]-.5, uv[..., 1]-.5
        patch = map_coordinates(img, [v, u], order=1, mode='constant', cval=np.nan,
                                prefilter=False).reshape(shape)
        patch -= patch.mean(axis=1, keepdims=True)
        norm = np.sqrt((patch**2).mean(axis=1, keepdims=True))
        patches.append(patch / np.where(norm > .022, norm, np.nan))
    pairs = list(combinations(range(len(patches)), 2))
    correlations = np.stack([(patches[a]*patches[b]).mean(axis=1) for a,b in pairs])
    return correlations


def peak(scores, heights):
    finite = np.isfinite(scores)
    if not finite.any():
        return np.nan, np.nan
    i = np.argmax(np.where(finite, scores, -np.inf))
    if i == 0 or i == len(scores)-1:
        return np.nan, np.nan
    return heights[i], scores[i]


def surface(images, ps, x, y, heights):
    def coarse(x0, y0):
        scores = match(images, ps, x0, y0, heights).mean(axis=0)
        return peak(scores, heights)[0]
    z = coarse(x,y)
    if not np.isfinite(z):
        return None
    step = .22
    around = np.arange(z-.14, z+.141, .005)
    def neighbor(dx,dy):
        return peak(match(images,ps,x+dx,y+dy,around).mean(axis=0), around)[0]
    slope = np.array([(neighbor(step,0)-neighbor(-step,0))/(2*step),
                      (neighbor(0,step)-neighbor(0,-step))/(2*step)])
    if not np.isfinite(slope).all() or np.linalg.norm(slope) > .7:
        return None
    fine = np.arange(z-.11, z+.111, .002)
    pairs = match(images,ps,x,y,fine,slope)
    best,score = peak(pairs.mean(axis=0), fine)
    # Six baseline fits, including two mutually disjoint opposing pairs.
    fits = np.asarray([peak(row,fine)[0] for row in pairs])
    if not np.isfinite(fits).all() or not np.isfinite(best) or score < .70:
        return None
    coarse_scores = match(images,ps,x,y,heights,slope).mean(axis=0)
    away = coarse_scores[np.abs(heights-best) > .09]
    other = away[np.isfinite(away)]
    if not len(other) or score - max(other) < .08:
        return None
    spread = float(np.max(np.abs(fits-best)))
    if spread > .04:
        return None
    return best,slope,score,spread


def estimate(info, track, terrain, output):
    cams = [c for c in info['projection'] if c.startswith('H')]
    if len(cams) != 4:
        raise ValueError('Expected four registered upper cameras')
    ps = [np.asarray(info['projection'][c]) for c in cams]
    images = [np.asarray(Image.open(terrain/(c+'.png')).convert('L'),float)/255 for c in cams]
    idx = [i for i,n in enumerate(track['names']) if n.endswith('_tarsus_tip')]
    pose,upper = (track[k][:,idx]*1000 for k in ('8_views','4_upper'))
    if not np.isfinite(pose).any():
        raise ValueError('No tracked toes')
    heights = np.arange(np.nanmin(pose[...,2])-2.,np.nanmax(pose[...,2])+.81,.02)
    fields = ['frame','time_ms','leg','views','toe_x_mm','toe_y_mm','toe_z_mm','ground_z_mm',
              'slope_x','slope_y','score','surface_spread_mm','marker_gap_mm','guard_mm',
              'low_mm','high_mm','state']
    rows = []
    for j,frame in enumerate(track['frames']):
        for k,i in enumerate(idx):
            tip = pose[j,k]
            leg = str(track['names'][i]).split('_')[0]
            views = int(track['views_used'][j,i])
            row = dict(zip(fields, [int(frame),float((frame-track['frames'][0])*1000/info['source_fps']),
                                    leg,views,*tip]))
            row['state'] = 'unresolved'
            if views >= 3 and np.isfinite([tip,upper[j,k]]).all():
                obs = surface(images,ps,*tip[:2],heights)
                if obs is not None:
                    z,slope,score,spread = obs
                    nz = 1 / np.sqrt(1 + slope @ slope)
                    gap = (tip[2]-z)*nz - RADIUS
                    # Conservative engineering guard, not a statistical confidence interval.
                    guard = max(.030, np.linalg.norm(tip-upper[j,k])) + max(.012, spread)
                    lo,hi = gap-guard,gap+guard
                    state = 'clear' if lo>0 else 'overlap' if hi<0 else 'contact_compatible'
                    row.update(zip(fields[7:],[z,*slope,score,spread,gap,guard,lo,hi,state]))
            rows.append(row)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('calibration','tracks','empty_rgb','output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    with np.load(a.tracks, allow_pickle=False) as track:
        estimate(json.loads(a.calibration.read_text()),track,a.empty_rgb,a.output)
