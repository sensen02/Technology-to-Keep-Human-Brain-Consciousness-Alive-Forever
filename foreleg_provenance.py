"""Foreleg attachment provenance: compare the upstream OpenSim file with our model sites.

The upstream FlyMimic OpenSim model is Apache-2.0 and carries one Millard2012 muscle per
MTU, each with PathPoint elements that name a parent body and a location in that body's
frame. Our model's tendon sites carry the same PathPoint names, so the ancestry can be
checked directly without loading MuJoCo.

Outputs: a per-muscle traceability table (source file + muscle block + path point names +
parent bodies + locations), and a scale/residual check against our own site coordinates.
No model load, no forward, no integration, no neural step.
"""
from pathlib import Path
import json, hashlib, re
D = Path('/home/sensen/Desktop/engine修复验收/s5-external-muscle-source/20261028T120000Z')
OSIM = D / 'best_combined_full.osim'
OURS = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z/MODEL.xml')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_osim(text):
    muscles = []
    for m in re.finditer(r'<Millard2012EquilibriumMuscle name="([^"]+)">(.*?)</Millard2012EquilibriumMuscle>', text, re.S):
        name, body = m.group(1), m.group(2)
        pts = []
        for p in re.finditer(r'<PathPoint name="([^"]+)">(.*?)</PathPoint>', body, re.S):
            pname, pbody = p.group(1), p.group(2)
            frame = re.search(r'<socket_parent_frame>(.*?)</socket_parent_frame>', pbody, re.S)
            loc = re.search(r'<location>(.*?)</location>', pbody, re.S)
            pts.append(dict(path_point=pname,
                            parent_frame=frame.group(1).strip().replace('/bodyset/', '') if frame else None,
                            location=[float(x) for x in loc.group(1).split()] if loc else None))
        fmax = re.search(r'<max_isometric_force>(.*?)</max_isometric_force>', body, re.S)
        lopt = re.search(r'<optimal_fiber_length>(.*?)</optimal_fiber_length>', body, re.S)
        tslack = re.search(r'<tendon_slack_length>(.*?)</tendon_slack_length>', body, re.S)
        muscles.append(dict(muscle=name, leg=name[:2], path_points=pts,
                            max_isometric_force=float(fmax.group(1)) if fmax else None,
                            optimal_fiber_length=float(lopt.group(1)) if lopt else None,
                            tendon_slack_length=float(tslack.group(1)) if tslack else None))
    return muscles


def parse_ours(text):
    root_sites = {}
    for m in re.finditer(r'<site name="([^"]+)" pos="([^"]+)"', text):
        root_sites[m.group(1)] = [float(x) for x in m.group(2).split()]
    return root_sites


def main():
    osim_text = OSIM.read_text()
    ours_text = OURS.read_text()
    muscles = parse_osim(osim_text)
    ours = parse_ours(ours_text)
    rows = []
    ratios = []
    for mu in muscles:
        for pt in mu['path_points']:
            key_candidates = [k for k in ours if k.endswith(pt['path_point'])]
            ourpos = ours[key_candidates[0]] if key_candidates else None
            ratio = None
            if ourpos and pt['location']:
                num = [abs(b) / abs(a) for a, b in zip(pt['location'], ourpos) if abs(a) > 1e-9]
                if num:
                    ratio = sum(num) / len(num)
                    ratios.append(ratio)
            rows.append(dict(muscle=mu['muscle'], leg=mu['leg'], path_point=pt['path_point'],
                             parent_body=pt['parent_frame'], osim_location=pt['location'],
                             our_site_name=key_candidates[0] if key_candidates else None,
                             our_local_pos=ourpos, scale_ratio=ratio,
                             max_isometric_force=mu['max_isometric_force']))
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    resid = []
    if mean_ratio:
        for r in rows:
            if r['osim_location'] and r['our_local_pos']:
                pred = [v * mean_ratio for v in r['osim_location']]
                err = max(abs(a - b) for a, b in zip(pred, r['our_local_pos']))
                r['predicted_mm'] = pred
                r['max_abs_error_mm'] = err
                resid.append(err)
    legs = {}
    for r in rows:
        legs.setdefault(r['leg'], []).append(r['path_point'])
    out = dict(
        scope='Foreleg attachment provenance only. No model load, no integration.',
        source=dict(file=str(OSIM), sha256=sha(OSIM), license='Apache-2.0 (github.com/gizemozd/FlyMimic)',
                    citation='Özdil et al. 2025, arXiv:2509.06426; model file best_combined_full.osim'),
        our_model=dict(file=str(OURS), sha256=sha(OURS)),
        muscle_count=len(muscles),
        legs_present=sorted(legs),
        path_points_per_leg={k: len(v) for k, v in legs.items()},
        scale_ratio_mean=mean_ratio,
        max_abs_error_mm=max(resid) if resid else None,
        sites_without_upstream_match=[r['path_point'] for r in rows if not r['our_site_name']],
        rows=rows)
    (D / 'FORLEG_TRACEABILITY.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(muscle_count=out['muscle_count'], legs=out['legs_present'],
                          points_per_leg=out['path_points_per_leg'], scale=mean_ratio,
                          max_err_mm=out['max_abs_error_mm'],
                          unmatched=out['sites_without_upstream_match']), indent=2))


if __name__ == '__main__':
    main()
