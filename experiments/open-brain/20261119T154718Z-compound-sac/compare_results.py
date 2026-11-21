"""Generate paired comparison and figures from stored SAC and MuJoCo results."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = ('flat', 'wide_bridge', 'slope_bridge', 'staggered', 'staggered_slope')
LABELS = ('Flat', 'Wide bridge', 'Slope + bridge', 'Offset gaps', 'Slopes + offset gaps')
COLORS = ('#6d7379', '#007d83', '#bd7010', '#3e5cad', '#a24673')


def read(path):
    return json.loads(path.read_text())


def show_training(root):
    final = read(root/'FINAL.json')
    name_order = STAGES
    snapshots = [final['initial']]
    snapshots += [json.loads(row) for row in (root/'CHECKPOINTS.jsonl').read_text().splitlines()]
    times = [row['elapsed']/60 for row in snapshots]
    fig, (ax, compare) = plt.subplots(1, 2, figsize=(13.,4.3), layout='constrained')
    for name, title, color in zip(name_order, LABELS, COLORS, strict=True):
        ax.plot(times, [row['scores'][name]['meal']/row['scores'][name]['n']*100
                        for row in snapshots], marker='o', markersize=3.5,
                color=color, label=title, linewidth=1.7)
    picked = final['selected']['elapsed']/60
    ax.axvline(picked, linestyle='--', linewidth=.9, color='#191e24')
    ax.set(title='Course development set', xlabel='Training window (min)', ylabel='Full proxy meals (%)')
    ax.set_ylim(-5,108)
    ax.legend(loc='center left', fontsize=8)
    for arm, shift, color, title in (
            ('heldout_before',-.18,'#9ba0a7','Previous policy'),
            ('heldout_after',.18,'#007d83','Selected checkpoint')):
        rates = final[arm]
        heights = [rates[n]['meal']/rates[n]['n']*100 for n in name_order]
        bars = compare.bar(np.arange(len(heights))+shift, heights, width=.34, color=color, label=title)
        compare.bar_label(bars, labels=[f"{rates[n]['meal']}/{rates[n]['n']}" for n in name_order],
                          padding=3, fontsize=8)
    compare.set_ylim(0,113)
    compare.set_title('Independent mirror-paired proxy tests')
    compare.set_xticks(range(len(LABELS)), LABELS, rotation=19, ha='right')
    compare.set_ylabel('Full proxy meals (%)')
    compare.legend(loc='upper right', fontsize=8)
    for pane in (ax, compare):
        pane.spines[['right','top']].set_visible(False)
        pane.grid(axis='y', alpha=.20)
        pane.set_axisbelow(True)
    fig.savefig(HERE/'TRAINING_EVIDENCE.png', dpi=170)
    plt.close(fig)


def sensor_summary(folder):
    with np.load(folder/'BODY.npz', allow_pickle=False) as data:
        trace = data['trace']
        obs = data['observation']
        commands = data['commands']
    detected = obs[obs[:,1] > .5]
    moving = np.flatnonzero((trace[:,0] >= .24) & (np.linalg.norm(trace[:,7:9],axis=1) > .12))
    request = np.flatnonzero(np.linalg.norm(obs[:,17:19],axis=1) > .02)
    measured = np.flatnonzero(np.linalg.norm(commands[:,3:5],axis=1) > .02)
    return dict(camera_updates=len(obs), fruit_observations=len(detected),
                first_seen_s=float(detected[0,0]) if len(detected) else None,
                first_forward_or_turn_request_s=float(obs[request[0],0]) if len(request) else None,
                first_motor_drive_s=float(commands[measured[0],0]) if len(measured) else None,
                first_body_motion_s=float(trace[moving[0],0]) if len(moving) else None,
                peak_risk=float(obs[:,5:11].max()), final_xy_mm=(10*trace[-1,1:3]).tolist())


def compare(baseline, candidate):
    records = [read(p/'AUDIT.json') for p in (baseline,candidate)]
    assert all(r['n'] == r['expected'] for r in records)
    cases = {}
    groups = defaultdict(list)
    for name, before in records[0]['cases'].items():
        after = records[1]['cases'][name]
        assert before['geometry'] == after['geometry']
        assert before['model_sha'] == after['model_sha'] and before['delays_ms'] == after['delays_ms']
        assert before['neural_seed'] == after['neural_seed']
        assert before['max_duration_s'] == after['max_duration_s']
        with np.load(baseline/name/'BODY.npz',allow_pickle=False) as a, np.load(candidate/name/'BODY.npz',allow_pickle=False) as b:
            assert np.array_equal(a['initial'],b['initial'])
            identical_commands = np.array_equal(a['commands'],b['commands'])
            identical_body_trace = np.array_equal(a['trace'],b['trace'])
        cases[name] = dict(identical_commands=identical_commands, identical_body_trace=identical_body_trace,
            challenge=after['challenge'], seed=after['seed'], side=after['side'],
            meal_before=before['validated_meal'], meal_after=after['validated_meal'],
            intake_before=before['intake'], intake_after=after['intake'],
            crosses_before=sum(g['valid'] for g in before['gates']),
            crosses_after=sum(g['valid'] for g in after['gates']),
            available_crosses=len(after['gates']),
            fall_before=before['first_fall_s'], fall_after=after['first_fall_s'],
            sensors_before=sensor_summary(baseline/name),
            sensors_after=sensor_summary(candidate/name))
        groups[after['challenge']].append(cases[name])
    result = dict(
        candidates={name:read(path/'IDENTITY.json') for name,path in
                    zip(('before','after'),(baseline,candidate),strict=True)},
        cases=cases, by_group={
            key: dict(n=len(rows), before=sum(row['meal_before'] for row in rows),
                      after=sum(row['meal_after'] for row in rows))
            for key,rows in groups.items()},
        native_steps=sum(sum(c['native_steps'] for c in run['cases'].values()) for run in records))
    (HERE/'PHYSICAL_COMPARISON.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_paths((baseline,candidate),records)
    return result


def plot_paths(roots, reports):
    names = tuple(reports[0]['cases'])
    fig, axes = plt.subplots(2, 3, figsize=(13.4,7.7), layout='constrained')
    for pane, name in zip(axes.flat, names):
        facts = reports[1]['cases'][name]
        geom = facts['geometry']
        bounds = [geom] if 'gates' not in geom else geom['gates']
        for g in bounds:
            x = [10*g['near'],10*g['far']]
            a,b = (-14,10*g['edge']) if g['side'] > 0 else (-10*g['edge'],14)
            pane.fill_between(x,a,b,color='#e0e5f0')
        goal = plt.Circle((10*geom['goal_x'],10*geom['goal_y']),4.9,facecolor='#f4b696',
                          edgecolor='#d87b54',alpha=.45,zorder=1)
        pane.add_patch(goal)
        for root, report, color, label in zip(roots,reports,
                                               ('#677482','#167768'),
                                               ('Previous policy','30-min SAC'),strict=True):
            fact = report['cases'][name]
            with np.load(root/name/'BODY.npz',allow_pickle=False) as saved:
                trajectory = saved['trace'][:,1:3]*10
            pane.plot(*trajectory.T, color=color, linewidth=2, zorder=2,
                      label=f"{label}  intake {fact['intake']:.3f}")
            marker = 'x' if fact['first_fall_s'] is not None else 'o'
            pane.scatter(*trajectory[-1],color=color,marker=marker,s=29,zorder=3)
        pane.legend(loc='lower right',fontsize=7)
        pane.set(title=f"{name}  ·  seed {facts['seed']}", xlabel='Forward (mm)', ylabel='Lateral (mm)',
                 xlim=(-1.,max(24,10*(geom['goal_x']+.52))),ylim=(-14,14))
        pane.axhline(0,color='#7d8790',linestyle=':',linewidth=.65,zorder=0)
        pane.spines[['right','top']].set_visible(False)
        pane.set_aspect('equal',adjustable='box')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.06,.86,'Measured trajectories',fontsize=14,weight='bold')
    axes.flat[-1].text(.06,.64,'Blue bands show unsupported ground.\nOrange circles show the food footprints.\nCross marks a fall. Circle marks the final position.',
                       linespacing=1.6,fontsize=10,va='top')
    fig.savefig(HERE/'PHYSICAL_PATHS.png',dpi=170)
    plt.close(fig)


def cue_diagnosis():
    details = {}
    for name in ('SLOPE_L','COMPOUND_L'):
        original = [HERE/'LIVE_BASELINE'/name, HERE/'LIVE_SELECTED'/name]
        raised = [HERE/'VISIBLE_BASELINE'/name, HERE/'VISIBLE_SELECTED'/name]
        data = dict(
            ground_food={arm:sensor_summary(path) for arm,path in
                         zip(('before','after'),original,strict=True)},
            visible_food={})
        for arm,ground,cue in zip(('before','after'),original,raised,strict=True):
            origin = read(ground/'RESULT.json')
            changes = read(cue/'AUDIT.json')
            assert origin['geometry'] == changes['geometry']
            assert origin['model_sha'] == changes['model_sha'] and Path(origin['policy']).resolve() == Path(changes['policy']).resolve()
            assert origin['neural_seed'] == changes['neural_seed']
            with np.load(ground/'BODY.npz',allow_pickle=False) as p, np.load(cue/'BODY.npz',allow_pickle=False) as q:
                assert np.array_equal(p['initial'],q['initial'])
            data['visible_food'][arm] = sensor_summary(cue)|dict(
                meal=changes['validated_meal'],fall_s=changes['first_fall_s'],
                gates=changes['gates'],intake=changes['intake'],stand=changes['ended_standing'])
        details[name] = data
    (HERE/'CUE_DIAGNOSIS.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    print(json.dumps(details['SLOPE_L']['visible_food'],allow_nan=False))


def plot_hazard():
    panel=read(HERE/'RISK_TRANSFER_BOTH.json')
    fig, axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained',sharey=True)
    for row,arm in enumerate(('BASELINE','SELECTED')):
        for col,name in enumerate(('SLOPE_L','COMPOUND_L')):
            ax=axes[row,col]
            records=panel[arm][name]
            t=records['t']
            ax.plot(t,np.max(records['geometric_reference'],axis=1),label='Geometry reference',
                    color='#6d7583',linestyle='--',linewidth=1.8)
            ax.plot(t,np.max(records['observed'],axis=1),label='Camera-derived risk',
                    color='#903358',linewidth=2)
            entry=records['lip']['entry_s']
            if entry is not None:
                ax.axvline(entry,color='#aa692b',linewidth=.9,label='First gap entry')
            if records['first_fall_s'] is not None:
                ax.axvline(records['first_fall_s'],color='#302426',linewidth=.9,
                           linestyle=':',label='Fall')
            ax.set(title=f"{arm.title()} · {name}",xlabel='Simulation time (s)',ylabel='Peak sector risk',
                   ylim=(-.05,1.05))
            ax.spines[['right','top']].set_visible(False)
            ax.grid(axis='y',alpha=.20)
            ax.legend(loc='upper left',fontsize=8)
    fig.savefig(HERE/'VISION_RISK.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--visible-cue', action='store_true')
    parser.add_argument('--hazard', action='store_true')
    args = parser.parse_args()
    if args.training:
        show_training(HERE/'BUDGET_1800')
    if args.physical:
        print(json.dumps(compare(HERE/'LIVE_BASELINE',HERE/'LIVE_SELECTED')['by_group']))
    if args.visible_cue:
        cue_diagnosis()
    if args.hazard:
        plot_hazard()
