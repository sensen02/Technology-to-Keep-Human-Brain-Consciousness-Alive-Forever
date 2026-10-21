"""Declared radius LADDER around the delivered pose A: at what radius does a declared-cone static equilibrium
stop being findable?

Background (measured, artifact S2-equilibrium-A-corrected-reissued-20261015T161259):
  * full declared box, 240 starts                                              -> best max|qacc| = 2.5865734771464476e-11 mm/s^2
  * nested around pose A: R = 0.20 (40 starts) / 0.10 (240) / 0.05 (40)        -> 14.340629240512852 / 66.6177212372147 / 235.55124711621315, no hits
  * the accepted equilibrium sits 0.42313245798143917 rad from pose A in the 42 leg joints, and 6 of 240 starts in
    the full box landed on a gate-passing equilibrium (roughly 2.5 per start in 100), so a probe with only 40 starts
    would be ambiguous: about a third of the time it would miss the basin for sampling reasons alone. Every probe here
    therefore uses 240 starts, the same number that already produced hits in the full box.

DECLARED PROTOCOL (written before any run and never relaxed afterwards):
  * geometry   : act in [1e-4, 1], root z +-0.05 mm, and the 42 leg hinge joints within R of the DELIVERED pose A
                 (clipped to joint range minus 5e-3 rad)
  * ladder     : R = 0.45, 0.40, 0.35, 0.30, 0.25 rad, run as five independent searches
  * starts     : 240 random starts per probe; the generator seed is fixed inside the search implementation, so the
                 start sets of the probes correspond to each other
  * criterion  : a probe "hits" iff max|qacc| <= 1e-3 mm/s^2 AND six tarsi in contact AND zero non-foot contacts AND
                 every contact inside the declared diamond u_diamond <= 1
  * result     : the smallest ladder radius with at least one hit, reported at 0.05 rad resolution; the hit counts are
                 reported as well, because "one hit in 240" and "forty hits in 240" are different statements
  * wording    : every number is a NEGATIVE OR POSITIVE SEARCH RESULT for a finite number of starts inside a declared
                 box. A miss is never a proof that no equilibrium exists inside that radius.
  * no touching: no sealed artifact directory is modified; each probe is its own run directory recording its own
                 report.json hash, and this batch records all of them plus their hashes.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
PY = str(ROOT / 'venv_body/bin/python')
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
CANONICAL_XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
GATE = 1e-3
STARTS = 240
LADDER = [0.45, 0.40, 0.35, 0.30, 0.25]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probe(radius):
    label = 'ladderR%03d' % int(round(radius * 100))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    t0 = time.time()
    cp = subprocess.run([PY, '-B', str(SEARCH), '--pose', str(POSE_A), '--label', label,
                         '--radius', '%.4f' % radius, '--multistart', str(STARTS)],
                        cwd=str(ROOT), env=env, capture_output=True, text=True)
    dirs = sorted(DESK.glob('S2-mujoco-equilibrium-%s-*' % label))
    entry = dict(radius_rad=radius, label=label, returncode=cp.returncode, seconds=round(time.time() - t0, 1),
                 stdout_tail=cp.stdout.strip().splitlines()[-2:], stderr_tail=cp.stderr.strip().splitlines()[-2:])
    if dirs:
        d = dirs[-1]
        rj = json.loads((d / 'report.json').read_text())
        ms = rj.get('multistart', {})
        hits = ms.get('per_equilibrium_hits') or []
        gate_hits = [h for h in hits if h['max_abs_qacc_mm_s2'] <= GATE and h['six_tarsi_in_contact']
                     and h['n_nonfoot_contacts'] == 0 and h['max_u_diamond'] is not None
                     and h['max_u_diamond'] <= 1 + 1e-9]
        entry.update(run_dir=str(d), report_json_sha256=sha(d / 'report.json'),
                     best_max_abs_qacc_mm_s2=ms.get('best_max_abs_qacc_mm_s2'), n_starts=ms.get('n_starts'),
                     n_below_1_mm_s2=ms.get('n_below_1_mm_s2'), n_below_1e_minus_3=ms.get('n_below_1e_minus_3'),
                     n_hits_all_four_gate_items=len(gate_hits),
                     per_equilibrium_hits=[dict(found_at_start_index=h['found_at_start_index'],
                                                max_abs_qacc_mm_s2=h['max_abs_qacc_mm_s2'], qacc_norm=h['qacc_norm'],
                                                n_contacts=h['n_contacts'], n_tarsus_contacts=h['n_tarsus_contacts'],
                                                n_nonfoot_contacts=h['n_nonfoot_contacts'],
                                                six_tarsi_in_contact=h['six_tarsi_in_contact'],
                                                max_u_diamond=h['max_u_diamond'],
                                                root_z_change_from_delivered_pose_A_mm=h['root_z_change_from_delivered_pose_A_mm'],
                                                max_abs_delta_q_rad_vs_delivered_pose_A=h['max_abs_delta_q_rad_vs_delivered_pose_A'])
                                            for h in hits])
    else:
        entry['run_dir'] = None
    entry['hit'] = bool(entry.get('n_hits_all_four_gate_items', 0) > 0)
    print('R = %.3f -> rc %s  best %s  gate hits %s  (%.1f s)'
          % (radius, entry['returncode'], entry.get('best_max_abs_qacc_mm_s2'),
             entry.get('n_hits_all_four_gate_items'), entry['seconds']), flush=True)
    return entry


def main():
    out = DESK / ('S2-equilibrium-radius-ladder-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    print('declared ladder', LADDER, 'starts', STARTS, flush=True)
    with ThreadPoolExecutor(max_workers=len(LADDER)) as ex:
        probes = list(ex.map(probe, LADDER))
    ordered = sorted(probes, key=lambda p: -p['radius_rad'])
    hits = [p for p in ordered if p['hit']]
    misses = [p for p in ordered if not p['hit']]
    smallest_hit = min((p['radius_rad'] for p in hits), default=None)
    largest_miss = max((p['radius_rad'] for p in misses), default=None)
    monotone = all(not (a['hit'] and not b['hit']) for a, b in zip(sorted(ordered, key=lambda p: p['radius_rad']),
                                                                   sorted(ordered, key=lambda p: p['radius_rad'])[1:]))
    verdict = dict(
        smallest_ladder_radius_with_a_hit=smallest_hit,
        largest_ladder_radius_without_a_hit=largest_miss,
        interval_rad=([largest_miss, smallest_hit] if (smallest_hit is not None and largest_miss is not None) else None),
        monotone_in_radius=bool(monotone), ladder=LADDER, starts_per_probe=STARTS,
        statement=('with %d identical starts per probe, the smallest ladder radius around the delivered pose A at which '
                   'a declared-cone static equilibrium was FOUND is %.2f rad (gate hits %s); the largest radius that '
                   'found none is %s. These are finite-start search results inside declared boxes, never a proof of '
                   'existence or non-existence.'
                   % (STARTS, smallest_hit, [p['n_hits_all_four_gate_items'] for p in ordered if p['hit']],
                      largest_miss) if smallest_hit is not None else
                   'no probe on this ladder hit the declared gate with %d starts, so no transition radius can be '
                   'reported; the search box would have to be widened or the start count increased' % STARTS))

    report = dict(status='PASS_RADIUS_LADDER_REPORTED', manual_research_not_product=True, standing=False, walking=False,
                  declared_protocol=dict(
                      geometry='act in [1e-4,1]; root z +-0.05 mm; 42 leg hinge joints within R of the DELIVERED pose A '
                               '(clipped to joint range minus 5e-3 rad)',
                      ladder=LADDER, starts_per_probe=STARTS,
                      seed='fixed inside run_mujoco_equilibrium_search.py (np.random.default_rng(20261015)), identical across probes so the start sets correspond',
                      criterion='max|qacc| <= 1e-3 mm/s^2 AND six tarsi in contact AND zero non-foot contacts AND every contact u_diamond <= 1',
                      resolution_rad=0.05,
                      wording_rule='finite-start search result inside a declared box; a miss is never a proof of non-existence'),
                  reference=dict(corrected_batch=str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'),
                                 previously_probed=['R = 0.20 rad (40 starts): no hit', 'R = 0.10 rad (240 starts): no hit',
                                                    'R = 0.05 rad (40 starts): no hit'],
                                 accepted_equilibrium_distance_from_pose_A_rad=0.42313245798143917),
                  probes=ordered, verdict=verdict,
                  inputs=dict(pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A), source_pose=str(SOURCE_POSE),
                              source_pose_sha256=sha(SOURCE_POSE), canonical_xml=str(CANONICAL_XML),
                              canonical_xml_sha256=sha(CANONICAL_XML), search_script=str(SEARCH),
                              search_script_sha256=sha(SEARCH), runner=str(Path(__file__)),
                              runner_sha256=sha(Path(__file__))),
                  not_established='absolute forces/units, biological validity, any behaviour',
                  note='manual research implementation, not a product')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    xs = [p['radius_rad'] for p in ordered]
    ys = [max(p['best_max_abs_qacc_mm_s2'], 1e-14) if p['best_max_abs_qacc_mm_s2'] is not None else 1e-14 for p in ordered]
    ax.semilogy(xs, ys, 'o-', color='#1f77b4', label='best max|qacc| over %d starts' % STARTS)
    for x, y, p in zip(xs, ys, ordered):
        ax.annotate('%d hits' % p.get('n_hits_all_four_gate_items', 0), (x, y), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=9,
                    color=('#2ca02c' if p['hit'] else '#d62728'))
    ax.axhline(GATE, color='black', ls=':', lw=1.6, label='declared gate 1e-3 mm/s^2')
    ax.axvline(0.20, color='gray', ls='--', lw=1.0, label='earlier probe R = 0.20 (no hit)')
    if smallest_hit is not None and largest_miss is not None:
        ax.axvspan(largest_miss, smallest_hit, color='orange', alpha=0.18,
                   label='transition interval [%.2f, %.2f] rad' % (largest_miss, smallest_hit))
    ax.set_xlabel('radius R around the delivered pose A  [rad]')
    ax.set_ylabel('best max|qacc|  [mm/s^2], log scale')
    ax.set_title('declared radius ladder: the smallest R around pose A where a declared-cone static equilibrium was\n'
                 'FOUND by %d identical starts  |  hit = qacc + six tarsi + no non-foot contact + inside the diamond\n'
                 'finite-start search result, never a proof; static equilibrium != standing; manual research, not a product'
                 % STARTS, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out / 'equilibrium-radius-ladder.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# S2 声明半径阶梯：交付姿态 A 周围多小的半径内还能"找到"声明锥静态平衡（手动研究，不是成品）',
        '',
        '## 声明协议（跑之前写死）',
        '- 几何：`act ∈ [1e-4,1]`；root z ±0.05 mm；42 个腿关节在**交付姿态 A** 的 R 以内（并对关节范围减 5e-3 rad 取交集）。',
        '- 阶梯：%s rad；每档 **%d 个随机起点**（生成器种子固定在同一实现里，各档起点集合相互对应）。' % (LADDER, STARTS),
        '- 判定"命中"：`max|qacc| ≤ 1e-3 mm/s²` **且** 六足跗节接触 **且** 零足外接触 **且** 所有接触 `u_diamond ≤ 1`。',
        '',
        '## 结果',
        *['- R = %.2f rad：最好 `max|qacc| = %s mm/s²`，四项门限命中 %s 个' %
          (p['radius_rad'], ('%.6g' % p['best_max_abs_qacc_mm_s2']) if p['best_max_abs_qacc_mm_s2'] is not None else 'None',
           p.get('n_hits_all_four_gate_items', 0)) for p in ordered],
        '- ⇒ %s' % verdict['statement'],
        '',
        '## 必须披露',
        '- 这是**有限起点、声明盒内的搜索结果**；某档"没找到"**不是**"该半径内不存在平衡"的证明。',
        '- 静态平衡只是**静态必要条件**：不等于稳定性、不等于站立、不等于"姿态 A 是平衡"。',
        '- 静态平衡**不等于**可动态维持（同一状态在真 100 帧档案下步 2739 = 68.475 ms 首次失去足接触）。',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；**手动研究实施，不是成品**。',
        '',
        '## 文件',
        '- `report.json`（含每档起点数、命中数、逐平衡明细与该档运行目录的 sha256）',
        '- `equilibrium-radius-ladder.png`（对数坐标 + 门限线）',
        '- `source_snapshot.zip`、`delivery-hashes.json`'])
    (out / 'README.zh-CN.md').write_text(readme)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (Path(__file__), SEARCH):
            z.write(p, Path(p).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))

    print(report['status'])
    print('out:', out)
    print('verdict:', json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
