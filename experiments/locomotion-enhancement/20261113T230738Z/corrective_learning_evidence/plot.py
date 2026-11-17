"""Only JSON + streamed file hashes until explicit parent plotting release."""
from pathlib import Path
import argparse, ast, builtins, dis, hashlib, json, math, os, resource, signal, sys, time, types

HERE = Path(__file__).resolve().parent
OUTPUTS = ('TRAINING_FIT_COMPARISON.png', 'PLOT_VERIFY.json', 'REPORT.zh-CN.md')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fresh(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class Ledger:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = str(Path(path).resolve())
        if path in self.hashes:
            require(expected is None or self.hashes[path] == expected, 'Conflicting source hash ' + path)
            return
        digest = sha(path)
        require(expected is None or digest == expected, 'Source hash mismatch ' + path)
        self.hashes[path] = digest

    def json(self, path, expected=None):
        self.bind(path, expected)
        return read(path)

    def after(self):
        result = {path: sha(path) for path in self.hashes}
        require(result == self.hashes, 'Inputs changed during evidence generation')
        return result


def proof_pass(proof):
    return proof.get('integrity_all_pass') is True and bool(proof.get('checks')) and all(v is True for v in proof['checks'].values())


def qualify(calibration, ledger, require_parent=True):
    root = Path(calibration)
    jobs = ledger.json(root / 'JOBS.json')
    require(len(jobs) == 3 and all(row.get('exit_code') == 0 and row.get('status') == 'COMPLETED' for row in jobs), 'All3 calibration stages must be complete: ' + str(root))
    stage = ledger.json(root / 'STAGE_VERIFICATION.json')
    require(stage['all_stages_exit0'] and stage['train_updates'] == 600, 'Final600 stage qualification')
    evaluation = ledger.json(root / 'EVALUATION.json', stage['evaluation_sha256'])
    audit = ledger.json(root / 'EVALUATION_VERIFY.json', stage.get('independent_audit_sha256'))
    require(proof_pass(audit) and audit['evaluation_sha256'] == ledger.hashes[str((root / 'EVALUATION.json').resolve())], 'Independent exact evaluation audit')
    train = ledger.json(root / 'TRAIN/RESULT.json')
    require(train['updates'] == 600 and evaluation['step'] == 600, 'Actual final600 only')
    pt = root / 'TRAIN/checkpoint_000600.pt'
    require(str(pt) == evaluation['checkpoint'] == train['checkpoint'] == stage['checkpoint'], 'Exact final checkpoint path')
    digest = evaluation['checkpoint_sha256']
    require(digest == audit['checkpoint_sha256'] == stage['checkpoint_sha256'] == train['sha256'], 'Checkpoint qualifications agree')
    ledger.bind(pt, digest)  # Byte hash only, never deserialize a model.
    for step in (0, 300, 600):
        saved = ledger.json(root / 'TRAIN' / f'VERIFY_{step:06d}.json')
        require(proof_pass(saved) and saved['step'] == step, 'Fresh saved checkpoint proof')
        ledger.bind(saved['checkpoint'], saved['checkpoint_sha256'])
        dev = root / 'TRAIN' / f'DEV_VERIFY_{step:06d}.json'
        if dev.exists():
            require(proof_pass(ledger.json(dev)), 'Fresh optional DEV audit')
    parent_path = root / 'PARENT_CALIBRATION_REVIEW.json'
    if require_parent or parent_path.exists():
        parent = ledger.json(parent_path)
        require(parent.get('updates', parent.get('train_updates')) == 600 and parent.get('sha256', parent.get('checkpoint_sha256')) == digest, 'Parent actual calibration review binding')
    ledger.bind(evaluation['arrays']['path'], evaluation['arrays']['sha256'])
    for path, expected in evaluation['source_hashes'].items():
        ledger.bind(path, expected)
    binding = ledger.json(root / 'INPUT_BINDINGS.json')
    rows = {row['id']: row for row in evaluation['rows']}
    require(len(rows) == len(evaluation['rows']) == len(binding['entries']), 'Unique actual fit sequence rows')
    require(set(rows) == {entry['id'] for entry in binding['entries']}, 'Actual source IDs agree')
    for row in rows.values():
        require(math.isfinite(row['action_RMSE']) and row['action_RMSE'] >= 0 and abs(row['action_RMSE']**2-row['action_MSE']) < 1e-12, 'JSON metric finite and MSE/RMSE agree')
    return dict(root=str(root), checkpoint_sha256=digest, rows=rows, entries=binding['entries'],
                evaluation_sha256=audit['evaluation_sha256'])


def body_review(path, digest, ledger):
    path = Path(path)
    review = ledger.json(path)
    require(review['checkpoint_sha256'] == digest, 'Body review checkpoint matches candidate')
    jobs = ledger.json(path.parent / 'JOBS.json')
    require(len(jobs) == 9 and all(row.get('exit_code') == 0 for row in jobs), 'All9 body stages completed')
    output = []
    for row in review['rows']:
        name = row.get('condition', row.get('case'))
        folder = path.parent / 'trials' / name
        result = ledger.json(folder / 'RESULT.json', row['result_sha256'])
        audit = ledger.json(folder / 'VERIFY.json')
        require(proof_pass(audit) and audit['result_sha256'] == row['result_sha256'], 'Fresh actual body result proof')
        require(result['candidate_metrics'] == row['candidate_metrics'], 'Parent body values match actual result')
        # Frozen actual source checkpoint is required, without importing its loader.
        require(result['checkpoint_sha256'] == digest, 'Actual body used same final checkpoint')
        metrics = result['candidate_metrics']
        output.append(dict(id=name, task_pass=metrics['task_pass_candidate'],
                           duration_s=metrics['duration_s'], yaw_deg=metrics['yaw_change_deg'],
                           max_COM_drift_BL=metrics['max_COM_xy_drift_bodylength'],
                           failed_gates=[key for key, value in metrics['gates'].items() if not value],
                           result_sha256=row['result_sha256'], review=str(path)))
    require({x['id'] for x in output} == {'walk', 'stop', 'left_pivot', 'right_pivot'}, 'Actual four body tasks')
    return output


def shared_sources(candidates):
    common = [entry['id'] for entry in candidates[0]['entries']]
    require(len(common) == 8 and set(common) == set(candidates[1]['rows']), 'Original eight common training IDs')
    keys = ('path', 'start', 'warm_start', 'length', 'teacher', 'result_sha256', 'audit_sha256')
    first = {entry['id']: entry for entry in candidates[0]['entries']}
    for candidate in candidates[1:]:
        entries = {entry['id']: entry for entry in candidate['entries']}
        for name in common:
            require(all(entries[name][key] == first[name][key] for key in keys), 'Same actual training source/target suffix ' + name)
    extras = [entry for entry in candidates[2]['entries'] if entry['id'] not in common]
    require(len(extras) == 4 and all(entry.get('collection') is True for entry in extras), 'Four extra actual collection sources')
    require(sum(entry['teacher'] is True for entry in extras) == 3, 'Three assisted teacher sources and one old control')
    for entry in candidates[2]['entries']:
        row = candidates[2]['rows'][entry['id']]
        require(row['valid_rows'] == entry['expected_valid_count'] and row['total_history_rows'] == 1920, 'Exact actual valid mask count')
        require(abs(row['action_RMSE']-row['masked_action_RMSE']) < 1e-12 and abs(row['action_MSE']-row['masked_action_MSE']) < 1e-12, 'Third metric alias is masked fit')
        if entry['id'] in common:
            require(row['valid_rows'] == 1920 and not entry.get('collection'), 'Common8 full valid comparable fit')
    return common, extras


def globals_used(code):
    result = {x.argval for x in dis.get_instructions(code) if x.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            result.update(globals_used(constant))
    return result


def codecheck():
    ledger = Ledger()
    prereg = ledger.json(HERE / 'PREREG.json')
    ledger.bind(HERE / 'plot.py')
    tree = ast.parse((HERE / 'plot.py').read_text())
    compile(tree, str(HERE / 'plot.py'), 'exec')
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        require(checks[name], 'CODECHECK ' + name)
    check('no_top_level_numeric_plot_import', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name.split('.')[0] in ('numpy', 'matplotlib', 'torch', 'mujoco') for a in n.names) for n in tree.body))
    records = {}
    functions = [(name, obj) for name, obj in list(globals().items()) if isinstance(obj, types.FunctionType)]
    functions += [('Ledger.'+name, obj) for name, obj in vars(Ledger).items() if isinstance(obj, types.FunctionType)]
    for name, obj in functions:
        used = globals_used(obj.__code__)
        missing = sorted(n for n in used if n not in obj.__globals__ and not hasattr(builtins, n))
        check(name + '_complete_nested_globals', not missing)
        records[name] = dict(global_names=sorted(used), unresolved=missing)
    old = [qualify(row['calibration'], ledger) for row in prereg['candidates'][:2]]
    for spec, candidate in zip(prereg['candidates'][:2], old):
        actual = body_review(spec['body_review'], candidate['checkpoint_sha256'], ledger)
        check(spec['label'] + '_actual4bodyFAIL', all(row['task_pass'] is False for row in actual))
    newroot = Path(prereg['candidates'][2]['calibration'])
    binding = ledger.json(newroot / 'INPUT_BINDINGS.json')
    for name in ('evaluate.py', 'audit_eval.py', 'learning.py', 'data.py', 'run.py', 'sources.py', 'PREREG.json'):
        ledger.bind(newroot / name)
    common_ids = [entry['id'] for entry in old[0]['entries']]
    extras = [entry for entry in binding['entries'] if entry['id'] not in common_ids]
    check('actual_third12_binding_extra4', len(binding['entries']) == 12 and len(extras) == 4 and all(e['collection'] for e in extras))
    first_entries = {entry['id']: entry for entry in old[0]['entries']}
    new_entries = {entry['id']: entry for entry in binding['entries']}
    source_keys = ('path', 'start', 'warm_start', 'length', 'teacher', 'result_sha256', 'audit_sha256')
    check('third_common8_exact_actual_source_suffix', all(all(first_entries[name][key] == new_entries[name][key] for key in source_keys) and new_entries[name]['expected_valid_count'] == 1920 for name in common_ids))
    check('never_freeze_live_third_progress', str(newroot/'JOBS.json') not in ledger.hashes)
    check('no_plot_outputs_yet', not any((HERE / n).exists() for n in OUTPUTS))
    check('no_numeric_or_plot_module_imports', not any(n in sys.modules for n in ('numpy', 'matplotlib', 'torch', 'mujoco', 'physics')))
    ledger.after()
    fresh(HERE / 'CODECHECK.json', dict(all_passed=True, checks=checks, source_hashes=ledger.hashes,
          recursive_global_analysis=records, actual_shared_ids=common_ids,
          actual_extra_source_metadata=[{k: e[k] for k in ('id', 'teacher', 'expected_valid_count', 'source_kind')} for e in extras],
          third_final600_qualification='DEFERRED_UNTIL_PARENT_EXECUTION',
          array_payload_reads=0, numeric_imports=0, plot_calls=0, NN_forward_calls=0, physics_steps=0))
    print(json.dumps(dict(status='READY_ONLYCODE', checks=len(checks), sources=len(ledger.hashes), unresolved=[], plot=0, NN=0, physics=0)))


def draw(candidates, common, extras, bodies, specs, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    cjk = next((font.name for font in font_manager.fontManager.ttflist
                if any(s in font.name.lower() for s in ('noto sans cjk', 'wenquanyi', 'source han sans'))), None)
    plt.rcParams.update({'font.family': cjk or 'DejaVu Sans', 'font.size': 11, 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(18, 13), facecolor='white')
    grid = fig.add_gridspec(3, 1, height_ratios=(3.1, 2.1, 3.2), hspace=.68)
    fig.suptitle('TRAINING FIT ONLY — PURE BODY ACCEPTANCE SEPARATE', fontsize=21, fontweight='bold', y=.985)
    fig.text(.5, .95, '仅比较训练拟合；纯身体验收单列，不代表已会转向' if cjk else 'Same saved training sources; lower fit error does not establish closed-loop body success.', ha='center', fontsize=14)
    ax = fig.add_subplot(grid[0])
    colors = ('#4978ae', '#df913b', '#529c82')
    width = .24
    for index, (candidate, spec) in enumerate(zip(candidates, specs)):
        xx = [i+(index-1)*width for i in range(len(common))]
        yy = [candidate['rows'][name]['action_RMSE'] for name in common]
        bars = ax.bar(xx, yy, width, color=colors[index], label=spec['label']+' / final600')
        for bar, value in zip(bars, yy):
            ax.annotate(f'{value:.4f}', (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, rotation=90)
    ax.set_xticks(range(len(common)), common, rotation=12)
    ax.set_ylabel('Action RMSE (42 outputs)')
    ax.set_title('A. Shared eight TRAINING sequences / full 1920 labels each — lower is closer to saved targets', loc='left', fontweight='bold')
    ax.legend(loc='upper right'); ax.margins(y=.35); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax2 = fig.add_subplot(grid[1])
    labels = []; values = []; extra_colors = []
    for entry in extras:
        row = candidates[2]['rows'][entry['id']]
        labels.append(entry['id']+'\n'+str(row['valid_rows'])+'/1920 qualified labels\n'+('ASSISTED teacher target' if entry['teacher'] else 'DIRECT old-control target'))
        values.append(row['masked_action_RMSE']); extra_colors.append('#a66b93' if entry['teacher'] else '#888888')
    bars = ax2.bar(range(len(extras)), values, color=extra_colors, width=.55)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x()+bar.get_width()/2, value, f'{value:.6f}', ha='center', va='bottom')
    ax2.set_xticks(range(len(extras)), labels, fontsize=10)
    ax2.set_ylabel('Masked action RMSE'); ax2.margins(y=.35); ax2.grid(axis='y', alpha=.2); ax2.set_axisbelow(True)
    ax2.set_title('B. C3 extra collection TRAINING fit only — excluded from panel A; no generalization score', loc='left', fontweight='bold')
    ax3 = fig.add_subplot(grid[2]); ax3.axis('off')
    ax3.set_title('C. Separate actual PURE BODY acceptance — archived original gates retained', loc='left', fontweight='bold', pad=15)
    table_rows = []
    for spec, rows in zip(specs, bodies):
        if rows is None:
            table_rows.append([spec['label'], '4 tasks', 'PENDING', '—', '—', '—', 'No pure-body review supplied; training fit is not acceptance'])
            continue
        for row in rows:
            failed = row['failed_gates']
            shown = ', '.join(failed[:3]) or 'All preregistered gates pass'
            if len(failed) > 3:
                shown += f'; +{len(failed)-3} more (full list in JSON)'
            table_rows.append([spec['label'], row['id'], 'PASS' if row['task_pass'] else 'FAIL', f'{row["duration_s"]:.3f}', f'{row["yaw_deg"]:+.2f}', f'{row["max_COM_drift_BL"]:.4f}', shown])
    table = ax3.table(cellText=table_rows, colLabels=['Candidate', 'Body task', 'Verdict', 'Time s', 'Yaw deg', 'COM drift BL', 'Actual failed gates / pending scope'], cellLoc='left', loc='upper center', colWidths=[.12,.12,.07,.065,.075,.10,.45])
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.72)
    for (r, c), cell in table.get_celld().items():
        if r == 0: cell.set_facecolor('#e5ebf0'); cell.set_text_props(weight='bold')
        elif c == 2: cell.set_text_props(color='#25714a' if cell.get_text().get_text() == 'PASS' else '#ad4936')
    fig.text(.04, .025, 'Fixed saved-input imitation fit is descriptive. Assisted labels are not pure ANN trials. No new model/NN/physics execution.\nActual source IDs are preserved; metric definitions and full SHA ledger are in PLOT_VERIFY.json. Parent visual review remains required.', fontsize=10)
    fig.subplots_adjust(left=.065, right=.975, top=.90, bottom=.07)
    with destination.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=150, facecolor='white')
    plt.close(fig)
    return dict(font=cjk or 'DejaVu Sans', CJK_available=bool(cjk), plot_title='TRAINING FIT ONLY — PURE BODY ACCEPTANCE SEPARATE')


def execute(third_body_review):
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError('Plot wall limit30s')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(30)
    require(not any((HERE / n).exists() for n in OUTPUTS), 'Exclusive evidence output paths required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', name+'=1 required')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    ledger = Ledger()
    proof = ledger.json(HERE / 'CODECHECK.json')
    require(proof['all_passed'] and all(proof['checks'].values()), 'ONLYCODE qualification')
    for path, expected in proof['source_hashes'].items():
        ledger.bind(path, expected)
    prereg = ledger.json(HERE / 'PREREG.json')
    specs = prereg['candidates']
    # Refuse missing third final qualification before importing matplotlib or creating outputs.
    candidates = [qualify(spec['calibration'], ledger, require_parent=i<2) for i, spec in enumerate(specs)]
    common, extras = shared_sources(candidates)
    bodies = [body_review(spec['body_review'], candidate['checkpoint_sha256'], ledger) for spec, candidate in zip(specs[:2], candidates[:2])]
    bodies.append(body_review(third_body_review, candidates[2]['checkpoint_sha256'], ledger) if third_body_review else None)
    style = draw(candidates, common, extras, bodies, specs, HERE / OUTPUTS[0])
    after = ledger.after()
    wall = time.monotonic()-started; peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    require(wall <= 30 and peak <= 512, 'Plot resource budget30s/512MiB')
    require(not any(k in sys.modules for k in ('torch','mujoco','physics','student_wave','flygym')), 'No neural/physical runtime import')
    lines = ['# 三候选训练拟合与独立身体验收', '', '该图只描述实际 final600 在保存训练输入上的动作拟合，不代表闭环转向、站立或行走验收。三个候选的共享八源使用相同训练目标；第三候选这些序列全部1920标签有效。', '']
    for name in common:
        values = [f'{spec["label"]}={candidate["rows"][name]["action_RMSE"]:.8f}' for spec,candidate in zip(specs,candidates)]
        lines.append('- '+name+'：'+ '；'.join(values))
    lines += ['', '额外四源单列，不混入共享八源或泛化成绩。辅助源使用未混合教师目标，但采集过程中有教师辅助，不能据此称纯ANN身体通过。', '']
    for entry in extras:
        row = candidates[2]['rows'][entry['id']]
        lines.append(f'- {entry["id"]}：合格标签 {row["valid_rows"]}/1920，masked RMSE={row["masked_action_RMSE"]:.8f}；'+('教师辅助目标' if entry['teacher'] else '旧控制直接目标'))
    lines += ['', '纯身体验收使用原始独立RESULT/VERIFY及父审核记录，图中逐任务保留FAIL/PASS与实际失败门。第三候选 '+('已按显式提供审核路径核验，详见JSON。' if bodies[2] is not None else 'PENDING：本次未提供纯身体审核，不从训练拟合推断通过。'), '', '所有来源SHA前后一致。未读取NPZ载荷、未加载模型、未执行NN forward/Core/物理；仅父执行Matplotlib绘图。', '', '图像：'+str(HERE/OUTPUTS[0]), '机器审核：'+str(HERE/OUTPUTS[1]), '父实际目视审核：PENDING。']
    with (HERE / OUTPUTS[2]).open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    fresh(HERE / OUTPUTS[1], dict(kind='ACTUAL_THREE_CANDIDATE_TRAINING_FIT_PLOT_V1', all_integrity_checks_passed=True,
          common_ids=common, common_metric_definition='Action RMSE on same1920 valid saved training labels /42 outputs; C3 masked equals fullvalid here',
          candidates=[dict(label=spec['label'], calibration=c['root'],checkpoint_sha256=c['checkpoint_sha256'],evaluation_sha256=c['evaluation_sha256'],actual_training_rows=c['rows']) for spec,c in zip(specs,candidates)],
          extra_source_metadata=extras, body_acceptance=bodies, third_body_status='QUALIFIED_REVIEW_SUPPLIED' if bodies[2] is not None else 'PENDING_NO_PURE_BODY_REVIEW_SUPPLIED',
          source_hashes_before=ledger.hashes, source_hashes_after=after, source_unchanged=True,
          outputs={str(HERE/n):sha(HERE/n) for n in (OUTPUTS[0],OUTPUTS[2])},
          style=style, wall_s=wall, peak_RSS_MiB=peak, parent_actual_review='PENDING',
          NPZ_payload_reads=0, NN_forward_calls=0, Core=0, physics_steps=0,
          scientific_scope='Training fit only, separate actual body verdicts, no generalization/neck/biological/NN success inference'))
    signal.alarm(0)
    print(str(HERE / OUTPUTS[1]))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-code-only', action='store_true')
    mode.add_argument('--execute-parent-after-calibration', action='store_true')
    parser.add_argument('--third-body-review', type=Path)
    args = parser.parse_args()
    if args.check_code_only:
        codecheck()
    else:
        execute(args.third_body_review)


if __name__ == '__main__':
    main()
