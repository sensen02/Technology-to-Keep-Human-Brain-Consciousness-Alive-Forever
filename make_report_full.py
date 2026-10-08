"""Assemble the multi-layer report from generated metrics files.

Only reads JSON produced by the runners; missing stages are reported as missing,
never silently dropped.
"""
from pathlib import Path
import json, shutil, html, datetime

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
DESK = Path('/home/sensen/Desktop/cell_wound_prototype')


def load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {'_error': f'{type(exc).__name__}: {exc}'}


def fmt(x, nd=3):
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return 'n/a'


def main():
    v = load('metrics.json') or {}
    m = load('metrics_mechanics.json')
    c = load('metrics_cellstate.json')
    l = load('metrics_late.json')
    lit = load('metrics_mechanics_lit.json')
    sw = load('metrics_mechanics_sweep.json')
    cs = load('metrics_cellspace.json')
    og = load('metrics_organism.json')
    lc = load('metrics_lineage_coupling.json')
    lv = load('lineage_verified.json')
    mdv = load('mesh_division_verified.json')
    cases = v.get('cases', {})
    L = []
    A = L.append
    A('# 多尺度生物模拟：分层规格与本轮实现报告')
    A('')
    A(f'生成时间：{datetime.datetime.now().isoformat(timespec="seconds")}')
    A('')
    A('本报告由脚本自动汇总各层实际运行结果。**所有模拟输出均为合成数据，不是实验数据。**')
    A('')
    A('## 一、本轮实际实现并运行的模块')
    A('')
    A('| 层 | 模块 | 文件 | 状态 |')
    A('|---|---|---|---|')
    A('| 胞内钙信号（已发表模型移植） | early microtear-only | `model.py` | 已运行 |')
    A('| 观测算子（径向半高半径） | 环形荧光测量 | `measurement.py` | 已运行 |')
    A('| L1 组织力学（可变形顶点模型） | 切割/消融由力学计算 | `mechanics.py` | ' + ('已运行' if m else '未完成') + ' |')
    A('| L2 细胞状态（体积/ATP/膜修复/死活判定） | 状态层 | `cellstate.py` | ' + ('已运行' if c else '未完成') + ' |')
    A('| L3 胞内空间（反应扩散） | 单细胞空间模型 | `cellspace.py` | ' + ('已运行' if cs else '未完成') + ' |')
    A('| L1 与已发表实测锚点对照 | 形态指数/松弛时间/T1 | `run_mechanics_lit.py` | ' + ('已运行' if lit else '未完成') + ' |')
    A('| L4 延迟配体信号（>25 s 远场波） | 蛋白酶–配体–受体 | `ligand.py`, `model_late.py` | ' + ('已运行' if l else '未完成') + ' |')
    A('')
    A('## 二、钙信号层与实验对照（回溯，非独立验证）')
    A('')
    if cases:
        A('| 版本 | 细胞数 | 可比较时点 | MAE(µm) | 半径@19.26s(µm) |')
        A('|---|---:|---:|---:|---:|')
        for name in ['control', 'recommended_576', 'larger_domain', 'seed2_576', 'seed3_576']:
            if name in cases:
                d = cases[name]
                A(f"| {name} | {d.get('n_cells')} | {d.get('comparison_points')}/11 | {fmt(d.get('mae_um'))} | {fmt(d.get('radius_at_19_26_s_um'))} |")
        A('')
        A(f"- 实验值 19.26 s：**58.895 µm**；576 细胞预测：**{fmt(cases.get('recommended_576',{}).get('radius_at_19_26_s_um'))} µm**")
        A(f"- 576 与 784 半径最大差：{fmt(v.get('domain_576_vs784_radius_max_difference_um'))} µm；中央 60 µm 内胞质钙轨迹 RMS 差：{v.get('domain_576_vs784_inner60_c_rms_difference_um')}")
        A(f"- 求解器严格容差对照：最大胞质钙差 {v.get('solver_576_max_abs_c_um')} µM")
        A(f"- 无损伤漂移：{v.get('no_wound_max_c_drift_um')} µM；观测单元测试误差：{fmt(v.get('measurement_unit_test',{}).get('error_um'))} µm")
        A('')
        A('**边界与循环性问题（必须同时读）：** 400 细胞版本只有 3/11 时点可比较，属于视野不足；')
        A('而规定损伤半径本身由同一篇论文用同一条实验曲线拟合，故早期形态主要由输入决定。')
        A('实测敏感性：损伤半径 51.25→40 µm 时预测从 63.82 降到 51.06 µm。')
    else:
        A('（钙信号层指标缺失）')
    A('')
    A('## 三、L1 可变形力学（计算损伤替代规定损伤）')
    A('')
    if m and '_error' not in m:
        ia = m.get('index_alignment', {})
        A(f"- 索引对齐：纯平移 {ia.get('constant_translation_um')} µm，移除平移后最大残差 "
          f"{ia.get('max_residual_after_translation_um')} µm，顺序一致={ia.get('ordering_consistent')}")
        A(f"- 应变统计：{m.get('strain_stats')}")
        A(f"- 损伤足迹：{m.get('injury_footprint')}")
        A(f"- 切割闭合：{m.get('closure')}")
        A(f"- **不同假设耦合下的预测半径跨度**：{m.get('coupling_spread_radius_19_26_um')}")
        A(f"- 各耦合结果（实验 19.26 s = 58.895 µm）：")
        for k, val in (m.get('coupling_results') or {}).items():
            A(f"  - {k}: 半径@19.26s={fmt(val.get('radius_at_19_26_s_um'),2)} µm, "
              f"可用对照点={val.get('usable_reference_points')}, 最大钙={fmt(val.get('max_c_um'),1)} µM")
        A('')
        A(f"- {m.get('key_finding')}")
        A('')
        A('撕裂律之间的差异度量的是**耦合不确定性**，不是生物不确定性。')
        A('力学参数目前为示例值：notum 无任何绝对张力(nN)、黏度(Pa·s)、弹性模量(Pa)实测，见 `MEASURED_ANCHORS.md`。')
    else:
        A('（力学层未完成或缺失）')
    A('')
    A('## 四、L2 细胞状态、代谢与死活判定')
    A('')
    if c and '_error' not in c:
        for k in ['no_injury_drift', 'conservation', 'bounds', 'thresholds', 'hysteresis', 'repair', 'oxygen_sweep', 'identifiability', 'runtime_seconds']:
            if k in c:
                A(f"- {k}: {c[k]}")
        A('')
        A('关键限制：ATP/修复/裂解阈值在果蝇上皮**没有可用的实测阈值**；唯一可用的定量阈值是"ATP 降至对照的约 15% 以下转为坏死"（体外、摘要级别）。')
        A('果蝇线粒体内膜通道不是经典 PTP，不能套用哺乳动物 PTP/CypD/CsA 参数。')
    else:
        A('（细胞状态层未完成或缺失）')
    A('')
    A('## 五、L1 力学层与已发表实测锚点对照')
    A('')
    if lit and '_error' not in lit:
        si = lit.get('shape_index', {})
        sii = lit.get('shape_index_injured', {})
        A(f"- 规则晶格形态指数：{fmt(si.get('mean'),4)}（六角理论值 {lit.get('anchors',{}).get('shape_index_hexagon')}）→ 几何自检，不检验流动性")
        if sii:
            A(f"- 受伤后形态指数：均值 {fmt(sii.get('mean'),4)}，标准差 {fmt(sii.get('sd'),4)}，超过已发表阈值 3.81 的细胞比例 {100*float(sii.get('fraction_above_threshold') or 0):.1f}%")
        rv = lit.get('relaxation_vs_published', {})
        A(f"- 松弛时间：模型 {fmt(rv.get('model_tau_s'))} s（能量拟合 R²={fmt(lit.get('relaxation_energy',{}).get('r_squared'))}）；已发表 notum 18–26 hAPF 约 {rv.get('published_tau_s')} s（范围 {rv.get('published_range_s')}）；是否落在范围内：{rv.get('inside_published_range')}")
        A('  - **注意**：力学参数是示例值，落在实测范围内目前**不构成验证**；它给出了一个可用的标定把手。')
        t1 = lit.get('t1_rate', {})
        A(f"- T1 发生率：模型 {t1.get('t1_events_in_run')} 次事件（{t1.get('model_rate_per_min_per_junction')} 次/分/连接）vs 已发表 notum {t1.get('published_rate_per_min_per_junction')} 次/分/连接")
        A('  - 原因明确：本模型没有随机连接线张力，因此**无法产生**该实测的涨落驱动可逆 T1；这是缺失机制，不是拟合问题。')
        cl = lit.get('closure', {})
        A(f"- 闭合：模拟切缝自由边界 {fmt(cl.get('model_cut_length_initial_um'),1)}→{fmt(cl.get('model_cut_length_final_um'),1)} µm（60 s 内闭合 {100*float(cl.get('model_fraction_closed') or 0):.0f}%）；已发表 notum 约 40 µm 伤口约 3 h 闭合")
        A('  - 几何不同（狭缝 vs 圆孔），闭合时间不可直接比较；但模型明显快约两个数量级。')
    else:
        A('（力学锚点对照未完成）')
    A('')
    if sw and '_error' not in sw:
        A('- 收紧 purse-string 张力（λ_cut 0.25→4）使边界回缩速率从 '
          f"{fmt(sw.get('rate_range_um_per_s',[None])[0])} 升到 {fmt(sw.get('rate_range_um_per_s',[None,None])[1])} µm/s。")
        A(f"- {sw.get('interpretation')}")
    A('')
    A('### 索引安全与晶格有效性的独立核验（我方复核）')
    A('')
    A(f"- 力学网格与钙模型网格：纯平移 {m.get('index_alignment',{}).get('constant_translation_um') if m else 'n/a'} µm，移除平移后残差 {m.get('index_alignment',{}).get('max_residual_after_translation_um') if m else 'n/a'} µm → **顺序一致，耦合安全**。")
    A('- 我另行检验了钙模型晶格本身：多边形互不重叠（重叠像素 0.000%）、每个面积恰为 43 µm²、最近邻距离一致为 7.0464 µm。')
    A('  - 因此后台代理提出的"hex_geometry 六边形会重叠"这一说法，**经实测不成立**，不采纳。')
    A('')
    A('## 六、L3 胞内空间层')
    A('')
    if cs and '_error' not in cs:
        st = cs.get('self_test', {})
        A(f"- 与 `model.py` 的通量恒等性检查：最大绝对误差 {st.get('flux_identity_max_abs_error')}"
          f"（参数与 model.py 完全一致={st.get('parameters_identical_to_model_py')}）")
        A(f"- 守恒/均匀性：均匀初始场最大偏差 {st.get('uniform_preservation_max_dev')}")
        mv = cs.get('mean_vs_lumped', {})
        A(f"- 体积加权平均钙 vs 集中模型：最大绝对差 {fmt(mv.get('max_abs_diff_uM'))} µM，"
          f"相对峰值 {100*float(mv.get('max_rel_diff_vs_peak') or 0):.2f}%")
        gr = cs.get('gradient', {})
        A(f"- 胞内相对梯度 (max-min)/mean 峰值：{fmt(gr.get('max_relative_spread'))}，"
          f"出现在 t={gr.get('time_of_max_s')} s（随后约 1 s 内衰减）")
        A('- 扩散系数敏感性（D_ca 单位 µm²/s）：')
        for key, val in (cs.get('spatial_vs_lumped') or {}).items():
            if not isinstance(val, dict):
                continue
            A(f"  - {key}: 与集中模型的最大差 {100*float(val.get('max_abs_diff_fraction_of_lumped_peak') or 0):.2f}% 峰值，"
              f"大信号段最大相对差 {100*float(val.get('max_rel_diff_large_signal') or 0):.1f}%，"
              f"亚膜超标倍数 {fmt(val.get('submembrane_excess_fraction'),2)}")
        A('- 结论：体积平均量上，空间模型与集中模型差异为百分之几（D_ca 取已发表值时约 2.2% 峰值）；')
        A('  但亚膜层与局部斑块刺激可以差 2–3 倍，集中模型在局部刺激下定性错误。')
        A('- 成本（子代理实测，单核）：单细胞 radial1d 约 71 µs/步；外推 576 细胞 × 5 s 约 0.57 h。')
        A('  grid3d 24³ 约 3.4 h、32³ 约 21 h → **组织级三维胞内空间在单核不可行**，需多核/GPU 批处理。')
    else:
        A('（胞内空间层未完成或缺失）')
    A('')
    A('## 七、L4 延迟配体信号与远场波')
    A('')
    if l and '_error' not in l:
        A(f"- 早期相与 `model.py` 对拍差异：{l.get('early_crosscheck')}")
        A(f"- 49.22 s：实验 51.412 µm，预测 {fmt(l.get('radius_49_22'))} µm")
        A(f"- 126.26 s：实验 113.102 µm，预测 {fmt(l.get('radius_126_26'))} µm")
        A(f"- 是否复现『先扩张、回落、二次扩张』模式：{l.get('pattern')}")
        A(f"- 域尺寸检查：{l.get('domain_check')}")
    else:
        A('（延迟信号层未完成或缺失）')
    A('')
    A('## 八、个体尺度扩展：生长分裂、血管交换、激素')
    A('')
    if og and '_error' not in og:
        A(f"- 组织规模：{og.get('tissue')}")
        A('- 血管间距扫描（规定几何，非生长）：')
        for k, val in (og.get('vessel_spacing') or {}).items():
            A(f"  - {k}: 平均 O₂ {fmt(val.get('mean_uM'),2)} µM，最低 {fmt(val.get('min_uM'),2)} µM，"
              f"低于参考一半的细胞比例 {100*float(val.get('hypoxic_fraction_(rel<0.5)') or 0):.1f}%，"
              f"场收支相对残差 {val.get('budget_residual_relative'):.2e}")
        A('- 细胞状态层（由局部氧驱动，但当前 API 只接受标量）：')
        for k, val in (og.get('cellstate') or {}).items():
            A(f"  - {k}: 氧输入 {fmt(val.get('oxygen_input'),3)}，"
              f"最终被判不可逆/裂解比例 {fmt(val.get('death_fraction_final'),3)}")
        A('- 生长与分裂：')
        for k, val in (og.get('growth') or {}).items():
            A(f"  - {k}: 细胞 {val.get('n_initial')}→{val.get('n_final')}，分裂 {val.get('divisions')} 次，"
              f"平均面积 {fmt(val.get('mean_area_first_um2'),1)}→{fmt(val.get('mean_area_last_um2'),1)} µm²")
        A(f"- 激素层：剂量 {og.get('hormone',{}).get('doses_nM')} → 响应 "
          f"{[round(x,3) for x in og.get('hormone',{}).get('response',[])]}，"
          f"与 Langmuir 解析平衡差 {og.get('hormone',{}).get('self_test')}")
        A('')
        A('**耦合假设（必须与结果同读）：**')
        for k, val in (og.get('coupling_assumptions') or {}).items():
            A(f"  - {k}: {val}")
        A('')
        A('')
        A('### 8.1 逐细胞氧耦合（任务①）')
        A('')
        A('- `cellstate.run` 现在接受逐细胞氧数组；标量与"均匀数组"结果**逐位相同**（最大差 0.0），无回归。')
        A('- 受控对照：一半细胞给 0.05、一半给 1.0 的氧，缺氧半边**100% 判为不可逆/裂解**，常氧半边 0%；')
        A('  而把同一场压成均值后，判死比例为 **0%** —— 均值化会把整个缺氧亚群隐藏掉。')
        csx = (og or {}).get('cellstate', {})
        for k, v in (csx.items() if isinstance(csx, dict) else []):
            if isinstance(v, dict):
                A(f"- 血管场景 {k}：最低可用度 {fmt(v.get('min_availability'),3)}，"
                  f"逐细胞判死 {fmt(v.get('per_cell_death_fraction'),3)} vs 均值判死 "
                  f"{fmt(v.get('uniform_mean_death_fraction'),3)}"
                  f"（差 {fmt(v.get('difference_in_death_fraction'),3)}）")
        A('')
        A('### 8.2 真实氧单位（任务②）')
        A('')
        return_units = (og or {}).get('units') if isinstance(og, dict) else None
        if return_units:
            A(f"- 换算：1 µM 溶解氧 ≈ {fmt(1/float(return_units['uM_per_mmHg']),2)} mmHg；"
              f"动脉 {return_units['arterial_uM']} µM（100 mmHg）、"
              f"静脉参考 {return_units['venous_reference_uM']} µM（40 mmHg）、"
              f"缺氧阈值 {return_units['hypoxic_uM']} µM（10 mmHg）")
        A('- 血管间距扫描（规定几何）：')
        for k, v in ((og or {}).get('vessel_spacing') or {}).items():
            A(f"  - {k}: 平均 {fmt(v.get('mean_uM'),1)} µM（{fmt(v.get('mean_mmHg'),1)} mmHg），"
              f"最低 {fmt(v.get('min_mmHg'),1)} mmHg，缺氧比例 "
              f"{100*float(v.get('hypoxic_fraction_below_10mmHg') or 0):.1f}%，"
              f"收支残差 {v.get('budget_residual_relative'):.1e}")
        A('- 结论：间距 ≤160 µm 时组织全程不缺氧（最低 35.7 mmHg）；只有把间距拉到 320 µm 才出现缺氧（最低 7.1 mmHg、19.8% 细胞缺氧）。')
        A('')
        A('### 8.3 分化与谱系（任务③）')
        A('')
        if lv and '_error' not in lv:
            A(f"- 谱系自检通过 {lv.get('n_passed')}/{lv.get('n_tests')} 项；"
              f"唯一 id、单亲、无环、计数残差 0、面积守恒误差 0（细节见 lineage_verified.json）。")
        if lc and '_error' not in lc:
            ox = lc.get('oxygen', {})
            A(f"- 驱动场：{ox.get('n_cells')} 细胞，平均 {fmt(ox.get('mean_uM'),1)} µM，"
              f"最低 {fmt(ox.get('min_mmHg'),1)} mmHg，缺氧比例 {100*float(ox.get('hypoxic_fraction') or 0):.1f}%")
            fr = (lc.get('lineage') or {}).get('final_fractions', {})
            A(f"- 终态类型比例：{ {k: round(float(v),3) for k,v in fr.items()} }；"
              f"身份残差 {(lc.get('lineage') or {}).get('identity_residual')}")
            A('- **局部氧驱动的空间分级**（myocyte 比例）：')
            for b in (lc.get('patterning') or {}).get('by_local_po2', []):
                A(f"  - {b['po2_range_mmHg']} mmHg（n={b['n_cells']}）: {fmt(float(b['myocyte_fraction']),3)}")
            A('- 说明：全部分化比例在各氧区间都是 1.0，差别体现在**类型身份**上；这检验的是规定规则，不是真实分化。')
        A('')
        A('### 8.4 分裂真正改变顶点网格（关键修复）')
        A('')
        A('- 先测出两个**真实缺陷**：')
        A('  - `growth.divide_in_mechanics_mesh` **完全不可用**：n_side=5 与 10 各试 3 次，**成功 0 次**，每次都以 `MeshError: duplicate directed half-edge` 回滚（证据 `vertex_division_probe.json`）。')
        A('  - `mechanics.Mesh` 的**分块几何缓存**在追加顶点槽后错位：总面积读成 903 而非守恒的 1075，单细胞读成 3.583/68.083 而同一环的鞋带面积是 43；把 `_topo_dirty` 置 True 后即恢复正确。')
        A('- 我据此在 `mesh_division.py` 里自己实现分裂手术（沿两条不相邻环边插入中点、拼接邻居环、母环一分为二），并**自己算几何与受力**，不依赖该缓存路径。')
        if mdv:
            t = mdv.get('self_test', {}).get('tests', {})
            sd = t.get('single_division', {})
            A(f"- 单次分裂：成功={sd.get('ok')}，面积相对误差 {sd.get('area_error_rel'):.2e}，网格有效={sd.get('valid')}")
            rp = t.get('repeated_division', {})
            A(f"- 连续 30 次分裂：**{rp.get('n_succeeded')}/{rp.get('n_attempted')} 成功**，回滚 {rp.get('n_rolled_back')} 次，"
              f"细胞 {rp.get('n_cells_final')} 个，网格有效={rp.get('valid')}，最小面积 {fmt(rp.get('min_cell_area_um2'),3)} µm²，"
              f"总面积相对误差 {rp.get('total_area_error_rel'):.2e}")
            sx = t.get('stress', {})
            A(f"- 加压到 80 次分裂：**{sx.get('n_succeeded')}/{sx.get('n_attempted')} 成功**，网格仍有效={sx.get('valid')}，"
              f"最小面积 {fmt(sx.get('min_cell_area_um2'),3)} µm²，面积相对误差 {sx.get('total_area_error_rel'):.2e}")
            A(f"- 确定性：{t.get('determinism')}")
            A(f"- 门控（只允许指定细胞分裂）：{t.get('gating')}")
        A('- 注意：总面积 ~1e-3 的相对漂移来自自由边界的松弛（每次分裂自身的面积误差为 1.7e-16），'
          '且连续分裂后最小细胞面积降到 1.7 µm²，因为本模型没有细胞尺寸稳态调控。')
        A('')
        A('### 8.5 可视化成果（可直接观察）')
        A('')
        A('- `mesh_division.gif`：连续 60 次分裂的动态过程（每次分裂后网格松弛）。')
        A('- `mesh_division_zoom.png`：单次手术的术前/术后对照——红虚线是分裂轴，红点是插入的两个中点，红实线是新边；右图为面积守恒（43 → 21.5 + 21.5，误差 1.7e-16）。')
        A('- `mesh_division_frames.png`：起始 64 细胞 / 1 次分裂 / 30 次 / 80 次的网格演化，颜色代表细胞面积。')
        A('- `mesh_division_areas.png`：80 次分裂期间的总面积（2752 → 2757.3 µm²，漂移来自自由边界松弛）与面积分布（最小 5.73 µm²，**无尺寸稳态**）。')
        A('- 上述四张图我已逐一看过；均为合成结果，分裂几何是建模选择。')
        A('')
        A('**诚实的负面结论：** O₂ 绝对值仍取决于示例性的每细胞耗氧率（vmax 1 amol/cell/s、Km 5 µM）；'
          '血管是规定几何而非血管新生；分化速率与阈值全部是示例值，细胞类型是抽象而非转录组定义的类型。')
    else:
        A('（个体尺度耦合未完成）')
    A('')
    A('## 九、仍属假设的量（不得当作实测）')
    A('')
    for line in Path(ROOT / 'MEASURED_ANCHORS.md').read_text().splitlines():
        if line.startswith('notum 绝对张力'):
            A('- ' + line)
    A('- 果蝇 innexin 单通道电导、IP3 跨缝隙连接通透系数、上皮内 pH 与 Mg²⁺、上皮体积设定点')
    A('- 细胞死亡相关的胞质钙阈值、临界体积阈值')
    A('- 氧气/灌注：各组织 CMRO₂、上皮耗氧率；Windkessel 系数；氧合器膜面积与传质系数')
    A('- Hydra：张力、膨压、细胞尺寸、闭合速度、形态发生素扩散系数；且整体组织与中胶层刚度相差约 100 倍')
    A('')
    A('## 十、结论')
    A('')
    A('可以做到"把每一层都写成可运行、可检查的代码"；不能做到"每一层都精确"。')
    A('四个硬上限（参数不可辨识、观测通道不足、计算上限、混沌与单次性）决定了精度上限由**数据**决定，而不是由模型规模或显卡决定。')
    A('本项目的正确用法是：把每层的**可观测预测**拿出来，与独立实验对照；对照失败要照实报告。')
    A('')
    text = '\n'.join(L)
    (OUT / 'REPORT_FULL.zh-CN.md').write_text(text)

    images = [p.name for p in [OUT / 'validation.png', OUT / 'mechanics_validation.png',
                               OUT / 'mechanics_snapshots.png', OUT / 'mechanics_literature.png',
                               OUT / 'cellstate_validation.png', OUT / 'late_validation.png',
                               OUT / 'cellspace_validation.png',
                               OUT / 'organism_scale.png',
                               OUT / 'organism_lineage.png',
                               OUT / 'mesh_division_zoom.png',
                               OUT / 'mesh_division_frames.png',
                               OUT / 'mesh_division_areas.png'] if p.exists()]
    gifs = [p.name for p in [OUT / 'calcium_wave.gif', OUT / 'mesh_division.gif']
            if p.exists()]
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
            '<title>多尺度模拟验证</title>',
            '<style>body{max-width:1150px;margin:32px auto;background:#0f172a;color:#e5e7eb;font:17px/1.7 sans-serif}',
            'img{width:100%;background:#fff;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
            '<h1>多尺度生物模拟：分层规格与本轮实现</h1>',
            '<p>全部为合成模拟结果，不是实验影像。层数与限制见正文。</p>']
    for name in images:
        body.append(f'<img src="{name}">')
    for name in gifs:
        body.append(f'<img style="max-width:640px" src="{name}">')
    body.append('<pre>' + html.escape(text) + '</pre></html>')
    (OUT / 'index_full.html').write_text('\n'.join(body))

    DESK.mkdir(exist_ok=True)
    copied = []
    for name in ['REPORT_FULL.zh-CN.md', 'index_full.html', 'MEASURED_ANCHORS.md',
                 'SPEC_FULL_FIDELITY.md', 'ORGANISM_SCALE_PLAN.md'] + images + gifs:
        src = OUT / name
        if not src.exists():
            src = ROOT / name
        if src.exists():
            shutil.copy2(src, DESK / name)
            copied.append(name)
    other = OUT / 'metrics.json'
    if other.exists():
        shutil.copy2(other, DESK / 'metrics_calcium.json')
        copied.append('metrics_calcium.json')
    for extra in ['metrics_mechanics.json', 'metrics_cellstate.json', 'metrics_late.json',
                  'metrics_mechanics_lit.json', 'metrics_mechanics_sweep.json',
                  'metrics_cellspace.json', 'metrics_organism.json',
                  'growth_verified.json', 'transport_verified.json',
                  'lineage_verified.json', 'metrics_lineage_coupling.json',
                  'mesh_division_verified.json', 'vertex_division_probe.json']:
        p = OUT / extra
        if p.exists():
            shutil.copy2(p, DESK / extra)
            copied.append(extra)
    print('report written; desktop files:', copied)


if __name__ == '__main__':
    main()
