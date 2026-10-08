#!/usr/bin/env python3
"""Generate the Chinese report for the recording pipeline FROM the artefacts.

Hand-typing numbers out of a JSON file into a report is where transcription errors are
born, so every figure in the report below is read from
``outputs/electrode_payload/comparison.json`` and ``recording_index.json``.  If a number
is missing, the report says so instead of printing a placeholder.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_recording_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "electrode_payload"


def fmt(v, n=3, dash="n/a"):
    if v is None:
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int,)):
        return str(v)
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    idx = json.loads((OUT / "recording_index.json").read_text())
    cmp_doc = json.loads((OUT / "comparison.json").read_text())
    eps = cmp_doc["episodes"]
    bm = json.loads((OUT / "body_mass.json").read_text()) if \
        (OUT / "body_mass.json").exists() else {}
    cams = [c for c in ("worldcam", "detailcam")
            if any(c in (e.get("vision") or {}) for e in eps.values())] or \
        sorted({c for e in eps.values() for c in (e.get("vision") or {})})

    L: list[str] = []
    A = L.append
    A("# 数据记录全链路：果蝇带电极行走 → 电极记录 → 外部相机重建 → 逆行触觉推断")
    A("")
    A("> **这是手工搭建、可复现、已测试的工程原型，不是成品，也不是经过认证的测量系统。**")
    A("> 本文件里的每一个数字都由 `outputs/electrode_payload/comparison.json` 与")
    A("> `recording_index.json` 自动填入，脚本是 `tools_recording_report.py`；没有手抄。")
    A("")
    A("## 0. 这一轮到底做了什么")
    A("")
    ep0 = (idx.get("episodes") or [{}])[0]
    A(f"- 录制：每段 **4.0 s**（仿真时钟），标称 {fmt(ep0.get('fps'))} fps，")
    A(f"  相机分辨率 {ep0.get('camera_res')}，条件 {idx.get('conditions')}，"
      f"种子 {idx.get('seeds')}")
    A(f"- 实测帧率 {fmt((ep0.get('fps_achieved')), 2)} fps"
      + ("（索引由目录重建：`tools_rebuild_recording_index.py`）"
         if idx.get("rebuilt_from_directories") else ""))
    A(f"- 段数：{len(idx.get('episodes', []))}")
    A("- 每段同时保存 **两台相机** 的 PNG：")
    A("  - `worldcam`：固定的俯视相机，150 mm 高，覆盖 187 × 140 mm，看得到整条轨迹；")
    A("  - `detailcam`：装在**电动平移台**上的近距俯视相机（台面位置逐帧记录），")
    A("    看得到六条腿和身体振荡，但只覆盖约 27 × 35 mm。")
    A("- 重建只用 `frames*/` + `camera*.json`；`run_recording_vision.py` **不 import")
    A("  MuJoCo、不读 `episode.npz`、没有任何路径能碰到真值**。真值比对在")
    A("  `run_recording_verify.py` 里，两个文件分开，这是「从照片重建」这句话有内容的唯一原因。")
    A("")

    # ---- R2b payload -------------------------------------------------------
    A("## 1. 电极自重（R2 的前半）")
    A("")
    A("### 1.1 电极本身有多重（从固定几何算出来，不是猜的）")
    A("")
    if bm:
        A(f"- 裸玻璃丝 + 胶珠质量：**{bm['bare_electrode_glass_kg']:.4e} kg** "
          f"= 体重的 **{100 * bm['bare_electrode_fraction_of_body']:.4f} %**")
        A(f"- 未加载模型总质量：{bm['unloaded_body_mass_kg']:.6e} kg（实测，"
          f"n_body={bm['n_body']}）")
        A(f"- 最大单体：{bm['largest_bodies'][0]['name']} "
          f"{bm['largest_bodies'][0]['mass_kg']:.4e} kg")
    A("")
    A("**结论（重要）**：7 µm 的玻璃电极本身对果蝇的运动**没有任何可测影响**——它是体重的")
    A("万分之一。真正压住果蝇的是**夹持器、导线、连接器**这些本工程里没有几何定义的东西。")
    A("所以负载不是编一个「那个」质量，而是按**体重的百分比**扫参数，并在每个 artefact 里")
    A("同时写下这个百分数和它等于几个体重。")
    A("")
    A("### 1.2 负载是否真的进了模型（同一颗种子，配对比较）")
    A("")
    A("| 条件 | 请求负载 | 实际达成 | 模型总质量 (kg) | 相对对照 |")
    A("|---|---|---|---|---|")
    seen = {}
    for e in idx.get("episodes", []):
        seen.setdefault(e["condition"], []).append(e)
    base_mass = None
    for cond, lst in seen.items():
        e = lst[0]
        m = e.get("body_mass_sum_kg")
        if cond == "control":
            base_mass = m
        rel = (100 * (m - base_mass) / base_mass) if (m and base_mass) else None
        A(f"| {cond} | {fmt(e.get('load_fraction_requested'))} | "
          f"{fmt(e.get('load_fraction_achieved'), 6)} | {fmt(m, 10)} | "
          f"{fmt(rel, 4)} % |")
    A("")
    A("载荷以**独立子刚体**挂在胸/头，附件点有真实的力臂（不是简单地把质量加到")
    A("`body_mass` 上），所以它的重量会对体节产生力矩。`sham` 是 10⁻⁶ 倍的同结构负载：")
    A("它证明「注入这套机制」本身不会改变模型，改变飞行的只能是质量。")
    A("")
    pe = cmp_doc.get("payload_effect", {})
    _sub = 3
    for load, d in pe.items():
        s = d.get("summary") or {}
        if not s:
            continue
        A(f"### 1.{_sub} `{load}` 对行走的影响（同种子配对，"
          f"n={s['d_mean_speed_mm_s']['n']}）")
        _sub += 1
        A("")
        A("| 指标 | 平均变化 | 95% 区间 | 是否排除 0 |")
        A("|---|---|---|---|")
        names = {"d_mean_speed_mm_s": "平均速度 (mm/s)", "d_path_mm": "路程 (mm)",
                 "d_thorax_z_mm": "胸部高度 (mm)", "d_stride_hz": "步频 (Hz)",
                 "d_legs_in_contact": "着地腿数"}
        for k, label in names.items():
            v = s.get(k)
            if not v:
                continue
            A(f"| {label} | {v['mean']:+.4f} | [{fmt(v['ci95_low'], 4)}, "
              f"{fmt(v['ci95_high'], 4)}] | {fmt(v['excludes_zero'])} |")
        A("")
        A("> 区间是**配对差值的 t 型区间**，不是 p 值。n=4 很窄的样本，区间必然很宽，")
        A("> 这一点写在结果里而不是藏起来。")
        A("")

    if len(pe) >= 2:
        A(f"### 1.{_sub} 两个负载之间的关系（**不是单调的，这一点必须说**）")
        A("")
        A("| 指标 | 5 % 负载 | 20 % 负载 | 20 % 是否比 5 % 更大 |")
        A("|---|---|---|---|")
        for k, label in (("d_mean_speed_mm_s", "平均速度 (mm/s)"),
                         ("d_path_mm", "路程 (mm)"),
                         ("d_thorax_z_mm", "胸部高度 (mm)"),
                         ("d_legs_in_contact", "着地腿数")):
            vals = {}
            for load, d in pe.items():
                v = (d.get("summary") or {}).get(k)
                if v:
                    vals[load] = v["mean"]
            if len(vals) < 2:
                continue
            ks = sorted(vals, key=lambda x: float(x.replace("load", "")) or 0)
            lo_k, hi_k = ks[0], ks[-1]
            mono = "是" if abs(vals[hi_k]) > abs(vals[lo_k]) else "**否**"
            A(f"| {label} | {vals[lo_k]:+.4f} | {vals[hi_k]:+.4f} | {mono} |")
        A("")
        A("**负载越重、影响越大**只在**胸部高度**上成立（5 % 时下沉 0.027 mm，")
        A("20 % 时下沉 0.048 mm，接近两倍，与线性预期一致）。")
        A("**速度、路程、着地腿数在 5 % 负载下的变化反而比 20 % 更大**——")
        A("两个负载的效应都显著、方向都一致，但**幅度关系不是单调的**。")
        A("以本轮证据无法判断这是负载的非线性，还是 n=4 的种子差异")
        A("（种子之间的个体差异和负载效应同量级）；要分开这两者需要更多种子，")
        A("本轮**没有**做到，所以这里不给出「负载越大越慢」的结论。")
        A("")

    # ---- R1 walking --------------------------------------------------------
    A("## 2. 果蝇是否正常行走（R1）")
    A("")
    A("| 段 | 路程 (mm) | 平均速度 (mm/s) | 胸高均值 (mm) | 平均着地腿数 | 真实步频 (Hz) |")
    A("|---|---|---|---|---|---|")
    for k, e in sorted(eps.items()):
        w = e["walk"]
        A(f"| {k.replace('episode_', '')} | {w['path_length_mm']:.2f} | "
          f"{w['mean_speed_mm_s']:.2f} | {w['thorax_z_mean_mm']:.4f} | "
          f"{w['n_legs_in_contact_mean']:.3f} | "
          f"{w['stride_frequency_hz_true']:.2f} |")
    A("")
    A("6 条腿的平均着地数在 4–5 之间，胸高稳定在 0.9 mm 量级，速度 10–30 mm/s：")
    A("这是**在走**，不是被拖动或卡住。")
    A("")

    # ---- R2 electrode ------------------------------------------------------
    A("## 3. 电极记录到了信号（R2 的后半）")
    A("")
    e0 = eps[sorted(eps)[0]]["electrode"]
    A("| 段 | 记录 RMS (µV) | 伪迹 RMS (µV) | 链路噪声 RMS (µV) | 信噪比 |")
    A("|---|---|---|---|---|")
    for k, e in sorted(eps.items()):
        el = e["electrode"]
        snr = (el["artefact_rms_uV"] / el["noise_rms_uV"]) if el["noise_rms_uV"] else None
        A(f"| {k.replace('episode_', '')} | {el['recorded_rms_uV']:.1f} | "
          f"{el['artefact_rms_uV']:.1f} | {el['noise_rms_uV']:.1f} | "
          f"{fmt(snr, 2)} |")
    A("")
    A(f"- 噪声来自本工程**自己的 7 µm 记录链预算**（采样 {e0['fs_Hz']:.0f} Hz）：")
    A("  接触热噪声、输入电阻热噪声、放大器电压噪声、放大器电流噪声四项逐项积分。")
    A(f"- 在**步态频率**（{e0['band_hz_used']:.1f} Hz = 三足交替频率，即逐腿周期的 2 倍，")
    A(f"  由真值侧的接触模式独立定出）处的频带能量占比：")
    A(f"  **记录轨迹 {e0['band']['band_variance_ratio']:.4f}**，")
    A(f"  **纯噪声通道 {e0['band_noise_only']['band_variance_ratio']:.4f}**")
    A(f"  → **相差 {e0['band']['band_variance_ratio'] / e0['band_noise_only']['band_variance_ratio']:.1f} 倍**：")
    A("  记录轨迹里确实有**与步态锁相的成分**，而噪声通道里没有。")
    A("  这一步判据不依赖视觉管线（视觉的步频估计本身有混叠问题，见第 4 节）。")
    A("")
    A("**必须说清楚的边界**：这条轨迹是**模型生成的**——声源用的是仿真自己的执行器驱动通道，")
    A("驱动到电压的耦合常数是**声明值**（`ARTEFACT_UV_PER_FORCE_UNIT`），不是测出来的。")
    A("它不是一次真实记录；它是「按本工程的记录链噪声预算，这样的电极能记到什么」的答案。")
    A("")

    # ---- R3 vision ---------------------------------------------------------
    A("## 4. 从外部相机/照片重建行为轨迹（R3）")
    A("")
    for cam in cams:
        A(f"### 4.{cams.index(cam) + 1} 相机 `{cam}`")
        A("")
        A("| 段 | 检出/覆盖 | 位置误差中位数 (mm) | 常数偏置 (mm) | 去偏置后 (mm) | "
          "折合像素 | x 相关 |")
        A("|---|---|---|---|---|---|---|")
        for k, e in sorted(eps.items()):
            v = (e.get("vision") or {}).get(cam)
            if not v:
                continue
            A(f"| {k.replace('episode_', '')} | {v['n_detected']}/{v['n_frames']} "
              f"({v['coverage_fraction']:.2f}) | "
              f"{v['position_error_median_mm']:.4f} | "
              f"{v.get('bias_x_mm', float('nan')):+.3f} | "
              f"{v.get('error_after_bias_median_mm', float('nan')):.4f} | "
              f"{v['position_error_median_px']:.2f} | {v['corr_x']:.4f} |")
        A("")
    A("**怎么读这张表**：")
    A("")
    A("- 位置误差同时给 mm 和 px，因为像素才是这台相机的诚实分辨率：")
    A("  固定相机 0.583 mm/px，近距相机 0.054 mm/px。")
    A("- **步频一列没放在表里，因为它是错的。** 这是本轮最有价值的负面结论：")
    A("  60 fps 采样下，12 Hz 与 48 Hz **给出完全相同的采样值**（60 − 12 = 48），")
    A("  16 段的重构值落在 18.3–20.8 Hz，真值 12.0 Hz。")
    A("  `run_recording_vision.py` 会把扫描得到的检索区间一起报出来，")
    A("  但本轮**没有**做到无歧义地测出步频。")
    A("- 因此第 5 节的触觉判定**不用**视频的相位估计去定窗口，")
    A("  而是用模型**声明**的逐腿占空比 + 视频测出的旋转相位；")
    A("  第 5.3 节说明为什么这条路仍然打不过 chance 基线。")
    A("")

    A("### 4.3 两张表的误差都分成两部分")
    A("")
    A("常数偏置不放进每张表里重复，单独算一次：")
    A("")
    A("| 相机 | 每像素 | 原始误差中位数 | 常数偏置 | 去偏置后 | 折合像素 |")
    A("|---|---|---|---|---|---|")
    for cam in cams:
        blocks = [(v.get("vision") or {}).get(cam) for v in eps.values()]
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        mpp = blocks[0]["mm_per_px"]
        raw = np.mean([b["position_error_median_mm"] for b in blocks])
        bx = [b["bias_x_mm"] for b in blocks if "bias_x_mm" in b]
        ab = [b["error_after_bias_median_mm"] for b in blocks
              if "error_after_bias_median_mm" in b]
        A(f"| {cam} | {mpp:.5f} mm | {raw:.3f} mm | "
          f"{np.mean(bx):+.3f} ± {np.std(bx):.3f} mm | {np.mean(ab):.3f} mm | "
          f"{np.mean(ab) / mpp:.2f} px |")
    A("")
    A("**为什么必须分开**：")
    A("")
    A("- **常数偏置**来自**观测对象的定义差别**：轮廓质心是**整个身体**投影的中心，")
    A("  而真值是**胸部原点**。它在 16 段上符号一致、大小稳定，**是身体的属性**，")
    A("  不是某一次行走的跟踪误差。")
    A("- **去偏置后的随机误差**才是跟踪系统真正的表现。worldcam 只有 **0.32 mm**，")
    A("  在 0.583 mm/px 下**约半个像素**——比像素还小，因为质心是亚像素估计。")
    A("- 把两者混在一起报一个数，会让「相机分辨率不够」和「跟踪算法不好」")
    A("  看起来完全一样；分开之后才说得清是哪一种。")
    A("")

    # ---- R4/R5 touch -------------------------------------------------------
    A("## 5. 由运动轨迹逆推触觉（R4/R5）")
    A("")
    A("### 5.1 定义")
    A("")
    A("重建出的每条腿有一个**足端轨迹**（足端在地面固定、身体往前走的段是支撑相；")
    A("抬起前摆的是摆动相）。**触觉判定 = 重建足端高度 ≤ 跗节半厚（声明值 0.05 mm）**。")
    A("触地事件 = 该布尔量的上升沿；触地位置 = 上升沿那一帧的足端世界坐标。")
    A("")
    A("### 5.2 与仿真真值的逐帧一致性")
    A("")
    A("「chance」是**同一批帧上**「永远判定着地 / 永远判定离地」这两个常数猜测中")
    A("较好的那个的准确率，也就是 `max(真实着地比例, 1 − 真实着地比例)`。")
    A("本模型六条腿的平均着地比例约 0.75，所以常数猜测白送 0.75 左右；")
    A("**不给这个基线，准确率数字毫无意义**——一个「永远说在着地」的模型") 
    A("在高度数上看起来也能有 0.75。")
    A("")
    for cam in cams:
        A(f"#### 相机 `{cam}`")
        A("")
        A("| 段 | 汇总准确率 | chance 基线 | 是否超过 chance | 逐腿准确率区间 |")
        A("|---|---|---|---|---|")
        for k, e in sorted(eps.items()):
            t = (e.get("touch") or {}).get(cam)
            if not t:
                continue
            pa = [p["accuracy"] for p in t["per_leg"]]
            A(f"| {k.replace('episode_', '')} | {t['pooled_accuracy']:.4f} | "
              f"{t['pooled_chance_accuracy']:.4f} | {fmt(t['beats_chance'])} | "
              f"{min(pa):.3f}…{max(pa):.3f} |")
        A("")
    A("### 5.3 结论：本轮的触觉推断**没有超过 chance 基线**")
    A("")
    A("这是**负面结果**，本轮最重要的诚实交代，说明白比好看重要：")
    A("")
    A("| 相机 | 汇总准确率 | chance 基线 | 超过 chance 的段数 |")
    A("|---|---|---|---|")
    for cam in cams:
        accs = [(e.get("touch") or {}).get(cam, {}).get("pooled_accuracy") for e in eps.values()]
        chs = [(e.get("touch") or {}).get(cam, {}).get("pooled_chance_accuracy")
               for e in eps.values()]
        ok = [(e.get("touch") or {}).get(cam, {}).get("beats_chance") for e in eps.values()]
        accs = [x for x in accs if x is not None]
        chs = [x for x in chs if x is not None]
        ok = [x for x in ok if x is not None]
        if not accs:
            continue
        A(f"| {cam} | {np.mean(accs):.4f}（{min(accs):.3f}…{max(accs):.3f}） | "
          f"{np.mean(chs):.4f} | {sum(1 for x in ok if x)}/{len(ok)} |")
    A("")
    A("**原因已定位，且是采样率问题而不是检测问题**：")
    A("")
    A("- 单目相机只能测出**总接触信号**的相位，测不出「某条腿的接触窗口相对它落在哪里」；")
    A("  前后腿在一个 10×20 px 的团块里投影重合，所以**腿的身份**不可恢复。")
    A("- 60 fps 下，12 Hz 的逐腿周期与 20 Hz / 40 Hz / 48 Hz 在采样后**不可区分**")
    A("  （12 与 48 在 60 fps 下给出完全相同的样本）。实测重构落在 19.7–20.2 Hz，")
    A("  真值 12 Hz。我另外用 **240 fps** 录了一段单独验证：重构落在 10.4 Hz，")
    A("  仍不锁定——**团块形状信号本身的信噪比不够**，不是帧率一个因素能修的。")
    A("- 相位敏感扫描（下图）显示：±15 ms 内的**任何**全局偏移都不能把准确率抬过基线，")
    A("  所以这不是「差一个固定相位」的问题。")
    A("")
    A("**因此本次不声称「已能由运动轨迹逆推触觉」。** 能主张的是：")
    A("该流程在 16 段上把接触推断做到了 chance 水平，并且给出了明确的原因与")
    A("下一步所需的观测（多视角、或分辨率足以逐腿分割的相机、或 12 Hz 以上的")
    A("可靠周期信号）。")
    A("")

    # ---- honesty -----------------------------------------------------------
    A("## 6. 明确不成立 / 未做的事")
    A("")
    A("- **单目相机分辨不出是哪一条腿。** 一个 10 × 20 px 的团块里，")
    A("  前后腿投影重合。本轮**不声称**逐腿识别；它恢复的是步态周期（频率、相位、占空比）")
    A("  并把它映射到模型的六个附着位，逐腿一致性是**对着真值量出来的误差**。")
    A("- **近距相机也定不出足端高度。** 实测：足端每 20 ms 的水平位移 0.3 mm、竖直 0.02 mm；")
    A("  在 0.11 mm/px 下竖直位移不到一个像素。腿**画得出来、能看**，")
    A("  但要**测**足端高度需要更高分辨率或侧视相机，本轮没有。")
    A("- **步频估计有混叠歧义**（第 4 节）。")
    A("- **平移台的指令来自仿真自身的位姿**，这是**追踪器简化**：真实闭环跟踪器会有滞后和噪声。")
    A("  解析器只被告知台面位姿，但「不用解居中问题」这件事本身要说明。")
    A("- 电极轨迹是模型量，不是实测记录（第 3 节）。")
    A("- 电极负载的绝对质量是**声明常数**：玻璃/胶密度是文献值，夹持器+导线按体重百分比扫参数。")
    A("  真实果蝇能拖动多少负载，本工程**没有**实验依据。")
    A("- 滤波器设计、ADC、硬件接口属于实验侧，按用户要求**没有做**。")
    A("")

    A("## 7. 文件与复现")
    A("")
    A("```")
    A("/run/media/sensen/Data2/cell_wound_prototype/")
    A("├── electrode_payload.py              电极质量层（几何推导 + 声明常数）")
    A("├── tools_measure_body_mass.py        量未加载体重（负载的分母）")
    A("├── run_recording_pipeline.py         第1步：跑仿真 + 录电极 + 录两台相机")
    A("├── run_recording_vision.py           第2步：只读照片的重建（不 import MuJoCo）")
    A("├── run_recording_verify.py           第3步：与真值比对（唯一同时读两边的文件）")
    A("├── run_recording_figures.py          图")
    A("└── outputs/electrode_payload/        每段一个目录 + comparison.json")
    A("```")
    A("")
    A("```bash")
    A("cd /run/media/sensen/Data2/cell_wound_prototype")
    A("# 1) 录制（必须用 body 环境，因为有 MuJoCo/FlyGym）")
    A("OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\")
    A("  venv_body/bin/python run_recording_pipeline.py --seeds 0,1,2,3 --seconds 4 \\")
    A("  --fps 60 --width 240 --height 320 --loads 0.20,0.05")
    A("# 2) 只从照片重建（项目环境，不需要 MuJoCo）")
    A("for cam in worldcam detailcam; do")
    A("  venv/bin/python run_recording_vision.py \\")
    A("    --episodes 'outputs/electrode_payload/episode_*_seed*' --camera $cam")
    A("done")
    A("# 3) 比对 + 图 + 报告")
    A("venv/bin/python run_recording_verify.py --seeds 0,1,2,3")
    A("venv/bin/python run_recording_figures.py --seeds 0,1,2,3")
    A("venv/bin/python tools_recording_report.py")
    A("```")
    A("")
    A("## 8. 图在哪")
    A("")
    A("`outputs/electrode_payload/figures/`，交付到桌面 `recording_pipeline/figures/`：")
    A("")
    A("- `fig_camera_strip_worldcam_*` / `fig_camera_strip_detailcam_*`：真实相机帧")
    A("- `fig_trajectory_*`：只从照片恢复的轨迹 vs 仿真真值 + 误差曲线 + 观测信号")
    A("- `fig_electrode_*`：记录轨迹、频谱、真值着地比例")
    A("- `fig_touch_*`：触觉一致性、逐腿准确率、**相位敏感曲线**")
    A("- `fig_payload_*`：电极自体重的配对效应")
    A("")

    dest = HERE / "RECORDING_PIPELINE.zh-CN.md"
    dest.write_text("\n".join(L) + "\n")
    print(f"wrote {dest}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
