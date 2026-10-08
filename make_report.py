from pathlib import Path
import json,shutil,html
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs';d=json.loads((OUT/'metrics.json').read_text());c=d['cases']
rows='\n'.join(f"|{name}|{c[name]['n_cells']}|{c[name]['comparison_points']}/11|{c[name]['mae_um']:.3f}|{c[name]['rmse_um']:.3f}|" for name in ['control','recommended_576','larger_domain','seed2_576','seed3_576'])
report=f'''# 果蝇上皮损伤信号：数百细胞模拟原型与验证报告

## 结论
已实际编写并运行 Python 原型。这是手动实施的研究原型，不是完整生物数字孪生或现成成熟软件。
选择果蝇蛹背部上皮，而不是果蝇大脑：有直接对应的损伤信号实验、开放方程、MIT 代码及数值数据。
目前可复现损伤后早期信号传播的大体趋势；仍存在约5微米的系统性半径高估，不能宣称精确还原现实。

## 实验依据
Stevens 等，2023，A mathematical model of calcium signals around laser-induced epithelial wounds。
论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC10208100/
代码与数据：https://github.com/mshutson/wound-calcium-LRCa
固定源版本：f0be2fa8e2da5ce59166866b72c819e3df2f874d。
使用作者提供的 controlCaRadData.m 单个伤口的已处理信号半径数据，不是我们生成的实验数据。微撕裂空间分布由作者公开表格按原算法拟合。
原始版本是 Mathematica；本版本为缩减 Python 移植，不是原软件直接运行。

## 模型包含什么
每细胞43平方微米固定六边形；胞质钙、内质网钙、IP3、IP3受体失活状态4个动态变量；内外膜通量、缓冲、细胞间隙连接、GCaMP荧光观测近似。
先300秒平衡，再25秒损伤响应。采用原文参数；没有针对半径对照再次拟合生物参数。
中心损伤半径24.5微米，微撕裂区域51.25微米，是规定输入，不是力学求解结果。
消融区钙强制保持1000微摩尔，代表外源储库，不表示存活细胞。
未建模：真实膜网格、细胞形变/撕裂、胞内空间细节、化学侵蚀、ATP、死亡判定、组织修复、远期蛋白酶/配体信号、大脑、意识。

## 与真实实验对照
比较2.14～23.54秒11个时点，使用荧光径向半高半径；不是伤口半径，也不是细胞死亡半径。
|版本|细胞数|可比较时点|MAE 微米|RMSE 微米|
|---|---:|---:|---:|---:|
{rows}

400细胞视野完整圆半径仅59.69微米，很多时点无法取得外侧半高交点，因此只剩3/11点，不能据此声称验证成功。
发现问题后公开增加576细胞版本，保留相同细胞大小和生物参数。784细胞仅作为边界对照。
19.26秒：实验58.895微米，576细胞预测{c['recommended_576']['radius_at_19_26_s_um']:.3f}微米。
576与784在可比较帧上的半径最大差{d['domain_576_vs784_radius_max_difference_um']:.3f}微米；说明该观测量对这次域扩展较稳定，不证明所有胞内量都准确。
576与784中央60微米内胞质钙轨迹RMS差{d['domain_576_vs784_inner60_c_rms_difference_um']:.6g}微摩尔。

## 机制对照（定性）
将间隙连接降至10%，19.26秒半径降至{c['gj_576']['radius_at_19_26_s_um']:.3f}微米，早期向外扩展明显受抑制。
将PLC降至30%，同一时点{c['plc_576']['radius_at_19_26_s_um']:.3f}微米，早期传播保留较多。
方向与论文对应扰动结论一致，但我们全域改变参数，原实验含空间异质敲低，不是完整复制实验。
没有获得对应敲低的独立数值实验数据；不能将它们相对正常对照的MAE称为敲低验证成绩。

## 数值检查
- 浓度有限、非负，h在0～1；几何面积与间隙交换守恒检查通过。
- 面积加权GJ交换守恒残差：{d['model_self_test']['conservation_error']:.3e}。
- 无损伤胞质钙最大漂移：{d['no_wound_max_c_drift_um']:.3e}微摩尔。
- 576细胞严格容差对照最大胞质钙差：{d['solver_576_max_abs_c_um']:.3e}微摩尔，RMS：{d['solver_576_rms_c_um']:.3e}。
- 合成高斯观测单元测试的半径误差：{d['measurement_unit_test']['error_um']:.3f}微米。它是测量代码测试，不是生物验证。
- CPU上576细胞单次平衡＋25秒轨迹计算：{c['recommended_576']['metadata']['total_seconds']:.3f}秒，不含绘图/导出。没有使用或测试H20/9070XT，不将此速度外推至三维细胞力学。
- NPZ缓存验证模型代码、输入参数及微撕裂数据哈希；记录源仓库版本。

## 证据边界与问题
1. 只有单个伤口的处理后曲线；作者参数已利用同一实验背景校准。这是回溯参考检查，不是独立外部验证，也不能给出“真实度百分比”。不同随机种子不是生物重复样本。
2. 原始Voronoi形状替换为规则六边形；无细胞形变。内部仅四个集中变量，不能叫完整细胞内部模拟。
3. 观测采用raw CtoF×GCaMP与径向像素半高近似；作者视频还有亮度仿射变换和裁剪，二者在表达异质时不严格等价。缺原始显微图像，观测差异可能贡献误差。
4. 数百微摩尔的近伤口自由钙是模型推算，不是实验测得。GCaMP已饱和，因此半径吻合不能验证这些绝对浓度，也不能证明细胞活着。
5. 省略延迟配体信号仅用于前25秒，不能预测完整远场波与长期修复。
6. t=0实验已见34.034微米信号，仿真微撕裂刚开始，无可比半高半径；没有通过偷调时间原点掩盖这一差异。统计窗口明确排除t=0。
7. 400细胞边界效应严重，所有原始结果保留。censored标记合并了无响应及找不到外侧交点，不能全部解释为边界裁剪。

## 下一阶段优先事项
先解决观察算子、引入真实Voronoi几何和更多独立实验曲线，检查约5微米系统偏差来源；不要直接以调参把这条曲线拟合得更漂亮作为验证。
随后建立局部可变形上皮力学，并使用独立激光切断后的回缩/细胞面积数据验证；再把计算的应力、膜破损和现有钙模型耦合。力学外观没有实验对照前，应明确标为假设。

## 查看与复现
主目录：/run/media/sensen/Data2/cell_wound_prototype
桌面样本：/home/sensen/Desktop/cell_wound_prototype
运行：
```bash
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_validation.py
```
核心代码：/run/media/sensen/Data2/cell_wound_prototype/model.py
对照程序：/run/media/sensen/Data2/cell_wound_prototype/run_validation.py
观测程序：/run/media/sensen/Data2/cell_wound_prototype/measurement.py
协议：/run/media/sensen/Data2/cell_wound_prototype/PROTOCOL.md
机器可读结果：/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics.json
原始实验CSV：/run/media/sensen/Data2/cell_wound_prototype/outputs/experimental_control.csv
各工况完整胞内轨迹保存在主目录outputs下NPZ。图中主空间演示为576细胞，非显微实验影像。
'''
(OUT/'REPORT.zh-CN.md').write_text(report)
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>细胞模拟验证</title><style>body{max-width:1100px;margin:32px auto;background:#111827;color:#e5e7eb;font:17px/1.7 sans-serif}img{width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#93c5fd}</style><h1>数百细胞果蝇上皮：实验对照原型</h1><p>手动实现的缩减研究模型；不是实际显微图像，不含细胞形变或机械撕裂。</p><img src="validation.png"><img src="cell_snapshots.png"><img style="max-width:650px" src="calcium_wave.gif"><pre>'+html.escape(report)+'</pre></html>'
(OUT/'index.html').write_text(body)
desktop=Path('/home/sensen/Desktop/cell_wound_prototype');desktop.mkdir(exist_ok=True)
for name in ['REPORT.zh-CN.md','index.html','validation.png','cell_snapshots.png','calcium_wave.gif','metrics.json','experimental_control.csv']:
    shutil.copy2(OUT/name,desktop/name)
print('Report and desktop delivery created:',desktop)
