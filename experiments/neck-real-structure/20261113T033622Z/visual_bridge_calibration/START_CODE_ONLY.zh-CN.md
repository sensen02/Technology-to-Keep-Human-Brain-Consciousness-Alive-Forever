# 实际RGB视觉神经标量经颈部校准：CODE_ONLY已就绪

本子只读旧源码／JSON、写新文件并执行标准库检查。实际codecheck.py exit0，37项全部通过，无numpy／Torch导入、数组payload读取、模型／Core构造、forward、优化器、物理或网络。所有旧目录未修改。数值正确性、源帧逐位重放及训练结果仍须父实际执行，不能把37项静态检查当功能通过。

新校准入口：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_bridge_calibration/run.py --execute-parent-visual-calibration > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_bridge_calibration/PIPELINE.log 2>&1
```

父用单个托管后台job执行以上命令。它串行执行预检、固定300训练、0／150／300新进程保存审核、最终11条归档指标及独立指标审核。每stage独立PID／log／exit记录在JOBS.json；失败后停止不自动重跑、不追加更新或缩batch。先读失败日志／已有部分输出，再另立恢复，不覆盖科学证据。

## 固定接口

99698冻结视觉实际RGB→CNN有序16缓存→GRU32→rawturn[-1,1]直接作为brain16标量输入，无sign／deadzone／除.8。脑另接已交付AN1，输出DN2经来源队列后进入身体神经decoder和冻结motor789f。新完整身体预期状态[128,64,64,32,16]=304，cache16／相机／端口队列另记。当前校准脚本不加载motor或Core，也不运行身体。

旧warm时序保持：相机真实捕获／20ms发布后缓存，视觉GRU32与brain16在预热保持0，ANN起5.8后step，不交接reset；actualAN热历史未来身体需保留。当前离线桥仍沿既有coldANN端口初值约定，首DN接收零；实际新身体warmAN历史必须单列验证，不能偷偷重置。

## 数据和监督

8条已有视觉档案先实际重放capturedRGB→frozenvision：6normal全部train，旧occlusion和actor-framefreeze两case只作已知归档开发对照。核对全部2880行storedheld16／rawturn／latent／visionh32逐位相同、20msrelease与actor/rawcaptureID及RGB SHA。flip只用已实际声明fid529冻结，实际raw帧继续；occlusion用真实遮挡画面，无伪造空图。

3条actualbody辅助只包含X238suffix1160:3080中的111通道与显式zero-neural-visual输入／目标；没有对应RGB或visualhidden，也不生成假图或hidden填充。元数据显式标“神经输入零干预”，不计视觉行为或freshholdout。6真实RGB配对ANNsuffix960:2880＋3辅助合9条，每条1920步。

目标y[k]=当前frozenvisualu[k−1]，首y0=0。currentu[k]只进脑，teacher只loss；同时报告对当前u误差但不将预测未来混训练。所有连续幅度保留，不再cue×.8。

## 参数、预算与验证

strict4153全state起点，冻结反馈encoder／reconstruction／normalizer／mask和全部视觉；trainable仅brainGRU4＋out2＋decoder4＝10张量。freshAdam3e-4／固定300／B9／完整1920，保存0／150／300只最终验证；wall1800s、RSS1800MiB，不自动加步／缩批。

预检会核parentallstate、全部冻结tensor、实际RGB bitexact、Python/JIT全历史forward／weightedloss／10梯度／Adam、111禁止通道、真实1ms来源队列9600tick与2ms释放／5ms消费。实际生产重放与保存结果不一致立即拒绝。保存审核按同initialize两次构造路线验证RNG，优化器只有10状态且计数等于对应更新数，冻结tensor逐字相同。

语义identity已更新脑输入为frozen_actual_RGB_visual_rawturn1，不遗留experimental_instruction；新格式严格SOURCE_NECK_VISUAL_SCALAR_CALIBRATION_V1。调用newcommon.load_checkpoint(path,digest,final=True)取得最终300，返回m/p。model.py无common依赖，但新common.initialize会临时alias旧model供严格4153祖先加载，finally恢复。身体集成时仍须隔离visual／bridge／motor各自model/common别名和sys.path，不改旧源。

已有空指令五工程门/slip.484657通过、左右转弯滑移FAIL和全部先前失败保持。视觉恢复不意味着这些指标自动改善，更不是完整颈部膜电生理、原生视网膜、电极或意识证明。
