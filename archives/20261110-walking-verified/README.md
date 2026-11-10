# 存档：替代ANN 全身物理行走验收（2026-11-10）

本目录是本次已验证行走结果的不可变存档，配套 git 提交与标签：

- 提交链：`037f6bc`（忽略可再生载荷）→ `3667bec`（协议/计划/状态）→ `7e41fa6`（全部实验代码与可读证据）→ `c9f1599`（本存档的README与校验和）
- 标签：`walking-verified-20261110`（annotated，指向已验证证据提交`7e41fa6`，已推送）
- 远端：`github-brain:sensen02/Technology-to-Keep-Human-Brain-Consciousness-Alive-Forever.git`；`main` = `c9f1599288b6d385f61227829198d57636daa44f`（`c9f1599`仅新增本目录说明，未改动任何实验证据）

## 结论与范围

受用户授权的**替代ANN／声明过的派生完整身体／有限增量PD**行走实施已验证通过：2500步检查点在1秒零命令稳定与4.8秒实际算法教师预热后，由纯ANN控制完整9.6秒，六腿完成换步并真实前进。撤辅处身体、执行器目标、上一实际命令与GRU hidden均不reset；无附着、无根轨迹写入、无外部力。

不声称：原连接组能力、冻结原身体认证、独立SI/生物有效性、任意时长或任意扰动稳健性。

## 关键数值（正式原生记录，12.5 µs）

| 指标 | 数值 |
|---|---:|
| 最大逐腿承重滑移比 | 0.49411203866010717（≤0.5） |
| 有符号前向速度 | 0.0713190529277483（≥0.05） |
| 最低COM | 1.2561640816296098（≥1.20 且 ≥AUX−0.1） |
| 非足正法向力峰值 | 0（原D1严格0） |
| 平均足载/W | 1.000000049933006 |
| 四六腿完整周期 | LF1，LH/LM/RF/RH/RM各2 |
| 额外体力/广义力/附着力 | 全部0 |
| 原生步数 | 1232000（80000稳定+384000教师+768000自主） |
| 接触事件 / 轨迹端点 | 19769188 / 15400（逐位匹配快评估） |

两个预登记20 ms真实电机脉冲初态回合同样通过全部工程门与六腿周期：滑移0.49269334091673184、0.494878086356009。

## 包内文件

- `01-weights-and-evidence.tar.zst`：2500与3000权重、最终合取`FINAL_ACCEPTANCE.json`、原生审计`NATIVE_ROLE_2500_AUDIT.json`、训练摘要、两次标称评估、两扰动RESULT/IDENTITY、同次视频与解码图、全部确认脚本、冻结评估器`native_acceptance.py`与`audited_eval.py`、测量冻结声明、编译模型XML、1ms/5ms/神经决策数组。
- `02-native-2500.tar.zst`：正式运行的完整12.5 µs原生子步数据流（602块，约5.8 GB未压缩）。每块绝对路径与SHA-256同时记录在包内`NATIVE_ROLE_2500/RESULT.json`的`native.chunks`。
- `SHA256SUMS.txt`：两个归档文件的校验和。
- `README.md`：本说明。

## 复核方法

```bash
cd /run/media/sensen/Data2/cell_wound_prototype/archives/20261110-walking-verified
sha256sum -c SHA256SUMS.txt
zstd -t 01-weights-and-evidence.tar.zst
tar -tf 01-weights-and-evidence.tar.zst | head    # 65个成员
tar -I zstd -tf 02-native-2500.tar.zst | head     # 602个原生块
```

解包后在项目原路径使用冻结评估器复核（读取只读证据，不重新积分物理）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python \
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152725Z/final_native_audit.py \
  --run  /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500 \
  --baseline /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/EVAL_ROLE_2500 \
  --out  /tmp/REAUDIT_2500.json
```

## 失败证据保留

300/1000/1500/2000步的实际失败（滑移1.351456／0.534655／0.516383／0.504026）及其checkpoint、轨迹和报告未被删除或改写，位于`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T114148Z`与`20261110T125712Z`等时间戳目录。桌面仅保留本次正式视频与图；被替代的旧桌面副本已清理，项目内原始失败证据完整。

## 桌面交付

- `/home/sensen/Desktop/engine修复验收/行走改进实施-20261110/自主行走验收通过-2500步-同次原生轨迹.mp4`
- `/home/sensen/Desktop/engine修复验收/行走改进实施-20261110/自主行走验收通过-同次六阶段实际画面.png`
- `/home/sensen/Desktop/engine修复验收/行走改进实施-20261110/续轮5-固定续训六腿行走通过工程门.png`
- `/home/sensen/Desktop/engine修复验收/行走改进实施-20261110/续轮5-2500步真实足载抬足与推进-候选通过.png`

生成时间：2026-11-10（本地时区），由本次会话在推送完成后立即建立。
