# P1 断面 CMOS 扩至 14×14 触点

V2 运行入口现在默认采用 196 个独立记录触点。选用与原 8×8 版相同的 136×136 µm 名义占地，中心距从 17 降至 9.714286 µm，单片方形金属从 12×12 µm 缩至 8×8 µm，绝缘间隙约 1.714 µm。BANC 切面固定在 y=432 µm。依据组织几何，在原中心附近各轴 ±5 µm、1 µm 步长范围内查找两前足投影全部面对金属且断面仍在芯片内的位置，优先最大化全截面直接面对金属的神经根数。选中 BANC 中心 x=490.2、z=135.6 µm，比原版分别移动 +3、+5 µm。最靠近焊盘边缘的目标像素距边缘约 0.62 µm，取自登记标本。未移动中心的密排布局仍覆盖 2753 根，左前足占 115/134、右侧占 207/207。

| 登记结果 | 原 64 触点 | 196 触点向外扩，保留原间距与焊盘 | 196 触点同占地密排 |
|---|---:|---:|---:|
| 直接面对金属的神经根，分母 3587 | 2417 | 2417 | 2829 |
| 前足标记投影位于金属的像素，左 / 右 | 0/134、207/207 | 0/134、207/207 | 134/134、207/207 |
| 左 / 右局部孤立偶极峰值/噪声 RMS | 2.01 / 5.49 | 2.00 / 5.52 | 6.49 / 7.65 |
| 0、6、24、72 h 含噪下行孤立事件检出，每组 324 次 | 141、91、66、65 | 118、71、37、36 | 322、292、246、244 |
| 独立空白窗误报，每组 256 窗 | 4 | 5 | 1 |
| 全切面局部探针 ≥5×RMS，3579 根可计分 | 787 | 783 | 2246 |

组间使用相同的前端模型及评分方法。通道数改变后，随机数分配与噪声样本会改变。每组内四个时点使用配对刺激和噪声；断后衰减参数来自平台工程假设。含噪试验用 27 根登记下行纤维及其已知模板判断窗口内是否有孤立事件。目标像素覆盖、局部偶极信号和下行检出均属记录端证据。选择性写入左右前足、全束重叠放电下的读出和实物芯片仍待验证。

## 入口与复验

模块 /run/media/sensen/Data1/omnisim/omnisim/scenes/neck_cmos.py 。使用当前缓存时将原生断面和薄层输入复制到新的输出目录以复用离线数据。下列命令在 /run/media/sensen/Data1/omnisim 中执行。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
project=/run/media/sensen/Data2/cell_wound_prototype
archive=/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck
record_dir=$(mktemp -d)
cp "$archive/cmos-17um-wide/"{native-face,section}.npz "$record_dir"
py="$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
"$py" -Bm omnisim.scenes.neck_cmos "$project" --output-dir "$record_dir" --extend-brain-100um --volume-cache "$archive/anterior100-baseline/neck-v888-anterior100.npz" --reference-array "$archive/single-array-sorting/audit.json"
```

不传布局为同占地 196 版。原版使用 `--layout v2-64`。外围扩展对照为 `--side 14 --pitch-um 17 --pad-width-um 12 --center-xz-um 487.2 130.6`。仅指定旧参数 `--pad-width-um` 的历史宽度扫描保留 64 布局语义。默认入口独立重放后，对 cmos-replay、cmos-local-probes、cmos-frontend、cmos-cut-response 四个 NPZ 的全部字段进行了逐值一致性核对。启用 HIP 的全套 92 项测试通过。

原始试验分别位于 /run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/cmos196-dense、cmos196-default-replay 和 cmos196-extended。当前目录保存几何选择、评分代码、对照图和结果。
