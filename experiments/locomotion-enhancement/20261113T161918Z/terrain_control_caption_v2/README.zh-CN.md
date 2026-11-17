# 平坦控制组字幕修正 v2（ONLYCODE，S2/S5）

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_control_caption_v2` 只修两个已完成控制组的三处字幕。旧完整 render.py AST 仅替换 header 最后元素、bottom 元素1、末行 footer 字符串；完整模块在独立 namespace 编译，依赖显式含 hashlib/json/argparse/os/sys/subprocess 及原 contracts 全部导出。compiled main 的 HERE 指向新目录；原 contracts 函数 __globals__、own_sources、全部三案例 admission 均保持旧目录，不通过 sourceclone 误写旧输出路径。

header 显示平坦表面 z=0，无本控制组粗障碍，SI NOT_ESTABLISHED。bottom 从实际 RESULT 读取 functional_stable_body、autonomous_cycles.all_legs_completed、engineering_pass 显示 Original5 PASS/FAIL，粗任务 N/A。footer 明确 block00/01 仅属于另一粗地形回合。原 rough_terrain_pass=false 仍原样写入 VIDEO 元数据；不修改科学判定。

实际两个 RESULT：plane slip=0.48924928654563793，flat_box slip=0.4892493307288685；原五门/六腿周期/身体稳定均 true。旧失败 JOBS 的第五项 exit1 原样保存。两个实际 XML SHA 分别 b67a690b2dca2a6a9a7af78a080e16bcee4d68f6613dc2bd8df6b7921f273995 和 7a8fea5629facbe90dd6dc0631b19ddefc98c748b65127311f3d29605672b43d。

本 child 只标准库 AST 编译、完整递归 LOAD_GLOBAL/LOAD_NAME 解析、JSON 与原始文件哈希；未执行数组、NumPy、Torch、MuJoCo、Core、渲染或神经 forward。新 CODECHECK.json 的 source_hashes 包含原 admission 全源、实际 runtime/terrain_core、原视频全部产物与新 PREREG/render/check/README；CODECHECK 自身在父实际 load 时加入完整前后哈希。旧 sourcecheck 不修改也不重跑会因已存在输出而失效的旧 nooutputs 条件；原准入保持旧 CODEONLY_CHECK 全资格。

父资源释放后，使用以下现有环境命令**串行**执行。每条命令内部固定到可用 CPU 的最小编号，1 CPU；LP/NUMEXPR 等均为1，使用 EGL。不得并行启动。目录已存在时旧代码 exclusive mkdir 会拒绝覆盖，保留原失败证据后再决定独立恢复，不能删除旧视频。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_control_caption_v2/render.py --condition plane_control --execute-after-parent-release

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_control_caption_v2/render.py --condition flat_box_control --execute-after-parent-release
```

两个新输出分别在：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_control_caption_v2/VIDEO_plane_control/VIDEO.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_control_caption_v2/VIDEO_flat_box_control/VIDEO.json`

每组完整保留旧 schema：VIDEO.json、FRAME_MAPPING.json、FFPROBE.json、DECODED_MONTAGE.png、ACTUAL_DERIVED_TERRAIN_NEURAL_BODY.mp4。旧 metadata key 与所有数值不变，只路径/新 source ledger/字幕引起的像素与产物哈希改变。原门/运行时 canonical compiled model/实际派生 TerrainCore/实际 RGB/336神经状态/相机/所有帧映射保持，770帧、50fps、15.4秒、1280×720、0积分/0NNforward；渲染内仍执行逐帧 pose/COM/head/yaw 检查与全770帧解码。

父须收集两个 exit code，实际打开两个新 montage，与原 FRAME_MAPPING 逐条比较（唯一预期改变字段 renderedRGB_sha256），核 VIDEO key schema、body_silhouette_bounds、actual XML、各科学数值、身份和全 source_hashes 不变；将原 ledger 视为新 ledger 的严格子集，新增文件逐 SHA 核。原真实 MP4 与图像必须保留。字幕区域外真实场景/RGB来自同一完整旧 AST，可在父执行后额外解码核对可见场景；有差异如实报告，不重设物理数值。显示修正不算运动增强已成功，粗地形 slip FAIL 与独立 NN 行为失败由父继续主线。
