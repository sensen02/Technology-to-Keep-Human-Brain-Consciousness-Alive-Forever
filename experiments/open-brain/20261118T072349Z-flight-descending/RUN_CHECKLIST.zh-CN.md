# P1 飞行实施与复验

- [x] 前置读取 GOAL_ROADMAP、GOAL_TRACKER 的飞行与当前 P1 状态。保全当前工作树既有修改。实验日志只写在本目录，研究追踪按交接要求另行追加。
- [x] 新验收标准先记录到 PROTOCOL。两种初始拍翼相位，具名同发放切断组、不同剂量、共发放、关闭气动对照以及逐步复算。
- [x] 飞行学源 FlyBody d015e9bfe441bd90ae431bac24c55cb74bdbce26，公开飞控资产取归档中的 SavedModel 和 fmech 翼循环。脚本没有写入上述资产。颈部使用 BANC v888 结构与归档固定参数，受试参数与来源哈希在 NECK/IDENTITY.json。所有脑端定时刺激按 Eon 映射身份命名。
- [x] Python 3.12，颈部用 canonical 的 banc-walking-recovery/20261110T125712Z/env，飞行用 open-brain/20261117T051249Z-bidirectional/flight_env，MuJoCo 3.1.6、TensorFlow 2.17.1、TFP 0.24。两环境通过经过 SHA-256 校验的 NPZ 交接。各依赖的绝对路径见 REPLAY.sh。
- [x] probe_neck.py 两次，fly.py 三批正式测试及种子 29 独立重放，legged_fixture.py、legged_air.py、transition.py A/B/C，assess.py、pictures.py 均退出 0。验证 100800 个原版正常步骤、39200 个有腿正常步骤另计。种子 29 的四组独立重放逐数组相等，颈端两个 NPZ 逐字节一致。无训练。
- [x] 失败结果保留。早期运行时改地面 geom 掩码没有更新编译的 body 掩码，因此改为复位编译前设置并重新生成 VALID_* 与 LEGGED_COLLIDABLE。仅正式目录用于接触结论。初始无腿参考未提供起降。
- [x] 初始调参 SMOKE_* 表明只差动振幅或翼俯仰不产生持续转弯，TRANSITION 与 TRANSITION_RETEST 留下早期落地碰撞。转向读出最终采用参考角速度。起降阶段按 A/B/C 分初态报告，未把更困难初态计作通过。
- [x] 实际查看 AIRBORNE.png、AIRBORNE_POSES.png、TRANSITIONS.png、TRANSITION_POSES.png。用户要求的桌面位置保留各图、报告和复算结果。
- [ ] 针对未验的完整起飞与着陆，在有腿模型上衔接支撑、蹬伸、拍翼与姿态接管，并以全程无复位验收。飞行上行感知、全脑自主选择和长期扰动适应尚未排程。
