# 固定动力学运行记录

- P1/S1/S3/S5。实际 UTC 开始20261117T164207Z。原临时目录20261118T-fixed-neck保留软链接，完工后已改为实际时间戳目录，RELOCATION.json记录，旧绝对输入路径继续可读取同一字节。实验期间全部输入固定。人工脑项目的根级NECK_CURRENT_STATE.json不调整。
- 先读AGENTS、roadmap、tracker、当前颈部运行源码，git status保留现有用户修改。候选设计DESIGN先落盘。SWC按注册切面和父子连续分支量弧长，首轮使用平面裁剪，后在正式前改为完整连续分支弧长，PARAMETERS_INITIAL及BEFORE_PATH_REFINEMENT保留。52根路径、原图、预设速度、端点范围和参数依据在PARAMETERS.json逐项登记。
- 环境归档Python3.12.14／Torch2.5.1 CPU／MuJoCo3.9／FlyGym2.1、BLAS/OMP/MKL单线程、EGL。无PPO、GAE或可塑性参数更新。神经和物理.1ms、动作1ms。正式前freeze.py核对242项依赖与体模、52SWC、原脑发放、静态读出和固定参数，最终复核一致。新模型使用的原始递质元数据漏入本次跑前清单，事后与旧源审计已固定的86ccf5...0098一致，文件mtime/ctime均在2026-10-11，见METADATA_IDENTITY_AUDIT，未伪作新增跑前登记。
- 参数生成、calibrate.py十三组神经标定、静态收集、DEV_RAW/DEV_MAP十条件均退出0。DEV_RAW DNg60行为FAIL，DEV_MAP全通过后选择。verify_dynamics初次错误地要求短弱受体输入必须让每条AN放电而退出1，保留。修正以实际发放逐根核对，并补持续受体输入，52条实际神经源均有过颈及恢复证据。核面积解析/数值、不应期、兴奋/抑制反转、静息恢复和.1/.05ms比较退出0。ASCENDING持续五通道、REVERSAL检查退出0。
- run_formal.sh FORMAL/REPLAY退出0，57条各自完整无教师关节介入，所有原通过动作工程门通过。物理共2,034,000步。audit_regression首次把含wall-time的RESULT.json摘要当输入身份比较，退出1，原失败保留。随后逐项核对输入身份、各自RESULT摘要以及57对存档所有数组，退出0。行为门与结果没有重写。单切、全断、恢复按各轴突delay_ticks逐项独立复算，16个额外切断配对和4个两速比值全过。
- sensitivity.py三类固定不确定参数、各两个范围共30条子进程退出0，29条行为PASS，漏电导1.2nS下DNg60停稳FAIL，summarize结果保留。无根据敏感性失败继续改正式模型。DNg60下一步需要膜参数约束或独立适配数据。所有旧通过及失败权重保留。
- plot_dynamics.py、原地面render_ground读取同次新轨迹、FFmpeg全帧解码、ffprobe均退出0。100帧30fps，2秒物理、3.333秒播放(.60x)，6栏完整和切断。实际打开DYNAMICS.png、MEDIA_GROUND/RESPONSE.png、MEDIA_GROUND/PHYSICAL_REVIEW.png。媒体未修改物理记录。桌面文件逐字节cmp和git diff --check退出0。
