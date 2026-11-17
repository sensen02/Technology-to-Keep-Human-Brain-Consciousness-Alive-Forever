# 固定影子教师：停止支撑／左右原地转向三段实际视频

ONLYCODE实际36项标准库检查全true、exit0，完整renderer及嵌套函数26个全局名称全部可解析；0数组payload、模型、Core、渲染、forward、训练或物理。仅读取NPZ内NPY头部确认真实schema，ACTUAL_ARCHIVE_HEADERS.json记录；执行源码自检查后冻结，本文纯说明不进source。

三实际源：stop_shadow_control/trials/stop、pivot_fixed_shadow_v2/trials/left_pivot和right_pivot。实际kind分别STOP_FIXED_SHADOW_MATCHED_CONTROL_V1／FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2，control均ALGORITHM_TEACHER_ONLY_NOT_ANN。资格保持stop2stage、pivot4stage全exit0／实际零积分PREFLIGHT／独立26checks／MATCHED PD-RNG-firstaction-prefix／全源与artifactSHA。不用旧第一次失稳教师作成功视频，不覆盖原失败。

父资源释放后，唯一CPU1先绘图：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_evidence/plot_sync.py --execute-after-parent-release
```

再父managed串行三片；以下stop，后续依次left_pivot、right_pivot只改condition与不同日志：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_evidence/render.py --condition stop --execute-after-parent-release > /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_evidence/RENDER_stop.log 2>&1
```

输出VIDEO_<case>/ACTUAL_ALGORITHM_TEACHER_STOP_PIVOT.mp4、VIDEO.json、FRAME_MAPPING.json、FFPROBE.json、DECODED_MONTAGE.png。每片完整1s稳定＋4.8s walkingteacher预热＋9.6s algorithmcandidate，总15.4s／770frames／50fps／1280×720。全身71opaque、禁止mj_step积分、逐帧qpos／qvel／COM、已记录head／thoraxrotation/yaw重建、完整轮廓不裁、source和编译参数前后精确、全770解码；父必须实际开三蒙太奇和同步图、核桌面副本SHA后交付。

这里geometry_1ms只有14400行，从1.001s到15.4s；Trace15400从.001s开始，teacher_samples2880从1s开始。映射tracei>=1000→geometryi−1000，首settle没有独立保存head字段，所以由同次savedqpos恢复FK并在元数据注明，不能假称15400条head都已archived核。5msteacher索引floorclosestpast，不用未来指令。首视频帧是实际.001秒不是t0。

标题持续ALGORITHM TEACHER — NOT ANN/NECK PIVOT；没有任何真实眼图／NNACh／颈部神经状态，禁止伪造。stop只表示已验证9.6秒支撑，不造学习停止或speedPASS；pivot新预登记10门实际通过但属教师示范，不是ANN部署通过。原五门按实际RESULT逐门展示且明确不是原地转任务的成功定义，不把实际leftlegacy5PASS错误标成FAIL。

侧栏为实际yaw／COM漂移／L_body、teacher请求yawrate单列、实时足载／原生NF峰与额外力、actualeventtime累计六腿周期与FINALcounts。L_body由compiled头胸腹几何长度定义，native2.902405...，SI／生物标定未知，不能写米。左右约74°且COM漂移约.32L_body是近原地旋转示范，不是位置锁住。主体仅旁观candidate起点COM地面小标，不修改物理根位置或轨迹。同步图用真实yaw／drift／support／事件时刻周期，不手画姿态。
