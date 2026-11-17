# 身体重训运行记录

- P1/S2/S3/S5，run_id为20261117T170820Z-body-readaptation。用户直接授权身体重训。新固定颈部、静态读出、体模、初态和CPG演化冻结。PREREG对应PROTOCOL.zh-CN.md。39训练、10开发、12额外条件及纠偏备选在TASKS.json预登记。两端对照按具名脉冲实际过颈，统一原身体关节顺序、尺度和1ms控制，神经/物理.1ms。无PPO、GAE或颈部可塑性。
- 归档Python3.12.14、Torch2.5.1 CPU、MuJoCo3.9、FlyGym2.1、单线程OMP/BLAS/MKL、EGL。任务入口drive.sh。smoke退出0。teacher与teacher_dev各子回合、各自batch退出0，39/39和10/10物理教师通过。训练前冻结350输入哈希，含实际用到的原始递质元数据。两基线桥接证实新evaluate入口与归档所有数组一致。
- 初次桥接在物理完成后生成metadata时将str传给要求Path的hash函数，两个子回合退出1、batch123。已修正，旧失败保留BRIDGE_MISSING_METADATA，两个新桥接和逐值检查退出0。首次verify_inputs因父清单含相对CALIBRATION路径而退出1，按父实验位置解析后输入核验退出0。该修复没有改神经或身体动力学。训练采样前修正Case导入，未影响任何采样回合。
- train.py FIT退出0，30轮每轮保存实际模型、Adam、rng、epoch、数据和日志哈希。实际输出读出和感觉末层的中心差分梯度误差分别约1.0e−16和4.9e−15。最终模型共2,760个AdamW更新；四ier和全部reflex张量变化已直接逐值核验。保存重载的304运动状态最大误差5.60e−8rad。最初10帧静态保存检查保留，更强的实际运动态检查单列MOVING_SAVE_RELOAD。
- 第10/20/30轮各10条无教师开发全部通过，按开发关节RMSE固定第30轮，SELECTION.json先于重训身体正式回合。无需触发纠偏追加，未执行CORRECT批。EXTRA_BASELINE的旧身体对照先采，未用于候选选择。
- regress.sh FORMAL/REPLAY、batch.py extra EXTRA/EXTRA_REPLAY、EXTRA_BASELINE、audit.py、verify_pairs.py各退出0。57既有和12新条件全PASS，69独立复跑所有数组一致，共2,718,000原生步含新条件旧身体配对。20组同源DN因果配对全部通过，读取的权重身份与磁盘所选一致。旧57组基线由父档案保持，额外12条旧身体重新在本轮运行。
- sensitivity.sh和汇总退出0，30种固定条件子进程全部正常，但身体指标29/30。漏电导+20%的DNg60为明确FAIL，RMS.8953mm/s，未按该失败修改训练或门限。Eon权重、固定颈部与原身体资产均通过终局SHA核验。
- MPLBACKEND=Agg、plot_training.py、原地面render_ground.py、ffmpeg全帧解码、ffprobe退出0。100帧、30fps、0.60x真实六栏同次物理，已实际打开训练图、响应图和物理关键帧。报告、权重、结果和视频优先交桌面并逐字节cmp。完整D1–D8和未见地形未在本轮评估。
