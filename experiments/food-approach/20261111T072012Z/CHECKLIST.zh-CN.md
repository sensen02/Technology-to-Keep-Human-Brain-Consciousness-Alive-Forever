# 左右实际视频交付检查（S2/S5）

- 已读项目入口；父已读规范、roadmap和tracker。本子任务不改tracker。
- 用户授权：立即交付实际左右视频，由用户目视裁定；不训练、不优化。
- 新证据目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T072012Z。
- 源checkpoint：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/g04_c00/candidate.pt；SHA256 6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f。
- 左：复用完整保存轨迹及独立审核。原slip 0.50573596241638 > 0.5，原工程FAIL保留。
- 右：补一个同一权重完整物理回合；沿用原物理、五门、六腿周期、转角、完整性及祖先right保持审核。
- 资源：开始前无本项目其他物理/渲染进程；最多一个物理或renderer，OPENBLAS/OMP/MKL/NUMEXPR线程均1。实测物理worker NLWP=1，RSS约762MB。
- 视频：保存状态每20ms采样、770帧、50fps实时、15.4s。包含1s settle、4.8s teacher、9.6s ANN；固定世界相机。
- 已完成：右独立93项审核全部通过，1,232,000实际物理步；right祖先完整数组保持all_exact；原五门及六腿完整周期通过。
- 已完成：两片全770帧71mesh不透明与轮廓边界核验、源hash与qpos/qvel/COM/yaw帧身份、ffprobe及770/770全解码。均960×640、50fps、15.4s，各验证退出0。
- 已完成：制作代理使用read_image实际打开左右DECODED_MONTAGE.png，目视完整身体、连续交接、地面参照及阶段标注清楚。
- 渲染失败保留：bash-2在首帧前因framebuffer宽度不足退出1，记录RENDER_ATTEMPT1_FAILURE.json；仅调整内存可视framebuffer并删除空staging，无物理更改。bash-3左、bash-4右随后退出0。
- 桌面副本hash比对和精简DELIVERY.json：由finalize.py串行核验并保存最终结果；只有全部断言通过才输出DELIVERY.json。
- 限制：替代ANN／派生完整身体／有限PD；未做本次原生高频D1—D8验收，不代表生物有效性或食物趋近成功，不代替用户目视裁定。
