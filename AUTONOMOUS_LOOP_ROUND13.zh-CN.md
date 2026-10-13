# 第十三轮：分离兴奋/抑制传播对照与上游PDF

目标未完成，手工仿真原型，飞行暂缓。未改正常网络参数或增加行为控制。

## 真实身体反馈的隔离干预

入口新增显式 --split-propagation-cuts，仅在指定时增加excitatory_cut(We=0,Wi不变)和inhibitory_cut(Wi=0,We不变)；记录各矩阵非零数。每组新建网络/身体，不遮蔽MN读出，不回写原连接。

0.3秒实测：完整125018脉冲；全传播切断823；仅兴奋切断823，全部命令与身体轨迹与全切断逐值一致；仅抑制切断770193脉冲约为完整6.16倍，主动命令饱和，最终高0.72986mm仍塌陷。六组360区间命令来源检查通过，矩阵互斥切断检查通过。图已核看。

这证明当前模型抑制通路确实在限制活动，不能声称是抑制未生效造成持续放电；也不证明抑制增益符合生理。后续不能为了站立随意增加抑制权重。新对照不是正常行为设置。

结果 /run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_split_round13/ 。

## 公开PDF的实际内容

已下载固定commit文件 https://raw.githubusercontent.com/gizemozd/FlyMimic/9ea1131626cd76f7203b74076ef8f0e9cab30bef/flymimic/assets/models/opensim/best_combined_full.pdf ，共63页，1,452,250字节。pdfinfo和完整pdftotext检查确认标题为MyoConverter Model Validation Report，不是原始分割/神经支配表。第一页已渲染查看，是OpenSim→MuJoCo运动学/力学转换误差摘要；全文可提取文字无extensor/sterno/tergo字符串（图中标签可能不进入提取，不能据此断言全部页面无任何标签）。没有从PDF取得a/b神经支配依据。

摘要打印cm和Fmax误差，但其模型缩放/单位语境未核清，不将厘米数直接移植为毫米身体实际误差或据此盲改缩放。

保存 /run/media/sensen/Data2/cell_wound_prototype/outputs/upstream_muscle_round13/model.pdf、model.txt、overview.png 。初始curl不可用，改用已有Python urllib成功获取，未安装工具。

## 结论

本轮落实的是可复用的分离传播干预与上游资料排查，不是站立修复。关键未实现的肌肉/感觉对应仍在。保持目标未完成。
桌面 /home/sensen/Desktop/cell_wound_prototype/autonomous_round13/ 。
