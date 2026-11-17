# 真实RGB帧库：112组双眼输入已生成并独立核验

父唯一托管bash73实际EGL生成已收集exit0。使用既有完整71opaque身体，从5个保存的真实中性姿态恢复后仅mj_forward，不积分；固定工程双眼参数沿用已验证camera.py，不改XML、碰撞或身体数组。112组双眼32×32×3 uint8，对应224次实际RGB渲染。

训练几何T0—T3各3个保存姿态、留出几何V0—V1各2个未用于相机首probe的姿态。每组合包含off/center/left/right/occluded_left/occluded_right/occluder_only七场景。训练序列只使用预先固定的T0pose0/T1pose1/T2pose2/T3pose0，验证只V0pose3/V1pose4；其余图像用于光学审阅，不假称增加独立训练轨迹。

独立新进程audit exit0，340项检查全部true。全部左右和中心目标与空场像素有实际差异；遮挡左右目标与同一遮挡物无目标的像素逐位相同，target_visible=false来自实测而非强行标签。所有row的原RGB SHA、文件SHA、配置/pose/split/眼顺序与实际物理身份逐项核验。物理步数0、身体compiled参数前后一致。

LIBRARY SHA256：e219edd435cefbdf7bd44502a08a4c2d32bab5f87263fb70d046cdc462840772。
RGB.npz SHA256：929977216b2942b91d08f3b40761c13e1aaff692a72faa05519a33de3c39ee87。

权威证据：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/frame_library_v1/LIBRARY.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/frame_library_v1/VERIFY.json`

父实际打开图像总览，并桌面copy2/sha核验：
`/home/sensen/Desktop/engine修复验收/视觉实施-20261112/真实双眼帧库-目标与遮挡总览.png`
SHA256 55f410c669071947ba6e962ade810a0ea592ed59a2daa303d7e52b8105c493cd。

这是静态头部图像钳制预训练库，不是身体运动中的动态视觉反馈。后续神经训练部署只RGB，场景坐标/方位/visibility/time只离线教师标签或审核，不进入actor，不通过颜色质心或几何规则生成动作。下一步数值预检与有限训练仍待完成，视觉行为或导航成功尚未建立。
