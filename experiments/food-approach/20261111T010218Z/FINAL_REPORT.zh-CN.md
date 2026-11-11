# 正向神经读出2000：保住中性和右转，左转滑移仍未通过

最终模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt`

SHA：`eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df`

按根目录PREREG，在旧原2-only慢残差1500父模型194878个参数及128+64记忆冻结后，只训练新增2730个神经读出参数。输入是当前残差hidden64，额外latent为`relu(actualturn)*.25*tanh(readout(hidden64))`，forward0；不读取时钟、步态相位、食品坐标、来源或未来。完整教师left_full2880行因果缓存，prime960不计损失，freshAdam.003固定2000，只实际测最终2000。

| 实际条件 | 最大滑移（要求<=.5） | 实际yaw rad | 原工程结果 | 独立检查 |
|---|---:|---:|---|---:|
| neutral | .49278883265360646 | -.006677607974098819 | 全5门/all6/NF0通过 | 126 |
| left | .5148415850460619 | +.061792138875233306 | 只有滑移失败，其余4门/all6/NF0/方向通过 | 102 |
| right | .49501104818453806 | -.07642541452275556 | 全5门/all6/NF0/方向通过 | 145 |

三实际回合和三审计由父bash14托管，exit0并已收集；`evaluation/FINAL_SUMMARY.json`COMPLETE、saved_trajectory_verificationTRUE、all_conditions_eligibleFALSE。右转完整15400端点身体/动作/几何与2880神经输出/192hidden链逐位复现已通过父右转，不只是架构推断；中性同样复现原通过基线。所有三回合均真实native12.5us积分、完整71opaque、42有限PD、原力矩/速率限，外部力/附着0、无根轨迹写入、交接连续，两侧命令都是1ssmoothstep，评分包含完整9.6s无删窗。

左转相对冻结父left .5380394078384092有改善，但.514841585仍严格FAIL，不能四舍五入/改变分母/门槛。新增容量和正向神经残差上限同时变化，不能把改善单独归因一个因素。尚无完整双向可用接口，未开始食品行为优化，速度未训练。

## 保存与元数据限制

- `train/PREFLIGHT.json`33项实现/单次临时优化检查PASS，原全部参数/状态冻结、负命令和中性功能精确；新head两组梯度有效，生产与因果缓存/梯度核对、负目标Jacobian保持。
- 固定训练bash10exit0，约7.15s。初始verify_readout仅finite/config检查，不能据此宣称强保存核验；后补`verify_final.py`、`audit_saved_full.py`对实际最终2000保存状态做独立19/25项检查，neutral/right完整流精确、实际Adam2counter2000/finite moments/shapes/coverage/reload、RNGroundtrip、curve1..2000和所有父194878参数一致均通过。无追加训练。
- config.parent_checkpoint字段错误地沿用了旧500祖先路径，而configSHA、模型identity及全部冻结权重实际正确对应c6b...1500；`PARENT_METADATA_CORRECTION.json`绑定最终SHA外部更正，不重写checkpoint或历史hash。
- 没有保存step0曲线；记录是1..2000连续，不能声称0..2000。live-before-save完整Adam/RNG/curve匹配记录缺失，后验保存加载一致性不能补造该证据。1000保存后构造加载消耗RNG，更新本身没有随机操作，因此不影响确定性梯度序列，但不声明完美随机续训。
- bash13尝试用final-only审计器验证1000时正确拒绝exit1，未运行物理/优化，也未生成新结果，不影响实际2000审计。

上述限制不改变实际最终模型身份和三条件物理结果。完整原生D1—D8审计、任意切换/初态稳健性、食品到达和因果对照尚未建立，不对这些作扩展结论。

## 下一项有限验证

因拟合曲线在预登记最终2000仍下降，已在新的`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T013852Z/`提前登记同架构/数据/神经范围的精确2000→固定10000续训，只将lr.003降.001，保留实际Adam2/RNG/已记录1..2000曲线。只实际测最终10000，不挑4000/7000。该试验未出结果，不可声称左转解决或后台已成功。
