# S4 真实2018钙观测适配协议（适配/预测前声明）

仅数据资格准备，不拟合、不运行作者模型、不修改生产接口或根身份。原Parquet字节保持不变。没有声称真实spike、nA标定或生物验证。

输入仅桌面S4-calcium-input中claw_magnet_Mamiya2018.parquet和hook_flexion_01_magnet_Mamiya2018.parquet，receipt摘要必须匹配。每文件动物编号与trial/ROI必须namespace化，不能把claw的animal1与hook的animal1未经证据合并。每组按来源行序保留，校验time严格递增/有限、analyze全1（否则拒绝而非静默剔除）、同ROI/动物/录制规则。

按动物划分，预声明每文件排序后的最高2个动物编号为holdout，其余train。claw holdout9/10，hook holdout13/14。无validation，不选模型、不看留出拟合效果。此拆分仅项目预声明工程资格，不能将作者曾用同数据调参的模型再算独立留出。数据关联不保证不同文件是不同真实个体，namespace只防止错误合并。

角度原degrees显式np.deg2rad至rad，原time秒保持，不取ceil采样率、不重采样/插值、不平移零点。角度符号与侧别保留来源定义待查，不把角度减90或原始绝对夹角映射本项目Δq。钙值原样保留，不额外最大值/全数据归一化，荧光指标/处理按准确方法证据，否则unknown。roi_population，不填BANC根，禁止spike字段。

sync所有未核公共时钟/误差/变换字段保持unknown；同表角度与荧光time使用来源配对列不是证明硬件同步误差零。单个契约最多100000数组元素；总数据大于预算，按每文件、split、animal独立分片，每片仍包含全部该动物ROI/试次，不截断。单片缺train/holdout不能直接套现有要求两边的契约，应将按文件train/holdout拆为成对的原始记录小包（同动物整体split保持一致），每包至少一train一holdout且总点预算内；若不支持预算保留失败，不放宽现有资源上限。

可选更简实现：每组1trial合成一个包并配一个来自相反split的组，允许同一原记录跨包出现但必须在全局split保持一致；独立全局登记表核重复记录是否一致，不把打包复制计作独立动物/样本。总报告原分组数与原始行数按唯一身份计，不用包内重复计数冒充。

输出新唯一目录，逐包资格报告及全局动物/录制拆分报告，calibration_ready/biological_validation始终false。独立核输入摘要、每包角度转换/原钙值/时轴、全局不泄漏，真实观测图实际查看。失配/非有限/侧别方法冲突保持缺口，不拟合制造表现。
