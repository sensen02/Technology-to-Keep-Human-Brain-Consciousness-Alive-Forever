# P3/S3 两个中腿侧别目标：flow筛选审计

手动只读研究原型，非成品。无模型/模拟/池替换/增益搜索/在线连接体查询。本轮新问题仅是：未解析中腿左sternal_adductor及中腿右pleural_remotor_and_abductor是否有记录被flow=efferent过滤掉？不是重做14个promotor或两根版本crosswalk。

## 实际结果

输入 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather` SHA25686ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098，188508行；扫描全表目标字段与cell_type关键词（不先筛flow），结果：

|精确目标|全flow记录|efferent记录|middle_leg缺侧全flow/efferent|
|---|---:|---:|---:|
|sternal_adductor_muscle|5|5|left 0/0|
|pleural_remotor_and_abductor_muscle|7|7|right 0/0|

另搜索cell_type/peripheral_target_type含sternal_adductor或pleural_remotor但不同target的行，0条。这里只覆盖这两个精确词干，不能称所有同义词/解剖存在性普查。

两条中腿已有相反侧记录：
- middle/right sternal_adductor，root_id=banc_888_id=720575941644474312，row21565，right_mesothoracic_leg_nerve。
- middle/left pleural_remotor_and_abductor，root_id=banc_888_id=720575941553503460，row12150，left_mesothoracic_leg_nerve。

**结论：在这份精确输入及字段范围内，取消flow筛选不会产生两条缺侧池；不是隐藏在非efferent记录中的可直接恢复输出。** 仍不能说动物缺神经元、两侧功能异名无候选、另release不会补齐。旧VFB/词表计数历史只特定nerve范围，不应与本次全target5/7混为矛盾。

## 额外资格警告（只观察，不推断）

12条精确目标记录的status均含LR_TYPE_CONFLICT。这是来源标签原文，不证明side错，也不证明可自动镜像；相反侧记录名字/nerve侧别相符并不足以消除该标记。旧报告“真正单细胞缺失”应只理解成当前字段下无法声明的侧别池，不当动物解剖缺失。当前74usable/65complete/9partial/16unresolved仍按原声明保持，不修改P1。

## 最小后续与停止条件

不重复词表/VFB候选检索。若继续此目标，只读官方status/LR_TYPE_CONFLICT定义及这两中腿对应cell_type的侧别/目标来源解释，先核已有材料是否覆盖；只在有明确来源时准备行级请求。不能把left/right候选互换、把MANC男性根搬到BANC或改side以增覆盖。若没有可用定义/行级证据，保留两个池unresolved，停止该支线。

## 核查范围

项目根venv_body +已有vendor/pylibs，pyarrow.feather读取11列、全行filter，前后hash一致，退出0；无安装/新测试/全量回归/图或过夜任务。records.json保存12原记录、两缺侧空结果、字符串ID及字段范围，不保存完整全表副本。记录来自本地输入，不称官方动态版本或逐根生物验收。

桌面证据 `/home/sensen/Desktop/engine修复验收/P3-side-target-filter-20261014-v1/records.json`。生成器来源 `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_11_mn_pools.py`119–131/146，flow=efferent及target/leg/side精确过滤；旧综合审计 `/run/media/sensen/Data2/cell_wound_prototype/P3_CONTROL_CHAIN_GAPS.zh-CN.md`78–86。

身体收敛、S3生物身份、S4标定、稳定站立/行走仍未达成。
