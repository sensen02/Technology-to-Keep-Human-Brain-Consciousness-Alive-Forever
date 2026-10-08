# 一体化项目工作台服务

入口：http://127.0.0.1:8766/ 。这是项目服务，不是DeepSeek Harness聊天页面。旧file://入口停用并提供服务链接，避免加载过期离线副本。

启动：`python3 /run/media/sensen/Data2/cell_wound_prototype/serve_workbench.py --port 8766`
桌面：`/home/sensen/Desktop/cell_wound_prototype/bench_visualizer/启动工作台.sh`

## 真正接入后端的功能

- GET /api/data：当前项目导出几何、历史模拟状态与最新台架评估。
- POST /api/bench/evaluate：提交配置JSON和测量CSV，经现有严格评估器检查并保存证据记录；非法输入400，不冒充通过。
- GET /api/status：项目服务、实际任务状态和最新台架结果。
- POST /api/jobs/recording：启动真实记录预算脚本，单任务并发，2GiB地址空间上限，120秒超时。
- GET /api/recording：读取实际计算结果，不是动画生成的假结果。

只监听127.0.0.1，不开放项目目录。静态资源白名单、Host/Origin检查、请求2MiB限制、未知API拒绝。不是公网多用户服务器，不承诺完成安全审计。

## 没有实现或不应混淆

身体/神经/上皮仍是历史回放与工程演示，不是实时生理联动；独立坐标和时基仍没有配准。后端任务当前接入记录噪声预算，不是点击电极就重新运行全脑或组织力学。真实硬件尚未连接，台架没有实测数据时保持NOT_RUN。页面显示任务进度而不是伪造实时模拟。

服务恢复不自动重启旧任务；重启后任务状态idle。状态输出和任务日志位于项目outputs/workbench。当前视觉模型不支持图像输入，浏览器截图可生成但未完成目视审查。软件是手工研究原型。
