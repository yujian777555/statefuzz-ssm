# StateFuzz-SSM Milestone 1验收报告

## 范围

本阶段仅交付CPU侧StateProbeIR核心：严格schema、规范化哈希、精确oracle、确定性编译器、符号重命名和模板切换。未下载模型、未占用GPU、未实现搜索器或StatePolicyIR。

## 验收条件

- pytest：18 passed，0 failed，0 errors；
- 100个不同seed生成100个不同probe hash；
- 编译器不修改Python全局RNG；
- 相同ProbeSpec完全确定；
- 两种变形保持精确答案；
- 仓库没有旧项目资产。

## 验收证据

- JUnit XML：`runs/milestone1/pytest.xml`；
- 全量测试输出：18项全部通过；
- 100-seed检查输出：`COMPILE_100_PASS`；
- `src/`与`tests/`隔离检查：`ISOLATION_CHECK=PASS`。

## 计划偏差与说明

- 为避免缺失模块导致pytest收集错误，三个功能模块增加了“模块存在性”红阶段守卫，因此实际测试数由计划的15增至18；
- 独立`python -c`检查显式设置`PYTHONPATH=src`，因为源码包未安装到共享环境；
- VM未安装`rg`，隔离检查使用系统已有`grep`扫描`src/`和`tests/`，未安装新依赖；
- `context_tokens`在Milestone 1仅为模型无关的目标长度预算，精确tokenizer计数与裁剪留到Milestone 2，不将字符单元数报告为真实token数。

## 下一步

只有本报告与测试证据一致后，才编写Milestone 2计划：原子缓存、参数域隔离、fake/HF后端、Mamba2Mixer/FalconH1Mixer状态hook和Gate 0 GPU冒烟。
