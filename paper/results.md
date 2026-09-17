# 结果

## 失败发现与配对 checkpoint 对照

Mamba structured-repetition 风险曲线在 256、1024、1408 和 1536 target budgets 上均为 0/16 失败，在 1664 上变为 3/16，在确认性 1792 endpoint 上达到 12/16，并在 1920 上达到 16/16。1792 endpoint 的实际输入为 1791 tokens，Mamba failure rate 为 0.75，Wilson 95% interval 为 [0.5050, 0.8982]。Pythia 在相同七个预算上均为 0/16；其 1792 endpoint Wilson interval 为 [0, 0.1936]。配对表包含 12 个 Mamba-only failures 和 0 个 Pythia-only failures，exact two-sided McNemar p=0.00048828125。

Lexically diverse negative control 没有复现 structured-repetition failure pattern。Mamba 在 256、1664、1792 和 1920 上均为 0/16 失败，Pythia 在 1792 primary endpoint 也为 0/16。这个结果支持“失败与冻结压力形式相关”，但不能说明其他 filler、模板或 checkpoint 不会失败，也不能把 Mamba/Pythia 差异单独归因于架构。

## 循环状态内容的因果诊断

Round20 的八个 seeds 中，normal-long recovery 为 0/8，mean margin 为 -1.238905；wrong-memory state recovery 为 0/8，mean margin 为 -1.848633；correct-memory state recovery 为 8/8，mean margin 为 2.155273；randomized-matched recovery 为 1/8，mean margin 为 -1.306641。正确状态的 memory-restoration effect mean 为 3.461914，state-specificity ratio 为 0.980434。正确、错误和随机状态的分离表明恢复取决于注入的记忆内容，而不是任意状态替换。

Round21 在新的 seeds 53–60 上复制 red/blue 条件：normal-long 0/8、wrong-memory 0/8、correct-memory 8/8、randomized-matched 0/8，specificity gap 为 1.0。预声明 cat/dog pair 中，normal-long 8/8、wrong-memory 0/8、correct-memory 8/8、randomized-matched 5/8。cat/dog 结果进一步支持正确和错误状态内容的区分，但其原始长上下文没有失败，因此不是 failure-recovery replication。综合两轮证据，获批的窄结论是：在结构化重复的远程记忆压力条件下，Mamba-130M 的循环状态内容会对远程记忆行为产生因果影响。

## 迁移、现实任务与 Hybrid checkpoint

StateFuzz 通过统一接口覆盖多类 stress family，并在 long-document retrieval、code-context dependency 和 agent-conversation memory 三类受控现实任务上得到 48 条有效记录。Zamba2 行为评估表明框架也能应用于指定 Hybrid checkpoint。后期 Round35/36 surface package 的 160 个 observed direction records 没有出现 failure boundary；这只描述该后期 cohort，而不是整个项目的零失败结论。

后期 surface 仍提供非单调响应证据。Zamba2 periodic_pattern 的 normalized margin retention 从 256 token 的 1.000 降至 768 token 的 0.103，随后在 1792 token 回升至 0.638，并在 3584 token 达到 3.058。该曲线反驳“所有响应都随长度单调退化”的简单描述，但不识别恢复来自 Hybrid 内部哪一路径。Round33 的严格 cache replay 验证为 0/40 合规重构，低于 90% 门槛，因此 Round32 path-swap 数值不用于 SSM-versus-Attention 因果归因。

## 次级 Mitigation 结果

Context anchor 在 code_context_dependency 上将 Mamba 从 4/8 提升到 8/8，将 Pythia 从 5/8 提升到 7/8。该策略在 aggregate realistic score 上并非对两个模型都优于 baseline，memory reinjection 和 retrieval reminder 也出现下降或不一致。因此 mitigation 只作为任务特异候选结果，适合次级表格或附录，而不是主论文的普遍修复结论。
