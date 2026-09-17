# 引言

长上下文模型通常通过固定任务集和若干预设长度进行评测。LongBench 提供双语、多任务长上下文 benchmark，RULER 关注模型的有效上下文大小，NeedleBench 将检索与推理扩展到超长输入 [3–5]。这些工具刻画了模型在标准任务上的能力，但固定分数不一定揭示失败由哪一种压力条件触发，也不直接回答失败是否与模型内部状态内容有关。

这一缺口对状态空间模型尤其重要。Mamba 以 selective state spaces 实现线性时间序列建模，后续 state space duality 工作进一步连接了 SSM 与 Transformer 计算结构 [1,2]。LongMamba 和针对 Mamba recall scaling 的近期研究从感受野、回忆容量和机制角度讨论长记忆 [6,7]。然而，平均性能、理论容量与特定条件下的行为 failure regime 并非同一问题。一个模型可以在多数长度上成功，却在特定 filler、value pair 与预算组合上发生稳定偏转。

StateFuzz 将这个问题表述为受控行为压力搜索。框架生成方向相反但候选集合一致的反事实提示，用 direction-correct signed margin 消除固定候选方向造成的符号混淆，并沿 stress family 与 token budget 搜索失败区域。在发现失败后，框架可以替换 Mamba 的循环状态内容，通过 correct-memory、wrong-memory 与 matched-random controls 区分记忆内容恢复和任意状态扰动。这个流程与 metamorphic testing 对输入变换后行为关系的关注相关 [8–10]，但 StateFuzz 的当前实证范围专门针对 remote-memory probe、failure localization 与状态内容干预。

本文的证据链按“发现—诊断—迁移”组织。Round18 在 Mamba-130M 中发现一个可复现的 structured-repetition failure regime，并以 Pythia-160M 提供配对 checkpoint 参考。Round20/21 在冻结的 Mamba 条件中完成循环状态干预与 fresh-seed replication。后续轮次则检验方法能否迁移到更广压力家族、受控现实任务与 Zamba2 Hybrid checkpoint，同时保留无 failure boundary 的负结果和非单调响应。Round28 mitigation 作为次级证据，用于说明诊断结果可以启发任务特异干预，但不构成通用修复方案。

本文刻意限制结论强度。Pythia 参考没有控制训练数据和架构以外的全部变量，跨 checkpoint 差异不能解释为架构因果；Mamba 状态干预只支持获批的窄条件；Zamba2 cache replay 未通过严格重构，因此 Hybrid 内部路径归因保持未证实。这样的边界使强阳性结果、后期零 failure-boundary cohort 和失败的机制协议可以在同一论文中共存，而不相互覆盖。
