# 相关工作

## SSM 与 Mamba 长记忆

Mamba 通过 selective state spaces 在输入相关的状态更新中实现线性时间序列建模 [1]，Mamba-2/structured state space duality 进一步从算法与结构角度连接 SSM 和 Transformer [2]。LongMamba 讨论无需重新训练的感受野扩展 [6]，近期的 Mamba recall scaling 研究则从 hashing 视角分析回忆容量与机制 [7]。这些工作关注模型结构、有效感受野或 recall capacity；StateFuzz 的定位不同，它搜索受控 filler、value pair 与 budget 组合中的行为 failure regime，并在发现后使用状态内容干预。对 2026 recall 工作的比较仍需完整新颖性核查，因此本文不作优先权判断。

## 长上下文 Benchmark 与有效上下文

LongBench 通过双语、多任务集合评价长上下文理解 [3]，RULER 估计模型在多类合成长上下文任务中的实际可用窗口 [4]，NeedleBench 将检索与推理扩展到百万 token 场景 [5]。它们提供固定 benchmark score 或 context-size diagnostics。StateFuzz 不以扩大 benchmark 覆盖为主要目标，而是把 stress condition 视为可搜索变量，以 counterfactual pair 和 direction-correct margin 定位 failure transition。两类方法互补：benchmark 描述标准任务能力，targeted stress discovery 寻找平均分数可能掩盖的条件性 failure regime。

## Metamorphic 与行为一致性测试

Manino 等从 systematicity、compositionality 与 transitivity 角度将 metamorphic testing 用于深度 NLP 模型 [8]。后续工作研究知识一致性测试 [9] 以及面向 NLP 的 LLM metamorphic testing [10]。这些方法通过输入变换与预期关系检测行为不一致。StateFuzz 同样依赖成对输入关系，但当前探针专门固定 remote-memory 值与候选方向，沿压力与预算搜索 failure surface，并在 Mamba failure condition 上追加 recurrent-state content intervention。该差异是方法定位，不是未经核查的优先权主张。

## 检测与诊断的边界

固定 benchmark、recall 分析和 metamorphic testing 均可发现性能或一致性问题，但行为检测本身不等于机制归因。StateFuzz 的 Round20/21 通过 correct-memory、wrong-memory 和 matched-random states 建立冻结 Mamba 条件内的内容特异性；相反，Round33 的 Zamba2 cache reconstruction 失败使 Hybrid 内部路径归因保持无效。这一区分是论文的关键边界：检测可以跨模型迁移，因果诊断必须逐模型、逐协议验证。
