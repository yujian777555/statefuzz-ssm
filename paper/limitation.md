# 局限性

当前证据的首要限制是 checkpoint 覆盖不足。冻结范围只有一个历史 pure SSM、一个历史 pure-Attention 参考和一个指令微调的 Hybrid checkpoint。Round37-A 在固定 VM 扫描根目录中没有发现 standalone Mamba2 或新的现代 pure-Attention checkpoint，因此本文不能验证 Mamba2 泛化，也不能将 Pythia-160M 的结果推广到现代 Attention 模型。该不可用性是资产条件，不是模型质量结论。

不同 checkpoint 的实验条件并不匹配。Mamba 与 Pythia 使用 seed 61–62、±16 token 容差和两个预算点，Zamba2 使用 seed 69–72、±8 token 容差并具有不均匀的预算覆盖。模型规模、预训练数据和指令微调同样存在混杂。模型内 normalized margin retention 能减少绝对 logit 标度差异，但不能消除这些混杂，因此架构角色相关的曲线差异不构成架构因果证据或通用模型排名。

压力生成也存在已知限制。periodic_pattern 不随 seed 改变提示文本，因此四个 seed 不是四个独立文本样本。interleaved_distractor 与 lexically_diverse 的若干 Zamba2 预算无法达到拟合容差，Mamba 与 Pythia 也缺少 768 和 3584 token 观测。这些单元被保留为 null；它们既不是失败，也不能用于支持预算外不存在失败的结论。

最后，当前所有已评估 cohort 的 failure probability 都为零。这个负结果使 StateFuzz 能排除“在已测试范围内已经观察到统一 failure boundary”的说法，却不能证明 failure boundary 永远不存在。早期 Hybrid cache replay 未通过严格重构验证，因此本文也不使用其结果进行 Hybrid 内部 SSM-versus-Attention 因果归因。后续工作需要在预注册条件下增加匹配规模与训练设置的 checkpoint、独立压力模板和更完整的预算网格，然后再检验架构因果与边界稳定性。
