# 结论

StateFuzz 将压力生成、成对行为探测、signed margin 测量和压力表面汇总连接为一个可审计的状态化序列模型评估流程。基于 Mamba-130M、Pythia-160M 和 Zamba2-1.2B-Instruct-v2 的冻结证据，StateFuzz 揭示了随 checkpoint、压力家族和上下文预算变化的非一致、部分非单调 margin-response 轨迹。与此同时，全部已评估 cohort 的 failure probability 均为零，说明当前证据支持描述性鲁棒性表面，而不支持统一 failure boundary。

本文的贡献边界与其方法同样重要。StateFuzz 不提供模型普遍优劣排名，当前跨 checkpoint 差异不构成架构因果证据，Zamba2 结果也不验证 standalone Mamba2、现代 Attention 或所有 Hybrid 模型。通过同时保留负结果、缺失预算和已知混杂，StateFuzz 为后续受控扩展提供了明确起点：新的模型和预算应在预先冻结协议后加入，而不是根据有利结果进行事后选择。
