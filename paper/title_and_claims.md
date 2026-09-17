# 工作标题与主张边界

## 工作标题

**StateFuzz: Discovering and Diagnosing Remote-Memory Failure Modes in Stateful Sequence Models**

## 核心主张

StateFuzz 通过受控压力生成、反事实成对行为 margin、失败区域定位和可选状态干预，将远程记忆评估从固定 benchmark 分数扩展为“发现—诊断—迁移”的证据链。主发现限定为 Round18 冻结协议中的 Mamba-130M structured-repetition failure regime；Pythia-160M 只提供配对行为参考，不构成架构因果控制。

获批的窄因果表述为：

> Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content causally influences remote-memory behavior.

其中文含义是：在结构化重复的远程记忆压力条件下，Mamba-130M 的循环状态内容会对远程记忆行为产生因果影响。该表述不外推到所有 SSM、所有值对、所有任务或所有长上下文失败。

## 次级主张

后续证据表明 StateFuzz 可以迁移到更广的压力家族、三类受控现实任务和指定的 Hybrid Zamba2 checkpoint。后期 Round35/36 surface cohort 在其观测范围内没有 failure boundary，但这不覆盖也不否定 Round18 的独立 failure-discovery cohort。Round28 的 context-anchor 只作为任务特异 mitigation 证据。

## 明确排除

本文不声称所有 SSM 具有相同 failure boundary，不声称 Mamba 普遍劣于 Pythia，不把未匹配 checkpoint 的行为差异归因于架构，不把 Zamba2 当作 standalone Mamba2 泛化证据，也不进行 Hybrid 内部 SSM-versus-Attention 路径归因。任何优先权表述均标记为待新颖性核查，当前不声称方法优先权。
