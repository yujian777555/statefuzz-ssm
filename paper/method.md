# 方法

## 总体流程

StateFuzz 将评估流程分为四个连续阶段：压力生成、行为探测、margin 测量和压力表面刻画。给定一个模型、一个压力家族和一个目标上下文预算，生成器构造一组成对提示，使两条提示需要保留不同的远程记忆值，同时尽量共享其余表面形式。执行器随后在相同候选集合上查询模型，不进行训练或权重修改。分析器把方向一致的候选偏好转换为 signed margin，并按模型与压力家族在不同预算上汇总。Round36 证据表直接编码 observed 与 missing 两种 availability；paper package 再联合 Round37 scope，将 Zamba2 已执行但预算拟合不可达的单元标记为 budget-unreachable，并将 Mamba/Pythia 未纳入历史预算网格的单元标记为 historical untested。该联合映射避免把两类 missing 都解释为行为失败。

## 压力生成

冻结实验包含四类压力家族。structured_repetitive 使用重复结构扩展上下文；periodic_pattern 在周期模式中放置与目标记忆竞争的规律；interleaved_distractor 将干扰内容交错到目标信息之间；lexically_diverse 增加词汇形式的多样性。每类压力通过 token 预算拟合器逼近目标长度。实际 token 数与目标预算同时记录，无法达到预算的单元标记为 budget-unreachable，而不被转换为行为失败。

## 成对行为探测

每个 probe 使用两条方向相反的提示和两个固定候选。对于 Prompt A，候选 A 是记忆一致答案；对于 Prompt B，候选 B 是记忆一致答案。该对称设计把固定候选的词汇偏好与记忆方向分开，使分析能够分别保存两个提示方向的 primitive logit 和 signed margin。历史 Mamba 与 Pythia 证据只保留了方向 margin；Zamba2 证据同时保留了候选 logit 与方向 margin，因此最终表对缺失 primitive 值保持空值，而不进行补算。

## Margin 测量

对每个提示方向，signed margin 定义为记忆一致候选的 logit 减去替代候选的 logit。正 margin 表示模型在该方向上偏向记忆一致答案，非正 margin 表示该方向未维持预期偏好。分析首先在模型、压力家族和预算内部聚合方向 margin，然后以该模型和压力家族最短可用预算的平均 margin 为基准计算 normalized margin retention。这个归一化只支持模型内轨迹描述；由于不同 checkpoint 的 logit 标度与训练条件不匹配，原始 margin 不用于跨模型排名。

## 压力表面刻画

每个压力表面以 architecture role、stress family 和 context budget 为三个索引轴，并保存 mean signed margin、normalized margin retention、failure probability、实际 token 数、seed 和 availability。Round36 表中的 missing 预算以 null 保存，不插值、不连接为观测曲线，也不计入 failure probability；其 missing reason 必须由 Round37 scope 映射为 budget-unreachable 或 historical untested，不能仅凭空值推断。StateFuzz 由此提供的是有限评估域上的行为响应表面，而不是对任意长度、任意 SSM 或任意 Hybrid 模型都成立的通用 failure detector。
