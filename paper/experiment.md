# 实验

## 评估设计

冻结证据按“architecture role × stress family × context budget”组织。模型范围包括 state-spaces/mamba-130m-hf、EleutherAI/pythia-160m 和 Zyphra/Zamba2-1.2B-Instruct-v2。前两者分别作为历史 pure-SSM 与 pure-Attention 参考，仅有 256 和 1792 token 预算、seed 61–62 与 ±16 token 容差。Zamba2 作为已测试的 Hybrid SSM+Attention checkpoint，使用 seed 69–72 与 ±8 token 容差；structured_repetitive 和 periodic_pattern 覆盖 256、768、1792、3584，interleaved_distractor 覆盖 256、768，lexically_diverse 覆盖 256、3584。不可达预算保持为未观测状态。

## 证据覆盖

最终证据表包含 180 行。160 行对应已观测的提示方向，20 行是未覆盖模型—压力—预算单元的空值占位。Mamba 与 Pythia 在四个压力家族中各有两个可用预算；Zamba2 的 structured_repetitive 和 periodic_pattern 各有四个可用预算，而 interleaved_distractor 和 lexically_diverse 各有两个可用预算。所有已评估 cohort 的 failure probability 均为 0，因而 first observed failure boundary 在全部表面中均为空。该结果表示在已测试范围内没有观测到 failure boundary，而不表示边界在更长预算或其他压力条件下不存在。

## 描述性 Margin 轨迹

Mamba-130M 在 1792 token 相对于 256 token 的 normalized margin retention 随压力家族而变化：structured_repetitive 为 0.473，periodic_pattern 为 0.866，interleaved_distractor 为 0.790，lexically_diverse 为 0.407。Pythia-160M 在相同两个历史预算上的对应值分别为 1.000、1.000、1.350 和 1.250。由于两者只覆盖两个预算点，这些轨迹只能说明端点之间的描述性差异，不能确定中间过程是否单调，也不能据此推断 Pythia 普遍优于 Mamba。

Zamba2 展示了更密集但仍不完整的预算响应。structured_repetitive 的 retention 从 256 token 的 1.000 变化为 768 token 的 0.733、1792 token 的 1.706 和 3584 token 的 1.349。periodic_pattern 的对应轨迹为 1.000、0.103、0.638 和 3.058，呈现明显的下降后恢复形状。interleaved_distractor 在 768 token 降至 0.410，后续预算不可达；lexically_diverse 在 3584 token 为 1.045，但 768 与 1792 token 未覆盖。上述数值表明响应并非统一的单调长度退化，但不能把恢复或下降归因于 Hybrid 内部的 SSM 或 Attention 路径。

## 跨架构角色比较边界

跨 checkpoint 比较仅用于展示不同描述性曲线形状。Mamba、Pythia 与 Zamba2 在参数规模、训练数据、指令微调、seed cohort 和 token 容差上不匹配，而且历史证据与 Zamba2 证据保存的 primitive logit 完整性不同。因此，实验不报告全局模型排名，不声称观察差异由架构单独造成，也不把 Zamba2 的 Hybrid Mamba2-Attention 标签视为 standalone Mamba2 泛化证据。
