# 方法

## 受控压力与反事实提示

StateFuzz 将远程记忆 probe 表示为两个方向相反的提示。Prompt A 要求恢复 value A，Prompt B 要求恢复 value B，而两个提示共享候选集合和大部分表面形式。压力生成器控制 filler style、目标信息位置、模板和值对，并通过 tokenizer 将提示拟合到目标 token budget。Round18 主条件固定为 template 1、structured_repetitive、red/blue、target position 0.0 和 ±16 token 容差。无法达到预算的样本记录为 budget-unreachable，不计为行为失败。

## Direction-correct margin 与失败事件

对于 Prompt A，方向 margin 为 logit(A|Prompt A) 减 logit(B|Prompt A)；对于 Prompt B，方向 margin 为 logit(B|Prompt B) 减 logit(A|Prompt B)。每个 seed 的失败事件定义为两个方向中最小 signed margin 不大于零。该定义使正值始终表示记忆一致候选占优，并避免早期实现中对两个方向同时使用 logit(B)-logit(A) 的符号错误。风险曲线按预算汇总 failure count、failure rate 和 Wilson 95% interval；Round18 的确认性 checkpoint contrast 在预声明 1792-token endpoint 使用配对 exact two-sided McNemar test。

## 失败定位与对照

StateFuzz 在同一 seed cohort 上比较多个 structured-repetition budgets，以定位 failure rate 从零转为非零的观测区域。Pythia-160M 在相同 Round18 提示与预算上作为行为参考，但由于 checkpoint、训练与架构变量没有分别控制，它不是架构因果对照。Lexically diverse filler 使用相同 red/blue probe 构成 negative control，用来判断失败是否只由上下文长度增加解释，还是与 structured repetition 的压力形式相关。

## 循环状态内容干预

因果诊断在冻结的 Mamba-130M failure condition 上运行。normal-long 保留原始长上下文状态；wrong-memory 使用与目标值相反的短上下文状态；correct-memory 使用与目标值一致的短上下文状态；randomized-matched 生成与状态均值和标准差匹配的随机状态。endpoint margin 定义为 logit(value B)-logit(value A)，大于零表示目标 value B 的行为恢复。正确状态相对于原长状态和随机状态的恢复差异用于检验内容特异性，而不是仅检验任意 cache 扰动是否改变输出。

## 迁移表面与现实任务

后续评估将生成接口扩展到 structured_repetitive、periodic_pattern、interleaved_distractor、semantic_distractor 与 lexically_diverse，并构建 long-document retrieval、code-context dependency 和 agent-conversation memory 三类受控现实任务。对不同 checkpoint 的 stress surface 使用模型内 normalized margin retention 描述曲线形状，不用 raw margin 对未匹配模型排名。Round36 表直接编码 observed/missing；Round37 scope 进一步区分 Zamba2 budget-unreachable 与历史 untested cells。

## 次级 Mitigation

Round28 比较 memory reinjection、context anchor 和 retrieval reminder。该实验不更新模型权重，只在输入中加入不同形式的记忆提示。由于不同任务和模型上的 aggregate effect 不一致，mitigation 被视为次级、任务特异证据，而不是 StateFuzz 主方法或通用修复算法。
