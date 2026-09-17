# 局限性

最强的 failure-discovery 与 causal-diagnosis 证据集中在一个 Mamba-130M checkpoint、一个 structured-repetition template 和一个 red/blue failure pair。虽然 Round21 使用 cat/dog 检验状态内容特异性，但该 pair 的 normal-long 已成功，因此不能作为第二个 failure-recovery replication。若要推广到其他模板、值对、任务或 SSM，需要新实验；当前稿件明确不作这种外推。

Pythia-160M 是 Round18 的配对行为参考，而不是架构因果控制。Mamba 与 Pythia 的训练数据、架构和 checkpoint 属性未被逐项匹配；Zamba2 又具有更大规模和 instruction tuning。后期历史 cohort 与 Zamba2 cohort 的 seeds、token tolerance 和预算覆盖也不同。因此跨 checkpoint 曲线只能描述行为差异，不能形成模型全局排名或架构因果结论。

后期 transfer evidence 存在覆盖和独立性限制。Mamba/Pythia surface 只有部分预算，Zamba2 的若干单元 budget-unreachable，periodic_pattern 不随 seed 变化提示文本。Round36 的 missing cells 必须联合 Round37 scope 才能区分 historical untested 与 budget-unreachable；二者都不是行为失败。后期 surface cohort 的零 observed failure boundary 只适用于自身观测范围。

机制证据也具有模型和协议边界。Round20/21 的 native Mamba recurrent-state intervention 支持冻结条件内的窄因果 claim，但不证明所有 SSM 都以相同方式存储记忆。Round33 对 Zamba2 的严格 cache replay 验证为 0/40，因而 Hybrid SSM-versus-Attention path attribution 未建立。这个失败不否定 Mamba intervention，却阻止将其机制迁移到 Hybrid 模型。

Round37 没有在冻结 VM scan root 中发现 standalone Mamba2 或新的现代 pure-Attention checkpoint。缺失资产不是模型质量结论，但限制了架构覆盖。除此之外，当前 related-work positioning 只基于计划冻结的 literature seeds；LongMamba、2026 Mamba recall study 和近期 LLM metamorphic testing 仍需 dedicated full-text novelty audit。主图尚未渲染，稿件也需要目标 venue 的英文格式化，因此当前版本不应标记为可直接投稿。
