# 实验设置

## Track A：失败发现

Round18 使用 state-spaces/mamba-130m-hf 与 EleutherAI/pythia-160m。确认性分析固定 seeds 29–44、structured_repetitive、red/blue、目标预算 1792、实际输入 1791 tokens 和 ±16 token 容差。失败定义为 min signed margin ≤ 0。描述性风险曲线还覆盖 256、1024、1408、1536、1664 和 1920；lexically_diverse negative control 覆盖 256、1664、1792 和 1920。两个 checkpoint 均以 float16 在 CUDA 上运行，Mamba 使用 Transformers sequential implementation，因为 fast CUDA kernels 不可用。

## Track B：因果诊断

Round20 使用 seeds 45–52，Round21 fresh-seed replication 使用 seeds 53–60。两轮都固定 Mamba-130M、template 1、structured_repetitive、target position 0.0，并比较 normal-long、wrong-memory short state、correct-memory short state 与 randomized-matched state。长提示包含 3884 tokens，短提示包含 556 tokens 左右。Round21 除 red/blue failure pair 外，还预声明 cat/dog support pair；由于 cat/dog 的 normal-long 已成功，它只用于状态内容特异性，不用于失败恢复复制。

## Track C：迁移与压力表面

Round25 扩展压力家族，Round26 在 Mamba 与 Pythia 上构建 cross-checkpoint matrix，Round27 增加 long_document_retrieval、code_context_dependency 和 agent_conversation_memory，得到 48 条有效现实任务记录。Round31/34 将行为评估迁移到 Zyphra/Zamba2-1.2B-Instruct-v2。Round35/36 将三 checkpoint 的后期证据包装为 180 行表，其中 160 行为 observed direction records，20 行为 missing-budget placeholders。后期历史 Mamba/Pythia cohort 使用 seeds 61–62 与 ±16 tolerance；Zamba2 使用 seeds 69–72 与 ±8 tolerance，因此只进行模型内曲线描述。

## Track D：Mitigation

Round28 在 Mamba 与 Pythia 上比较三种输入级 mitigation。合成任务包含 16 条有效记录，受控现实任务包含 48 条有效记录。主要观察对象是策略相对于相同模型、相同 task template 的成功率变化，未进行权重更新，也未把不同策略的 token overhead 视为相同成本。

## 证据协调原则

四条 track 使用不同 seeds、预算和问题定义。Round18 回答“能否发现 failure regime”，Round20/21 回答“冻结 Mamba failure condition 是否受循环状态内容因果影响”，Round25–37 回答“框架能否迁移并刻画更广表面”，Round28 回答“输入级干预是否在特定任务上有候选收益”。因此，后期 surface cohort 零 failure boundary 不会覆盖 Round18 failure evidence，Zamba2 replay failure 也不会追溯否定 Mamba native-state intervention。
