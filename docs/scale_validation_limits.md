# Round30 规模验证边界

## 已验证

- `state-spaces/mamba-130m-hf`：已有完整压力族、因果干预和现实任务证据。
- `EleutherAI/pythia-160m`：作为行为 Transformer 控制，未将 KV cache 等同于 recurrent state。

## 本轮未完成

计划中的 Mamba-370M、Mamba-790M 和 Mamba2 没有可用权重。VM 共享盘未发现对应 checkpoint；Hugging Face 连接超时；ModelScope 客户端安装因 DNS 无法解析 `pypi.ngc.nvidia.com` 失败。

## 结论限制

当前不能判断压力行为是否随 SSM 规模保持、减弱或增强，也不能声称大模型与小模型具有相同机制。后续获得权重后，应沿用相同 family、seed、tokenizer 计数和干预协议重新验证。
