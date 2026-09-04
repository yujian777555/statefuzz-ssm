"""Mamba/Mamba2风格模型的可注入执行器。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from statefuzz.probes.compiler import CompiledProbe
from statefuzz.runner.base import ProbeRunner


@dataclass(frozen=True)
class MambaExperimentConfig:
    """真实模型实验所需的可复现配置。"""

    model_id: str
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    max_new_tokens: int = 8
    seed: int = 0
    trust_remote_code: bool = False
    local_files_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }


def _copy_hidden_state(value: Any) -> Any:
    """对张量执行脱离计算图的CPU复制，对其他对象执行深复制。"""
    if hasattr(value, "detach") and hasattr(value, "clone"):
        detached = value.detach().clone()
        return detached.cpu() if hasattr(detached, "cpu") else detached
    return copy.deepcopy(value)


class MambaRunner:
    """通过预测函数注入模型，避免在协议层绑定具体权重或框架。"""

    def __init__(
        self,
        predictor: Callable[[str], str] | None = None,
        hidden_state_provider: Callable[[], Any] | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        experiment_config: MambaExperimentConfig | None = None,
    ) -> None:
        if predictor is None and model is None:
            raise ValueError("必须提供predictor或model")
        if model is not None and tokenizer is None:
            raise ValueError("model路径必须提供tokenizer")
        self._predictor = predictor
        self._hidden_state_provider = hidden_state_provider
        self._model = model
        self._tokenizer = tokenizer
        self.experiment_config = experiment_config
        self._last_hidden_state: Any = None

    @classmethod
    def from_pretrained(cls, config: MambaExperimentConfig) -> "MambaRunner":
        """按固定配置加载真实Mamba类因果语言模型。"""
        if config.trust_remote_code:
            raise ValueError("StateFuzz真实模型路径禁止trust_remote_code")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError("dtype必须是torch支持的类型名")
        revision = config.revision
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=revision,
            dtype=dtype,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        model.to(config.device)
        model.eval()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer, experiment_config=config)

    @property
    def is_model_backed(self) -> bool:
        return self._model is not None

    def _run_model_probe(self, probe: CompiledProbe) -> str:
        import torch

        assert self._model is not None
        assert self._tokenizer is not None
        config = self.experiment_config or MambaExperimentConfig("unknown")
        encoded = self._tokenizer(probe.prompt, return_tensors="pt")
        device = getattr(self._model, "device", config.device)
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                self._last_hidden_state = _copy_hidden_state(hidden_states[-1])
            generated = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(
                    self._tokenizer,
                    "pad_token_id",
                    getattr(self._tokenizer, "eos_token_id", None),
                ),
            )
        prompt_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0, prompt_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def run_probe(self, probe: CompiledProbe) -> str:
        """执行探针并在预测完成后捕获隐藏状态。"""
        if not isinstance(probe, CompiledProbe):
            raise TypeError("probe必须是CompiledProbe")
        prediction = (
            self._run_model_probe(probe)
            if self._model is not None
            else self._predictor(probe.prompt)  # type: ignore[union-attr]
        )
        if not isinstance(prediction, str):
            raise TypeError("模型预测必须是字符串")
        if self._model is None and self._hidden_state_provider is not None:
            self._last_hidden_state = _copy_hidden_state(self._hidden_state_provider())
        elif self._model is None:
            self._last_hidden_state = None
        return prediction

    def capture_hidden_state(self) -> Any:
        """返回最近一次捕获结果的再次复制，防止调用方修改内部状态。"""
        return _copy_hidden_state(self._last_hidden_state)

    def score_next_token(
        self, prompt: str, target_token_id: int | None = None
    ) -> dict[str, Any]:
        """返回真实模型对下一token的概率和最后一层隐藏状态证据。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("score_next_token需要model-backed runner")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt必须是非空字符串")
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", "cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                self._last_hidden_state = _copy_hidden_state(hidden_states[-1])
            logits = outputs.logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token_id = int(torch.argmax(probabilities).item())
            target = (
                predicted_token_id if target_token_id is None else target_token_id
            )
            if not isinstance(target, int) or not 0 <= target < probabilities.numel():
                raise ValueError("target_token_id超出词表范围")
            target_probability = float(probabilities[target].item())
            predicted_probability = float(probabilities[predicted_token_id].item())
        return {
            "target_token_id": target,
            "predicted_token_id": predicted_token_id,
            "target_probability": target_probability,
            "predicted_probability": predicted_probability,
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "hidden_state_shape": list(self._last_hidden_state.shape)
            if hasattr(self._last_hidden_state, "shape")
            else None,
        }

    def score_next_token_instances(
        self, prompts: list[str], target_token_id: int | None = None
    ) -> list[dict[str, Any]]:
        """按相同目标token逐个评分，保留每个实例的独立证据。"""
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts必须是非空列表")
        return [
            self.score_next_token(prompt, target_token_id=target_token_id)
            for prompt in prompts
        ]


assert isinstance(MambaRunner(lambda _: ""), ProbeRunner)

