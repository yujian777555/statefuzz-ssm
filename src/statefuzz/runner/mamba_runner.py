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


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """尽可能返回单token可读文本，避免把不可打印字节当作证据。"""
    try:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))
    except (AttributeError, TypeError, ValueError):
        return str(token_id)


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
        self._last_layer_states: dict[str, Any] = {}
        self._last_recurrent_state: dict[str, list[Any]] | None = None
        self._state_source = "unavailable"

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
            self._record_outputs(outputs)
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

    def capture_layer_states(self) -> dict[str, Any]:
        """返回各层最后时间点隐藏状态的独立副本。"""
        return {
            name: _copy_hidden_state(state)
            for name, state in self._last_layer_states.items()
        }

    def capture_recurrent_state(self) -> dict[str, list[Any]] | None:
        """返回模型明确暴露的SSM cache副本；不可用时返回None。"""
        if self._last_recurrent_state is None:
            return None
        return {
            name: [_copy_hidden_state(value) for value in values]
            for name, values in self._last_recurrent_state.items()
        }

    def capture_recurrent_state_summary(self) -> dict[str, Any]:
        """返回cache来源、各层形状和范数，不存储完整大张量。"""
        if self._last_recurrent_state is None:
            return {"state_source": "unavailable", "layers": []}
        import torch

        layers = []
        for index, (conv_state, ssm_state) in enumerate(
            zip(
                self._last_recurrent_state.get("conv_states", []),
                self._last_recurrent_state.get("ssm_states", []),
                strict=False,
            )
        ):
            layers.append(
                {
                    "layer": index,
                    "conv_shape": list(conv_state.shape)
                    if hasattr(conv_state, "shape")
                    else None,
                    "ssm_shape": list(ssm_state.shape)
                    if hasattr(ssm_state, "shape")
                    else None,
                    "conv_l2_norm": float(torch.linalg.vector_norm(conv_state.float()))
                    if hasattr(conv_state, "float")
                    else None,
                    "ssm_l2_norm": float(torch.linalg.vector_norm(ssm_state.float()))
                    if hasattr(ssm_state, "float")
                    else None,
                }
            )
        return {"state_source": self._state_source, "layers": layers}

    def _record_outputs(self, outputs: Any) -> None:
        """从一次forward中分别记录层激活和显式cache状态。"""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            self._last_hidden_state = _copy_hidden_state(hidden_states[-1])
            self._last_layer_states = {
                f"layer_{index}": _copy_hidden_state(state[:, -1, :])
                for index, state in enumerate(hidden_states)
            }
        cache = getattr(outputs, "cache_params", None)
        if cache is not None and all(
            hasattr(cache, name) for name in ("conv_states", "ssm_states")
        ):
            self._last_recurrent_state = {
                "conv_states": [
                    _copy_hidden_state(value) for value in cache.conv_states
                ],
                "ssm_states": [
                    _copy_hidden_state(value) for value in cache.ssm_states
                ],
            }
            self._state_source = "direct_recurrent_cache"
        elif cache is not None and hasattr(cache, "layers"):
            layers = list(cache.layers)
            if layers and all(
                hasattr(layer, name)
                for layer in layers
                for name in ("conv_states", "recurrent_states")
            ):
                self._last_recurrent_state = {
                    "conv_states": [
                        _copy_hidden_state(layer.conv_states) for layer in layers
                    ],
                    "ssm_states": [
                        _copy_hidden_state(layer.recurrent_states) for layer in layers
                    ],
                }
                self._state_source = "direct_recurrent_cache"
            else:
                self._last_recurrent_state = None
                self._state_source = "unavailable"
        else:
            self._last_recurrent_state = None
            self._state_source = "unavailable"

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
            self._record_outputs(outputs)
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
            target_rank = int((probabilities > target_probability).sum().item()) + 1
        return {
            "target_token_id": target,
            "predicted_token_id": predicted_token_id,
            "target_probability": target_probability,
            "predicted_probability": predicted_probability,
            "target_token_text": _decode_token(self._tokenizer, target),
            "predicted_token_text": _decode_token(
                self._tokenizer, predicted_token_id
            ),
            "target_rank": target_rank,
            "top1_margin": predicted_probability - target_probability,
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

    def single_token_id(self, text: str) -> int | None:
        """返回文本在当前tokenizer下的唯一token id，不满足时返回None。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("single_token_id需要model-backed runner")
        if not isinstance(text, str) or not text:
            raise ValueError("text必须是非空字符串")
        encoded = self._tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        token_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        if token_ids is None:
            raise ValueError("tokenizer输出缺少input_ids")
        if hasattr(token_ids, "shape"):
            if token_ids.ndim != 2 or token_ids.shape[0] != 1:
                raise ValueError("tokenizer必须返回单批次input_ids")
            if int(token_ids.shape[-1]) != 1:
                return None
            return int(token_ids[0, 0].item())
        values = list(token_ids[0]) if token_ids and isinstance(token_ids[0], list) else list(token_ids)
        return int(values[0]) if len(values) == 1 else None

    def score_candidate_tokens(
        self, prompt: str, candidate_token_ids: list[int]
    ) -> dict[str, Any]:
        """在同一次forward中评分显式候选，不使用模型argmax替换候选。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("score_candidate_tokens需要model-backed runner")
        if not isinstance(candidate_token_ids, list) or not candidate_token_ids:
            raise ValueError("candidate_token_ids必须是非空列表")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in candidate_token_ids
        ):
            raise ValueError("candidate_token_ids必须为整数")
        import torch

        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
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
            self._record_outputs(outputs)
            logits = outputs.logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            top1_token_id = int(torch.argmax(probabilities).item())
            top1_probability = float(probabilities[top1_token_id].item())
            candidates: list[dict[str, Any]] = []
            for token_id in candidate_token_ids:
                if not 0 <= token_id < probabilities.numel():
                    raise ValueError("candidate_token_id超出词表范围")
                probability = float(probabilities[token_id].item())
                log_probability = float(torch.log(probabilities[token_id]).item())
                candidates.append(
                    {
                        "token_id": token_id,
                        "token_text": _decode_token(self._tokenizer, token_id),
                        "probability": probability,
                        "logit": float(logits[token_id].item()),
                        "log_probability": log_probability,
                        "rank": int((probabilities > probability).sum().item()) + 1,
                    }
                )
        return {
            "candidates": candidates,
            "top1_token_id": top1_token_id,
            "top1_token_text": _decode_token(self._tokenizer, top1_token_id),
            "top1_probability": top1_probability,
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "hidden_state_shape": list(self._last_hidden_state.shape)
            if hasattr(self._last_hidden_state, "shape")
            else None,
            "state_source": self._state_source,
        }

    def score_remote_memory_pair(
        self, pair: Any, *, include_state_copies: bool = True
    ) -> dict[str, Any]:
        """评分远程记忆A/B对，并保留候选、长度和循环状态证据。"""
        if not hasattr(pair, "prompt_a") or not hasattr(pair, "prompt_b"):
            raise TypeError("pair必须是RemoteMemoryPair")
        value_a = str(pair.value_a)
        value_b = str(pair.value_b)
        token_id_a = self.single_token_id(value_a)
        token_id_b = self.single_token_id(value_b)
        counts = [self.count_tokens(pair.prompt_a), self.count_tokens(pair.prompt_b)]
        result: dict[str, Any] = {
            "seed": pair.seed,
            "template_id": pair.template_id,
            "target_position": pair.target_position,
            "candidate_values": [value_a, value_b],
            "candidate_token_ids": [token_id_a, token_id_b],
            "prompt_token_counts": counts,
            "matched": counts[0] == counts[1],
            "actual_input_tokens": counts[0] if counts[0] == counts[1] else None,
            "candidate_valid": token_id_a is not None and token_id_b is not None and token_id_a != token_id_b,
            "state_source": "unavailable",
        }
        if not result["matched"]:
            result["reason"] = "token_length_mismatch"
            return result
        if not result["candidate_valid"]:
            result["reason"] = "candidate_not_single_token_or_duplicate"
            return result
        candidate_ids = [int(token_id_a), int(token_id_b)]
        score_a = self.score_candidate_tokens(pair.prompt_a, candidate_ids)
        recurrent_a = self.capture_recurrent_state()
        layers_a = self.capture_layer_states()
        score_b = self.score_candidate_tokens(pair.prompt_b, candidate_ids)
        recurrent_b = self.capture_recurrent_state()
        layers_b = self.capture_layer_states()
        result.update(
            {
                "prompt_a": score_a,
                "prompt_b": score_b,
                "score_a": score_a,
                "score_b": score_b,
                "state_source": score_b.get("state_source", self._state_source),
                "recurrent_state_a": recurrent_a if include_state_copies else None,
                "recurrent_state_b": recurrent_b if include_state_copies else None,
                "layer_states_a": layers_a if include_state_copies else {},
                "layer_states_b": layers_b if include_state_copies else {},
                "recurrent_state_summary_a": self._summarize_recurrent_state(recurrent_a),
                "recurrent_state_summary_b": self._summarize_recurrent_state(recurrent_b),
            }
        )
        return result

    def _state_dict_to_cache(self, state_override: dict[str, list[Any]]) -> Any:
        """把安全复制的循环状态装入新的DynamicCache，不修改调用方对象。"""
        if not isinstance(state_override, dict):
            raise TypeError("state_override必须是循环状态对象")
        import torch
        from transformers import DynamicCache

        cache = DynamicCache(config=getattr(self._model, "config", None))
        conv_states = state_override.get("conv_states", [])
        ssm_states = state_override.get("ssm_states", [])
        if not conv_states or not ssm_states:
            raise ValueError("state_override层数不能为空")
        if len(conv_states) != len(ssm_states) or len(conv_states) != len(cache.layers):
            raise ValueError("state_override层数与模型cache不一致")
        first_conv = conv_states[0]
        first_ssm = ssm_states[0]
        cache.early_initialization(
            batch_size=int(first_conv.shape[0]),
            num_heads=int(first_conv.shape[1]),
            head_dim=int(first_ssm.shape[-1]),
            dtype=first_ssm.dtype if hasattr(first_ssm, "dtype") else torch.float16,
            device=self._model.device,
        )
        for layer, conv_state, ssm_state in zip(
            cache.layers, conv_states, ssm_states, strict=True
        ):
            layer.conv_states = (
                conv_state.detach().clone().to(self._model.device)
                if hasattr(conv_state, "detach")
                else torch.as_tensor(conv_state, device=self._model.device).clone()
            )
            layer.recurrent_states = (
                ssm_state.detach().clone().to(self._model.device)
                if hasattr(ssm_state, "detach")
                else torch.as_tensor(ssm_state, device=self._model.device).clone()
            )
            # DynamicCache layer在forward中会依据这些标志决定是否重置状态；
            # 仅赋值张量而不显式标记会导致所谓override被静默清零。
            layer.is_conv_states_initialized = True
            layer.is_recurrent_states_initialized = True
            layer.has_previous_state = True
        return cache

    def run_with_state_override(
        self, prompt: str, state_override: dict[str, list[Any]] | None = None
    ) -> dict[str, Any]:
        """在显式复制的循环cache上执行一次行为forward。

        默认路径不传cache；干预路径创建全新DynamicCache并记录来源，避免
        静默修改runner内部最近状态或调用方传入的张量。
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("run_with_state_override需要model-backed runner")
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = getattr(self._model, "device", "cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        cache = self._state_dict_to_cache(state_override) if state_override is not None else None
        with torch.inference_mode():
            if cache is not None and int(inputs["input_ids"].shape[-1]) > 1:
                # DynamicCache已有历史时，Mamba slow path只接受单步decode；
                # 逐token推进，确保override真的参与每一步状态更新。
                outputs = None
                for index in range(int(inputs["input_ids"].shape[-1])):
                    step_inputs = {
                        key: value[:, index : index + 1]
                        if hasattr(value, "shape") and value.ndim >= 2
                        else value
                        for key, value in inputs.items()
                    }
                    outputs = self._model(
                        **step_inputs,
                        cache_params=cache,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = getattr(outputs, "cache_params", cache)
            else:
                kwargs = {
                    **inputs,
                    "output_hidden_states": True,
                    "use_cache": True,
                    "return_dict": True,
                }
                if cache is not None:
                    kwargs["cache_params"] = cache
                outputs = self._model(**kwargs)
            self._record_outputs(outputs)
            logits = outputs.logits[0, -1].float().detach().cpu()
        return {
            "logits": logits,
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "state_source": (
                "override_recurrent_cache"
                if state_override is not None
                else self._state_source
            ),
            "recurrent_state": self.capture_recurrent_state(),
        }

    @staticmethod
    def _summarize_recurrent_state(state: dict[str, list[Any]] | None) -> dict[str, Any]:
        """为实验产物提供不含完整张量的循环状态摘要。"""
        if state is None:
            return {"state_source": "unavailable", "layers": []}
        import torch

        conv_states = state.get("conv_states", [])
        ssm_states = state.get("ssm_states", [])
        layers = []
        for index, (conv_state, ssm_state) in enumerate(
            zip(conv_states, ssm_states, strict=False)
        ):
            layers.append(
                {
                    "layer": index,
                    "conv_shape": list(conv_state.shape) if hasattr(conv_state, "shape") else None,
                    "ssm_shape": list(ssm_state.shape) if hasattr(ssm_state, "shape") else None,
                    "conv_l2_norm": float(torch.linalg.vector_norm(conv_state.float()))
                    if hasattr(conv_state, "float")
                    else None,
                    "ssm_l2_norm": float(torch.linalg.vector_norm(ssm_state.float()))
                    if hasattr(ssm_state, "float")
                    else None,
                }
            )
        return {"state_source": "direct_recurrent_cache", "layers": layers}

    def count_tokens(self, prompt: str) -> int:
        """使用真实tokenizer计数，不把字符数当作token数。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("count_tokens需要model-backed runner")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt必须是非空字符串")
        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        return int(encoded["input_ids"].shape[-1])

    def score_paired_next_token(
        self,
        control_prompt: str,
        stressed_prompt: str,
        target_token_id: int | None = None,
    ) -> dict[str, Any]:
        """评分长度匹配的控制/干扰对，长度不一致时不宣称可比。"""
        control_count = self.count_tokens(control_prompt)
        stressed_count = self.count_tokens(stressed_prompt)
        result: dict[str, Any] = {
            "control_token_count": control_count,
            "stressed_token_count": stressed_count,
            "matched": control_count == stressed_count,
        }
        if control_count != stressed_count:
            result["reason"] = "token_length_mismatch"
            return result
        control = self.score_next_token(control_prompt, target_token_id=target_token_id)
        control_recurrent = self.capture_recurrent_state_summary()
        control_layers = self.capture_layer_states()
        effective_target = control["target_token_id"]
        stressed = self.score_next_token(
            stressed_prompt, target_token_id=effective_target
        )
        stressed_recurrent = self.capture_recurrent_state_summary()
        stressed_layers = self.capture_layer_states()
        result.update(
            {
                "control": control,
                "stressed": stressed,
                "control_layers": control_layers,
                "stressed_layers": stressed_layers,
                "control_recurrent_state": control_recurrent,
                "stressed_recurrent_state": stressed_recurrent,
                "state_source": stressed_recurrent["state_source"],
            }
        )
        return result


assert isinstance(MambaRunner(lambda _: ""), ProbeRunner)
