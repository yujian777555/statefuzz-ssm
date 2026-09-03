from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from statefuzz.probes.oracle import join_values
from statefuzz.probes.schema import ProbeSpec, TaskFamily


@dataclass(frozen=True)
class CompiledProbe:
    spec: ProbeSpec
    prompt: str
    answer: str
    provenance: dict[str, Any]
    probe_hash: str


_RECORDS = (
    "记录 {key} = {value}",
    "映射 {key} -> {value}",
    "条目[{key}]的值为{value}",
    "保存键{key}，对应值{value}",
    "数据 {key}:{value}",
    "关系 {key} 等于 {value}",
    "缓存项 {key} 取值 {value}",
    "事实({key},{value})",
)
_QUERIES = (
    "请按键顺序仅输出对应值并用竖线连接：{keys}",
    "依序返回这些键的值，不要输出其他文字：{keys}",
    "恢复这些键的值，格式为值1|值2：{keys}",
    "严格按顺序回答并使用|：{keys}",
    "检索键{keys}，只返回组合答案。",
    "读取{keys}对应的数据并用|连接。",
    "找出{keys}的值，只输出最终序列。",
    "执行精确检索：{keys}。只返回值。",
)


def _make_pairs(rng: random.Random, n_items: int) -> list[tuple[str, str]]:
    numbers = rng.sample(range(10000, 100000), n_items)
    pairs = [(f"K{i:04d}", f"V{n:05d}") for i, n in enumerate(numbers)]
    rng.shuffle(pairs)
    return pairs


def compile_probe(spec: ProbeSpec) -> CompiledProbe:
    """将规范编译为提示、精确答案和可审计来源。"""
    rng = random.Random(spec.seed)
    pairs = _make_pairs(rng, spec.n_items)
    primary = round(spec.target_position * (spec.n_items - 1))
    indices = [primary]
    if spec.task is TaskFamily.MULTI_KEY:
        pool = [i for i in range(spec.n_items) if i != primary]
        indices.extend(sorted(rng.sample(pool, spec.query_fanout - 1)))
    queried = [pairs[i] for i in indices]
    lines = [_RECORDS[spec.template_id].format(key=k, value=v) for k, v in pairs]
    filler_count = max(0, spec.context_tokens - len(lines) * 4)
    filler = " ".join(f"中性词{i % 97:02d}" for i in range(filler_count))
    keys = [k for k, _ in queried]
    values = [v for _, v in queried]
    query = _QUERIES[spec.template_id].format(keys=",".join(keys))
    prompt = "\n".join([*lines, filler, query]).strip() + "\n"
    answer = join_values(values)
    provenance: dict[str, Any] = {
        "config_hash": spec.config_hash,
        "primary_index": primary,
        "queried_indices": indices,
        "queried_keys": keys,
        "queried_values": values,
        "pairs": pairs,
    }
    payload = json.dumps(
        {"spec": spec.canonical_payload(), "prompt": prompt, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CompiledProbe(
        spec=spec,
        prompt=prompt,
        answer=answer,
        provenance=provenance,
        probe_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
