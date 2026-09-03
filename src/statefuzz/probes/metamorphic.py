from __future__ import annotations

import hashlib
import json

from statefuzz.probes.compiler import CompiledProbe
from statefuzz.probes.schema import ProbeSpec


def switch_template(spec: ProbeSpec, template_id: int) -> ProbeSpec:
    """切换提示模板，同时保持探针语义参数不变。"""
    if template_id == spec.template_id:
        raise ValueError("新模板必须不同于原模板")
    if not 0 <= template_id <= 7:
        raise ValueError("template_id必须位于0到7")
    return ProbeSpec(**{**spec.canonical_payload(), "template_id": template_id})


def rename_symbols(compiled: CompiledProbe, salt: str) -> CompiledProbe:
    """对全部键执行确定性双射重命名，保持答案不变。"""
    if not salt:
        raise ValueError("salt不能为空")
    old_keys = [key for key, _ in compiled.provenance["pairs"]]
    new_keys = [
        "R" + hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8]
        for key in old_keys
    ]
    if len(set(new_keys)) != len(new_keys):
        raise RuntimeError("键重命名发生碰撞")
    mapping = dict(zip(old_keys, new_keys, strict=True))
    prompt = compiled.prompt
    for key in old_keys:
        prompt = prompt.replace(key, mapping[key])
    provenance = dict(compiled.provenance)
    provenance["pairs"] = [(mapping[k], v) for k, v in provenance["pairs"]]
    provenance["queried_keys"] = [mapping[k] for k in provenance["queried_keys"]]
    provenance["metamorphic_parent"] = compiled.probe_hash
    payload = json.dumps(
        {"prompt": prompt, "answer": compiled.answer, "parent": compiled.probe_hash},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CompiledProbe(
        spec=compiled.spec,
        prompt=prompt,
        answer=compiled.answer,
        provenance=provenance,
        probe_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
