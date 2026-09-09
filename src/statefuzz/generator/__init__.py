"""面向SSM状态限制的确定性压力样例生成器。"""

from __future__ import annotations

import hashlib
import json

from statefuzz.probes.compiler import CompiledProbe


def _tag_probe(probe: CompiledProbe, pattern: str, **metadata: object) -> CompiledProbe:
    """为探针附加研究模式元数据，不改变其提示、答案或哈希。"""
    provenance = dict(probe.provenance)
    provenance.update({"stress_pattern": pattern, **metadata})
    return CompiledProbe(
        spec=probe.spec,
        prompt=probe.prompt,
        answer=probe.answer,
        provenance=provenance,
        probe_hash=probe.probe_hash,
    )


def _rebuild_probe(
    probe: CompiledProbe, prompt: str, **metadata: object
) -> CompiledProbe:
    """在改变提示后重新计算审计哈希并保留原答案。"""
    provenance = dict(probe.provenance)
    provenance.update(metadata)
    payload = json.dumps(
        {
            "spec": probe.spec.canonical_payload(),
            "prompt": prompt,
            "answer": probe.answer,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CompiledProbe(
        spec=probe.spec,
        prompt=prompt,
        answer=probe.answer,
        provenance=provenance,
        probe_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )

