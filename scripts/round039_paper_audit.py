"""Round39：只读核对论文 claim、证据轨和图表映射。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NARROW_CAUSAL_CLAIMS = (
    "Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content causally influences remote-memory behavior.",
    "在结构化重复的远程记忆压力条件下，Mamba-130M 的循环状态内容会对远程记忆行为产生因果影响。",
)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _contains_path(value: object, path: str) -> bool:
    if isinstance(value, str):
        return value == path
    if isinstance(value, dict):
        return any(_contains_path(item, path) for item in value.values())
    if isinstance(value, list):
        return any(_contains_path(item, path) for item in value)
    return False


def _paper_text(root: Path) -> str:
    paper = root / "paper"
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(paper.glob("*.md")))


def _assert_no_unqualified(text: str, phrases: tuple[str, ...], label: str) -> None:
    negations = ("不", "不能", "不得", "未", "not ", "no ", "does not", "cannot", "unsupported")
    for line in text.splitlines():
        lowered = line.casefold()
        if any(phrase.casefold() in lowered for phrase in phrases) and not any(
            negation.casefold() in lowered for negation in negations
        ):
            raise ValueError(f"检测到未限定的{label}: {line.strip()}")


def verify(root: Path) -> dict[str, object]:
    results = root / "results"
    reconciliation = _load_object(results / "evidence_reconciliation_round_039.json")
    _load_object(results / "paper_claim_graph_round_039.json")
    reviewer = _load_object(results / "reviewer_audit_round_039.json")
    figures = _load_object(results / "paper_figure_plan_round_039.json")

    tracks = reconciliation.get("tracks")
    if not isinstance(tracks, dict):
        raise ValueError("evidence reconciliation 缺少 tracks")
    if not _contains_path(tracks.get("failure_discovery"), "results/result_round_018.json"):
        raise ValueError("Track A 必须引用 results/result_round_018.json")
    causal = tracks.get("causal_diagnosis")
    for required in ("results/result_round_020.json", "results/result_round_021.json"):
        if not _contains_path(causal, required):
            raise ValueError(f"Track B 必须引用 {required}")

    abstract = (root / "paper/abstract.md").read_text(encoding="utf-8")
    abstract_lower = abstract.casefold()
    zero_failure_language = any(
        phrase in abstract_lower
        for phrase in ("all evaluated cohorts", "全部已评估", "zero failures", "failure probability 均为 0")
    )
    scoped_to_later = any(
        phrase in abstract_lower
        for phrase in ("later transfer", "later surface", "round35/36", "后续迁移", "后续压力表面", "后期迁移")
    )
    if zero_failure_language and not scoped_to_later:
        raise ValueError("摘要将零失败错误推广到全部 cohort，必须限定为后续 surface cohort")

    paper_text = _paper_text(root)
    _assert_no_unqualified(
        paper_text,
        ("cross-model differences are caused by architecture", "跨模型差异由架构导致", "架构造成了跨模型差异"),
        "跨模型架构因果 claim",
    )
    _assert_no_unqualified(
        paper_text,
        ("validates mamba2 generalization", "证明 mamba2 泛化", "验证 mamba2 泛化"),
        "Mamba2 泛化 claim",
    )
    if not any(claim in paper_text for claim in NARROW_CAUSAL_CLAIMS):
        raise ValueError("论文完全遗漏获批的窄 Mamba 循环状态因果 claim")

    figure_three = next(
        (item for item in figures.get("figures", []) if isinstance(item, dict) and item.get("figure_id") == "figure_3"),
        None,
    )
    if figure_three is None or not all(
        _contains_path(figure_three, required)
        for required in ("results/result_round_020.json", "results/result_round_021.json")
    ):
        raise ValueError("Figure 3 必须映射 Round20/21 intervention evidence")

    related = (root / "paper/related_work.md").read_text(encoding="utf-8")
    required_topics = ("LongBench", "RULER", "NeedleBench", "metamorphic")
    if len(related.strip()) < 200 or not all(topic.casefold() in related.casefold() for topic in required_topics):
        raise ValueError("related_work.md 仍是占位文本或缺少冻结定位维度")

    priority_pattern = re.compile(r"\b(first|novel)\b|首次|首个", re.IGNORECASE)
    for line in paper_text.splitlines():
        if priority_pattern.search(line) and not any(
            marker in line.casefold() for marker in ("todo", "novelty verification", "待新颖性核查", "不声称")
        ):
            raise ValueError(f"未经 novelty verification 的优先权 claim: {line.strip()}")

    if reviewer.get("round") != 39:
        raise ValueError("reviewer audit 缺失或 round 不正确")
    return {"verify": "passed", "tracks": sorted(tracks), "figure_count": len(figures.get("figures", []))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        parser.error("仅支持 --verify")
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root), ensure_ascii=False))


if __name__ == "__main__":
    main()
