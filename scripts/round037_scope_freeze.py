"""Round37-C：只读校验论文范围冻结产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_UNSUPPORTED_CLAIMS = {
    "universal failure boundaries across SSMs",
    "general superiority or inferiority of Mamba, Pythia, or Zamba2",
    "architecture-causal claims from cross-model comparisons",
    "Mamba2 generalization",
    "modern-Attention generalization beyond the historical Pythia reference",
    "universal claims about all SSMs or all Hybrid architectures",
    "Hybrid SSM-vs-Attention internal causal attribution",
    "any claim that no failure exists outside evaluated token budgets",
}
REQUIRED_EVALUATED_MODELS = {
    "state-spaces/mamba-130m-hf",
    "EleutherAI/pythia-160m",
    "Zyphra/Zamba2-1.2B-Instruct-v2",
}


def _load_json(path: Path) -> dict[str, object]:
    """读取一个必需的 JSON 对象。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def verify(root: Path) -> dict[str, object]:
    """只读校验 Round37-C 的两份范围冻结产物。"""
    results = root / "results"
    scope = _load_json(results / "final_scope_round_037.json")
    result = _load_json(results / "result_round_037.json")
    unavailable_roles = {
        str(entry.get("requested_role", "")).casefold()
        for entry in scope.get("planned_but_unavailable", [])
        if isinstance(entry, dict)
    }
    if not any("mamba2" in role for role in unavailable_roles):
        raise ValueError("planned_but_unavailable 缺少 Mamba2")
    if not any("modern pure-attention" in role for role in unavailable_roles):
        raise ValueError("planned_but_unavailable 缺少 modern pure-Attention baseline")
    unsupported_claims = set(scope.get("unsupported_claims", []))
    missing_claims = REQUIRED_UNSUPPORTED_CLAIMS - unsupported_claims
    if missing_claims:
        raise ValueError(f"unsupported_claims 缺少禁止主张: {sorted(missing_claims)}")
    evaluated_models = scope.get("evaluated_models", [])
    evaluated_ids = {
        str(model.get("model_id", ""))
        for model in evaluated_models
        if isinstance(model, dict)
    }
    missing_models = REQUIRED_EVALUATED_MODELS - evaluated_ids
    if missing_models:
        raise ValueError(f"evaluated_models 缺少冻结模型角色: {sorted(missing_models)}")
    if any(
        isinstance(model, dict) and model.get("newly_evaluated_in_round037") is True
        for model in evaluated_models
    ):
        raise ValueError("scope falsely reports a newly evaluated Round37 model")
    extra_models = evaluated_ids - REQUIRED_EVALUATED_MODELS
    if extra_models or len(evaluated_models) != len(REQUIRED_EVALUATED_MODELS):
        raise ValueError(
            f"unexpected evaluated_models: extras={sorted(extra_models)}, "
            f"entry_count={len(evaluated_models)}"
        )
    if result.get("new_models_added") != 0:
        raise ValueError("new_models_added 必须为 0")
    paper_transition = scope.get("paper_transition")
    if not isinstance(paper_transition, dict) or paper_transition.get("ready") is not True:
        raise ValueError("paper_transition.ready 必须为 true")
    return {"verify": "passed"}


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
