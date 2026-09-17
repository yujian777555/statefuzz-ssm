"""Round39 论文证据审计器测试。"""

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_workspace(root: Path) -> None:
    _write_json(
        root / "results/evidence_reconciliation_round_039.json",
        {
            "tracks": {
                "failure_discovery": {"artifacts": ["results/result_round_018.json"]},
                "causal_diagnosis": {
                    "artifacts": ["results/result_round_020.json", "results/result_round_021.json"]
                },
                "transfer_surface": {},
                "mitigation": {},
            }
        },
    )
    _write_json(root / "results/paper_claim_graph_round_039.json", {"claims": []})
    _write_json(root / "results/reviewer_audit_round_039.json", {"round": 39})
    _write_json(
        root / "results/paper_figure_plan_round_039.json",
        {
            "figures": [
                {
                    "figure_id": "figure_3",
                    "source_artifacts": ["results/result_round_020.json", "results/result_round_021.json"],
                }
            ]
        },
    )
    paper = root / "paper"
    paper.mkdir(parents=True)
    (paper / "abstract.md").write_text(
        "Later transfer surface cohorts had no observed failure boundary. "
        "Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content "
        "causally influences remote-memory behavior.",
        encoding="utf-8",
    )
    (paper / "related_work.md").write_text(
        "LongBench, RULER, and NeedleBench provide fixed long-context evaluation points, whereas targeted stress discovery "
        "searches a controlled condition space. Metamorphic testing evaluates behavioral relations under transformed inputs, "
        "while remote-memory counterfactual probes additionally support failure localization and a separate causal intervention stage.",
        encoding="utf-8",
    )


def test_valid_round039_package_passes(tmp_path: Path) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    assert verify(tmp_path)["verify"] == "passed"


def test_unscoped_zero_failure_abstract_fails(tmp_path: Path) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    (tmp_path / "paper/abstract.md").write_text(
        "All evaluated cohorts had zero failures. " + NARROW_CLAIM,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surface cohort"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "unsupported_sentence",
    [
        "Cross-model differences are caused by architecture.",
        "This validates Mamba2 generalization.",
    ],
)
def test_unqualified_unsupported_claim_fails(tmp_path: Path, unsupported_sentence: str) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    with (tmp_path / "paper/abstract.md").open("a", encoding="utf-8") as stream:
        stream.write("\n" + unsupported_sentence)
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_missing_narrow_causal_claim_fails(tmp_path: Path) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    (tmp_path / "paper/abstract.md").write_text(
        "Later surface cohorts had no observed failure boundary.", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Mamba"):
        verify(tmp_path)


def test_figure_three_without_round21_fails(tmp_path: Path) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    figure_path = tmp_path / "results/paper_figure_plan_round_039.json"
    payload = json.loads(figure_path.read_text(encoding="utf-8"))
    payload["figures"][0]["source_artifacts"] = ["results/result_round_020.json"]
    _write_json(figure_path, payload)
    with pytest.raises(ValueError, match="Figure 3"):
        verify(tmp_path)


def test_unverified_priority_claim_fails(tmp_path: Path) -> None:
    from scripts.round039_paper_audit import verify

    _valid_workspace(tmp_path)
    with (tmp_path / "paper/abstract.md").open("a", encoding="utf-8") as stream:
        stream.write("\nStateFuzz is the first method to find these failures.")
    with pytest.raises(ValueError, match="novelty verification"):
        verify(tmp_path)


NARROW_CLAIM = (
    "Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content "
    "causally influences remote-memory behavior."
)
