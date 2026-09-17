# Round 39 Plan: Evidence Reconciliation, Literature Positioning, and Reviewer-Level Paper Strengthening

> **For agentic workers:** REQUIRED SUB-SKILL: use a disciplined task-by-task execution workflow. Do not add new experiments, checkpoints, or claims beyond the frozen repository evidence and the literature seeds listed here.

**Goal:** Rebuild the paper around the strongest already-validated StateFuzz evidence, reconcile the apparent conflict between early failure/causal results and later zero-failure stress-surface cohorts, add verified related-work positioning, and produce a reviewer-auditable paper package.

**Architecture:** Round39 is repository analysis and paper revision only. It has four evidence tracks: (A) failure discovery and cross-model contrast, (B) recurrent-state causal diagnosis, (C) transfer/generalization via broader stress families, realistic workloads, and Hybrid Zamba2, and (D) mitigation as secondary evidence. No GPU inference or checkpoint download is allowed.

**Tech Stack:** Python, JSON, Markdown, pytest. Optional BibTeX/Markdown reference generation from the frozen literature seed list below.

**Spec:** `docs/PLANNER_EXECUTION_CONTRACT.md`, `docs/paper_claims.md`, `results/final_experiment_manifest.json`, and the evidence sources listed below.

---

## Global constraints

1. **NO new model inference.**
2. **NO checkpoint download or network-dependent experiment.**
3. **NO new seeds, prompts, stress families, or token budgets.**
4. **NO claim may be introduced unless mapped to an existing repository artifact.**
5. **Do not treat cross-checkpoint differences as architecture-causal.**
6. **Do not state that all SSMs share a failure boundary.**
7. **Do not state that failure probability is universally zero.** Round18/20/21 contain a separate frozen failure condition that must be reconciled with Round35/36 zero-failure cohorts.
8. Historical missing values and budget-unreachable values remain distinct from behavioral failures.
9. The current Round38 paper is a draft, not a frozen scientific scope. It may be rewritten when earlier validated evidence was accidentally omitted.
10. Preserve negative results and limitations even when strengthening the narrative.

---

# Scientific issue to resolve

Round38 currently centers the paper on Round35/36 stress-surface cohorts and says all evaluated cohorts have zero failure probability. That statement is only true for the **Round35/36 packaged surface cohort**.

The repository also contains stronger, independently validated evidence that is absent from the Round38 narrative:

- Round18: fresh-seed structured-repetition remote-memory failure contrast between Mamba-130M and Pythia-160M, including exact paired McNemar statistics.
- Round20: Mamba recurrent-state intervention in a frozen long-context failure condition; correct short-state content restores behavior while wrong-memory and randomized controls do not.
- Round21: fresh-seed replication and an additional value-pair specificity test.
- `docs/paper_claims.md`: narrow causal claim explicitly approved after the intervention rounds.
- `results/final_experiment_manifest.json`: frozen failure/causal experiment manifest.
- Round27: controlled realistic-workload transfer evidence.
- Round28: task-specific mitigation candidate.

Round39 must reconcile, not overwrite, these evidence tracks.

---

# Frozen repository evidence

## Track A — Failure discovery and contrast

Read at minimum:

- `results/result_round_018.json`
- any Round18 helper artifacts referenced by that result
- `docs/reviewer_attack.md` if present

Extract exact values programmatically from source JSON rather than copying from this plan.

Required audit items:

- tested model names;
- exact seed cohort;
- stress/filler style;
- value pair;
- target and actual token ranges;
- Mamba failure count/rate;
- Pythia failure count/rate;
- discordant pair count;
- exact two-sided McNemar p-value;
- negative-control result;
- all protocol/tolerance differences relevant to comparison with later rounds.

The audit should confirm whether Round18 reports the known paired result with 12 discordant Mamba-only failures and exact two-sided p = 0.00048828125. **Do not hard-code this as truth; verify from JSON and fail if the source differs.**

## Track B — Causal diagnosis

Read:

- `results/result_round_020.json`
- `results/result_round_021.json`
- `docs/paper_claims.md`
- `results/final_experiment_manifest.json`

Extract exactly:

- normal-long recovery rate and mean margin;
- wrong-memory state recovery rate;
- correct-memory state recovery rate;
- randomized-matched recovery rate;
- restoration effect / specificity metrics;
- fresh-seed replication status;
- support-pair behavior and why it is specificity evidence rather than failure-recovery replication.

Principal causal wording must remain narrow:

> Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content causally influences remote-memory behavior.

Do not generalize this to all SSMs or all long-context failures.

## Track C — Transfer and stress surfaces

Read:

- `results/result_round_025.json` if present
- `results/result_round_026.json` if present
- `results/result_round_027.json`
- `results/result_round_031.json`
- `results/result_round_034.json`
- `results/result_round_035.json`
- `results/stress_surface_round_035.json`
- `results/evidence_table_round_036.json`
- `results/figure_data_round_036.json`
- `results/paper_evidence_summary_round_036.json`
- `results/final_scope_round_037.json`

This track must be framed as **transfer / scope extension / descriptive surface evidence**, not as replacement for Track A/B.

Required conclusions to audit:

- broader stress-family evaluation exists;
- controlled realistic tasks exist (long-document retrieval, code context dependency, agent conversation memory);
- Zamba2 Hybrid behavioral stress evaluation exists;
- later surface cohorts can legitimately have zero observed failure boundaries while Round18 has failures because they are different frozen cohorts/protocol slices;
- Zamba2 response can be non-monotonic;
- cross-checkpoint model scale/training/tuning/seed/tolerance confounds prevent architecture-causal ranking.

## Track D — Mitigation

Read:

- `results/result_round_028.json`
- `results/mitigation_comparison_round_028.json` if present

Use only as a secondary contribution / appendix-level result unless the artifact supports a stronger statement.

Allowed wording:

- context-anchor mitigation showed task-specific improvement on the tested code-context dependency task;
- no mitigation strategy was universally beneficial.

Do not promote this to a universal mitigation claim.

---

# Task 1 — Build an evidence reconciliation artifact

Create:

`results/evidence_reconciliation_round_039.json`

Required schema:

```json
{
  "round": 39,
  "status": "complete",
  "tracks": {
    "failure_discovery": {},
    "causal_diagnosis": {},
    "transfer_surface": {},
    "mitigation": {}
  },
  "apparent_conflicts": [],
  "resolved_interpretations": [],
  "paper_mainline": [],
  "paper_secondary": [],
  "excluded_claims": []
}
```

For each apparent conflict, record:

- source A;
- source B;
- exact differing protocol fields;
- whether the difference is scientifically compatible, a true contradiction, or unresolved;
- wording permitted in the paper.

At minimum reconcile:

1. Round18 long-context Mamba failures vs Round35/36 zero-failure surface cohort.
2. `architecture-specific/dependent` language vs acknowledged cross-checkpoint confounds.
3. Round20/21 causal state intervention vs invalid Hybrid cache replay from Round32/33.

Expected interpretation for item 3: Mamba-130M recurrent-state causal evidence may remain valid; failed Zamba2 cache reconstruction only blocks **Hybrid internal path attribution**, not the earlier Mamba intervention result. Verify this from artifacts before writing it.

---

# Task 2 — Build a claim-to-evidence graph

Create:

`results/paper_claim_graph_round_039.json`

Each claim entry must contain:

- `claim_id`
- `claim_text`
- `claim_type`: one of `primary`, `secondary`, `negative`, `limitation`, `unsupported`
- `supporting_artifacts`
- `supporting_fields`
- `scope`
- `known_confounds`
- `paper_sections`

Required primary claims:

### C1 — Discovery

StateFuzz can expose a reproducible structured-repetition remote-memory failure regime in the tested Mamba-130M checkpoint, with a paired behavioral contrast against the tested Pythia-160M reference under the Round18 frozen protocol.

Do not convert this into an architecture-universal statement.

### C2 — Diagnosis

Under the frozen Mamba-130M structured-repetition condition, recurrent-state content has a causal influence on the remote-memory behavior: correct-memory state transfer restores the target behavior, wrong-memory state does not, and matched random-state controls are separated; fresh-seed replication exists.

### C3 — Transfer / surface

StateFuzz can be applied beyond the single discovery condition to broader stress families, controlled realistic workloads, and the tested Hybrid Zamba2 checkpoint, where response surfaces can be stress-dependent and non-monotonic and may contain no failure boundary in the observed range.

### C4 — Mitigation (secondary)

A context-anchor intervention provides task-specific improvement in the tested code-context setting, but no mitigation strategy is universally beneficial.

Required negative/unsupported claims include:

- all SSMs have the same failure boundary;
- Mamba is globally worse than Pythia;
- Zamba2 proves standalone Mamba2 behavior;
- cross-model differences are architecture-causal;
- Hybrid SSM-vs-Attention path attribution is established;
- zero failures in Round35/36 means no failure exists elsewhere;
- causal intervention results apply to all value pairs/tasks/models.

---

# Task 3 — Repair the paper framing and title

The current `architecture-specific` framing is too strong if it implies architecture causality. Use safer wording unless a sentence explicitly says `architecture role` and immediately states the confounds.

Preferred working title:

**StateFuzz: Discovering and Diagnosing Remote-Memory Failure Modes in Stateful Sequence Models**

Acceptable alternate title if the draft needs broader emphasis:

**StateFuzz: Behavioral Stress Testing and Causal Diagnosis of Remote-Memory Failures in Stateful Sequence Models**

Do **not** use `Architecture-Specific ...` as the main title in Round39.

Create/update:

- `paper/title_and_claims.md`
- `paper/abstract.md`
- `paper/introduction.md`
- `paper/method.md`
- `paper/experiment.md`
- `paper/results.md` (new; separate results from setup)
- `paper/limitation.md`
- `paper/conclusion.md`

Required narrative order:

1. Problem: average accuracy / single context length hides condition-dependent memory failure regimes.
2. StateFuzz method: controlled stress generation + counterfactual paired behavioral margins + failure localization.
3. Discovery: narrow reproducible Mamba failure regime and paired Pythia contrast.
4. Diagnosis: state-content intervention provides causal evidence in Mamba-130M.
5. Transfer: broader stress families, realistic workloads, and Hybrid Zamba2 show the framework is not tied to a single probe; results are heterogeneous and sometimes non-monotonic.
6. Mitigation: optional secondary evidence, explicitly task-specific.
7. Limitations: small checkpoint set, unmatched checkpoints, unavailable Mamba2/modern Attention, seed/tolerance differences, periodic prompt non-independence, invalid Hybrid cache replay.

The abstract must no longer imply that the entire project observed zero failures. It may say that **later transfer/surface cohorts** contained no observed failure boundary, while the original discovery condition did contain failures.

---

# Task 4 — Related-work positioning from frozen literature seeds

Round38 intentionally left Related Work empty. Round39 must add a real positioning section without inventing priority claims.

Use the following verified literature seeds. If exact citation metadata cannot be confirmed from the identifier, leave a TODO rather than fabricate metadata.

## SSM / Mamba foundations

1. Albert Gu, Tri Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** arXiv:2312.00752, 2023.
2. Tri Dao, Albert Gu. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** ICML 2024 / arXiv:2405.21060.

## Long-context evaluation

3. Yushi Bai et al. **LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.** ACL 2024, DOI 10.18653/v1/2024.acl-long.172.
4. Cheng-Ping Hsieh et al. **RULER: What's the Real Context Size of Your Long-Context Language Models?** arXiv:2404.06654, 2024.
5. Mo Li, Songyang Zhang, Yunxin Liu, Kai Chen. **NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?** arXiv:2407.11963, 2024.

## Mamba long-memory / recall context

6. Zhifan Ye et al. **LongMamba: Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement.** arXiv:2504.16053, 2025.
7. Yuval Koren, Assaf Ben-Kish, Raja Giryes, Lior Wolf, Itamar Zimerman. **On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing.** arXiv:2609.07681, 2026. Treat as very recent work and avoid unsupported comparisons.

## Metamorphic / automated behavioral testing

8. Edoardo Manino, Julia Rozanova, Danilo Carvalho, Andre Freitas, Lucas Cordeiro. **Systematicity, Compositionality and Transitivity of Deep NLP Models: a Metamorphic Testing Perspective.** Findings of ACL 2022, DOI 10.18653/v1/2022.findings-acl.185.
9. Sai Sathiesh Rajan, Ezekiel Soremekun, Sudipta Chattopadhyay. **Knowledge-based Consistency Testing of Large Language Models.** Findings of EMNLP 2024, DOI 10.18653/v1/2024.findings-emnlp.596.
10. Steven Cho, Stefano Ruberto, Valerio Terragni. **Metamorphic Testing of Large Language Models for Natural Language Processing.** arXiv:2511.02108, 2025.

Create:

- `paper/related_work.md`
- `paper/references.md`

Required comparison dimensions:

- fixed benchmark score vs targeted stress discovery;
- retrieval/long-context evaluation vs failure-surface search;
- behavioral/metamorphic consistency tests vs remote-memory-specific counterfactual probes;
- behavioral detection vs causal state intervention;
- generic Mamba recall analysis vs StateFuzz's stress-search + intervention workflow.

Do not claim `first` or `novel` without a dedicated novelty audit.

---

# Task 5 — Redesign the main figures around the actual contributions

Replace the Round38 figure plan with:

`results/paper_figure_plan_round_039.json`

Required main figures:

## Figure 1 — StateFuzz workflow

Panels:

- counterfactual paired prompt construction;
- stress dimensions;
- direction-correct signed margins;
- failure detection;
- optional causal state intervention;
- transfer/surface analysis.

## Figure 2 — Discovery: failure transition and paired model contrast

Source primarily from Round18.

Must show:

- Mamba failure rate vs tested context/budget region;
- Pythia reference under the matched Round18 protocol;
- the fresh-seed long-context point used for paired significance;
- exact seed count and confidence/statistical annotation;
- negative control if the artifact supports a compact panel.

Do not merge Round18 with Round35 cohorts as if they were identical.

## Figure 3 — Diagnosis: recurrent-state intervention

Source Round20/21.

Required groups:

- normal long failed state;
- wrong-memory short state;
- correct-memory short state;
- randomized matched state.

Show recovery rate and/or signed margin with seed-level points where possible. Include fresh-seed replication as a separate panel or visual marker.

## Figure 4 — Transfer and scope

Use Round27/31/34/35/36.

Show broader stress-family / realistic workload / Zamba2 transfer evidence and explicitly mark missing/unreachable budgets.

Mitigation may be Table 1 or appendix figure unless space allows.

---

# Task 6 — Novelty and reviewer attack audit

Create:

`results/reviewer_audit_round_039.json`

Required fields:

```json
{
  "round": 39,
  "major_strengths": [],
  "major_weaknesses": [],
  "likely_reviewer_questions": [],
  "answerable_from_existing_evidence": [],
  "requires_new_experiment": [],
  "claim_rewrites": [],
  "submission_blockers": []
}
```

At minimum address:

1. Is StateFuzz just another long-context benchmark?
2. Is the failure effect only a single prompt/value pair artifact?
3. Why is Pythia a valid reference but not an architecture-causal control?
4. Does the causal intervention genuinely establish recurrent-state content specificity?
5. Why do Round35/36 show zero failures when Round18 shows failures?
6. Does invalid Zamba2 cache replay invalidate Mamba causal evidence? (It should not unless source artifacts show otherwise.)
7. Are periodic-pattern seeds independent?
8. Does the paper need Mamba2 before submission, or can this remain a clearly scoped limitation?
9. What differentiates StateFuzz from LongBench/RULER/NeedleBench and metamorphic testing?
10. Are mitigation results strong enough for the main paper or better as secondary/appendix evidence?

Do not answer a reviewer question more strongly than the artifacts permit.

---

# Task 7 — Verification

Create:

`scripts/round039_paper_audit.py`

Supported command:

```bash
python scripts/round039_paper_audit.py --verify
```

The verifier must be read-only and must fail if:

1. `results/evidence_reconciliation_round_039.json` is missing or malformed.
2. `results/paper_claim_graph_round_039.json` is missing.
3. Track A does not reference Round18.
4. Track B does not reference both Round20 and Round21.
5. The paper abstract says or implies all evaluated cohorts had zero failures without scoping that statement to later surface cohorts.
6. The paper claims cross-model architecture causality.
7. The paper claims Mamba2 generalization.
8. The paper omits the approved narrow Mamba causal claim entirely.
9. Figure 3 is not mapped to Round20/21 intervention evidence.
10. `paper/related_work.md` remains a placeholder.
11. Any `first`, `novel`, or priority claim appears without being explicitly marked for novelty verification.
12. `results/reviewer_audit_round_039.json` is absent.

Run:

```bash
python scripts/round039_paper_audit.py --verify
python -m pytest
```

If `python` is unavailable, use the repository's active Python interpreter consistently.

---

# Task 8 — Round result and handoff

Create:

`results/result_round_039.json`

Required fields:

```json
{
  "round": 39,
  "status": "evidence_reconciled_paper_strengthened",
  "new_experiments_run": false,
  "new_models_added": 0,
  "primary_evidence_tracks": [
    "failure_discovery",
    "causal_diagnosis",
    "transfer_surface"
  ],
  "secondary_evidence_tracks": ["mitigation"],
  "paper_ready_for_finalization": false,
  "remaining_submission_blockers": [],
  "tests": {}
}
```

`paper_ready_for_finalization` should only be true if the reviewer audit has no material evidence/novelty blocker. It is acceptable and preferable for it to be false if literature novelty or experimental scope still requires work.

Finally update `status.json` to:

```json
{
  "last_plan": "plans/latest_plan.md",
  "last_result": "results/result_round_039.json",
  "next": "gpt",
  "previous_actor": "codex",
  "round": 39,
  "phase": "paper_evidence_reconciled"
}
```

Do not start Round40.

---

# Success criterion

Round39 succeeds when the paper no longer loses the strongest validated StateFuzz evidence, every major claim has a traceable artifact, apparent evidence conflicts are explicitly reconciled, related work is no longer a placeholder, and the draft is strong enough for a reviewer-level go/no-go decision without running new experiments.