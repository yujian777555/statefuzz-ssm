"""验证产物可从原始logits重算，拒绝旧版850-token门禁。"""
import json
from pathlib import Path
from statefuzz.analyzer.memory_dependence import decompose_pairwise_preference

ROOT=Path(__file__).resolve().parents[1]


def test_round031_gate_and_full_frozen_matrix():
    result=json.loads((ROOT/'results/result_round_031.json').read_text(encoding='utf-8'))
    raw=json.loads((ROOT/'results/hybrid_stress_round_031.json').read_text(encoding='utf-8'))['raw']
    assert result['short_context_gate']['passed']
    assert {r['seed'] for r in result['short_context_gate']['rows']}=={65,66,67,68}
    assert all(248<=r['actual_input_tokens']<=264 and r['min_signed_margin']>0 for r in result['short_context_gate']['rows'])
    assert len(raw['records'])==120
    keys={(r['stress_family'],r['target_budget'],r['seed']) for r in raw['records']}
    assert len(keys)==120
    for r in raw['records']:
        if r['runtime_status']!='ok':
            assert 'failure' not in r
            continue
        da=r['logit_prompt_a_candidate_a']-r['logit_prompt_a_candidate_b']
        db=r['logit_prompt_b_candidate_b']-r['logit_prompt_b_candidate_a']
        expected=decompose_pairwise_preference(dict(direction_a_margin=da,direction_b_margin=db))
        assert r['memory_signal']==expected['memory_signal']
        assert r['lexical_bias']==expected['lexical_bias']
        assert r['failure']==(min(da,db)<=0)
        assert abs(r['actual_input_tokens']-r['target_budget'])<=8
    assert not result['architecture_causality_confirmed']
    assert result['instruction_tuning_confound']
