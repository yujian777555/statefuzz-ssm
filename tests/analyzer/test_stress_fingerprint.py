import pytest
from statefuzz.analyzer.stress_fingerprint import build_stress_fingerprint, compare_stress_fingerprints


def row(seed,budget,signal,margin):
    return dict(seed=seed, stress_family='structured_repetitive', target_budget=budget, actual_input_tokens=budget, runtime_status='ok', memory_signal=signal, min_signed_margin=margin, lexical_bias=signal-margin)


def test_normalization_and_first_risk():
    result=build_stress_fingerprint([row(s,b,v,m) for s in (65,66) for b,v,m in ((256,2,1),(768,1,-0.5))])
    family=result['structured_repetitive']
    assert family['curve'][1]['normalized_memory_retention']==0.5
    assert family['curve'][1]['failure_rate']==1
    assert family['first_observed_risk_region']==[768,768]
    assert not compare_stress_fingerprints(result,result)['architecture_causality_confirmed']


def test_unreachable_is_not_failure_and_missing_seed_not_normalized():
    rows=[row(65,256,2,1),row(66,256,2,1),row(65,768,1,0.5),dict(seed=66,stress_family='structured_repetitive',target_budget=768,runtime_status='budget_unreachable')]
    point=build_stress_fingerprint(rows)['structured_repetitive']['curve'][1]
    assert point['failure_rate']==0
    assert point['normalized_memory_retention'] is None
    assert not point['complete_seed_set']


def test_rejects_duplicate_and_off_budget():
    r=row(65,256,2,1)
    with pytest.raises(ValueError): build_stress_fingerprint([r,r])
    with pytest.raises(ValueError): build_stress_fingerprint([{**r,'actual_input_tokens':850}])
