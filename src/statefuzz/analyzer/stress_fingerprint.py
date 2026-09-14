"""严格预算记录的模型内归一化指纹。"""
import math
from statistics import mean, median


def build_stress_fingerprint(records):
    families = {}
    seen = set()
    for row in records:
        key = (row['stress_family'], row['target_budget'], row['seed'])
        if key in seen:
            raise ValueError('重复实验单元')
        seen.add(key)
        if row['runtime_status'] == 'ok':
            if abs(row['actual_input_tokens']-row['target_budget']) > row.get('budget_tolerance',8):
                raise ValueError('实际token超出预算')
            if not all(math.isfinite(row[k]) for k in ('memory_signal','lexical_bias','min_signed_margin')):
                raise ValueError('非有限指标')
        families.setdefault(row['stress_family'], []).append(row)
    output = {}
    for family, rows in families.items():
        curve = []
        seed_set = {r['seed'] for r in rows}
        baseline_signal = baseline_margin = None
        first_risk = None
        for budget in sorted({r['target_budget'] for r in rows}):
            cells = [r for r in rows if r['target_budget']==budget]
            valid = [r for r in cells if r['runtime_status']=='ok']
            complete = {r['seed'] for r in valid} == seed_set
            point = dict(target_budget=budget, valid_count=len(valid), expected_seed_count=len(seed_set), complete_seed_set=complete, unavailable_cells=[r for r in cells if r['runtime_status']!='ok'])
            if valid:
                signal, margin = mean(r['memory_signal'] for r in valid), mean(r['min_signed_margin'] for r in valid)
                if budget == 256 and complete and all(r['min_signed_margin']>0 for r in valid):
                    baseline_signal, baseline_margin = signal, margin
                failure_count = sum(r['min_signed_margin']<=0 for r in valid)
                point.update(actual_token_range=[min(r['actual_input_tokens'] for r in valid),max(r['actual_input_tokens'] for r in valid)], failure_count=failure_count, failure_rate=failure_count/len(valid), mean_memory_signal=signal, median_memory_signal=median(r['memory_signal'] for r in valid), mean_min_signed_margin=margin, median_min_signed_margin=median(r['min_signed_margin'] for r in valid), mean_lexical_bias=mean(r['lexical_bias'] for r in valid), normalized_memory_retention=signal/baseline_signal if complete and baseline_signal else None, normalized_margin_retention=margin/baseline_margin if complete and baseline_margin else None)
                if failure_count and first_risk is None:
                    first_risk=point['actual_token_range']
            curve.append(point)
        violations=[]
        for seed in sorted(seed_set):
            history=sorted([r for r in rows if r['seed']==seed and r['runtime_status']=='ok'],key=lambda r:r['target_budget'])
            for a,b in zip(history,history[1:]):
                if b['memory_signal']>a['memory_signal']:
                    violations.append(dict(seed=seed, from_budget=a['target_budget'], to_budget=b['target_budget'], kind='signal_increase'))
                if a['min_signed_margin']<=0<b['min_signed_margin']:
                    violations.append(dict(seed=seed, from_budget=a['target_budget'], to_budget=b['target_budget'], kind='failure_recovery'))
        complete_points=[p for p in curve if p['complete_seed_set'] and 'actual_token_range' in p]
        output[family]=dict(curve=curve, short_context_valid=baseline_signal is not None, first_observed_risk_region=first_risk, observed_all_pass_lower_bound=min(complete_points[-1]['actual_token_range']) if complete_points and first_risk is None else None, monotonicity_violations=violations, negative_control=family=='lexically_diverse')
    return output


def compare_stress_fingerprints(reference, candidate):
    """只比较相同预算的模型内归一化量，保留混杂因素。"""
    comparisons=[]
    for family in sorted(set(reference)&set(candidate)):
        ref={p['target_budget']:p for p in reference[family]['curve']}
        for p in candidate[family]['curve']:
            old=ref.get(p['target_budget'],{})
            a,b=old.get('normalized_memory_retention'),p.get('normalized_memory_retention')
            if a is not None and b is not None:
                comparisons.append(dict(family=family,target_budget=p['target_budget'],reference_retention=a,candidate_retention=b))
    return dict(classification='inconclusive_due_to_scale_training_tuning_confounds', normalized_comparisons=comparisons, architecture_causality_confirmed=False, scale_confound=True, training_data_confound=True, instruction_tuning_confound=True)
