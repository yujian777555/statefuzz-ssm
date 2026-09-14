"""按修订后的Round31协议离线运行，原子保存可恢复记录。"""
import os
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[name] = '1'
import json
import sys
import socket
import platform
import hashlib
from pathlib import Path
import torch
import transformers
from statefuzz.runner.hybrid_causal_lm_runner import HybridCausalLMRunner, HybridCausalLMExperimentConfig
from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget

SEEDS = [65, 66, 67, 68]
PAIRS = [(' red', ' blue'), (' one', ' two'), (' cat', ' dog')]
FAMILIES = ['structured_repetitive', 'periodic_pattern', 'interleaved_distractor', 'semantic_distractor', 'lexically_diverse']
BUDGETS = [256, 768, 1280, 1792, 2560, 3584]
OUTPUT = Path('/202532803004/statefuzz_ssm_20260901/runs/round031_compliant.json')


def save(data):
    tmp = OUTPUT.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, OUTPUT)


def evaluate(runner, family, budget, seed, pair):
    row = dict(seed=seed, stress_family=family, target_budget=budget, budget_tolerance=8, value_pair=list(pair))
    fit = fit_remote_memory_pair_to_token_budget(runner.count_tokens, budget, tolerance_tokens=8, seed=seed, template_id=1, value_a=pair[0], value_b=pair[1], target_position=0.0, filler_style=family)
    row.update(runtime_status=fit.status, actual_input_tokens=fit.actual_tokens)
    if fit.pair is None:
        return row, None
    probe = fit.pair
    result = runner.score_remote_memory_pair(probe)
    ids = result['candidate_token_ids']
    row.update(candidate_token_ids=ids, prompt_token_counts=result['prompt_token_counts'])
    if not result['matched'] or not result['candidate_valid']:
        row['runtime_status'] = 'tokenizer_or_pair_invalid'
        return row, probe
    if abs(fit.actual_tokens-budget) > 8:
        raise ValueError('实际token超出冻结预算')
    for direction in ('a', 'b'):
        candidates = {c['token_id']: c['logit'] for c in result['score_'+direction]['candidates']}
        for index, candidate in enumerate(('a', 'b')):
            row[f'logit_prompt_{direction}_candidate_{candidate}'] = candidates[ids[index]]
    for key in ('direction_a_margin', 'direction_b_margin', 'raw_preference_prompt_a', 'raw_preference_prompt_b', 'memory_signal', 'lexical_bias', 'min_signed_margin'):
        row[key] = result[key]
    row.update(failure=result['min_signed_margin'] <= 0, runtime_status='ok', prompt_sha256=[hashlib.sha256(p.encode()).hexdigest() for p in (probe.prompt_a, probe.prompt_b)])
    return row, probe


def main():
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    out = dict(protocol_revision='3158afc', execution_environment=dict(expected_location='user_experiment_vm', hostname=socket.gethostname(), python_executable=sys.executable, python_version=platform.python_version(), gpu=torch.cuda.get_device_name(0)), runtime_versions=dict(torch=torch.__version__, transformers=transformers.__version__), candidate_validation=[], records=[], failures=[])
    runner = HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig())
    out['model_metadata'] = runner.model_metadata()
    supported = out['model_metadata'].get('max_context_tokens') or out['model_metadata']['tokenizer_model_max_length']
    ids = runner._tokenizer('The remembered word is', return_tensors='pt', add_special_tokens=False)['input_ids'].to(runner._model.device)
    with torch.inference_mode():
        first = runner._model(input_ids=ids, use_cache=True).logits
        second = runner._model(input_ids=ids, use_cache=True).logits
    out['runtime_gate'] = dict(finite=bool(torch.isfinite(first).all()), deterministic=bool(torch.equal(first, second)))
    if not all(out['runtime_gate'].values()):
        out['status']='hybrid_runtime_incompatible'; save(out); return
    selected = None
    for pair in PAIRS:
        rows = [evaluate(runner, 'structured_repetitive', 256, seed, pair)[0] for seed in SEEDS]
        valid = all(r['runtime_status']=='ok' and r['min_signed_margin']>0 for r in rows)
        out['candidate_validation'].append(dict(value_pair=list(pair), rows=rows, passed=valid))
        print(json.dumps({'gate_pair':pair, 'passed':valid, 'rows':rows}), flush=True)
        if valid:
            selected=pair; break
    out['selected_value_pair']=selected
    if selected is None:
        out['status']='hybrid_task_invalid'; save(out); return
    # 长扫描仅在独立短门禁通过后进行。
    out['status']='gate_passed_pending_sweep'
    if previous.get('protocol_revision') == out['protocol_revision'] and previous.get('selected_value_pair') == list(selected):
        out['records'] = previous.get('records', [])
    save(out)
    completed = {(r['stress_family'], r['target_budget'], r['seed']) for r in out['records']}
    for family in FAMILIES:
        for budget in BUDGETS:
            for seed in SEEDS:
                if (family, budget, seed) in completed:
                    continue
                if budget > supported:
                    row = dict(stress_family=family, target_budget=budget, seed=seed, runtime_status='unsupported_by_model_context_limit')
                else:
                    row, _ = evaluate(runner, family, budget, seed, selected)
                out['records'].append(row)
                save(out)
                print(json.dumps({'cell':[family,budget,seed], 'status':row['runtime_status']}), flush=True)
    out['cache_state_observability'] = {}
    for budget in (256, 3584):
        fit = fit_remote_memory_pair_to_token_budget(runner.count_tokens, budget, tolerance_tokens=8, seed=65, template_id=1, value_a=selected[0], value_b=selected[1], target_position=0.0, filler_style='structured_repetitive')
        if fit.pair is None:
            out['cache_state_observability'][str(budget)] = {'runtime_status':fit.status}
            continue
        encoded = runner._tokenizer(fit.pair.prompt_a, add_special_tokens=False, return_tensors='pt').to(runner._model.device)
        with torch.inference_mode():
            cache = runner._model(**encoded, use_cache=True, return_dict=True).past_key_values
        observations = []
        for i, layer in enumerate(getattr(cache, 'layers', [])):
            shapes = {name:list(getattr(layer,name).shape) for name in ('keys','values','conv_states','recurrent_states','ssm_states') if isinstance(getattr(layer,name,None), torch.Tensor)}
            observations.append(dict(layer=i, type=type(layer).__name__, shapes=shapes))
        out['cache_state_observability'][str(budget)] = dict(actual_input_tokens=fit.actual_tokens, cache_type=type(cache).__name__, layers=observations, attention_cache_observed=any('keys' in l['shapes'] for l in observations), ssm_recurrent_state_observed=any('recurrent_states' in l['shapes'] or 'ssm_states' in l['shapes'] for l in observations))
        save(out)
    out['status']='hybrid_validation_complete'
    save(out)


if __name__=='__main__':
    main()
