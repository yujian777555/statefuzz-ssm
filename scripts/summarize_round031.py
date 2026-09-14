"""从原始证据重建Round31结果，不手工转录科学指标。"""
import argparse
import json
import hashlib
from pathlib import Path
from statefuzz.analyzer.stress_fingerprint import build_stress_fingerprint, compare_stress_fingerprints
from statefuzz.analyzer.memory_dependence import decompose_pairwise_preference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--reference', required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    raw=json.loads(Path(args.raw).read_text(encoding='utf-8'))
    reference=json.loads(Path(args.reference).read_text(encoding='utf-8'))
    fingerprint=build_stress_fingerprint(raw['records'])
    comparisons={}
    for model,payload in reference['models'].items():
        rows=[]
        for r in payload['records']:
            row=dict(seed=r['seed'],stress_family=r['family'],target_budget=r['target_budget_tokens'],actual_input_tokens=r['actual_input_tokens'],budget_tolerance=16,runtime_status='ok',min_signed_margin=r['min_signed_margin'])
            row.update(decompose_pairwise_preference(dict(direction_a_margin=r['signed_margin_a'],direction_b_margin=r['signed_margin_b'])))
            rows.append(row)
        comparisons[model]=compare_stress_fingerprints(build_stress_fingerprint(rows),fingerprint)
        comparisons[model]['limitations']=['历史Round26使用种子61/62和±16 token容差；本轮使用65—68和±8。周期族本身不随种子变化，四个种子不是四个独立文本。']
    data=dict(raw=raw,summary=fingerprint,comparison_to_existing_models=comparisons)
    for r in raw['records']:
        if r['runtime_status']!='ok': continue
        da=r['logit_prompt_a_candidate_a']-r['logit_prompt_a_candidate_b']
        db=r['logit_prompt_b_candidate_b']-r['logit_prompt_b_candidate_a']
        assert r['direction_a_margin']==da and r['direction_b_margin']==db
        assert r['memory_signal']==(da+db)/2 and r['lexical_bias']==(da-db)/2
        assert r['failure']==(min(da,db)<=0)
        assert r['prompt_token_counts']==[r['actual_input_tokens']]*2
        assert abs(r['actual_input_tokens']-r['target_budget'])<=8
    gate=raw['candidate_validation'][-1]
    assert gate['passed'] and len(gate['rows'])==4
    assert all(248<=r['actual_input_tokens']<=264 and r['min_signed_margin']>0 for r in gate['rows'])
    assert len(raw['records'])==120
    result=dict(round=31,status=raw['status'],execution_environment=raw['execution_environment'],hybrid_model='Zyphra/Zamba2-1.2B-Instruct-v2',local_checkpoint='/202532803004/models/Zamba2-1.2B-Instruct-v2',checkpoint_visible_from_executor=True,checkpoint_source='user_downloaded_modelscope_directory',offline_loading=True,instruction_tuning_confound=True,architecture_causality_confirmed=False,runtime_versions=raw['runtime_versions'],runtime_gate=raw['runtime_gate'],metric_definition=dict(direction_a_margin='logit(A|PromptA)-logit(B|PromptA)',direction_b_margin='logit(B|PromptB)-logit(A|PromptB)',memory_signal='(dA+dB)/2',lexical_bias='(dA-dB)/2'),short_context_gate=dict(target_tokens=256,tolerance_tokens=8,acceptable_interval=[248,264],passed=gate['passed'],rows=gate['rows']),selected_value_pair=raw['selected_value_pair'],stress_fingerprint=fingerprint,comparison_to_existing_models=comparisons,cache_state_observability=raw['cache_state_observability'],realistic_spot_check=dict(status='not_run_optional',reason='本轮完成冻结合成指纹；现实任务生成器不支持负控参数，未扩展任务模板。'),claim_scope='仅支持指定指令微调混合检查点的描述性压力指纹；规模、训练数据和微调混杂均未控制。',prior_scan=dict(protocol_role='exploratory_noncompliant_scan',counts_as_round031_short_context_gate=False,reason='旧版短门禁850 tokens、长扫描±16且派生公式不合规，未并入本轮。'),failures=[dict(step='旧版实施',error='850-token门禁与错误派生公式；已废弃并全量重跑'),dict(step='跨shell命令',error='早期引号错误产生SyntaxError及python未找到；改用上传脚本和明确解释器'),dict(step='GPU快速路径',error='缺少快速kernel，使用naive实现'),dict(step='token拟合提示',error='搜索中间文本超4096触发tokenizer警告；只做计数，实际forward受±8预算限制')],tests=dict(status='pending'),artifacts=['results/hybrid_stress_round_031.json','results/reference_round026_for_round031.json'],source_hashes=dict(raw=hashlib.sha256(Path(args.raw).read_bytes()).hexdigest(),reference=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest()))
    for name,payload in [('hybrid_stress_round_031.json',data),('result_round_031.json',result),('reference_round026_for_round031.json',reference)]:
        (root/'results'/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({f:{'valid':sum(p['valid_count'] for p in v['curve']),'risk':v['first_observed_risk_region']} for f,v in fingerprint.items()}))


if __name__=='__main__': main()
