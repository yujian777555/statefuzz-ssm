def test_calibrated_prompt_is_deterministic_and_scales() -> None:
    from statefuzz.generator.calibrated import generate_calibrated_prompt

    short = generate_calibrated_prompt(context_tokens=64, seed=7)
    long = generate_calibrated_prompt(context_tokens=512, seed=7)
    assert short == generate_calibrated_prompt(context_tokens=64, seed=7)
    assert short != long
    assert short.endswith("The next symbol is")
    assert len(long) > len(short)


def test_calibrated_prompt_instances_are_deterministic_and_distinct() -> None:
    from statefuzz.generator.calibrated import generate_calibrated_prompts

    prompts = generate_calibrated_prompts(context_tokens=64, seed=7, instances=3)
    assert prompts == generate_calibrated_prompts(context_tokens=64, seed=7, instances=3)
    assert len(prompts) == 3
    assert len(set(prompts)) == 3

