from chains.config import ModelConfig, build_chat_model


def test_openrouter_langchain_model_uses_compatible_endpoint():
    config = ModelConfig(
        classifier_model="openai/gpt-4.1-mini",
        evaluator_model="openai/gpt-4.1",
        api_key="test-key",
        provider="openrouter",
        api_base_url="https://openrouter.ai/api/v1",
        openrouter_site_url=None,
        openrouter_app_name="DHAGA FITLENS",
        classifier_temperature=0,
        evaluator_temperature=0,
        classifier_input_cost_per_million=None,
        classifier_output_cost_per_million=None,
        evaluator_input_cost_per_million=None,
        evaluator_output_cost_per_million=None,
    )
    model = build_chat_model(config, config.classifier_model, 0)
    assert str(model.openai_api_base).rstrip("/") == "https://openrouter.ai/api/v1"
