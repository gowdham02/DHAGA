"""Central, environment-backed model and optional cost configuration."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    classifier_model: str
    evaluator_model: str
    api_key: str | None
    provider: str
    api_base_url: str | None
    openrouter_site_url: str | None
    openrouter_app_name: str
    classifier_temperature: float
    evaluator_temperature: float
    classifier_input_cost_per_million: float | None
    classifier_output_cost_per_million: float | None
    evaluator_input_cost_per_million: float | None
    evaluator_output_cost_per_million: float | None

    @property
    def has_live_llm(self) -> bool:
        return bool(self.api_key) and self.provider.lower() in {"openai", "openrouter"}


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number when set.") from exc
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative.")
    return parsed


def load_model_config() -> ModelConfig:
    load_dotenv()
    return ModelConfig(
        classifier_model=os.getenv("CLASSIFIER_MODEL", "gpt-4.1-mini"),
        evaluator_model=os.getenv("EVALUATOR_MODEL", "gpt-4.1"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        provider=os.getenv("LLM_PROVIDER", "openai"),
        api_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if os.getenv("LLM_PROVIDER", "openai").lower() == "openrouter" else None,
        openrouter_site_url=os.getenv("OPENROUTER_SITE_URL") or None,
        openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "DHAGA FITLENS"),
        classifier_temperature=float(os.getenv("CLASSIFIER_TEMPERATURE", "0")),
        evaluator_temperature=float(os.getenv("EVALUATOR_TEMPERATURE", "0")),
        classifier_input_cost_per_million=_optional_float("CLASSIFIER_INPUT_COST_PER_MILLION"),
        classifier_output_cost_per_million=_optional_float("CLASSIFIER_OUTPUT_COST_PER_MILLION"),
        evaluator_input_cost_per_million=_optional_float("EVALUATOR_INPUT_COST_PER_MILLION"),
        evaluator_output_cost_per_million=_optional_float("EVALUATOR_OUTPUT_COST_PER_MILLION"),
    )


def build_chat_model(config: ModelConfig, model_name: str, temperature: float):
    """Build the LangChain chat-model integration for OpenAI or OpenRouter."""
    if not config.has_live_llm:
        raise RuntimeError("Live LLM configuration is unavailable. Add LLM_API_KEY to use live analysis.")
    if config.provider.lower() not in {"openai", "openrouter"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'openrouter'.")
    from langchain_openai import ChatOpenAI

    options = {
        "model": model_name,
        "api_key": config.api_key,
        "temperature": temperature,
        "timeout": 45,
        "max_retries": 1,
    }
    if config.provider.lower() == "openrouter":
        # OpenRouter has an OpenAI-compatible /chat/completions endpoint.
        options["base_url"] = config.api_base_url
        headers = {"X-OpenRouter-Title": config.openrouter_app_name}
        if config.openrouter_site_url:
            headers["HTTP-Referer"] = config.openrouter_site_url
        options["default_headers"] = headers
    return ChatOpenAI(**options)
