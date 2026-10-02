"""First LangChain chain: bulk high-level return classification."""

from langchain_core.runnables import Runnable

from chains.config import ModelConfig, build_chat_model
from models.schemas import ReturnClassification
from prompts.classifier_prompt import CLASSIFIER_PROMPT


def build_classifier_chain(config: ModelConfig) -> Runnable:
    """Create a Pydantic-validated LangChain chain using the economical model."""
    model = build_chat_model(config, config.classifier_model, config.classifier_temperature)
    return CLASSIFIER_PROMPT | model.with_structured_output(ReturnClassification)
