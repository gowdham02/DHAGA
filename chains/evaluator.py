"""Independent evaluator chain that deliberately uses the stronger model."""

from langchain_core.runnables import Runnable

from chains.config import ModelConfig, build_chat_model
from models.schemas import EvaluationResult
from prompts.evaluator_prompt import EVALUATOR_PROMPT


def build_evaluator_chain(config: ModelConfig) -> Runnable:
    model = build_chat_model(config, config.evaluator_model, config.evaluator_temperature)
    return EVALUATOR_PROMPT | model.with_structured_output(EvaluationResult)
