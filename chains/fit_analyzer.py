"""Second stage of the FIT route, implemented as a LangChain chain."""

from langchain_core.runnables import Runnable

from chains.config import ModelConfig, build_chat_model
from models.schemas import FitAnalysis
from prompts.fit_prompt import FIT_PROMPT


def build_fit_analysis_chain(config: ModelConfig) -> Runnable:
    model = build_chat_model(config, config.classifier_model, config.classifier_temperature)
    return FIT_PROMPT | model.with_structured_output(FitAnalysis)
