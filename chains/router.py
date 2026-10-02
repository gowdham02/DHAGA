"""LangChain RunnableBranch routing: FIT records take the detailed analysis path."""

from collections.abc import Callable

from langchain_core.runnables import RunnableBranch, RunnableLambda

from models.schemas import ReturnCategory


def build_return_router(fit_handler: Callable[[dict], dict]):
    """Return a RunnableBranch, keeping FIT handling explicit and testable.

    The payload contains a `classification` object plus original row fields. Non-FIT
    results pass directly to the final-result branch.
    """
    return RunnableBranch(
        (
            lambda payload: payload["classification"].category == ReturnCategory.FIT,
            RunnableLambda(fit_handler),
        ),
        RunnableLambda(lambda payload: {**payload, "route": "NON_FIT"}),
    )
