from langchain_core.prompts import ChatPromptTemplate


EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an independent quality evaluator for return-comment classifications.
Check whether the proposed category and subcategory are supported by the original
customer text. Do not infer facts not in the text. Reject ambiguous, unsupported,
or contradictory classifications and require human review. Approve only when the
evidence supports the proposal. Consider English, Hinglish, informal and mixed
language comments.""",
        ),
        (
            "human",
            "Original dropdown: {return_reason}\nOriginal text: {return_reason_detail}\n\n"
            "Proposed category: {category}\nProposed subcategory: {subcategory}\n"
            "Classifier confidence: {classifier_confidence}\n"
            "Classifier explanation: {explanation}",
        ),
    ]
)
