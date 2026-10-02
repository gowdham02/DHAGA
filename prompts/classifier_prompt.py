from langchain_core.prompts import ChatPromptTemplate


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You classify D2C fashion return comments. Understand English, Hinglish,
Tamil/vernacular phrases, informal text, spelling errors, and short comments.
Choose exactly one category: FIT, QUALITY, PRODUCT, DELIVERY, OTHER.
For FIT, use a best-effort fit label; detailed FIT classification occurs later.
For non-FIT, use a concise uppercase label. Base the answer only on the text.
Low-information or ambiguous messages should have lower confidence.""",
        ),
        ("human", "Return reason dropdown: {return_reason}\nFree-text detail: {return_reason_detail}"),
    ]
)
