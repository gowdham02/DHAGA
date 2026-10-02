from langchain_core.prompts import ChatPromptTemplate


FIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You analyze fit-related D2C fashion returns. Classify the specific fit issue
as exactly one of TOO_TIGHT, TOO_LOOSE, SIZE_MISMATCH, SHOULDER_FIT,
LENGTH_ISSUE, OTHER_FIT. Understand English, Hinglish, Tamil/vernacular terms,
spelling mistakes, and short comments. If the fit complaint is vague, choose
OTHER_FIT and lower confidence. Use only evidence from the text.""",
        ),
        ("human", "Return reason dropdown: {return_reason}\nFree-text detail: {return_reason_detail}"),
    ]
)
