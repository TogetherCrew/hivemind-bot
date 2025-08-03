from llama_index.core import PromptTemplate
from utils.globals import NO_ANSWER_REFERENCE_PLACEHOLDER

qa_prompt = PromptTemplate(
    "Below is the context for this query:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\nTask: Using *only* the information above, answer the user's question.\n"
    "If the answer does *not* appear in the context, reply **exactly**:\n\n"
    f"'{NO_ANSWER_REFERENCE_PLACEHOLDER}'\n\n"
    "Do not add any other text.\n"
    "\n--------------------------------\n\n"
    "Query:\n {query_str}\n"
    "Answer (concisely): "
)
