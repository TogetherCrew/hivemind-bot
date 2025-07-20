from llama_index.core import PromptTemplate
from utils.globals import NO_ANSWER_REFERENCE

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "give a short answer to the query.\n"
    "If the query is not related to the context, respond with "
    f"'{NO_ANSWER_REFERENCE}'\n"
    "Query: {query_str}\n"
    "Answer (concisely): "
)
