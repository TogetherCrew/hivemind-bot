from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "give a short answer to the query.\n"
    "Query: {query_str}\n"
    "Answer (concisely): "
)
