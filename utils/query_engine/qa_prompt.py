from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query very concisely in less than two paragraphs.\n"
    "Query: {query_str}\n"
    "Answer (concisely in less than two paragraphs): "
)
