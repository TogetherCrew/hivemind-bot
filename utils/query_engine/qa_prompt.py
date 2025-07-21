from llama_index.core import PromptTemplate
from utils.globals import NO_ANSWER_REFERENCE

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "give a short answer to the query.\n"
    "Instructions: \n"
    "1. Using only the information provided above (summary and messages), answer the following query concisely.\n"
    f"2. If the answer cannot be determined from the context, respond only with: '{NO_ANSWER_REFERENCE}'. \n"
    "3. Do not use any prior knowledge or assumptions.\n"
    "4. Do not do any recommendations or suggestions.\n\n"
    "Query: {query_str}\n"
    "Answer (concisely): "
)
