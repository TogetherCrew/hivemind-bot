# the theshold to skip nodes of being included in an answer
RETRIEVER_THRESHOLD = 0
REFERENCE_SCORE_THRESHOLD = 0
INVALID_QUERY_RESPONSE = (
    "We're unable to process your query. Please refine it and try again."
)
QUERY_ERROR_MESSAGE = "Sorry, we're unable to process your question at the moment. Please try again later."
NO_ANSWER_REFERENCE = (
    "Current documentation and community chats draw a blank. "
    "Please ask the community manager for intel and we'll take notes for next time 🙂"
)
# An easier prompt for the LLM to understand
NO_ANSWER_REFERENCE_PLACEHOLDER = "I don't have enough information to answer that question."
NO_DATA_SOURCE_SELECTED = "No data source is currently selected. Please choose a data source from the dashboard and try again."

EXCLUDED_DATE_MARGIN = 5  # minutes
K1_RETRIEVER_SEARCH=100
K2_RETRIEVER_SEARCH=50
D_RETRIEVER_SEARCH=7

RERANK_TOP_K=10