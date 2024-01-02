from llama_index import QueryBundle
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from utils.query_engine.discord_query_engine import prepare_discord_engine_auto_filter


def query_discord(
    community_id: str,
    query: str,
) -> str:
    """
    query the llm using the query engine

    Parameters
    ------------
    query_engine : BaseQueryEngine
        the prepared query engine
    query : str
        the string question
    """
    query_engine = prepare_discord_engine_auto_filter(
        community_id=community_id,
        query=query,
    )
    query_bundle = QueryBundle(
        query_str=query, embedding=CohereEmbedding().get_text_embedding(text=query)
    )
    response = query_engine.query(query_bundle)
    return response
