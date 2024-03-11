from llama_index.core.query_engine import BaseQueryEngine

from .level_based_platform_query_engine import LevelBasedPlatformQueryEngine


def prepare_discord_engine(
    community_id: str,
    filters: list[dict[str, str]],
    **kwargs,
) -> BaseQueryEngine:
    """
    query the platform database using filters given
    and give an anwer to the given query using the LLM

    Parameters
    ------------
    community_id : str
        the discord community id data to query
    query : str
        the query (question) of the user
    filters : list[dict[str, str]] | None
        the list of filters to be applied when retrieving data
        if `None` then set no filtering on PGVectorStore
    ** kwargs :
        testing : bool
            whether to setup the PGVectorAccess in testing mode

    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """

    testing = kwargs.get("testing", False)
    query_engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
        community_id=community_id,
        platform_table_name="discord",
        filters=filters,
        testing=testing,
    )
    return query_engine


def prepare_discord_engine_auto_filter(
    community_id: str,
    query: str,
) -> BaseQueryEngine:
    """
    get the query engine and do the filtering automatically.
    By automatically we mean, it would first query the summaries
    to get the metadata filters

    Parameters
    -----------
    community_id : str
        the discord community data to query
    query : str
        the query (question) of the user
        this query will be used to fetch the filters from similar summaries nodes


    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """

    engine = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
        community_id=community_id,
        query=query,
        platform_table_name="discord",
        level1_key="channel",
        level2_key="thread",
        date_key="date",
        include_summary_context=True,
    )
    return engine
