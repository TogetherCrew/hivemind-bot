from llama_index.query_engine import BaseQueryEngine
from .level_based_platform_query_engine import LevelBasedPlatformQueryEngine


def prepare_discord_engine(
    community_id: str,
    thread_names: list[str],
    channel_names: list[str],
    days: list[str],
    **kwarg,
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
    level1_names : list[str]
        the given categorys to search for
    level2_names : list[str]
        the given topics to search for
    days : list[str]
        the given days to search for
    ** kwargs :
        similarity_top_k : int | None
            the k similar results to use when querying the data
            if not given, will load from `.env` file
        testing : bool
            whether to setup the PGVectorAccess in testing mode

    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    query_engine_preparation = get_discord_level_based_platform_query_engine()
    query_engine = query_engine_preparation.prepare_platform_engine(
        community_id=community_id,
        level1_names=thread_names,
        level2_names=channel_names,
        days=days,
        **kwarg,
    )
    return query_engine


def prepare_discord_engine_auto_filter(
    community_id: str,
    query: str,
    similarity_top_k: int | None = None,
    d: int | None = None,
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
    similarity_top_k : int | None
        the value for the initial summary search
        to get the `k2` count similar nodes
        if `None`, then would read from `.env`
    d : int
        this would make the secondary search (`prepare_discord_engine`)
        to be done on the `metadata.date - d` to `metadata.date + d`


    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """

    query_engine_preparation = get_discord_level_based_platform_query_engine()
    query_engine = query_engine_preparation.prepare_engine_auto_filter(
        community_id=community_id,
        query=query,
        similarity_top_k=similarity_top_k,
        d=d,
    )

    return query_engine


def get_discord_level_based_platform_query_engine() -> LevelBasedPlatformQueryEngine:
    level_based_query_engine = LevelBasedPlatformQueryEngine(
        level1_key="thread",
        level2_key="channel",
        platform_table_name="discord",
    )
    return level_based_query_engine
