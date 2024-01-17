from llama_index.query_engine import BaseQueryEngine
from .level_based_platform_query_engine import LevelBasedPlatformQueryEngine


def prepare_discourse_engine(
    community_id: str,
    category_names: list[str],
    topic_names: list[str],
    days: list[str],
    **kwarg,
) -> BaseQueryEngine:
    """
    query the discourse database using filters given
    and give an anwer to the given query using the LLM

    Parameters
    ------------
    community_id : str
        the discourse community data to query
    query : str
        the query (question) of the user
    category_names : list[str]
        the given categorys to search for
    topic_names : list[str]
        the given topics to search for
    days : list[str]
        the given days to search for
    similarity_top_k : int | None
        the k similar results to use when querying the data
        if `None` will load from `.env` file
    ** kwargs :
        testing : bool
            whether to setup the PGVectorAccess in testing mode

    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    level_based_query_engine = get_discourse_level_based_platform_query_engine(
        table_name="discourse",
    )

    query_engine = level_based_query_engine.prepare_platform_engine(
        community_id=community_id,
        level1_names=category_names,
        level2_names=topic_names,
        days=days,
        **kwarg,
    )

    return query_engine


def prepare_discourse_engine_auto_filter(
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
    guild_id : str
        the discourse guild data to query
    query : str
        the query (question) of the user
    similarity_top_k : int | None
        the value for the initial summary search
        to get the `k2` count simliar nodes
        if `None`, then would read from `.env`
    d : int
        this would make the secondary search (`prepare_discourse_engine`)
        to be done on the `metadata.date - d` to `metadata.date + d`

    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    level_based_query_engine = get_discourse_level_based_platform_query_engine(
        table_name="discourse_summary"
    )

    query_engine = level_based_query_engine.prepare_engine_auto_filter(
        community_id=community_id,
        query=query,
        similarity_top_k=similarity_top_k,
        d=d,
    )
    return query_engine


def get_discourse_level_based_platform_query_engine(
    table_name: str,
) -> LevelBasedPlatformQueryEngine:
    """
    perpare the `LevelBasedPlatformQueryEngine` to use

    Parameters
    -----------
    table_name : str
        the postgresql data table to use

    Returns
    ---------
    level_based_query_engine : LevelBasedPlatformQueryEngine
        the query engine creator class
    """
    level_based_query_engine = LevelBasedPlatformQueryEngine(
        level1_key="category", level2_key="topic", platform_table_name=table_name
    )
    return level_based_query_engine
