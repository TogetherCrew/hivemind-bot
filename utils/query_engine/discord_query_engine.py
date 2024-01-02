from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.process_dates import process_dates
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import BaseQueryEngine
from llama_index.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters
from tc_hivemind_backend.pg_vector_access import PGVectorAccess


def prepare_discord_engine(
    community_id: str,
    thread_names: list[str],
    channel_names: list[str],
    days: list[str],
    similarity_top_k: int | None = None,
    **kwarg,
) -> BaseQueryEngine:
    """
    query the discord database using filters given
    and give an anwer to the given query using the LLM

    Parameters
    ------------
    guild_id : str
        the discord guild data to query
    query : str
        the query (question) of the user
    thread_names : list[str]
        the given threads to search for
    channel_names : list[str]
        the given channels to search for
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
    table_name = "discord"
    dbname = f"community_{community_id}"

    testing = kwarg.get("testing", False)

    pg_vector = PGVectorAccess(table_name=table_name, dbname=dbname, testing=testing)
    index = pg_vector.load_index()
    if similarity_top_k is None:
        _, similarity_top_k, _ = load_hyperparams()

    thread_filters: list[ExactMatchFilter] = []
    channel_filters: list[ExactMatchFilter] = []
    day_filters: list[ExactMatchFilter] = []

    for channel in channel_names:
        channel_updated = channel.replace("'", "''")
        channel_filters.append(ExactMatchFilter(key="channel", value=channel_updated))

    for thread in thread_names:
        thread_updated = thread.replace("'", "''")
        thread_filters.append(ExactMatchFilter(key="thread", value=thread_updated))

    for day in days:
        day_filters.append(ExactMatchFilter(key="date", value=day))

    all_filters: list[ExactMatchFilter] = []
    all_filters.extend(thread_filters)
    all_filters.extend(channel_filters)
    all_filters.extend(day_filters)

    filters = MetadataFilters(filters=all_filters, condition=FilterCondition.OR)

    query_engine = index.as_query_engine(
        filters=filters, similarity_top_k=similarity_top_k
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
    guild_id : str
        the discord guild data to query
    query : str
        the query (question) of the user
    similarity_top_k : int | None
        the value for the initial summary search
        to get the `k2` count simliar nodes
        if `None`, then would read from `.env`
    d : int
        this would make the secondary search (`prepare_discord_engine`)
        to be done on the `metadata.date - d` to `metadata.date + d`


    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    table_name = "discord_summary"
    dbname = f"community_{community_id}"

    if d is None:
        _, _, d = load_hyperparams()
    if similarity_top_k is None:
        similarity_top_k, _, _ = load_hyperparams()

    discord_retriever = ForumBasedSummaryRetriever(table_name=table_name, dbname=dbname)

    channels, threads, dates = discord_retriever.retreive_metadata(
        query=query,
        metadata_group1_key="channel",
        metadata_group2_key="thread",
        metadata_date_key="date",
        similarity_top_k=similarity_top_k,
    )

    dates_modified = process_dates(list(dates), d)

    engine = prepare_discord_engine(
        community_id=community_id,
        query=query,
        thread_names=list(threads),
        channel_names=list(channels),
        days=dates_modified,
    )
    return engine
