from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.process_dates import process_dates
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.query_engine import BaseQueryEngine
from llama_index.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess


class LevelBasedPlatformQueryEngine:
    def __init__(
        self,
        level1_key: str,
        level2_key: str,
        platform_table_name: str,
        date_key: str = "date",
    ) -> None:
        """
        A two level based platform query engine preparation tools.

        Parameters
        ------------
        level1_key : str
            first hierarchy of the discussion.
            the platforms can be discord or discourse. for example in discord
            the level1 is `channel` and level2 is `thread`.
        level2_key : str
            the second level of discussion in the hierarchy.
            e.g.: In discourse the level1 can be `category` and level2 would be `topic`
        platform_table_name : str
            the postgresql table name for the platform. Can be only the platform name
            as `discord` or `discourse`
        date_key : str
            the day key which the date is saved under the field in postgresql table.
            for default is is `date` which was the one that we used previously
        """
        self.level1_key = level1_key
        self.level2_key = level2_key
        self.platform_table_name = platform_table_name
        self.date_key = date_key

    def prepare_platform_engine(
        self,
        community_id: str,
        level1_names: list[str],
        level2_names: list[str],
        days: list[str],
        **kwarg,
    ) -> BaseQueryEngine:
        """
        query the platform database using filters given
        and give an anwer to the given query using the LLM

        Parameters
        ------------
        community_id : str
            the community id data to query
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
        dbname = f"community_{community_id}"

        testing = kwarg.get("testing", False)
        similarity_top_k = kwarg.get("similarity_top_k", None)

        pg_vector = PGVectorAccess(
            table_name=self.platform_table_name,
            dbname=dbname,
            testing=testing,
            embed_model=CohereEmbedding(),
        )
        index = pg_vector.load_index()
        if similarity_top_k is None:
            _, similarity_top_k, _ = load_hyperparams()

        level2_filters: list[ExactMatchFilter] = []
        level1_filters: list[ExactMatchFilter] = []
        day_filters: list[ExactMatchFilter] = []

        for level1 in level1_names:
            level1_name_value = level1.replace("'", "''")
            level1_filters.append(
                ExactMatchFilter(key=self.level1_key, value=level1_name_value)
            )

        for level2 in level2_names:
            levle2_value = level2.replace("'", "''")
            level2_filters.append(
                ExactMatchFilter(key=self.level2_key, value=levle2_value)
            )

        for day in days:
            day_filters.append(ExactMatchFilter(key=self.date_key, value=day))

        all_filters: list[ExactMatchFilter] = []
        all_filters.extend(level1_filters)
        all_filters.extend(level2_filters)
        all_filters.extend(day_filters)

        filters = MetadataFilters(filters=all_filters, condition=FilterCondition.OR)

        query_engine = index.as_query_engine(
            filters=filters, similarity_top_k=similarity_top_k
        )

        return query_engine

    def prepare_engine_auto_filter(
        self,
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
        community id : str
            the community id to process its platform data
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
        dbname = f"community_{community_id}"

        if d is None:
            _, _, d = load_hyperparams()
        if similarity_top_k is None:
            similarity_top_k, _, _ = load_hyperparams()

        discourse_retriever = ForumBasedSummaryRetriever(
            table_name=self.platform_table_name, dbname=dbname
        )

        level1_names, level2_names, dates = discourse_retriever.retreive_metadata(
            query=query,
            metadata_group1_key=self.level1_key,
            metadata_group2_key=self.level2_key,
            metadata_date_key=self.date_key,
            similarity_top_k=similarity_top_k,
        )

        dates_modified = process_dates(list(dates), d)

        engine = self.prepare_platform_engine(
            community_id=community_id,
            query=query,
            level1_names=list(level1_names),
            level2_names=list(level2_names),
            days=dates_modified,
        )
        return engine
