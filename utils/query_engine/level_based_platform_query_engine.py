import logging

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from utils.query_engine.utils import (
    LevelBasedPlatformUtils,
)
from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index import VectorStoreIndex
from llama_index.query_engine import CustomQueryEngine
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.retrievers import BaseRetriever
from llama_index.schema import NodeWithScore
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


class LevelBasedPlatformQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        """Doing custom query"""
        # first retrieving similar nodes in summary
        retriever = RetrieveSimilarNodes(
            self._raw_vector_store,
            self._similarity_top_k,
        )

        similar_nodes = retriever.query_db(
            query=query_str, filters=self._filters, date_interval=self._d
        )

        context_str = self._prepare_context_str(similar_nodes, self.summary_nodes)
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(fmt_qa_prompt)

        return str(response)

    @classmethod
    def prepare_platform_engine(
        cls,
        community_id: str,
        platform_table_name: str,
        filters: list[dict[str, str]] | None = None,
        testing=False,
        **kwargs,
    ) -> "LevelBasedPlatformQueryEngine":
        """
        query the platform database using filters given
        and give an anwer to the given query using the LLM

        Parameters
        ------------
        community_id : str
            the community id data to query
        platform_table_name : str
            the postgresql table name for the platform. Can be only the platform name
            as `discord` or `discourse`
        filters : list[dict[str, str]] | None
            the list of filters to be applied when retrieving data
            if `None` then set no filtering on PGVectorStore
        testing : bool
            if `True` it is in test phase and nothing must be changed
        similarity_top_k : int | None
            the k similar results to use when querying the data
            if not given, will load from `.env` file
        **kwargs :
            llm : llama-index.LLM
                the LLM to use answering queries
                default is gpt-4
            synthesizer : llama_index.response_synthesizers.base.BaseSynthesizer
                the synthesizers to use when creating the prompt
                default is to get from `get_response_synthesizer(response_mode="compact")`
            qa_prompt : llama-index.prompts.PromptTemplate
                the Q&A prompt to use
                default would be the default prompt of llama-index

        Returns
        ---------
        query_engine : BaseQueryEngine
            the created query engine with the filters
        """
        dbname = f"community_{community_id}"

        synthesizer = kwargs.get(
            "synthesizer", get_response_synthesizer(response_mode="compact")
        )
        llm = kwargs.get("llm", OpenAI("gpt-4"))
        qa_prompt_ = kwargs.get("qa_prompt", qa_prompt)

        index = cls._setup_vector_store_index(platform_table_name, dbname, testing)
        retriever = index.as_retriever()
        _, similarity_top_k, d = load_hyperparams()
        cls._d = d

        cls._raw_vector_store = index._vector_store
        cls._summary_vector_store = cls._setup_vector_store_index(
            platform_table_name + "_summary", dbname, testing
        )._vector_store
        cls._similarity_top_k = similarity_top_k
        cls._filters = filters

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt_,
        )

    @classmethod
    def prepare_engine_auto_filter(
        cls,
        community_id: str,
        query: str,
        platform_table_name: str,
        level1_key: str,
        level2_key: str,
        date_key: str = "date",
        include_summary_context: bool = False,
    ) -> "LevelBasedPlatformQueryEngine":
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
            this query would be used for filters preparation
            which filters are based on available summaries.
        platform_table_name : str
            the postgresql table name for the platform. Can be only the platform name
            as `discord` or `discourse`
        level1_key : str
            first hierarchy of the discussion.
            the platforms can be discord or discourse. for example in discord
            the level1 is `channel` and in discourse it can be `category`
        level2_key : str
            the second level of discussion in the hierarchy.
            For example in discord level2 is `thread`,
            and on discourse level2 would be `topic`
        date_key : str
            the day key which the date is saved under the field in postgresql table.
            for default is is `date` which was the one that we used previously

        Returns
        ---------
        query_engine : BaseQueryEngine
            the created query engine with the filters
        """
        dbname = f"community_{community_id}"
        summary_similarity_top_k, _, d = load_hyperparams()

        # For summaries data a posfix `summary` would be added
        platform_retriever = ForumBasedSummaryRetriever(
            table_name=platform_table_name + "_summary", dbname=dbname
        )

        nodes = platform_retriever.get_similar_nodes(query, summary_similarity_top_k)

        filters = platform_retriever.define_filters(
            nodes,
            metadata_group1_key=level1_key,
            metadata_group2_key=level2_key,
            metadata_date_key=date_key,
        )
        # saving to add summaries to the context of prompt
        if include_summary_context:
            cls.summary_nodes = nodes
        else:
            cls.summary_nodes = []

        cls._utils_class = LevelBasedPlatformUtils(level1_key, level2_key, date_key)
        cls._level1_key = level1_key
        cls._level2_key = level2_key
        cls._date_key = date_key
        cls._d = d
        cls._platform_table_name = platform_table_name

        logging.debug(f"COMMUNITY_ID: {community_id} | summary filters: {filters}")

        engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
            community_id=community_id,
            platform_table_name=platform_table_name,
            filters=filters,
        )
        return engine

    def _prepare_context_str(
        self, raw_nodes: list[NodeWithScore], summary_nodes: list[NodeWithScore]
    ) -> str:
        """
        prepare the prompt context using the raw_nodes for answers and summary_nodes for additional information
        """
        context_str: str = ""

        if summary_nodes == []:
            logging.warning(
                "Empty context_nodes. Cannot add summaries as context information!"
            )

            context_str += self._utils_class.prepare_prompt_with_metadata_info(
                nodes=raw_nodes
            )
        else:
            # grouping the data we have so we could
            # get them per each metadata without search
            (
                grouped_raw_nodes,
                grouped_summary_nodes,
            ) = self._group_summary_and_raw_nodes(raw_nodes, summary_nodes)

            # first using the available summary nodes try to create prompt
            context_data, (
                summary_nodes_to_fetch_filters,
                raw_nodes_missed,
            ) = self._utils_class.prepare_context_str_based_on_summaries(
                grouped_raw_nodes, grouped_summary_nodes
            )
            context_str += context_data

            logging.info(
                f"summary_nodes_to_fetch_filters {summary_nodes_to_fetch_filters}"
            )
            # then if there was some missing summaries
            if len(summary_nodes_to_fetch_filters):
                retriever = RetrieveSimilarNodes(
                    self._summary_vector_store,
                    similarity_top_k=None,
                )
                fetched_summary_nodes = retriever.query_db(
                    query="",
                    filters=summary_nodes_to_fetch_filters,
                    ignore_sort=True,
                )
                logging.info(f"len(fetched_summary_nodes) {len(fetched_summary_nodes)}")
                logging.info(f"fetched_summary_nodes {fetched_summary_nodes}")
                grouped_summary_nodes = self._utils_class.group_nodes_per_metadata(
                    fetched_summary_nodes
                )
                logging.info(f"grouped_summary_nodes {grouped_summary_nodes}")
                logging.info(f"len(grouped_summary_nodes) {len(grouped_summary_nodes)}")
                context_data, (
                    summary_nodes_to_fetch_filters,
                    _,
                ) = self._utils_class.prepare_context_str_based_on_summaries(
                    raw_nodes_missed, grouped_summary_nodes
                )
                context_str += context_data

        logging.debug(f"context_str of prompt\n" f"{context_str}")

        return context_str

    @classmethod
    def _setup_vector_store_index(
        cls, platform_table_name: str, dbname: str, testing: str
    ) -> VectorStoreIndex:
        """
        prepare the vector_store for querying data
        """
        pg_vector = PGVectorAccess(
            table_name=platform_table_name,
            dbname=dbname,
            testing=testing,
            embed_model=CohereEmbedding(),
        )
        index = pg_vector.load_index()
        return index

    def _group_summary_and_raw_nodes(
        self, raw_nodes: list[NodeWithScore], summary_nodes: list[NodeWithScore]
    ) -> tuple[
        dict[str, dict[str, dict[str, list[NodeWithScore]]]],
        dict[str, dict[str, dict[str, list[NodeWithScore]]]],
    ]:
        """a wrapper to do the grouping of given nodes"""
        grouped_raw_nodes = self._utils_class.group_nodes_per_metadata(raw_nodes)
        grouped_summary_nodes = self._utils_class.group_nodes_per_metadata(
            summary_nodes
        )

        return grouped_raw_nodes, grouped_summary_nodes
