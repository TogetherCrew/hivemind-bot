import logging

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from utils.query_engine.base_pg_engine import BasePGEngine
from utils.query_engine.level_based_platforms_util import LevelBasedPlatformUtils

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

        context_str = self._prepare_context_str(similar_nodes, summary_nodes=None)
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(fmt_qa_prompt)
        logging.debug(f"fmt_qa_prompt:\n{fmt_qa_prompt}")

        return Response(response=str(response), source_nodes=similar_nodes)

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
            index_raw : VectorStoreIndex
                the vector store index for raw data
                If not passed, it would just create one itself
            index_summary : VectorStoreIndex
                the vector store index for summary data
                If not passed, it would just create one itself
            summary_nodes_filters : list[dict[str, str]]
                a list of filters to fetch the summary nodes
                for default, not passing this would mean to use previous nodes
                but if passed we would re-fetch nodes.
                This could be benefitial in case we want to do some manual
                processing with nodes

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
        base_engine = BasePGEngine(platform_table_name, community_id)
        index: VectorStoreIndex = kwargs.get(
            "index_raw",
            base_engine._setup_vector_store_index(
                testing=testing,
            ),
        )
        summary_nodes_filters = kwargs.get("summary_nodes_filters", None)

        retriever = index.as_retriever()
        cls._summary_vector_store = kwargs.get(
            "index_summary",
            base_engine._setup_vector_store_index(
                table_name=platform_table_name + "_summary",
                dbname=dbname,
                testing=testing,
            ),
        )._vector_store

        _, similarity_top_k, d = load_hyperparams()
        cls._d = d

        cls._raw_vector_store = index._vector_store

        cls._similarity_top_k = similarity_top_k
        cls._filters = filters
        cls._summary_nodes_filters = summary_nodes_filters

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

        base_engine = BasePGEngine(
            platform_table_name + "_summary",
            community_id,
        )

        index_summary = base_engine._setup_vector_store_index(
            testing=False,
        )
        vector_store = index_summary._vector_store

        retriever = RetrieveSimilarNodes(
            vector_store,
            summary_similarity_top_k,
        )
        # getting nodes of just thread summaries
        nodes = retriever.query_db(query, [{"type": "thread"}])

        # For summaries data a posfix `summary` would be added
        platform_retriever = ForumBasedSummaryRetriever(
            table_name=platform_table_name + "_summary", dbname=dbname
        )

        raw_nodes_filters = platform_retriever.define_filters(
            nodes,
            metadata_group1_key=level1_key,
            metadata_group2_key=level2_key,
            metadata_date_key=date_key,
        )
        summary_nodes_filters = platform_retriever.define_filters(
            nodes,
            metadata_group1_key=level1_key,
            metadata_group2_key=level2_key,
            metadata_date_key=date_key,
            # we will always use thread summaries
            and_filters={"type": "thread"},
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

        logging.debug(
            f"COMMUNITY_ID: {community_id} | raw filters: {raw_nodes_filters}"
        )

        engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
            community_id=community_id,
            platform_table_name=platform_table_name,
            filters=raw_nodes_filters,
            index_summary=index_summary,
            summary_nodes_filters=summary_nodes_filters,
        )
        return engine

    def _prepare_context_str(
        self, raw_nodes: list[NodeWithScore], summary_nodes: list[NodeWithScore] | None
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
        elif summary_nodes is None:
            retriever = RetrieveSimilarNodes(
                self._summary_vector_store,
                similarity_top_k=None,
            )
            # Note: `self._summary_nodes_filters` must be set before
            fetched_summary_nodes = retriever.query_db(
                query="",
                filters=self._summary_nodes_filters,
                aggregate_records=True,
                ignore_sort=True,
                group_by_metadata=["thread", "date", "channel"],
                date_interval=self._d,
            )
            grouped_summary_nodes = self._utils_class.group_nodes_per_metadata(
                fetched_summary_nodes
            )
            grouped_raw_nodes = self._utils_class.group_nodes_per_metadata(raw_nodes)
            context_data, (
                summary_nodes_to_fetch_filters,
                _,
            ) = self._utils_class.prepare_context_str_based_on_summaries(
                grouped_raw_nodes, grouped_summary_nodes
            )
            context_str += context_data
        else:
            # grouping the data we have so we could
            # get them per each metadata without looping over them
            grouped_raw_nodes = self._utils_class.group_nodes_per_metadata(raw_nodes)
            grouped_summary_nodes = self._utils_class.group_nodes_per_metadata(
                summary_nodes
            )

            # first using the available summary nodes try to create prompt
            context_data, (
                summary_nodes_to_fetch_filters,
                raw_nodes_missed,
            ) = self._utils_class.prepare_context_str_based_on_summaries(
                grouped_raw_nodes, grouped_summary_nodes
            )
            context_str += context_data

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
                grouped_summary_nodes = self._utils_class.group_nodes_per_metadata(
                    fetched_summary_nodes
                )
                context_data, (
                    summary_nodes_to_fetch_filters,
                    _,
                ) = self._utils_class.prepare_context_str_based_on_summaries(
                    raw_nodes_missed, grouped_summary_nodes
                )
                context_str += context_data

        logging.debug(f"context_str of prompt\n" f"{context_str}")

        return context_str
