import logging

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.process_dates import process_dates
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
)
from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.schema import NodeWithScore
from llama_index.schema import MetadataMode


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
            self._vector_store,
            self._similarity_top_k,
        )
        similar_nodes = retriever.query_db(query=query_str, filters=self._filters)

        context_str = self._prepare_context_str(similar_nodes)
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(fmt_qa_prompt)
        # logging.info(f"fmt_qa_prompt {fmt_qa_prompt}")
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
                default is gpt-3.5-turbo
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
        llm = kwargs.get("llm", OpenAI("gpt-3.5-turbo"))
        qa_prompt_ = kwargs.get("qa_prompt", qa_prompt)

        pg_vector = PGVectorAccess(
            table_name=platform_table_name,
            dbname=dbname,
            testing=testing,
            embed_model=CohereEmbedding(),
        )
        index = pg_vector.load_index()
        retriever = index.as_retriever()
        _, similarity_top_k, _ = load_hyperparams()

        cls._vector_store = index.vector_store
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

        filters = platform_retriever.retreive_filtering(
            query=query,
            metadata_group1_key=level1_key,
            metadata_group2_key=level2_key,
            metadata_date_key=date_key,
            similarity_top_k=summary_similarity_top_k,
        )

        # getting all the metadata dates from filters
        dates: list[str] = [f[date_key] for f in filters]
        dates_modified = process_dates(list(dates), d)
        dates_filter = [{date_key: date} for date in dates_modified]
        filters.extend(dates_filter)

        logging.info(f"COMMUNITY_ID: {community_id} | summary filters: {filters}")

        engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
            community_id=community_id,
            platform_table_name=platform_table_name,
            filters=filters,
        )
        return engine

    def _prepare_context_str(self, nodes: list[NodeWithScore]) -> str:
        context_str = "\n\n".join(
            [
                node.get_content()
                + "\n"
                + node.node.get_metadata_str(mode=MetadataMode.LLM)
                for node in nodes
            ]
        )
        return context_str
