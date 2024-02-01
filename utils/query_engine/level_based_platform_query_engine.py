from dateutil import parser
import logging

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.process_dates import process_dates
from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.retrievers import BaseRetriever
from llama_index.schema import MetadataMode, NodeWithScore
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
            self._vector_store,
            self._similarity_top_k,
        )
        similar_nodes = retriever.query_db(query=query_str, filters=self._filters)

        context_str = self._prepare_context_str(similar_nodes, self.summary_nodes)
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(fmt_qa_prompt)
        logging.info(f"fmt_qa_prompt {fmt_qa_prompt}")
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

        cls._level1_key = level1_key
        cls._level2_key = level2_key
        cls._date_key = date_key

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

            context_str += self._prepare_prompt_with_metadata_info(nodes=raw_nodes)
        else:
            grouped_raw_nodes = self._group_nodes_per_metadata(raw_nodes)
            for summary_node in summary_nodes:
                # can be thread_title for discord
                level1_title = summary_node.metadata[self._level1_key]
                # can be channel_title for discord
                level2_title = summary_node.metadata[self._level2_key]
                date = summary_node.metadata[self._date_key]

                # intiialization
                node_context: str = ""

                nested_dict = grouped_raw_nodes.get(level1_title, {}).get(
                    level2_title, {}
                )

                if date in nested_dict:
                    raw_nodes = grouped_raw_nodes[level1_title][level2_title][date]
                    node_context: str = (
                        f"{self._level1_key}: {level1_title}\n"
                        f"{self._level2_key}: {level2_title}\n"
                        f"{self._date_key}: {date}\n"
                        f"summary: {summary_node.text}\n"
                        "messages:\n"
                    )
                    node_context += self._prepare_prompt_with_metadata_info(
                        raw_nodes, prefix="  "
                    )

                context_str += node_context

        logging.info(f"||||||||context_str|||||||| {context_str} |||||||")
        return context_str

    def _group_nodes_per_metadata(
        self, raw_nodes: list[NodeWithScore]
    ) -> dict[str, dict[str, dict[str, list[NodeWithScore]]]]:
        """
        group all nodes based on their level1 and level2 metadata

        Parameters
        -----------
        raw_nodes : list[NodeWithScore]
            a list of raw nodes

        Returns
        ---------
        grouped_nodes : dict[str, dict[str, dict[str, list[NodeWithScore]]]]
            a list of nodes grouped by
            - `level1_key`
            - `level2_key`
            - and the last dict key `date_key`

            The values of the nested dictionary are the nodes grouped
        """
        grouped_nodes: dict[str, dict[str, dict[str, list[NodeWithScore]]]] = {}
        for node in raw_nodes:
            level1_title = node.metadata[self._level1_key]
            # TODO: remove the _name when the data got updated
            level2_title = node.metadata[self._level2_key + "_name"]
            date_str = node.metadata[self._date_key]
            date = parser.parse(date_str).strftime("%Y-%m-%d")

            # defining an empty list (if keys weren't previously made)
            grouped_nodes.setdefault(level1_title, {}).setdefault(
                level2_title, {}
            ).setdefault(date, [])
            # Adding to list
            grouped_nodes[level1_title][level2_title][date].append(node)

        return grouped_nodes

    def _prepare_prompt_with_metadata_info(
        self, nodes: list[NodeWithScore], prefix: str = ""
    ) -> str:
        """
        prepare a prompt with given nodes including the nodes' metadata
        Note: the prefix is set before each text!
        """
        context_str = "\n".join(
            [
                "author: "
                + node.metadata["author_username"]
                + "\n"
                + prefix
                + "message_date: "
                + node.metadata["date"]
                + "\n"
                + prefix
                + f"message {idx + 1}: "
                + node.get_content()
                for idx, node in enumerate(nodes)
            ]
        )

        return context_str
