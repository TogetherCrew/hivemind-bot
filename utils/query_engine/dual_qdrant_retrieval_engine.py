from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.response.schema import Response
from llama_index.llms.openai import OpenAI
from schema.type import DataType
from tc_hivemind_backend.qdrant_vector_access import QDrantVectorAccess
from utils.query_engine.qdrant_query_engine_utils import QdrantEngineUtils

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


class DualQdrantRetrievalEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        if self.summary_retriever is None:
            response = self._process_basic_query(query_str)
        else:
            response = self._process_summary_query(query_str)
        return response

    @classmethod
    def setup_engine(
        cls,
        llm: OpenAI,
        synthesizer: BaseSynthesizer,
        platform_name: str,
        community_id: str,
    ):
        """
        setup the custom query engine on qdrant data

        Parameters
        ------------
        llm : OpenAI
            the llm to be used for RAG pipeline
        synthesizer : BaseSynthesizer
            the process of generating response using an LLM
        qa_prompt : PromptTemplate
            the prompt template to be filled and passed to an LLM
        platform_name : str
            specifying the platform data to identify the data collection
        community_id : str
            specifying community_id to identify the data collection
        """
        collection_name = f"{community_id}_{platform_name}"

        _, raw_data_top_k, date_margin = load_hyperparams()
        cls._date_margin = date_margin

        cls._vector_store_index: VectorStoreIndex = cls._setup_vector_store_index(
            collection_name=collection_name
        )
        retriever = VectorIndexRetriever(
            index=cls._vector_store_index,
            similarity_top_k=raw_data_top_k,
        )

        cls.summary_retriever = None

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
        )

    @classmethod
    def setup_engine_with_summaries(
        cls,
        llm: OpenAI,
        synthesizer: BaseSynthesizer,
        platform_name: str,
        community_id: str,
        metadata_date_key: str,
        metadata_date_format: DataType,
        metadata_date_summary_key: str,
        metadata_date_summary_format: DataType,
    ):
        """
        setup the custom query engine on qdrant data

        Parameters
        ------------
        use_summary : bool
            whether to use the summary data or not
            note: the summary data should be available before
            for this option to be enabled
        llm : OpenAI
            the llm to be used for RAG pipeline
        synthesizer : BaseSynthesizer
            the process of generating response using an LLM
        platform_name : str
            specifying the platform data to identify the data collection
        community_id : str
            specifying community_id to identify the data collection
        metadata_date_summary_key : str | None
            the date key name in summary documents' metadata
            In case of `use_summary` equal to be true this shuold be passed
        metadata_date_summary_format : DataType | None
            the date format in metadata
            In case of `use_summary` equal to be true this shuold be passed
            NOTE: this should be always a string for the filtering of it to work.
        """
        collection_name = f"{community_id}_{platform_name}"
        summary_data_top_k, raw_data_top_k, date_margin = load_hyperparams()
        cls._date_margin = date_margin
        cls._raw_data_top_k = raw_data_top_k

        cls._vector_store_index: VectorStoreIndex = cls._setup_vector_store_index(
            collection_name=collection_name
        )
        retriever = VectorIndexRetriever(
            index=cls._vector_store_index,
            similarity_top_k=raw_data_top_k,
        )

        summary_collection_name = collection_name + "_summary"
        summary_vector_store_index = cls._setup_vector_store_index(
            collection_name=summary_collection_name
        )

        cls.summary_retriever = VectorIndexRetriever(
            index=summary_vector_store_index,
            similarity_top_k=summary_data_top_k,
        )
        cls.metadata_date_summary_key = metadata_date_summary_key
        cls.metadata_date_summary_format = metadata_date_summary_format
        cls.metadata_date_key = metadata_date_key
        cls.metadata_date_format = metadata_date_format

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
        )

    @classmethod
    def _setup_vector_store_index(
        cls,
        collection_name: str,
    ) -> VectorStoreIndex:
        """
        prepare the vector_store for querying data

        Parameters
        ------------
        collection_name : str
            to override the default collection_name
        """
        qdrant_vector = QDrantVectorAccess(collection_name=collection_name)
        index = qdrant_vector.load_index()
        return index

    def _process_basic_query(self, query_str: str) -> Response:
        nodes: list[NodeWithScore] = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(prompt)

        # return final_response
        return Response(response=str(response), source_nodes=nodes)

    def _process_summary_query(self, query_str: str) -> Response:
        summary_nodes = self.summary_retriever.retrieve(query_str)
        utils = QdrantEngineUtils(
            metadata_date_key=self.metadata_date_key,
            metadata_date_format=self.metadata_date_format,
            date_margin=self._date_margin,
        )

        dates = [
            node.metadata[self.metadata_date_summary_key]
            for node in summary_nodes
            if self.metadata_date_summary_key in node.metadata
        ]

        if not dates:
            return self._process_basic_query(query_str)

        filter = utils.define_raw_data_filters(dates=dates)

        retriever: BaseRetriever = self._vector_store_index.as_retriever(
            vector_store_kwargs={"qdrant_filters": filter},
            similarity_top_k=self._raw_data_top_k,
        )
        raw_nodes = retriever.retrieve(query_str)

        context_str = utils.combine_nodes_for_prompt(summary_nodes, raw_nodes)
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(prompt)

        return Response(response=str(response), source_nodes=raw_nodes)
