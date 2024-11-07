from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from tc_hivemind_backend.qdrant_vector_access import QDrantVectorAccess
from schema.type import DataType


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
            nodes = self.retriever.retrieve(query_str)
            context_str = "\n\n".join([n.node.get_content() for n in nodes])
            response = self.llm.complete(
                qa_prompt.format(context_str=context_str, query_str=query_str)
            )
        else:
            summary_nodes = self.summary_retriever.retrieve(query_str)
            # the filters that will be applied on qdrant
            should_filters = []
            for node in summary_nodes:
                date_value = node.metadata[self.metadata_date_key]
            # TODO: filter on raw data for extraction
            # and then prepare the prompt

        return str(response)

    @classmethod
    def setup_engine(
        cls,
        use_summary: bool,
        llm: OpenAI,
        synthesizer: BaseSynthesizer,
        qa_prompt: PromptTemplate,
        platform_name: str,
        community_id: str,
        metadata_date_key: str | None = None,
        metadata_date_format: DataType | None = None,
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
        qa_prompt : PromptTemplate
            the prompt template to be filled and passed to an LLM
        platform_name : str
            specifying the platform data to identify the data collection
        community_id : str
            specifying community_id to identify the data collection      
        metadata_date_key : str | None
            the date key name in summary documents' metadata
            In case of `use_summary` equal to be true this shuold be passed
        metadata_date_format : DataType | None
            the date format in metadata
            In case of `use_summary` equal to be true this shuold be passed
        """
        if use_summary and (metadata_date_key is None or metadata_date_format is None):
            raise ValueError(
                "`metadata_date_key` and `metadata_date_format` "
                "should be given in case if use_summary=True!"
            )

        collection_name = f"{community_id}_{platform_name}"

        summary_data_top_k, raw_data_top_k, interval_margin = load_hyperparams()
        cls._interval_margin = interval_margin

        vector_store_index = cls._setup_vector_store_index(
            collection_name=collection_name
        )
        retriever = VectorIndexRetriever(
            index=vector_store_index,
            similarity_top_k=raw_data_top_k,
        )

        if use_summary:
            summary_collection_name = collection_name + "_summary"
            summary_vector_store_index = cls._setup_vector_store_index(
                collection_name=summary_collection_name
            )

            cls.summary_retriever = VectorIndexRetriever(
                index=summary_vector_store_index,
                similarity_top_k=summary_data_top_k,
            )
            cls.metadata_date_key = metadata_date_key
            cls.metadata_date_format = metadata_date_format
        else:
            cls.summary_retriever = None

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
        )

    def _setup_vector_store_index(
        self,
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
