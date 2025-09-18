import logging
import os
from utils.globals import (
    K1_RETRIEVER_SEARCH,
    K2_RETRIEVER_SEARCH,
    D_RETRIEVER_SEARCH,
    RERANK_TOP_K
)
import cohere
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from schema.type import DataType
from tc_hivemind_backend.qdrant_vector_access import QDrantVectorAccess
from utils.globals import (
    REFERENCE_SCORE_THRESHOLD,
    RETRIEVER_THRESHOLD,
)
from utils.query_engine.qa_prompt import qa_prompt
from utils.query_engine.combined_qdrant_retriever import CombinedQdrantRetriever
from utils.query_engine.qdrant_query_engine_utils import QdrantEngineUtils


class DualQdrantRetrievalEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate
    enable_reranking: bool = False
    reranker_model: str = "rerank-english-v3.0"

    def custom_query(self, query_str: str):
        retriever = self.retriever
        if isinstance(retriever, CombinedQdrantRetriever) and retriever.has_summary:
            return self._process_summary_query(query_str)
        return self._process_basic_query(query_str)

    def _rerank_nodes(self, query_str: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Rerank nodes using Cohere's reranking API.
        
        Parameters
        ----------
        query_str : str
            The original query string
        nodes : list[NodeWithScore]
            List of nodes to rerank
            
        Returns
        -------
        list[NodeWithScore]
            Reranked list of nodes
        """
        if not self.enable_reranking or not nodes:
            return nodes
            
        try:
            # Initialize Cohere client
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                logging.warning("COHERE_API_KEY not found, skipping reranking")
                return nodes
                
            co = cohere.Client(cohere_api_key)
            
            # Prepare documents for reranking
            documents = [node.node.get_content() for node in nodes]
            
            # Perform reranking
            rerank_response = co.rerank(
                query=query_str,
                documents=documents,
                model=self.reranker_model,
                top_n=min(RERANK_TOP_K, len(documents)),
            )
            
            # Reorder nodes based on reranking results
            reranked_nodes = []
            for result in rerank_response.results:
                original_node = nodes[result.index]
                # Update the score with the reranking score
                original_node.score = result.relevance_score
                reranked_nodes.append(original_node)
                
            logging.info(f"Reranked {len(nodes)} nodes to top {len(reranked_nodes)} nodes")
            return reranked_nodes
            
        except Exception as e:
            logging.error(f"Error during reranking: {str(e)}")
            # Return original nodes if reranking fails
            return nodes

    @classmethod
    def setup_engine(
        cls,
        llm: OpenAI,
        synthesizer: BaseSynthesizer,
        platform_id: str,
        community_id: str,
        enable_answer_skipping: bool,
        metadata_date_key: str | None = None,
        metadata_date_format: DataType | None = None,
        enable_reranking: bool = True,
        reranker_model: str = "rerank-english-v3.0",
        rerank_top_k: int = RERANK_TOP_K,
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
        platform_id : str
            specifying the platform ID to identify the data collection
        community_id : str
            specifying community_id to identify the data collection
        """
        collection_name = f"{community_id}_{platform_id}"

        vector_store_index: VectorStoreIndex = cls._setup_vector_store_index(
            collection_name=collection_name
        )
        retriever = CombinedQdrantRetriever(
            raw_index=vector_store_index,
            raw_top_k=K2_RETRIEVER_SEARCH,
            summary_index=None,
            summary_top_k=None,
            metadata_date_key=metadata_date_key,
            metadata_date_format=metadata_date_format,
            metadata_date_summary_key=None,
            metadata_date_summary_format=None,
            date_margin=D_RETRIEVER_SEARCH,
            enable_answer_skipping=enable_answer_skipping,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
            enable_reranking=enable_reranking,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
        )

    @classmethod
    def setup_engine_with_summaries(
        cls,
        llm: OpenAI,
        synthesizer: BaseSynthesizer,
        platform_id: str,
        community_id: str,
        metadata_date_key: str,
        metadata_date_format: DataType,
        metadata_date_summary_key: str,
        metadata_date_summary_format: DataType,
        enable_answer_skipping: bool,
        summary_type: str | None = None,
        enable_reranking: bool = True,
        reranker_model: str = "rerank-english-v3.0",
        rerank_top_k: int = 10,
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
        platform_id : str
            specifying the platform ID to identify the data collection
        community_id : str
            specifying community_id to identify the data collection
        metadata_date_summary_key : str | None
            the date key name in summary documents' metadata
            In case of `use_summary` equal to be true this shuold be passed
        metadata_date_summary_format : DataType | None
            the date format in metadata
            In case of `use_summary` equal to be true this shuold be passed
            NOTE: this should be always a string for the filtering of it to work.
        enable_answer_skipping : bool
            skip answering questions with non-relevant retrieved nodes
            having this, it could provide `None` for response and source_nodes
        summary_type : str, optional
            Optional label describing the type of the summary collection.
            Default is None meaning no filter is applied to the summary index.
        """
        collection_name = f"{community_id}_{platform_id}"

        vector_store_index: VectorStoreIndex = cls._setup_vector_store_index(
            collection_name=collection_name
        )

        summary_collection_name = collection_name + "_summary"
        summary_vector_store_index = cls._setup_vector_store_index(
            collection_name=summary_collection_name
        )

        retriever = CombinedQdrantRetriever(
            raw_index=vector_store_index,
            raw_top_k=K2_RETRIEVER_SEARCH,
            summary_index=summary_vector_store_index,
            summary_top_k=K1_RETRIEVER_SEARCH,
            metadata_date_key=metadata_date_key,
            metadata_date_format=metadata_date_format,
            metadata_date_summary_key=metadata_date_summary_key,
            metadata_date_summary_format=metadata_date_summary_format,
            date_margin=D_RETRIEVER_SEARCH,
            enable_answer_skipping=enable_answer_skipping,
            summary_type=summary_type,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
            enable_reranking=enable_reranking,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
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
        logging.info("=== BASIC QUERY MODE ===")

        # Delegate to retriever (combined retriever applies cutoff internally)
        nodes: list[NodeWithScore] = self.retriever.retrieve(query_str)
        logging.info(f"Retrieved {len(nodes)} nodes with cutoff filter applied")

        # Apply reranking if enabled
        nodes = self._rerank_nodes(query_str, nodes)

        nodes_filtered = [node for node in nodes if node.score >= RETRIEVER_THRESHOLD]
        logging.info(
            f"Filtered to {len(nodes_filtered)} nodes (threshold: {RETRIEVER_THRESHOLD})"
        )

        raw_scores = [
            node.score for node in nodes if node.score >= REFERENCE_SCORE_THRESHOLD
        ]

        enable_skip = (
            isinstance(self.retriever, CombinedQdrantRetriever)
            and self.retriever.enable_answer_skipping
        )
        if not raw_scores and enable_skip:
            logging.warning("No high-quality nodes found, skipping answer")
            raise ValueError(
                f"All nodes are below threhsold: {REFERENCE_SCORE_THRESHOLD}"
                " Returning empty response"
            )

        context_str = "\n\n".join([n.node.get_content() for n in nodes_filtered])
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(prompt)

        logging.info("=== BASIC QUERY MODE COMPLETED ===")

        # return final_response
        return Response(response=str(response), source_nodes=nodes_filtered)

    def _process_summary_query(self, query_str: str) -> Response:
        logging.info("=== SUMMARY QUERY MODE ===")
        # Retrieve summary nodes via combined retriever
        combined = self.retriever
        assert isinstance(combined, CombinedQdrantRetriever)
        summary_nodes = combined.retrieve_summary(query_str)
        
        # Apply reranking to summary nodes if enabled
        summary_nodes = self._rerank_nodes(query_str, summary_nodes)
        
        summary_nodes_filtered = [
            node for node in summary_nodes if node.score >= RETRIEVER_THRESHOLD
        ]
        dates = []
        if combined.metadata_date_summary_key is not None:
            dates = [
                node.metadata[combined.metadata_date_summary_key]
                for node in summary_nodes_filtered
                if combined.metadata_date_summary_key in node.metadata
            ]

        if not dates:
            logging.info("No dates found in summary nodes, proceeding to basic query")
            return self._process_basic_query(query_str)

        raw_nodes = combined.retrieve_raw_with_dates(query_str, dates)
        logging.info(f"Retrieved {len(raw_nodes)} raw nodes")

        # Apply reranking to raw nodes if enabled
        raw_nodes = self._rerank_nodes(query_str, raw_nodes)

        raw_nodes_filtered = [
            node for node in raw_nodes if node.score >= RETRIEVER_THRESHOLD
        ]

        # Checking nodes threshold
        summary_scores = [
            node.score
            for node in summary_nodes_filtered
            if node.score >= REFERENCE_SCORE_THRESHOLD
        ]
        raw_scores = [
            node.score
            for node in raw_nodes_filtered
            if node.score >= REFERENCE_SCORE_THRESHOLD
        ]
        enable_skip = (
            isinstance(self.retriever, CombinedQdrantRetriever)
            and self.retriever.enable_answer_skipping
        )
        if not summary_scores and not raw_scores and enable_skip:
            raise ValueError(
                f"All nodes are below threhsold: {REFERENCE_SCORE_THRESHOLD}"
                " Returning empty response"
            )
        else:
            logging.info(f"Filtered {len(summary_nodes_filtered)} summary nodes and {len(raw_nodes_filtered)} raw nodes")
            # if we had something above our threshold then try to answer
            # Use QdrantEngineUtils to build combined prompt text
            utils_helper = QdrantEngineUtils(
                metadata_date_key=combined.metadata_date_key,
                metadata_date_format=combined.metadata_date_format,
                date_margin=combined.date_margin,
            )
            context_str = utils_helper.combine_nodes_for_prompt(
                summary_nodes_filtered, raw_nodes_filtered
            )
            prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
            # TODO: remove this after testing
            # logging.error(f"Prompt: {prompt}")
            response = self.llm.complete(prompt)

            logging.info("=== SUMMARY QUERY MODE COMPLETED ===")
            return Response(
                response=str(response),
                source_nodes=raw_nodes_filtered,
                metadata={
                    "summary_nodes": summary_nodes_filtered,
                },
            )
