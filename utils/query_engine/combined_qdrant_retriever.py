from __future__ import annotations

from typing import Optional

from qdrant_client.http import models
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from schema.type import DataType
from utils.globals import EXCLUDED_DATE_MARGIN
from utils.query_engine.qdrant_query_engine_utils import QdrantEngineUtils


class CombinedQdrantRetriever(BaseRetriever):
    def __init__(
        self,
        raw_index: VectorStoreIndex,
        raw_top_k: int,
        *,
        summary_index: Optional[VectorStoreIndex] = None,
        summary_top_k: Optional[int] = None,
        metadata_date_key: Optional[str] = None,
        metadata_date_format: Optional[DataType] = None,
        metadata_date_summary_key: Optional[str] = None,
        metadata_date_summary_format: Optional[DataType] = None,
        date_margin: int = EXCLUDED_DATE_MARGIN,
        enable_answer_skipping: bool = False,
        summary_type: str | None = None,
    ) -> None:
        """
        Prepare the combined retriever

        Parameters
        ----------
        raw_index : VectorStoreIndex
            Primary vector store index used for raw document retrieval.
        raw_top_k : int
            Number of top similar nodes to retrieve from the raw index.
        summary_index : VectorStoreIndex, optional
            Optional summary index used by `retrieve_summary`. Default is None.
        summary_top_k : int, optional
            Number of top similar nodes to retrieve from the summary index.
            Required if `summary_index` is provided. Default is None.
        metadata_date_key : str, optional
            Metadata key in the payload that stores the document date/time used
            for raw retrieval filtering. Default is None.
        metadata_date_format : DataType, optional
            Format/type of the values stored under `metadata_date_key`.
            Default is None.
        metadata_date_summary_key : str, optional
            Metadata key for dates in the summary collection. Default is None.
        metadata_date_summary_format : DataType, optional
            Format/type of the values stored under
            `metadata_date_summary_key`. Default is None.
        date_margin : int, optional
            Safety margin in days to exclude very recent items from raw
            retrievals. Default is `EXCLUDED_DATE_MARGIN`.
        enable_answer_skipping : bool, optional
            Reserved for future use to skip answering based on heuristics.
            Default is False.
        summary_type : str, optional
            Optional label describing the type of the summary collection.
            Default is None meaning no filter is applied to the summary index.
        """
        super().__init__()
        self.raw_index = raw_index
        self.raw_top_k = raw_top_k
        self.summary_index = summary_index
        self.summary_top_k = summary_top_k
        self.metadata_date_key = metadata_date_key
        self.metadata_date_format = metadata_date_format
        self.metadata_date_summary_key = metadata_date_summary_key
        self.metadata_date_summary_format = metadata_date_summary_format
        self.date_margin = date_margin
        self.enable_answer_skipping = enable_answer_skipping
        self.summary_type = summary_type

    @property
    def has_summary(self) -> bool:
        return self.summary_index is not None and self.summary_top_k is not None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Default behavior: raw retrieval, with cutoff filter if configured
        filter = self._build_cutoff_filter()
        retriever = self._build_raw_retriever(filter=filter, top_k=self.raw_top_k)
        return retriever.retrieve(query_bundle.query_str)

    def retrieve_summary(self, query_str: str) -> list[NodeWithScore]:
        if not self.has_summary:
            return []
        assert self.summary_index is not None
        assert self.summary_top_k is not None

        if self.summary_type is not None:
            filter = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value=self.summary_type))
                ]
            )

            retriever = self.summary_index.as_retriever(
                vector_store_kwargs={"qdrant_filters": filter},
                similarity_top_k=self.summary_top_k
            )
        else:
            retriever = self.summary_index.as_retriever(
                similarity_top_k=self.summary_top_k
            )
        return retriever.retrieve(query_str)

    def retrieve_raw_with_dates(self, query_str: str, dates: list[str | float]) -> list[NodeWithScore]:
        if self.metadata_date_key is None or self.metadata_date_format is None:
            return self.retrieve(query_str)
        utils = QdrantEngineUtils(
            metadata_date_key=self.metadata_date_key,
            metadata_date_format=self.metadata_date_format,
            date_margin=self.date_margin,
        )
        filter = utils.define_raw_data_filters(dates)
        retriever = self._build_raw_retriever(filter=filter, top_k=self.raw_top_k)
        return retriever.retrieve(query_str)

    def _build_raw_retriever(
        self, *, filter: Optional[models.Filter], top_k: int
    ) -> VectorIndexRetriever:
        if filter is not None:
            return self.raw_index.as_retriever(
                vector_store_kwargs={"qdrant_filters": filter},
                similarity_top_k=top_k,
            )
        return self.raw_index.as_retriever(similarity_top_k=top_k)

    def _build_cutoff_filter(self) -> Optional[models.Filter]:
        # Safety cutoff to exclude very recent messages from raw retrieval
        if self.metadata_date_key is None or self.metadata_date_format is None:
            return None
        # Delegate to QdrantEngineUtils with an empty date list and only cutoff
        # Note: define_raw_data_filters always includes a global cutoff must-filter
        utils = QdrantEngineUtils(
            metadata_date_key=self.metadata_date_key,
            metadata_date_format=self.metadata_date_format,
            date_margin=self.date_margin,
        )
        # An empty list yields only the global cutoff
        return utils.define_raw_data_filters(dates=[])


