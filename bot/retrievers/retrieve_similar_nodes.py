from llama_index.embeddings import BaseEmbedding
from llama_index.schema import NodeWithScore
from llama_index.vector_stores import PGVectorStore, VectorStoreQueryResult
from llama_index.vector_stores.postgres import DBEmbeddingRow
from sqlalchemy import Date, and_, cast, or_, select, text
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding


class RetrieveSimilarNodes:
    """Retriever similar nodes over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        similarity_top_k: int,
        embed_model: BaseEmbedding = CohereEmbedding(),
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def query_db(
        self, query: str, filters: list[dict[str, str]] | None = None
    ) -> list[NodeWithScore]:
        """
        query database with given filters (similarity search is also done)

        Parameters
        -------------
        query : str
            the user question
        filters : list[dict[str, str]] | None
            a list of filters to apply with `or` condition
            the dictionary would be applying `and`
            operation between keys and values of json metadata_
            if `None` then no filtering would be applied
        """
        self._vector_store._initialize()
        embedding = self._embed_model.get_text_embedding(text=query)
        stmt = select(  # type: ignore
            self._vector_store._table_class.id,
            self._vector_store._table_class.node_id,
            self._vector_store._table_class.text,
            self._vector_store._table_class.metadata_,
            self._vector_store._table_class.embedding.cosine_distance(embedding).label(
                "distance"
            ),
        ).order_by(text("distance asc"))

        if filters is not None and filters != []:
            conditions = []
            for condition in filters:
                filters_and = []
                for key, value in condition.items():
                    if key == "date":
                        # Apply ::date cast when the key is 'date'
                        filter_condition = cast(
                            self._vector_store._table_class.metadata_.op("->>")(key),
                            Date,
                        ) == cast(value, Date)
                    else:
                        filter_condition = (
                            self._vector_store._table_class.metadata_.op("->>")(key)
                            == value
                        )

                    filters_and.append(filter_condition)

                conditions.append(and_(*filters_and))

            stmt = stmt.where(or_(*conditions))

            stmt = stmt.limit(self._similarity_top_k)

        with self._vector_store._session() as session, session.begin():
            res = session.execute(stmt)

        results = [
            DBEmbeddingRow(
                node_id=item.node_id,
                text=item.text,
                metadata=item.metadata_,
                similarity=(1 - item.distance) if item.distance is not None else 0,
            )
            for item in res.all()
        ]
        query_result = self._vector_store._db_rows_to_query_result(results)
        nodes = self._get_nodes_with_score(query_result)
        return nodes

    def _get_nodes_with_score(
        self, query_result: VectorStoreQueryResult
    ) -> list[NodeWithScore]:
        """get nodes from a query_results"""
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: float | None = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
