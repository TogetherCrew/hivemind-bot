"""
We're going to override the `_build_node_list_from_query_result`
since it is raising errors having the llama-index legacy & newer version together
"""

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.schema import Node, NodeWithScore, ObjectType
from llama_index.core.vector_stores.types import VectorStoreQueryResult


class CustomVectorStoreRetriever(VectorIndexRetriever):
    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> list[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                source_node = query_result.nodes[i].source_node
                if (not self._vector_store.stores_text) or (
                    source_node is not None and source_node.node_type != ObjectType.TEXT
                ):
                    node_id = query_result.nodes[i].node_id
                    if self._docstore.document_exists(node_id):
                        query_result.nodes[i] = self._docstore.get_node(
                            node_id
                        )  # type: ignore[index]

        log_vector_store_query_result(query_result)
        node_with_scores: list[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: float | None = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            # This is the part we updated
            node_new = Node.from_dict(node.to_dict())
            node_with_score = NodeWithScore(node=node_new, score=score)

            node_with_scores.append(node_with_score)

        return node_with_scores
