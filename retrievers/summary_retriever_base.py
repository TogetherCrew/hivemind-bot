from tc_hivemind_backend.embeddings.cohere import CohereEmbedding

from tc_hivemind_backend.pg_vector_access import PGVectorAccess
from llama_index import VectorStoreIndex
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore


class BaseSummarySearch:
    def __init__(
        self,
        table_name: str,
        dbname: str,
        embedding_model: BaseEmbedding = CohereEmbedding(),
    ) -> None:
        """
        initialize the base summary search class

        In this class we're doing a similarity search
        for available saved nodes under postgresql

        Parameters
        -------------
        table_name : str
            the table that summary data is saved
            *Note:* Don't include the `data_` prefix of the table,
            cause lamma_index would original include that.
        dbname : str
            the database name to access
        similarity_top_k : int
            the top k nodes to get as the retriever.
            default is set as 20
        embedding_model : llama_index.embeddings.BaseEmbedding
            the embedding model to use for doing embedding on the query string
            default would be CohereEmbedding that we've written
        """
        self.index = self._setup_index(table_name, dbname)
        self.embedding_model = embedding_model

    def get_similar_nodes(
        self, query: str, similarity_top_k: int = 20
    ) -> list[NodeWithScore]:
        """
        get k similar nodes to the query.
        Note: this funciton wold get the embedding
        for the query to do the similarity search.

        Parameters
        ------------
        query : str
            the user query to process
        similarity_top_k : int
            the top k nodes to get as the retriever.
            default is set as 20
        """
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

        query_embedding = self.embedding_model.get_text_embedding(text=query)

        query_bundle = QueryBundle(query_str=query, embedding=query_embedding)
        nodes = retriever._retrieve(query_bundle)

        return nodes

    def _setup_index(self, table_name: str, dbname: str) -> VectorStoreIndex:
        """
        setup the llama_index VectorStoreIndex
        """
        pg_vector_access = PGVectorAccess(table_name=table_name, dbname=dbname)
        index = pg_vector_access.load_index()
        return index
