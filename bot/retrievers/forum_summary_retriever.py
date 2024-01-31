from bot.retrievers.summary_retriever_base import BaseSummarySearch
from llama_index.embeddings import BaseEmbedding
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding


class ForumBasedSummaryRetriever(BaseSummarySearch):
    def __init__(
        self,
        table_name: str,
        dbname: str,
        embedding_model: BaseEmbedding | CohereEmbedding = CohereEmbedding(),
    ) -> None:
        """
        the class for forum based data like discord and discourse
        by default CohereEmbedding will be used.
        """
        super().__init__(table_name, dbname, embedding_model=embedding_model)

    def retreive_filtering(
        self,
        query: str,
        metadata_group1_key: str,
        metadata_group2_key: str,
        metadata_date_key: str,
        similarity_top_k: int = 20,
    ) -> list[dict[str, str]]:
        """
        retrieve filtering that can be done based on the retrieved similar nodes with the query

        Parameters
        -----------
        query : str
            the user query to process
        metadata_group1_key : str
            the conversations grouping type 1
            in discord can be `channel`, and in discourse can be `category`
        metadata_group2_key : str
            the conversations grouping type 2
            in discord can be `thread`, and in discourse can be `topic`
        metadata_date_key : str
            the daily metadata saved key
        similarity_top_k : int
            the top k nodes to get as the retriever.
            default is set as 20


        Returns
        ---------
        filters : list[dict[str, str]]
            a list of filters to apply with `or` condition
            the dictionary would be applying `and`
            operation between keys and values of json metadata_
        """
        nodes = self.get_similar_nodes(query=query, similarity_top_k=similarity_top_k)

        filters: list[dict[str, str]] = []

        for node in nodes:
            # the filter made by given node
            filter: dict[str, str] = {}
            if node.metadata[metadata_group1_key]:
                filter[metadata_group1_key] = node.metadata[metadata_group1_key]
            if node.metadata[metadata_group2_key]:
                filter[metadata_group2_key] = node.metadata[metadata_group2_key]
            # date filter
            filter[metadata_date_key] = node.metadata[metadata_date_key]

            filters.append(filter)

        return filters
