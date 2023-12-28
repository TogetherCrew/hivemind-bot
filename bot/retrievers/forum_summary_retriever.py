from llama_index.embeddings import BaseEmbedding
from bot.retrievers.summary_retriever_base import BaseSummarySearch
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

    def retreive_metadata(
        self,
        query: str,
        metadata_group1_key: str,
        metadata_group2_key: str,
        metadata_date_key: str,
        similarity_top_k: int = 20,
    ) -> tuple[set[str], set[str], set[str]]:
        """
        retrieve the metadata information of the similar nodes with the query

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
        group1_data : set[str]
            the similar summary nodes having the group1_data.
            can be an empty set meaning no similar thread
            conversations for it was available.
        group2_data : set[str]
            the similar summary nodes having the group2_data.
            can be an empty set meaning no similar channel
            conversations for it was available.
        dates : set[str]
            the similar daily conversations to the given query
        """
        nodes = self.get_similar_nodes(query=query, similarity_top_k=similarity_top_k)

        group1_data: set[str] = set()
        dates: set[str] = set()
        group2_data: set[str] = set()

        for node in nodes:
            if node.metadata[metadata_group1_key]:
                group1_data.add(node.metadata[metadata_group1_key])
            if node.metadata[metadata_group2_key]:
                group2_data.add(node.metadata[metadata_group2_key])
            dates.add(node.metadata[metadata_date_key])

        return group1_data, group2_data, dates
