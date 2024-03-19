from bot.retrievers.summary_retriever_base import BaseSummarySearch
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore
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

        filters = self.define_filters(
            nodes=nodes,
            metadata_group1_key=metadata_group1_key,
            metadata_group2_key=metadata_group2_key,
            metadata_date_key=metadata_date_key,
        )

        return filters

    def define_filters(
        self,
        nodes: list[NodeWithScore],
        metadata_group1_key: str,
        metadata_group2_key: str,
        metadata_date_key: str,
        **kwargs,
    ) -> list[dict[str, str]]:
        """
        define dictionary filters based on metadata of retrieved nodes

        Parameters
        ----------
        nodes : list[dict[llama_index.schema.NodeWithScore]]
            a list of retrieved similar nodes to define filters based
        metadata_group1_key : str
            the metadata name 1 to use
        metadata_group2_key : str
            the metadata name 2 to use
        metadata_date_key : str
            the date key in metadata
        **kwargs :
            and_filters : dict[str, str]
                more `AND` filters to be applied to each

        Returns
        ---------
        filters : list[dict[str, str]]
            a list of filters to apply with `or` condition
            the dictionary would be applying `and`
            operation between keys and values of json metadata_
        """
        and_filters: dict[str, str] | None = kwargs.get("and_filters", None)
        filters: list[dict[str, str]] = []

        for node in nodes:
            filter: dict[str, str] = {}
            filter[metadata_group1_key] = node.metadata[metadata_group1_key]
            filter[metadata_group2_key] = node.metadata[metadata_group2_key]
            filter[metadata_date_key] = node.metadata[metadata_date_key]

            # if more and filters were given
            if and_filters:
                for key, value in and_filters.items():
                    filter[key] = value

            filters.append(filter)

        return filters
