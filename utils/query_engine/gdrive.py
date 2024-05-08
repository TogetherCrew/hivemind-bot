from bot.retrievers.custom_retriever import CustomVectorStoreRetriever
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from utils.query_engine.base_engine import BaseEngine


class GDriveQueryEngine(BaseEngine):
    platform_name = "gdrive"

    def __init__(self, community_id: str) -> None:
        dbname = f"community_{community_id}"
        self.dbname = dbname

    def prepare(self, testing=False):
        index = self._setup_vector_store_index(
            platform_table_name=self.platform_name,
            dbname=self.dbname,
            testing=testing,
        )
        _, similarity_top_k, _ = load_hyperparams()

        retriever = CustomVectorStoreRetriever(
            index=index, similarity_top_k=similarity_top_k
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(),
        )
        return query_engine
