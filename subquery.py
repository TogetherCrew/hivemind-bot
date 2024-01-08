from guidance.models import OpenAIChat
from llama_index import QueryBundle, ServiceContext
from llama_index.core import BaseQueryEngine
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
from llama_index.schema import NodeWithScore
from llama_index.tools import QueryEngineTool, ToolMetadata
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from utils.query_engine import prepare_discord_engine_auto_filter


def query_multiple_source(
    query: str,
    community_id: str,
    discord: bool,
    discourse: bool,
    gdrive: bool,
    notion: bool,
    telegram: bool,
    github: bool,
) -> tuple[str, list[NodeWithScore]]:
    """
    query multiple platforms and get an answer from the multiple

    Parameters
    ------------
    query : str
        the user question
    community_id : str
        the community id to get their data
    discord : bool
        if `True` then add the engine to the subquery_generator
    discourse : bool
        if `True` then add the engine to the subquery_generator
    gdrive : bool
        if `True` then add the engine to the subquery_generator
    notion : bool
        if `True` then add the engine to the subquery_generator
    telegram : bool
        if `True` then add the engine to the subquery_generator
    github : bool
        if `True` then add the engine to the subquery_generator


    Returns
    --------
    response : str,
        the response to the user query from the LLM
        using the engines of the given platforms (pltform equal to True)
    source_nodes : list[NodeWithScore]
        the list of nodes that were source of answering
    """
    query_engine_tools: list[QueryEngineTool] = []
    tools: list[ToolMetadata] = []

    discord_query_engine: BaseQueryEngine
    # discourse_query_engine: BaseQueryEngine
    # gdrive_query_engine: BaseQueryEngine
    # notion_query_engine: BaseQueryEngine
    # telegram_query_engine: BaseQueryEngine
    # github_query_engine: BaseQueryEngine

    # query engine perparation
    # tools_metadata and query_engine_tools
    if discord:
        discord_query_engine = prepare_discord_engine_auto_filter(
            community_id,
            query,
            similarity_top_k=None,
            d=None,
        )
        tool_metadata = ToolMetadata(
            name="Discord",
            description="Contains messages and summaries of conversations from the Discord platform of the community",
        )

        tools.append(tool_metadata)
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=discord_query_engine,
                metadata=tool_metadata,
            )
        )

    if discourse:
        raise NotImplementedError
    if gdrive:
        raise NotImplementedError
    if notion:
        raise NotImplementedError
    if telegram:
        raise NotImplementedError
    if github:
        raise NotImplementedError

    question_gen = GuidanceQuestionGenerator.from_defaults(
        guidance_llm=OpenAIChat("gpt-3.5-turbo"),
        verbose=False,
    )
    embed_model = CohereEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    s_engine = SubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=query_engine_tools,
        use_async=False,
        service_context=service_context,
    )
    query_embedding = embed_model.get_text_embedding(text=query)
    response = s_engine.query(QueryBundle(query_str=query, embedding=query_embedding))

    return response.response, response.source_nodes
