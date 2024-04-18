from guidance.models import OpenAIChat
from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from utils.query_engine import (
    DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
    prepare_discord_engine_auto_filter,
)


def query_multiple_source(
    query: str,
    community_id: str,
    discord: bool = False,
    discourse: bool = False,
    gdrive: bool = False,
    notion: bool = False,
    telegram: bool = False,
    github: bool = False,
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
        default is set to False
    discourse : bool
        if `True` then add the engine to the subquery_generator
        default is set to False
    gdrive : bool
        if `True` then add the engine to the subquery_generator
        default is set to False
    notion : bool
        if `True` then add the engine to the subquery_generator
        default is set to False
    telegram : bool
        if `True` then add the engine to the subquery_generator
        default is set to False
    github : bool
        if `True` then add the engine to the subquery_generator
        default is set to False


    Returns
    --------
    response : str,
        the response to the user query from the LLM
        using the engines of the given platforms (platform equal to True)
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

    embed_model = CohereEmbedding()
    llm = OpenAI("gpt-3.5-turbo")
    Settings.embed_model = embed_model
    Settings.llm = llm

    question_gen = GuidanceQuestionGenerator.from_defaults(
        guidance_llm=OpenAIChat("gpt-4"),
        verbose=False,
        prompt_template_str=DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
    )
    s_engine = SubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=query_engine_tools,
        use_async=False,
    )
    query_embedding = embed_model.get_text_embedding(text=query)
    response = s_engine.query(QueryBundle(query_str=query, embedding=query_embedding))

    return response.response, response.source_nodes
