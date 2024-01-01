from utils.query_engine import prepare_discord_engine_auto_filter
from llama_index.core import BaseQueryEngine
from guidance.models import OpenAI as GuidanceOpenAI
from llama_index import QueryBundle
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator


def query_multiple_source(
    query: str,
    community_id: str,
    discord: bool,
    discourse: bool,
    gdrive: bool,
    notion: bool,
    telegram: bool,
    github: bool,
) -> str:
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
    reponse : str
        the response to the user query from the LLM
        using the engines of the given platforms (pltform equal to True)
    """
    query_engine_tools: list[QueryEngineTool] = []
    tools: list[ToolMetadata] = []

    discord_query_engine: BaseQueryEngine
    discourse_query_engine: BaseQueryEngine
    gdrive_query_engine: BaseQueryEngine
    notion_query_engine: BaseQueryEngine
    telegram_query_engine: BaseQueryEngine
    github_query_engine: BaseQueryEngine

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
            description="Provides the discord platform conversations data.",
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
        guidance_llm=GuidanceOpenAI("text-davinci-003"), verbose=False
    )

    s_engine = SubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=query_engine_tools,
    )
    reponse = s_engine.query(QueryBundle(query))

    return reponse.response
