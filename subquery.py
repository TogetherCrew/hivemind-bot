from guidance.models import OpenAIChat
from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from utils.qdrant_utils import QDrantUtils
from utils.query_engine import (
    DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
    GDriveQueryEngine,
    GitHubQueryEngine,
    MediaWikiQueryEngine,
    NotionQueryEngine,
    prepare_discord_engine_auto_filter,
)


def query_multiple_source(
    query: str,
    community_id: str,
    discord: bool = False,
    discourse: bool = False,
    google: bool = False,
    notion: bool = False,
    telegram: bool = False,
    github: bool = False,
    mediaWiki: bool = False,
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
    google : bool
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
    qdrant_utils = QDrantUtils(community_id)

    discord_query_engine: BaseQueryEngine
    github_query_engine: BaseQueryEngine
    # discourse_query_engine: BaseQueryEngine
    google_query_engine: BaseQueryEngine
    notion_query_engine: BaseQueryEngine
    mediawiki_query_engine: BaseQueryEngine
    # telegram_query_engine: BaseQueryEngine

    # wrapper for more clarity
    check_collection = qdrant_utils.check_collection_exist

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
    if google and check_collection("google"):
        google_query_engine = GDriveQueryEngine(community_id=community_id).prepare()
        tool_metadata = ToolMetadata(
            name="Google-Drive",
            description=(
                "Stores and manages documents, spreadsheets, presentations,"
                " and other files for the community."
            ),
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=google_query_engine,
                metadata=tool_metadata,
            )
        )
    if notion and check_collection("notion"):
        notion_query_engine = NotionQueryEngine(community_id=community_id).prepare()
        tool_metadata = ToolMetadata(
            name="Notion",
            description=(
                "Centralizes notes, wikis, project plans, and to-dos for the community."
            ),
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=notion_query_engine,
                metadata=tool_metadata,
            )
        )
    if telegram and check_collection("telegram"):
        raise NotImplementedError
    if github and check_collection("github"):
        github_query_engine = GitHubQueryEngine(community_id=community_id).prepare()
        tool_metadata = ToolMetadata(
            name="GitHub",
            description=(
                "Hosts commits and conversations from Github issues and"
                " pull requests from the selected repositories"
            ),
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=github_query_engine,
                metadata=tool_metadata,
            )
        )
    if mediaWiki and check_collection("mediawiki"):
        mediawiki_query_engine = MediaWikiQueryEngine(
            community_id=community_id
        ).prepare()
        tool_metadata = ToolMetadata(
            name="WikiPedia",
            description="Hosts articles about any information on internet",
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=mediawiki_query_engine,
                metadata=tool_metadata,
            )
        )

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
