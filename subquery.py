from guidance.models import OpenAIChat
from llama_index.core import QueryBundle, Settings
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from tc_hivemind_backend.db.utils.preprocess_text import BasePreprocessor
from tc_hivemind_backend.embeddings.cohere import CohereEmbedding
from utils.globals import INVALID_QUERY_RESPONSE, NO_ANSWER_REFERENCE
from utils.qdrant_utils import QDrantUtils
from utils.query_engine import (
    DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
    CustomSubQuestionQueryEngine,
    GDriveQueryEngine,
    GitHubQueryEngine,
    MediaWikiQueryEngine,
    NotionQueryEngine,
    TelegramDualQueryEngine,
    TelegramQueryEngine,
    WebsiteQueryEngine,
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
    website: bool = False,
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
        # checking if the summaries was available
        if check_collection("telegram_summary"):
            telegram_query_engine = TelegramDualQueryEngine(
                community_id=community_id
            ).prepare()
        else:
            telegram_query_engine = TelegramQueryEngine(
                community_id=community_id
            ).prepare()

        tool_metadata = ToolMetadata(
            name="Telegram",
            description=(
                "Contains messages, conversations, and media from the Telegram platform,"
                " used for group discussions within the community."
            ),
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=telegram_query_engine,
                metadata=tool_metadata,
            )
        )

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

    if website and check_collection("website"):
        website_query_engine = WebsiteQueryEngine(community_id=community_id).prepare()
        tool_metadata = ToolMetadata(
            name="Website",
            description=(
                "Hosts a diverse collection of crawled data from various "
                "online sources to facilitate community insights and analysis."
            ),
        )
        query_engine_tools.append(
            QueryEngineTool(
                query_engine=website_query_engine,
                metadata=tool_metadata,
            )
        )
    if not BasePreprocessor().extract_main_content(text=query):
        response = INVALID_QUERY_RESPONSE
        source_nodes = []
        return response, source_nodes

    embed_model = CohereEmbedding()
    llm = OpenAI("gpt-4o-mini")
    Settings.embed_model = embed_model
    Settings.llm = llm

    question_gen = GuidanceQuestionGenerator.from_defaults(
        guidance_llm=OpenAIChat("gpt-4"),
        verbose=False,
        prompt_template_str=DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
    )
    s_engine = CustomSubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=query_engine_tools,
        use_async=False,
        verbose=False,
    )
    query_embedding = embed_model.get_text_embedding(text=query)

    result: tuple[RESPONSE_TYPE, list[NodeWithScore]] = s_engine.query(
        QueryBundle(query_str=query, embedding=query_embedding)
    )
    response, source_nodes = result

    if source_nodes == []:
        return NO_ANSWER_REFERENCE, source_nodes
    else:
        return response.response, source_nodes
