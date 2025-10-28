import logging
import re
import html
import requests
from typing import Any
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode, MetadataMode
from llama_index.core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


def _fetch_page_content(api_url: str, title: str) -> str:
    """Fetch content for a single page title."""
    try:
        # Try to get plain text extract first
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }
        page_res = requests.get(api_url, params=params, timeout=20).json()

        # Try to read the extract safely
        pages = page_res.get("query", {}).get("pages", {})
        page_obj = next(iter(pages.values()), {})
        extract = page_obj.get("extract")

        if not extract:
            # Fallback: use action=parse and strip HTML
            parse_params = {
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "redirects": 1,
                "formatversion": 2,
            }
            parse_res = requests.get(api_url, params=parse_params, timeout=20).json()
            parse_text = (
                parse_res.get("parse", {}).get("text")
                if "parse" in parse_res
                else parse_res.get("text")
            )

            if parse_text:
                # Basic HTML to text conversion
                text = html.unescape(parse_text)
                # remove scripts/styles
                text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
                text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
                # replace <br> and <p> with newlines
                text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
                text = re.sub(r"</\s*p\s*>", "\n\n", text, flags=re.IGNORECASE)
                # strip all remaining tags
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                extract = text

        if not extract:
            return "(No extract available. The page may be a redirect, disambiguation, or non-extractable.)"

        return extract
    
    except requests.Timeout:
        logger.warning(f"Timeout error while fetching content for '{title}'")
        return "(Error: Request timeout while fetching page content.)"
    except requests.RequestException as e:
        logger.warning(f"Request error while fetching content for '{title}': {e}")
        return f"(Error: Request failed - {str(e)})"
    except Exception as e:
        logger.warning(f"Unexpected error while fetching content for '{title}': {e}")
        return f"(Error: {str(e)})"


def _create_mediawiki_search_tool(api_url: str, max_pages: int = 10):
    """Create a MediaWiki search tool with the given API URL."""
    
    @tool("mediawiki_search", return_direct=False)
    def mediawiki_search(query: str) -> dict:
        """Search MediaWiki and return relevant text content per page.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with page titles as keys and content as values
        """
        # 1. Search for pages
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srwhat": "text",
        }
        results = []

        try:
            while True:
                res = requests.get(api_url, params=params, timeout=30).json()
                search_results = res.get("query", {}).get("search", [])
                results.extend(search_results)

                # check if there is a "continue" field
                if "continue" not in res:
                    break

                # update params with the next offset
                params.update(res["continue"])

            logger.info(f"Got {len(results)} results total.")
        except requests.Timeout:
            logger.warning(f"Timeout error during search for query '{query}'")
            if len(results) > 0:
                logger.info(f"Partial results retrieved: {len(results)} pages")
            else:
                return {"error": "Search request timed out with no results."}
        except requests.RequestException as e:
            logger.warning(f"Request error during search: {e}")
            return {"error": f"Search request failed - {str(e)}"}
        except Exception as e:
            logger.warning(f"Unexpected error during search: {e}")
            return {"error": f"Search failed - {str(e)}"}

        if len(results) == 0:
            return {}

        # 2. Fetch content for each page
        pages_content = {}
        for result in results[:max_pages]:
            title = result.get("title")
            if not title:
                continue
            
            logger.info(f"Fetching content for: {title}")
            content = _fetch_page_content(api_url, title)
            pages_content[title] = content

        return pages_content
    
    return mediawiki_search


class MediaWikiQueryEngine:
    """Agent-based MediaWiki query engine that searches MediaWiki directly via API."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to MediaWiki search capabilities.
When answering questions, follow the given ruleset:

## RULES:
- Use the mediawiki_search tool to find relevant information.
- Provide clear, concise, and accurate responses based on the retrieved information.
- ALWAYS include a "References:" section at the end of your answer listing all page titles you used.

## CITATION FORMAT:
When you retrieve information from mediawiki_search, the tool returns a dictionary with page titles as keys.
You MUST list these page titles at the end of your answer.

Example format:
"Optimum theory is a concept that describes the optimal allocation of resources in economic systems. 
It involves analyzing trade-offs and making decisions that maximize efficiency while considering constraints.

References:
- Optimum Theory
- Economic Models
- Resource Allocation"

## IMPORTANT:
Your final answer MUST end with a "References:" section listing all the page titles you retrieved information from."""

    def __init__(
        self, 
        community_id: str, 
        platform_id: str = None,
        api_url: str = "https://wiki.p2pfoundation.net/api.php",
        max_pages: int = 10,
        system_prompt: str = None,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
    ) -> None:
        """
        Initialize the MediaWiki query engine with agent-based search.
        
        Parameters:
        -----------
        community_id : str
            The community ID (kept for interface compatibility)
        platform_id : str
            The platform ID (kept for interface compatibility)
        api_url : str
            The MediaWiki API URL to search
        max_pages : int
            Maximum number of pages to retrieve per search
        system_prompt : str
            Custom system prompt for the agent
        llm_model : str
            The LLM model to use
        temperature : float
            Temperature for LLM responses
        """
        self.community_id = community_id
        self.platform_id = platform_id or "mediawiki"
        self.api_url = api_url
        self.max_pages = max_pages
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.llm_model = llm_model
        self.temperature = temperature
        self._agent = None
        
        # Add llama-index compatibility attributes
        self.callback_manager = CallbackManager([])
        
        logger.info(
            f"MediaWikiQueryEngine initialized for community {community_id}, "
            f"platform {self.platform_id}, API: {api_url}"
        )
    
    def prepare(self, enable_answer_skipping: bool = False, testing: bool = False):
        """
        Prepare the query engine by initializing the agent.
        
        This method maintains interface compatibility with the old BaseQdrantEngine.
        
        Returns:
        --------
        self : MediaWikiQueryEngine
            Returns self to maintain the same interface
        """
        logger.info("Preparing MediaWiki agent-based query engine...")
        
        # Create the MediaWiki search tool with configured API URL
        mediawiki_search_tool = _create_mediawiki_search_tool(
            self.api_url, 
            self.max_pages
        )
        
        # Initialize LLM
        llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        
        # Initialize agent with custom system message
        self._agent = initialize_agent(
            [mediawiki_search_tool],
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,  # Set to False for production, True for debugging
            agent_kwargs={
                "system_message": SystemMessage(content=self.system_prompt),
                "suffix": """Begin!

Question: {input}

IMPORTANT REMINDER: End your final answer with a "References:" section listing all page titles you used.

Thought: {agent_scratchpad}"""
            }
        )
        
        logger.info("MediaWiki agent-based query engine prepared successfully.")
        return self
    
    def query(self, query_str: str) -> Response:
        """
        Query the MediaWiki using the agent.
        
        This maintains compatibility with llama-index QueryEngine interface.
        
        Parameters:
        -----------
        query_str : str
            The query string to search for
            
        Returns:
        --------
        Response
            A llama-index Response object containing the answer and source nodes
        """
        if self._agent is None:
            # Auto-prepare if not prepared yet
            self.prepare()
        
        logger.info(f"Querying MediaWiki with: {query_str}")
        
        try:
            # Run the agent
            response_text = self._agent.run(query_str)
            
            # Extract page titles from the References section
            source_nodes = self._extract_source_nodes_from_response(response_text)
            
            # Create llama-index Response object
            response = Response(
                response=response_text,
                source_nodes=source_nodes
            )
            
            logger.info(f"Query completed. Found {len(source_nodes)} source nodes.")
            return response
            
        except Exception as e:
            logger.error(f"Error during MediaWiki query: {e}")
            # Return error response
            error_response = Response(
                response=f"Error querying MediaWiki: {str(e)}",
                source_nodes=[]
            )
            return error_response
    
    def _extract_source_nodes_from_response(self, response_text: str) -> list[NodeWithScore]:
        """
        Extract page titles from the References section and create NodeWithScore objects.
        
        Parameters:
        -----------
        response_text : str
            The response text from the agent
            
        Returns:
        --------
        list[NodeWithScore]
            List of source nodes with metadata
        """
        source_nodes = []
        
        # Look for References section
        references_pattern = r"References?:\s*\n((?:[-*]\s*.+\n?)+)"
        match = re.search(references_pattern, response_text, re.IGNORECASE)
        
        if match:
            references_text = match.group(1)
            # Extract individual page titles
            page_titles = re.findall(r"[-*]\s*(.+)", references_text)
            
            for idx, title in enumerate(page_titles):
                title = title.strip()
                
                # Create URL for the page
                url_route = title.replace(" ", "_")
                url = f"{self.api_url.replace('/api.php', '')}/{url_route}"
                
                # Create a TextNode for each reference
                node = TextNode(
                    text=f"Reference: {title}",
                    metadata={
                        "title": title,
                        "url": url,
                        "platform": self.platform_id,
                        "community_id": self.community_id,
                    }
                )
                
                # Wrap in NodeWithScore
                node_with_score = NodeWithScore(
                    node=node,
                    score=1.0 - (idx * 0.1)  # Decreasing scores for ordering
                )
                source_nodes.append(node_with_score)
        
        return source_nodes
