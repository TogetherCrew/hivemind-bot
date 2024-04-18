import json
from typing import Sequence

from llama_index.core.prompts.guidance_utils import convert_to_handlebars
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata


def build_tools_text(tools: Sequence[ToolMetadata]) -> str:
    tools_dict = {}
    for tool in tools:
        tools_dict[tool.name] = tool.description
    return json.dumps(tools_dict, indent=4)


PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question. \
Define the sub-questions as search queries that can be used for vector similarity search:
"""


example_query_str = (
    "What was decided about the token allocation budget for the "
    "next airdrop and what did the community think of this?"
)
example_tools = [
    ToolMetadata(
        name="Discord",
        description="Contains messages and summaries of conversations from the Discord platform of the community",
    ),
    ToolMetadata(
        name="Discourse",
        description="Contains messages and summaries of discussions from the Discourse platform of the community",
    ),
]
example_tools_str = build_tools_text(example_tools)
example_output = [
    SubQuestion(
        sub_question="Decision token allocation budget airdrop", tool_name="Discourse"
    ),
    SubQuestion(
        sub_question="Opinion token allocation budget airdrop", tool_name="Discord"
    ),
]
example_output_str = json.dumps({"items": [x.dict() for x in example_output]}, indent=4)

EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```

""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

SUFFIX = """\
# Example 2
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

DEFAULT_SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX
DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL = convert_to_handlebars(
    DEFAULT_SUB_QUESTION_PROMPT_TMPL
)
