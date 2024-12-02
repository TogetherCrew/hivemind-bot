from typing import Sequence
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.query_engine import SubQuestionQueryEngine, SubQuestionAnswerPair
from llama_index.core.question_gen.types import BaseQuestionGenerator
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utils import get_color_mapping, print_text
from llama_index.core.async_utils import run_async_tasks
from typing import List, Optional, Sequence, cast


class CustomSubQuestionQueryEngine(SubQuestionQueryEngine):
    def __init__(
        self,
        question_gen: BaseQuestionGenerator,
        response_synthesizer: BaseSynthesizer,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: CallbackManager | None = None,
        verbose: bool = True,
        use_async: bool = False,
    ) -> None:
        super().__init__(
            question_gen,
            response_synthesizer,
            query_engine_tools,
            callback_manager,
            verbose,
            use_async,
        )

    def _query(
        self, query_bundle: QueryBundle
    ) -> tuple[RESPONSE_TYPE, list[NodeWithScore]]:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            sub_questions = self._question_gen.generate(self._metadatas, query_bundle)

            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

            if self._verbose:
                print_text(f"Generated {len(sub_questions)} sub questions.\n")

            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
                qa_pairs_all = cast(List[Optional[SubQuestionAnswerPair]], qa_pairs_all)
            else:
                qa_pairs_all = [
                    self._query_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

            # filter out sub questions that failed
            qa_pairs: List[SubQuestionAnswerPair] = list(filter(None, qa_pairs_all))

            nodes = [self._construct_node(pair) for pair in qa_pairs]

            source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response, qa_pairs_all

    def query(
        self, str_or_query_bundle: str | QueryBundle
    ) -> tuple[RESPONSE_TYPE, list[NodeWithScore]]:
        return super().query(str_or_query_bundle)
