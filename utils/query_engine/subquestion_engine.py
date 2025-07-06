import logging
from typing import List, Optional, Sequence, cast

import llama_index.core.instrumentation as instrument
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.instrumentation.events.query import QueryEndEvent, QueryStartEvent
from llama_index.core.query_engine import SubQuestionAnswerPair, SubQuestionQueryEngine
from llama_index.core.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.utils import get_color_mapping, print_text

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


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
        # Store metadata from individual query engines
        self._engine_metadata = {}

    def _query(
        self, query_bundle: QueryBundle
    ) -> tuple[RESPONSE_TYPE, list[Optional[SubQuestionAnswerPair]]]:
        # Clear previous metadata
        self._engine_metadata = {}

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
            if qa_pairs:
                nodes = [self._construct_node(pair) for pair in qa_pairs]

                source_nodes = [
                    node for qa_pair in qa_pairs for node in qa_pair.sources
                ]
                response = self._response_synthesizer.synthesize(
                    query=query_bundle,
                    nodes=nodes,
                    additional_source_nodes=source_nodes,
                )
                # Add response metadata from each query engine
                response.metadata = self._engine_metadata.copy()
            else:
                response = Response(
                    response=None, source_nodes=[], metadata=self._engine_metadata
                )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response, qa_pairs_all

    @dispatcher.span
    def query(
        self, str_or_query_bundle: str | QueryBundle
    ) -> tuple[RESPONSE_TYPE, list[Optional[SubQuestionAnswerPair]]]:
        dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            query_result, qa_pairs_all = self._query(str_or_query_bundle)
        dispatcher.event(
            QueryEndEvent(query=str_or_query_bundle, response=query_result)
        )
        return query_result, qa_pairs_all

    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQuestionAnswerPair]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = query_engine.query(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                # Store metadata from the individual query engine
                if hasattr(response, "metadata") and response.metadata:
                    self._engine_metadata[sub_q.tool_name] = response.metadata

                qa_pair = SubQuestionAnswerPair(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except Exception as exp:
            logger.warning(
                f"[{sub_q.tool_name}] Failed to run {sub_q.sub_question}: {exp}"
            )
            return None

    def get_engine_metadata(self) -> dict:
        """
        Get metadata from individual query engines.

        Returns
        -------
        dict
            Dictionary containing metadata from each query engine.
            Keys are tool names, values are metadata dictionaries.
        """
        return self._engine_metadata.copy()
