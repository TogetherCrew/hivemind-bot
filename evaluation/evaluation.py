import logging
import json
import argparse
import os
from typing import Any
from dotenv import load_dotenv
import pandas as pd

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
)
from langchain_openai import ChatOpenAI
from ragas.integrations.llama_index import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.llms import LangchainLLMWrapper
from ragas.executor import Executor
from ragas.run_config import RunConfig
from ragas.dataset_schema import EvaluationDataset
from ragas.evaluation import evaluate as ragas_evaluate
from worker.tasks import query_data_sources
from ragas_experimental import Dataset
from ragas.testset.synthesizers.testset_schema import Testset
from ast import literal_eval
from ragas.evaluation import EvaluationResult
from ragas.cost import TokenUsage
from utils.globals import NO_ANSWER_REFERENCE



class StartEvaluation:
    def __init__(self, community_id: str, platform_id: str, model: str):
        load_dotenv()
        self.community_id = community_id
        self.platform_id = platform_id
        self.model = model
        
        logging.basicConfig(level=logging.INFO)
        logging.info(
            f"Starting evaluation for community_id: {community_id} and platform_id: {platform_id}!"
            f" Using model: {model}!"
        )

        logging.info(f"Preparing engine...")
        self.engine = QueryDataSourcesAdapter(
            community_id=community_id,
            platform_id=platform_id,
            enable_answer_skipping=False,
        )

        logging.info(f"Loading dataset...")
        data_root = os.getenv("EVAL_DATA_ROOT", "evaluation")
        self.dataset = Dataset.load(
            name="testset_hybrid_extended_ooc_data",
            backend="local/csv",
            root_dir=data_root,
        )

    def evaluate(self):
        _df = self.dataset.to_pandas()

        # Build items for all rows
        _items_all = []
        for _, r in _df.iterrows():
            _items_all.append({
                "user_input": r["user_input"],
                "reference_contexts": self._parse_contexts(r["reference_contexts"]),
                "reference": r["reference"],
                "synthesizer_name": r.get("synthesizer_name", "unknown"),
            })

        # Convert to evaluation dataset
        evaluation_dataset_all = Testset.from_list(_items_all).to_evaluation_dataset()

        # the engine combining the summary and the source nodes
        wrapped_engine = SourceMergingQueryEngine(self.engine)

        # First, run queries on all rows to get responses
        logging.info(f"Running queries to get responses for all rows...")
        exec = Executor(
            desc="Running Query Engine",
            keep_progress_bar=True,
            show_progress=True,
            raise_exceptions=False,
            run_config=RunConfig(),
        )
        
        queries = [sample.user_input for sample in evaluation_dataset_all.samples]
        for i, q in enumerate(queries):
            exec.submit(wrapped_engine.aquery, q, name=f"query-{i}")
        
        # Get responses and retrieved contexts
        responses = []
        retrieved_contexts = []
        results = exec.results()
        for r in results:
            responses.append(r.response)
            retrieved_contexts.append([n.node.text for n in r.source_nodes])
        
        # Append to dataset samples
        for i, sample in enumerate(evaluation_dataset_all.samples):
            sample.response = responses[i]
            sample.retrieved_contexts = retrieved_contexts[i]

        # Now filter: skip rows where BOTH reference AND response equal NO_ANSWER_REFERENCE
        # Create dataframe with responses
        df_with_responses = pd.DataFrame([
            {
                "user_input": sample.user_input,
                "reference": sample.reference,
                "response": sample.response,
                "reference_contexts": sample.reference_contexts,
                "retrieved_contexts": sample.retrieved_contexts,
            }
            for sample in evaluation_dataset_all.samples
        ])
        
        # Mark rows to skip for core metrics (both reference AND response == NO_ANSWER_REFERENCE)
        df_with_responses["_skip_core_metrics"] = (
            (df_with_responses["reference"].astype(str) == NO_ANSWER_REFERENCE) &
            (df_with_responses["response"].astype(str) == NO_ANSWER_REFERENCE)
        )
        
        # Create filtered dataset for core metrics
        filtered_samples = []
        for i, sample in enumerate(evaluation_dataset_all.samples):
            if not df_with_responses.iloc[i]["_skip_core_metrics"]:
                filtered_samples.append(sample)
        
        evaluation_dataset_core = EvaluationDataset(samples=filtered_samples)

        logging.info(f"Evaluating factual_correctness over all {len(evaluation_dataset_all)} rows...")
        result_fc = self._evaluate_metrics_only(
            evaluation_dataset_all,
            metrics_override=["factual_correctness"],
        )

        logging.info(
            f"Evaluating core metrics (faithfulness, answer_relevancy, context_precision, context_recall) "
            f"over {len(evaluation_dataset_core)} filtered rows (excluding rows where both reference and response are NO_ANSWER_REFERENCE)..."
        )
        result_core = self._evaluate_metrics_only(
            evaluation_dataset_core,
            metrics_override=[
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
        )

        # Merge results: left join core metrics back onto the full set by user_input + reference
        df_fc = result_fc.to_pandas()
        df_core = result_core.to_pandas()

        core_cols = [
            c
            for c in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            if c in df_core.columns
        ]

        merged = df_fc.merge(
            df_core[["user_input", "reference", "response"] + core_cols],
            on=["user_input", "reference", "response"],
            how="left",
        )

        logging.info(f"Persisting results to results.csv")
        merged.to_csv("results.csv", index=False)

        logging.info(f"Persisting cost information to results_cost.json...")
        self._persist_cost([result_fc, result_core], "results_cost.json")

    def _persist_cost(self, results: list[EvaluationResult] | EvaluationResult, results_path: str) -> None:
        results_list: list[EvaluationResult] = results if isinstance(results, list) else [results]
        cbs = [getattr(r, "cost_cb", None) for r in results_list]
        cbs = [cb for cb in cbs if cb is not None]
        if not cbs:
            logging.warning("No cost callback found; skipping cost persistence.")
            return

        # Allow environment overrides; fall back to defaults used in notebook
        def _env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        # Prefer EVAL_ prefixed vars; fall back to INPUT_RATE/OUTPUT_RATE; then hardcoded defaults
        input_rate = _env_float("EVAL_INPUT_RATE", _env_float("INPUT_RATE", 0.00000015))
        output_rate = _env_float("EVAL_OUTPUT_RATE", _env_float("OUTPUT_RATE", 0.0000006))

        # Sum tokens and costs across callbacks
        total_tokens_obj: TokenUsage | None = None
        total_cost: float = 0.0
        for cb in cbs:
            try:
                tokens = cb.total_tokens()
                if tokens is not None:
                    if total_tokens_obj is None:
                        total_tokens_obj = tokens
                    else:
                        # merge
                        total_tokens_obj.input_tokens += tokens.input_tokens
                        total_tokens_obj.output_tokens += tokens.output_tokens
            except Exception:
                pass
            try:
                total_cost += cb.total_cost(
                    cost_per_input_token=input_rate,
                    cost_per_output_token=output_rate,
                )
            except Exception:
                logging.exception("Failed computing total cost from cost callback")
                return

        payload = {
            "model": self.model,
            "input_rate": input_rate,
            "output_rate": output_rate,
            "total_tokens": (
                total_tokens_obj.input_tokens + total_tokens_obj.output_tokens
                if total_tokens_obj
                else None
            ),
            "total_cost": total_cost,
        }
        logging.info(f"Persisted cost info: {payload}")

        try:
            with open(results_path, "w") as f:
                json.dump(payload, f)
            logging.info(f"Persisted cost info: {payload}")
        except Exception:
            logging.exception("Failed to write results_cost.json")


    def _evaluate(self, wrapped_engine, evaluation_dataset, metrics_override: list[str] | None = None) -> EvaluationResult:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model))
        name_to_metric = {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(llm=evaluator_llm),
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
            "factual_correctness": FactualCorrectness(llm=evaluator_llm),
        }
        if metrics_override is None:
            metrics = list(name_to_metric.values())
        else:
            metrics = [name_to_metric[m] for m in metrics_override]

        result = evaluate(
            query_engine=wrapped_engine,
            metrics=metrics,
            dataset=evaluation_dataset,
            token_usage_parser=get_token_usage_for_openai,
        )
        return result

    def _evaluate_metrics_only(self, evaluation_dataset, metrics_override: list[str] | None = None) -> EvaluationResult:
        """
        Evaluate metrics on a dataset that already has responses populated.
        Does not re-query the engine.
        """
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model))
        name_to_metric = {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(llm=evaluator_llm),
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
            "factual_correctness": FactualCorrectness(llm=evaluator_llm),
        }
        if metrics_override is None:
            metrics = list(name_to_metric.values())
        else:
            metrics = [name_to_metric[m] for m in metrics_override]

        # Use ragas evaluate directly (not llama_index evaluate) since we already have responses
        result = ragas_evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            token_usage_parser=get_token_usage_for_openai,
        )
        return result


    def _parse_contexts(self, val):
        if isinstance(val, list):
            return val
        try:
            return literal_eval(val)
        except Exception:
            return [str(val)]


class SourceMergingQueryEngine:
    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def _normalize_nodes(self, items: list[Any] | None) -> list[Any]:
        """
        Normalize heterogeneous references into a flat list of objects that expose `.node`.

        Accepts lists containing:
        - NodeWithScore-like objects (having `.node`)
        - SubQuestionAnswerPair-like objects (having `.sources` which is a list of NodeWithScore)
        - Ignores None or unexpected types
        """
        if not items:
            return []

        normalized: list[Any] = []
        for item in items:
            if item is None:
                continue
            # Already a NodeWithScore-like object
            if hasattr(item, "node"):
                normalized.append(item)
                continue
            # SubQuestionAnswerPair-like: flatten its `.sources`
            sources = getattr(item, "sources", None)
            if isinstance(sources, list):
                for s in sources:
                    if s is None:
                        continue
                    if hasattr(s, "node"):
                        normalized.append(s)
                continue
            # Unknown type; skip defensively
        return normalized

    def _merge(self, response):
        try:
            summary_nodes = []
            if hasattr(response, "metadata") and response.metadata:
                summary_nodes = self._normalize_nodes(response.metadata.get("summary_nodes", []) or [])
            orig_nodes = self._normalize_nodes(getattr(response, "source_nodes", []) or [])
            combined = list(orig_nodes) + list(summary_nodes)

            seen_ids = set()
            merged = []
            for n in combined:
                nid = getattr(n.node, "node_id", None) or getattr(n.node, "id_", None)
                key = nid or id(n.node)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                merged.append(n)

            response.source_nodes = merged
        except Exception:
            # be conservative; never break the engine
            pass
        return response

    def query(self, *args, **kwargs):
        resp = self._inner.query(*args, **kwargs)
        return self._merge(resp)

    async def aquery(self, *args, **kwargs):
        resp = await self._inner.aquery(*args, **kwargs)
        return self._merge(resp)


class _SimpleResponse:
    def __init__(self, response: str | None, source_nodes: list[Any] | None, metadata: dict | None = None):
        self.response = response
        self.source_nodes = source_nodes or []
        self.metadata = metadata or {}


class QueryDataSourcesAdapter:
    def __init__(self, community_id: str, platform_id: str, enable_answer_skipping: bool = False):
        self._community_id = community_id
        self._platform_id = platform_id
        self._enable_answer_skipping = enable_answer_skipping

    def query(self, query: str, *_, **__):
        response, refs, meta = query_data_sources(
            community_id=self._community_id,
            query=query,
            enable_answer_skipping=self._enable_answer_skipping,
            return_metadata=True,
        )
        return _SimpleResponse(response=response, source_nodes=refs, metadata=meta)

    async def aquery(self, query: str, *_, **__):
        return self.query(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation against a community/platform.")
    parser.add_argument("--community-id", required=True, type=str, help="Community ID to evaluate.")
    parser.add_argument("--platform-id", required=True, type=str, help="Platform ID to evaluate.")
    parser.add_argument(
        "--model",
        required=False,
        default="gpt-4o-mini",
        type=str,
        help="LLM model name for evaluation (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    evaluation = StartEvaluation(
        community_id=args.community_id,
        platform_id=args.platform_id,
        model=args.model
    )
    evaluation.evaluate()
