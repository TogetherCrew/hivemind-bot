import logging
import json
import argparse
import os
from typing import Any
from dotenv import load_dotenv

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from ragas.integrations.llama_index import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.llms import LangchainLLMWrapper
from utils.query_engine.prepare_discord_query_engine import prepare_discord_engine_auto_filter
from ragas_experimental import Dataset
from ragas.testset.synthesizers.testset_schema import Testset
from ast import literal_eval
from ragas.evaluation import EvaluationResult
from ragas.cost import TokenUsage



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
        self.engine = prepare_discord_engine_auto_filter(
            community_id,
            platform_id,
            enable_answer_skipping=True
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
        _items = []
        for _, r in _df.iterrows():
            _items.append({
                "user_input": r["user_input"],
                "reference_contexts": self._parse_contexts(r["reference_contexts"]),
                "reference": r["reference"],
                "synthesizer_name": r.get("synthesizer_name", "unknown"),
            })

        _testset = Testset.from_list(_items)

        # TODO: remove the [:2] after testing
        evaluation_dataset = _testset.to_evaluation_dataset()[:2]

        # the engine combining the summary and the source nodes
        wrapped_engine = SourceMergingQueryEngine(self.engine)

        logging.info(f"Evaluating...")
        results = self._evaluate(wrapped_engine, evaluation_dataset)
        logging.info(f"Results: {results}")

        logging.info(f"Persisting results to results.csv")
        results.to_pandas().to_csv("results.csv")

        logging.info(f"Persisting cost information to results_cost.json...")
        self._persist_cost(results, "results_cost.json")

    def _persist_cost(self, results: EvaluationResult, results_path: str) -> None:
        cb = getattr(results, "cost_cb", None)
        if cb is None:
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

        total_tokens: TokenUsage | None
        try:
            total_tokens = cb.total_tokens()
        except Exception:
            total_tokens = None

        try:
            total_cost: float = cb.total_cost(
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
            "total_tokens": total_tokens.input_tokens + total_tokens.output_tokens if total_tokens else None,
            "total_cost": total_cost,
        }
        logging.info(f"Persisted cost info: {payload}")

        try:
            with open(results_path, "w") as f:
                json.dump(payload, f)
            logging.info(f"Persisted cost info: {payload}")
        except Exception:
            logging.exception("Failed to write results_cost.json")


    def _evaluate(self, wrapped_engine, evaluation_dataset) -> EvaluationResult:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model))
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]

        result = evaluate(
            query_engine=wrapped_engine,
            metrics=metrics,
            dataset=evaluation_dataset,
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

    def _merge(self, response):
        try:
            summary_nodes = []
            if hasattr(response, "metadata") and response.metadata:
                summary_nodes = response.metadata.get("summary_nodes", []) or []
            orig_nodes = getattr(response, "source_nodes", []) or []
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
