# hivemind-bot

This repository is made for TogetherCrew's LLM bot.

### Evaluations

Run our RAG evaluations locally or in GitHub Actions. Results are written to `results.csv` and `results_cost.json`.

## Run locally (Docker Compose)

Prerequisites:

- Create a `.env` file with your `OPENAI_API_KEY` (and any other required envs).

Run:

```bash
docker compose -f docker-compose.evaluation.yml up --build
```

This will:

- Start a local Qdrant at port 6333
- Run `evaluation/evaluation.py --community-id 1234 --platform-id 4321`
- Persist `results.csv` and `results_cost.json` to the repo root on your host

## Run in GitHub Actions (manual)

- Workflow: RAG Evaluation (manual trigger)
- Steps performed:
  - Boot a Qdrant service
  - Install Python dependencies and spaCy model
  - Run the evaluation
  - Compute and publish averages (faithfulness, answer_relevancy, context_precision, context_recall) to the job summary
  - Upload `results.csv` and `results_cost.json` as artifacts

Ensure `OPENAI_API_KEY` is set as a repository secret.

## Outputs

- `results.csv`: exact evaluation results (per-sample)
- `results_cost.json`: aggregate token/cost info

## TODOs

1. Fetch the Qdrant snapshot from S3 and persist it in Docker Compose evaluation
2. Fetch the test dataset from S3 and update `evaluation/evaluation.py` to load from S3 (configurable root)
