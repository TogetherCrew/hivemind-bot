# It's recommended that we use `bullseye` for Python (alpine isn't suitable as it conflcts with numpy)
FROM python:3.11-bullseye AS base
WORKDIR /project
COPY . .
RUN pip3 install -r requirements.txt --no-cache-dir
RUN python -m spacy download en_core_web_sm

FROM base AS test
RUN chmod +x docker-entrypoint.sh
CMD ["./docker-entrypoint.sh"]

FROM base AS dev-server
CMD ["fastapi", "run", "dev", "--port", "3000"]

FROM base AS prod
CMD ["celery", "-A", "worker", "worker", "-l", "INFO"]

FROM base AS dev-temporal
CMD ["python", "temporal/temporal_worker.py"]
