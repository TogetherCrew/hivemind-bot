from datetime import datetime

from faststream.rabbit.fastapi import Logger, RabbitRouter  # type: ignore
from faststream.rabbit.schemas.queue import RabbitQueue
from pydantic import BaseModel
from schema import ResponseModel, RouteModelPayload
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials
from utils.persist_payload import PersistPayload
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from utils.traceloop import init_tracing
from worker.tasks import query_data_sources
from worker.utils.fire_event import job_send
from bot.evaluations.answer_relevance import AnswerRelevanceEvaluation
from bot.evaluations.schema import AnswerRelevanceSuccess

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Payload(BaseModel):
    event: str
    date: datetime | str
    content: RouteModelPayload


@router.subscriber(queue=RabbitQueue(name=Queue.HIVEMIND, durable=True))
async def ask(payload: Payload, logger: Logger):
    if payload.event == Event.HIVEMIND.QUESTION_RECEIVED:
        try:
            question = payload.content.question.message
            community_id = payload.content.communityId
            init_tracing()
            logger.info(f"COMMUNITY_ID: {community_id} Received job")

            if payload.content.metadata:
                enable_answer_skipping = payload.content.metadata.get(
                    "enableAnswerSkipping", False
                )
            else:
                enable_answer_skipping = False

            response, references = query_data_sources(
                community_id=community_id,
                query=question,
                enable_answer_skipping=enable_answer_skipping,
            )
            prepare_answer = PrepareAnswerSources()
            answer_reference = prepare_answer.prepare_answer_sources(nodes=references)

            logger.info(f"COMMUNITY_ID: {community_id} Job finished")

            eval_result = await AnswerRelevanceEvaluation().evaluate(
                question=question, answer=response
            )

            response_payload = RouteModelPayload(
                communityId=community_id,
                route=payload.content.route,
                question=payload.content.question,
                response=ResponseModel(message=f"{response}\n\n{answer_reference}"),
                metadata={
                    **payload.content.metadata,
                    "answer_relevance_score": (
                        eval_result.score
                        if isinstance(eval_result, AnswerRelevanceSuccess)
                        else eval_result.error
                    ),
                    "answer_relevance_explanation": (
                        eval_result.explanation
                        if isinstance(eval_result, AnswerRelevanceSuccess)
                        else eval_result.error
                    ),
                },
            )
            # dumping the whole payload of question & answer to db
            persister = PersistPayload()
            persister.persist_payload(response_payload)

            if response is None:
                raise ValueError("not confident in answering!")

            job_send(
                event=payload.content.route.destination.event,
                queue_name=payload.content.route.destination.queue,
                content=response_payload.model_dump(),
            )
        except Exception as e:
            logger.exception(f"Errors While processing job! {e}")
    else:
        logger.error(
            f"No such `{payload.event}` event available for {Queue.HIVEMIND} queue!"
        )
