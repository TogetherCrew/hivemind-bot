from fastapi import FastAPI
from routers.http import router as httpRouter
from routers.amqp import router as amqpRouter

app = FastAPI()

app.include_router(httpRouter)
app.include_router(amqpRouter)