from fastapi import FastAPI
from routers.amqp import router as amqpRouter
from routers.http import router as httpRouter

app = FastAPI()

app.include_router(httpRouter)
app.include_router(amqpRouter)
