from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from api.ingest import router as ingest_router
from api.query import router as query_router


@logger.catch
def generate_application() -> FastAPI:
    application = FastAPI(title='chatdatademo-api', version='v0.0.1', description='Created by antx-code.', redoc_url=None)
    application.debug = False

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # register_exception(application)

    application.include_router(
        ingest_router,
        prefix="/api",
        tags=["Upload file to server"],
        responses={404: {"description": "Not found"}}
    )

    application.include_router(
        query_router,
        prefix="/api",
        tags=["Chat with file"],
        responses={404: {"description": "Not found"}}
    )

    return application


app = generate_application()
