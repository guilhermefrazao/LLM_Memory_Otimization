import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from uvicorn import run

from api.retriver import router as retriver_router
from api.state import lifespan

app = FastAPI(lifespan=lifespan)
app.include_router(retriver_router)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8009"))
    run(app, host="0.0.0.0", port=port)
