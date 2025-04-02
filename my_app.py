import os
from loguru import logger
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
from main import load_model, offload_model, tts_func

# from dotenv import load_dotenv

# load_dotenv()
tts_zh_api_key = os.environ.get("TTS_ZH_API_KEY")

# Configure logging
log_level = "INFO"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <lvl>{level: <8}</lvl> | - <b>{message}</b>"
logger.remove()
logger.add(
    "logs/tts_zh.log",
    level=log_level,
    format=log_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
    rotation="10 MB",
    retention="2 weeks",
    compression="tar.gz",
)

# Global variable to track model load status
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    load_model()
    model_loaded = True
    yield
    model_loaded = False
    offload_model()


app = FastAPI(title="TTS_zh Service API", version="0.0.1", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return dict({"status": "ok"})


@app.get("/readiness")
async def readiness_check():
    if model_loaded:
        return {"status": "ready"}
    return {"status": "loading"}, 503


def verify_key(key: str = ""):
    if key != tts_zh_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return key


@app.get("/tts", response_description="Generate audio for given text")
async def text_to_speech(
    text: str,
    language: str,
    model: Optional[str] = "v3",
    key: str = Depends(verify_key),
):

    if text in [None, ""]:
        raise HTTPException(status_code=422, detail="Parameter 'text' is required.")
    
    if (language in [None, ""]) :
        raise HTTPException(status_code=422, detail="Parameter 'language' is required.") 
    elif language.lower() not in ["en", "zh"]:
        raise HTTPException(status_code=422, detail=f"Support only 'en' and 'zh' languages, but'{language}' is given.")
    
    if model not in ["v2", "v3"]:
        raise HTTPException(status_code=422, detail=f"Support only 'v2' and 'v3' version GPT-SoVITS models, but'{model}' is given.")

    audio_path = ""
    try:
        audio_path = tts_func(text, language, model)
    except Exception as e:
        logger.exception("Unexpected exception: {}".format(str(e)))
        raise HTTPException(
            status_code=500, detail="TTS API exception: {}".format(str(e))
        )
    return FileResponse(path=audio_path, media_type="audio/mp3")