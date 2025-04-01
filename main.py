import soundfile as sf

from loguru import logger
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


tts_pipeline = None


def load_model():
    global tts
    global normalizer
    if config_path in [None, ""]:
        config_path = "GPT-SoVITS/configs/tts_infer.yaml"

    tts_config = TTS_Config(config_path)
    logger.info(tts_config)
    tts_pipeline = TTS(tts_config)
    logger.info("Loaded models.")
    return


def offload_model():
    global tts_pipeline
    del tts_pipeline
    logger.info("Offloaded models")
    return


def tts_func(text: str, language: str, model: str):
    # Check parameters
    media_type = "wav"
    ref_audio_path = 
    prompt_lang =
    text_split_method = "cut5"
    
    # if text_split_method not in cut_method_names:
    #     return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})
    
    try:
        tts_generator = tts_pipeline.run(req)
        sr, audio_data = next(tts_generator)
        
        # Save to file directly using soundfile
        sf.write("tts_output/output.wav", audio_data, sr, format='wav')
        return 
    except Exception as e:
        raise Exception(f"TTS generation failed: {str(e)}")
    return audio_path