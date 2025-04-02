import soundfile as sf
import os
import sys
from loguru import logger
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
# logger.info(f"sys.path: {sys.path}")
# logger.info(f"Current working directory: {os.getcwd()}")
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


tts_pipeline = None


def load_model():
    global tts_pipeline
    config_path = "GPT_SoVITS/configs/tts_infer.yaml"
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
    
    # if text_split_method not in cut_method_names:
    #     return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})
    
    if language == "en":
        ref_audio_path = "ref_audio/tts_ref_en_male.wav"
        prompt_text = "The Munich metropolitan area has 3 million inhabitants; and the city's metropolitan region is home to about 6.2 million people."
    else:
        ref_audio_path = "ref_audio/tts_ref_zh_female.wav"
        prompt_text = "黄山位于中国安徽省南部黄山市境内，南北长约40公里，东西宽约30公里，主体以花岗岩构成"

    param = {
                "text": text,                 # str.(required) text to be synthesized
                "text_lang": language,        # str.(required) language of the text to be synthesized
                "ref_audio_path": ref_audio_path, # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": prompt_text,   # str.(optional) prompt text for the reference audio
                "prompt_lang": language,      # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                "return_fragment": False,     # bool. step by step return the audio fragment.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "parallel_infer": True,       # bool. whether to use parallel inference.
                "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
            }
    tts_generator = tts_pipeline.run(param)
    sr, audio_data = next(tts_generator)
    output_path = "tts_output.wav"
    # Save to file directly using soundfile
    sf.write("tts_output.wav", audio_data, sr, format='wav')
    return output_path
