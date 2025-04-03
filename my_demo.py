import os
import sys
import argparse
import soundfile as sf
from tools.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (  # noqa: E402
    get_method_names as get_cut_method_names,
)


# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument(
    "-c",
    "--tts_config",
    type=str,
    default="GPT_SoVITS/configs/tts_infer.yaml",
    help="tts_infer路径",
)
parser.add_argument(
    "-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1"
)
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        print("ref_audio_path is required")
    if text in [None, ""]:
        print("text is required")
    if text_lang in [None, ""]:
        print("text_lang is required")
    elif text_lang.lower() not in tts_config.languages:
        print(
            f"text_lang: {text_lang} is not supported in version {tts_config.version}"
        )
    if prompt_lang in [None, ""]:
        print("prompt_lang is required")
    elif prompt_lang.lower() not in tts_config.languages:
        print(
            f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        print(f"media_type: {media_type} is not supported")
    elif media_type == "ogg" and not streaming_mode:
        print("ogg format is not supported in non-streaming mode")

    if text_split_method not in cut_method_names:
        print(f"text_split_method:{text_split_method} is not supported")

    return None


def generate_tts_wav(
    text: str,
    text_lang: str,
    ref_audio_path: str,
    prompt_lang: str,
    prompt_text: str = "",
    aux_ref_audio_paths: list = None,
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    """
    Generate TTS .wav file and save to disk.

    Args:
        text (str): Text to be synthesized
        text_lang (str): Language of the text to be synthesized
        ref_audio_path (str): Reference audio path
        prompt_lang (str): Language of the prompt text
        prompt_text (str, optional): Prompt text for the reference audio. Defaults to "".
        aux_ref_audio_paths (list, optional): Auxiliary reference audio paths. Defaults to None.
        top_k (int, optional): Top k sampling. Defaults to 5.
        top_p (float, optional): Top p sampling. Defaults to 1.
        temperature (float, optional): Temperature for sampling. Defaults to 1.
        text_split_method (str, optional): Text split method. Defaults to "cut0".
        batch_size (int, optional): Batch size for inference. Defaults to 1.
        batch_threshold (float, optional): Threshold for batch splitting. Defaults to 0.75.
        split_bucket (bool, optional): Whether to split batch into buckets. Defaults to True.
        speed_factor (float, optional): Control speed of synthesized audio. Defaults to 1.0.
        fragment_interval (float, optional): Control interval of audio fragment. Defaults to 0.3.
        seed (int, optional): Random seed. Defaults to -1.
        parallel_infer (bool, optional): Whether to use parallel inference. Defaults to True.
        repetition_penalty (float, optional): Repetition penalty for T2S model. Defaults to 1.35.
        sample_steps (int, optional): Number of sampling steps for VITS model V3. Defaults to 32.
        super_sampling (bool, optional): Whether to use super-sampling. Defaults to False.
        output_dir (str, optional): Directory to save the output file. Defaults to "output_audio".
        output_filename (str, optional): Name for the output file. If None, generates a timestamp-based name.

    Returns:
        str: Path to the saved WAV file
    """
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "speed_factor": speed_factor,
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": sample_steps,
        "super_sampling": super_sampling,
    }

    # Check parameters
    err = check_params(req)
    if err is not None:
        raise ValueError(err.body.decode("utf-8"))

    try:
        tts_generator = tts_pipeline.run(req)
        sr, audio_data = next(tts_generator)

        # Save to file directly using soundfile
        sf.write("tts_output/output.wav", audio_data, sr, format="wav")
        return
    except Exception as e:
        raise Exception(f"TTS generation failed: {str(e)}")


if __name__ == "__main__":
    # text = "1979年邓小平游览黄山，说“这里是发展旅游的好地方，是你们发财的地方，要有点雄心壮志，把黄山的牌子打出去”。"
    # text = "黄山四千仞，三十二莲峰。丹崖夹石柱，菡萏金芙蓉。伊昔升绝顶，下窥天目松。仙人炼玉处，羽化留馀踪。亦闻温伯雪，独往今相逢。"
    # text = "自古至今，许多的文人墨客都写下了许多吟咏黄山的佳作，包括了李白、贾岛、范成大、老舍、郭沫若、龚自珍等。"
    # text = "黄山位于中国安徽省南部黄山市境内，南北长约40公里，东西宽约30公里，山脉面积1200平方公里，核心景区面积约160.6平方公里，主体以花岗岩构成，最高处莲花峰，海拔1864.8米。"
    # text_lang = "zh"

    # prompt_lang = "zh"
    # ref_audio_path = "tts_output/ref_tts_speech_sambert-hifigan_9s.wav"
    # prompt_text = "黄山位于中国安徽省南部黄山市境内，南北长约40公里，东西宽约30公里，主体以花岗岩构成"

    #  prompt_lang = "zh"
    # ref_audio_path = "tts_output/record_out.wav"
    # prompt_text = "熊哥哥和熊弟弟在地上捡到了一块奶酪，高兴极了"

    text = "From the 12th century until the 16th century, Milan was one of the largest European cities and a major trade and commercial centre, as the capital of the Duchy of Milan, one of the greatest political, artistic and fashion forces in the Renaissance."
    text_lang = "en"
    prompt_lang = "en"
    ref_audio_path = "tts_output/tts_ref_en.wav"
    prompt_text = "The Munich metropolitan area has 3 million inhabitants; and the city's metropolitan region is home to about 6.2 million people."

    generate_tts_wav(
        text=text,
        text_lang=text_lang,
        ref_audio_path=ref_audio_path,
        prompt_lang=prompt_lang,
        prompt_text=prompt_text,
    )
