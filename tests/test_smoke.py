import os
import requests

# from loguru import logger

# from dotenv import load_dotenv
# load_dotenv()

# BASE_URL = "http://localhost:9503"
BASE_URL = "https://tts-zh.livelyisland-33d28559.centralus.azurecontainerapps.io"


def test_tts_zh():
    api_key = os.environ.get("TTS_ZH_API_KEY")
    params = {
        "text": "黄山四千仞，三十二莲峰。丹崖夹石柱，菡萏金芙蓉。伊昔升绝顶，下窥天目松。仙人炼玉处，羽化留馀踪。亦闻温伯雪，独往今相逢。",
        "language": "zh",
        "key": api_key,
    }
    response = requests.get(f"{BASE_URL}/tts", params=params)
    assert response.status_code == 200


def test_tts_en():
    api_key = os.environ.get("TTS_ZH_API_KEY")
    params = {
        "text": "The Wurm is a river in the state of North Rhine-Westphalia in western Germany.",
        "language": "en",
        "key": api_key,
    }
    response = requests.get(f"{BASE_URL}/tts", params=params)
    assert response.status_code == 200



if __name__ == "__main__":
    test_tts_zh()
    test_tts_en()