import requests
import os
from dotenv import load_dotenv
from time import monotonic

load_dotenv()
tts_zh_api_key = os.environ.get("TTS_ZH_API_KEY")
service_url = "http://localhost:9503/"

# test in 'ch'
text = "1979年邓小平游览黄山，说“这里是发展旅游的好地方，是你们发财的地方，要有点雄心壮志，把黄山的牌子打出去”。"
params = {
    "text": text,
    "model": "v2",
    "language": "zh",
    "key": tts_zh_api_key,
}
start_time = monotonic()
response = requests.get(f"{service_url}/tts", params=params)
print(f"Time cost: {monotonic() - start_time} s.")

# # test in 'en'
# text = "Beijing is a global city and one of the world's leading centres for culture, diplomacy, politics, finance, business and economics, education, research, language, tourism, media, sport, science and technology and transportation and art."
# params = {
#     "text": text,
#     "model": "v2",
#     "language": "en",
#     "key": tts_zh_api_key,
# }
# start_time = monotonic()
# response = requests.get(f"{service_url}/tts", params=params)
# print(f"Time cost: {monotonic() - start_time} s.")

if response.status_code == 200:
    # Save the content as a .wav file
    with open("tts_recovered_audio.wav", "wb") as f:
        f.write(response.content)
    print("File saved as tts_recovered_audio.wav")
else:
    print("Error:", response.status_code, response.text)
