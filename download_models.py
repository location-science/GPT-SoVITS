from huggingface_hub import snapshot_download
import nltk



# Download G2PWModel_1.1
snapshot_download(repo_id="L-jasmine/GPT_Sovits", local_dir="/src/GPT_SoVITS/text/G2PWModel", 
                  allow_patterns="G2PWModel_1.1.zip")
                  
# Download GPT_SoVITS model
snapshot_download(repo_id="lj1995/GPT-SoVITS", local_dir="/src/GPT_SoVITS/pretrained_models", 
                  ignore_patterns=["s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s2D488k.pth", "s2G488k.pth"])

# Download nltk packages
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("averaged_perceptron_tagger")
nltk.download("cmudict")