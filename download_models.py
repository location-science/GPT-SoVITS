from huggingface_hub import snapshot_download


# snapshot_download(
#     repo_id="Thorsten-Voice/Tacotron2-DDC",
#     local_dir="/src/tts_model"
# )

snapshot_download(repo_id="Thorsten-Voice/VITS", local_dir="/src/tts_model")