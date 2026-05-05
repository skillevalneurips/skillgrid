from huggingface_hub import snapshot_download
import os

# model_id = "Qwen/Qwen3-4B-Base"
# model_id = "Qwen/Qwen3-8B-Base"
# model_id = "Alibaba-NLP/WebSailor-3B"
model_id = "Alibaba-NLP/WebSailor-7B"

# Move HF caches off /home as well
os.environ.setdefault("HF_HOME", "/deepfreeze/yav13/.cache/huggingface")

base_dir = "/deepfreeze/yav13/models"
target_dir = os.path.join(base_dir, model_id.split("/")[-1])
os.makedirs(target_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
)

print("Model downloaded to:", target_dir)