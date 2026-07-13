import os
import urllib.request
import ssl

# Disable SSL verification for this download
ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://hf-mirror.com/sentence-transformers/clip-ViT-B-32/resolve/main"

# Only create the directories we need
os.makedirs("clip-model", exist_ok=True)
os.makedirs("clip-model/0_CLIPModel", exist_ok=True)

# The correct file mapping for clip-ViT-B-32
model_files = {
    "config_sentence_transformers.json": f"{BASE_URL}/config_sentence_transformers.json",
    "modules.json": f"{BASE_URL}/modules.json",
    "README.md": f"{BASE_URL}/README.md",
    "0_CLIPModel/config.json": f"{BASE_URL}/0_CLIPModel/config.json",
    "0_CLIPModel/preprocessor_config.json": f"{BASE_URL}/0_CLIPModel/preprocessor_config.json",
    "0_CLIPModel/special_tokens_map.json": f"{BASE_URL}/0_CLIPModel/special_tokens_map.json",
    "0_CLIPModel/tokenizer_config.json": f"{BASE_URL}/0_CLIPModel/tokenizer_config.json",
    "0_CLIPModel/vocab.txt": f"{BASE_URL}/0_CLIPModel/vocab.txt",
    "0_CLIPModel/pytorch_model.bin": f"{BASE_URL}/0_CLIPModel/pytorch_model.bin",
}

print("Starting manual CLIP model download from mirror...")

for relative_path, url in model_files.items():
    local_path = os.path.join("clip-model", relative_path)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"[EXISTS] {relative_path}")
        continue
    
    print(f"Downloading {relative_path} from mirror...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"[SUCCESS] Downloaded {relative_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {relative_path}: {e}")

print("Manual download process finished.")
