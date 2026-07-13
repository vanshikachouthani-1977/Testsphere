import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTHONHTTPSVERIFY"] = "0"

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class VectorEmbeddingAgent:
    _model = None

    def __init__(self):
        # Lazily load the model on first initialization to save memory/startup time
        if VectorEmbeddingAgent._model is None:
            local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clip-model")
            if os.path.exists(os.path.join(local_model_path, "0_CLIPModel", "pytorch_model.bin")):
                print(f"Loading CLIP model locally from: {local_model_path}")
                VectorEmbeddingAgent._model = SentenceTransformer(local_model_path)
            else:
                print("Loading CLIP model from Hugging Face...")
                VectorEmbeddingAgent._model = SentenceTransformer('clip-ViT-B-32')

    def compare(self, mockup_img: Image.Image, screenshot_img: Image.Image) -> float:
        """
        Generates CLIP embeddings for both images and computes their cosine similarity.
        """
        # Encode the images
        embeddings = VectorEmbeddingAgent._model.encode([mockup_img, screenshot_img])
        
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)

        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0

        similarity = float(dot_product / (norm_emb1 * norm_emb2))
        return similarity
