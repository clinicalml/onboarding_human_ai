from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import torch

def get_text_embeddings_clip(text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # create random sample Image
    img = Image.new("RGB", (224, 224))
    images = [np.array(img)]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeds = outputs.text_embeds.numpy()
    return embeds[0]


def get_text_embeddings_text(text):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(text, batch_size=16)
    embeddings = np.array(embeddings)
    return embeddings
