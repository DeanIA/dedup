import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import faiss

# 1. Settings
IMAGE_DIR = "your_image_folder" 
MODEL_NAME = "openai/clip-vit-base-patch32"  
SIMILARITY_THRESHOLD = 0.99  # For cosine similarity (1.0 is identical)

# Device config
if torch.cuda.is_available() : 
    DEVICE = "cuda:1"
    print(f"device:{DEVICE}")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print(f"device:{DEVICE}")
else:
    print(f"Plain ol' CPU")

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# 3. Gather image paths
image_paths = [
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
]

# Compute embeddings
embeddings = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**{k: v.to(DEVICE) for k, v in inputs.items()})
    emb = feats.squeeze().cpu().numpy()
    embeddings.append(emb)

embeddings = np.stack(embeddings).astype("float32")  # shape: (N, 512)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP cosine similarity (for normalized vectors)
index.add(embeddings)

# Search for duplicates
lims, D, I = index.range_search(embeddings, SIMILARITY_THRESHOLD)

duplicates = []
N = embeddings.shape[0]
for i in range(N):
    # each neighbor for query i lives in I[lims[i]:lims[i+1]]
    for pos in range(lims[i], lims[i+1]):
        j = int(I[pos])
        score = float(D[pos])
        # skip self‐match and repeating pairs
        if i < j:
            duplicates.append((image_paths[i], image_paths[j], score))

# Print results
print("Duplicate pairs found:")
for img1, img2, score in duplicates:
    print(f"{img1} <--> {img2} (cosine similarity: {score:.4f})")

# Save to CSV
import pandas as pd
if duplicates:
    df = pd.DataFrame(duplicates, columns=["Image1", "Image2", "CosineSimilarity"])
    df.to_csv("duplicates.csv", index=False)
    print("Saved to duplicates.csv")
else:
    print("No duplicates found.")