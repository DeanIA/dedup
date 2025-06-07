import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import faiss

# 1. Settings
IMAGE_DIR = "your_image_folder"  # Path to your folder with images
MODEL_NAME = "openai/clip-vit-base-patch32"  # You can use open_clip or openai CLIP
SIMILARITY_THRESHOLD = 0.99  # For cosine similarity (1.0 is identical)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# 3. Gather image paths
image_paths = [
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
]

# 4. Compute embeddings
embeddings = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**{k: v.to(DEVICE) for k, v in inputs.items()})
    emb = feats.squeeze().cpu().numpy()
    embeddings.append(emb)

embeddings = np.stack(embeddings).astype("float32")  # shape: (N, 512)

# 5. Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# 6. Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = inner product = cosine similarity (for normalized vectors)
index.add(embeddings)

# 7. Search for duplicates
D, I = index.search(embeddings, k=2)  # self-match + closest neighbor

duplicates = []
for i, (score, neighbor_idx) in enumerate(zip(D[:,1], I[:,1])):  # Skip first (self)
    if score > SIMILARITY_THRESHOLD and i != neighbor_idx:
        duplicates.append((image_paths[i], image_paths[neighbor_idx], float(score)))

# 8. Print results
print("Duplicate pairs found:")
for img1, img2, score in duplicates:
    print(f"{img1} <--> {img2} (cosine similarity: {score:.4f})")

# 9. Optionally, save to CSV
import pandas as pd
if duplicates:
    df = pd.DataFrame(duplicates, columns=["Image1", "Image2", "CosineSimilarity"])
    df.to_csv("duplicates.csv", index=False)
    print("Saved to duplicates.csv")
else:
    print("No duplicates found.")