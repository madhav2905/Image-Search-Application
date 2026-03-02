import os
import torch
import numpy as np
import faiss
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# 1. Setup Paths
RAW_DATA_DIR = "data/raw/coco-val-2017-500"
PROCESSED_DIR = "data/processed"
INDEX_PATH = os.path.join(PROCESSED_DIR, "visual_index.faiss")
MAPPING_PATH = os.path.join(PROCESSED_DIR, "image_mapping.npy")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Load ResNet50 (Pre-trained)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the last FC layer
model = model.to(device)
model.eval()

# 3. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_index():
    image_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    embeddings = []
    
    print(f"Extracting ResNet50 features from {len(image_files)} images on {device}...")

    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(RAW_DATA_DIR, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            
            # Get 2048-dim embedding
            embedding = model(img_t).cpu().numpy().flatten()
            embeddings.append(embedding)

    # 4. Build FAISS Index
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)  # L2 normalize the embedding
    dimension = embeddings.shape[1] # 2048
    
    # Using Inner Product for cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # 5. Save everything
    faiss.write_index(index, INDEX_PATH)
    np.save(MAPPING_PATH, np.array(image_files)) # To map index ID back to filename
    
    print(f"Visual Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    create_index()