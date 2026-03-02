import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import faiss
from torchvision import models, transforms
from ultralytics import YOLO

from src.search_metadata import search_images, load_metadata

# PAGE CONFIG
st.set_page_config(page_title="YOLO Hybrid Search", layout="wide")

# MODEL & INDEX LOADERS 
@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    yolo = YOLO("yolo11n.pt")
    
    # ResNet50 for Visual Similarity (Backbone only)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet = resnet.to(device)
    resnet.eval()
    
    return yolo, resnet, device

@st.cache_resource
def load_visual_index():
    index = faiss.read_index("data/processed/visual_index.faiss")
    image_mapping = np.load("data/processed/image_mapping.npy")
    return index, image_mapping

yolo_model, resnet_model, device = load_models()
visual_index, image_mapping = load_visual_index()
metadata = load_metadata()

resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image):
    img_t = resnet_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(img_t).cpu().numpy().flatten().astype('float32')
    # L2 Normalization for consistency with index
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding

# UI: HEADER & DATASET STATS
st.title("YOLO-Powered Image Search App")
st.markdown("Object-aware logical filtering + ResNet50 Visual Similarity Ranking")

all_classes = sorted(list(set(cls for data in metadata.values() for cls in data.get("class_counts", {}).keys())))

# SIDEBAR FILTERS 
st.sidebar.header("Search Filters")
selected_classes = st.sidebar.multiselect("Select Classes", options=all_classes)
mode = st.sidebar.radio("Logical Mode", options=["AND", "OR"], 
                        format_func=lambda x: "All of the selected classes (AND)" if x == "AND" else "Any of the selected classes (OR)")

count_thresholds = {}
if selected_classes:
    st.sidebar.markdown("### Count Thresholds")
    for cls in selected_classes:
        count_thresholds[cls] = st.sidebar.number_input(f"Minimum {cls} count", min_value=1, value=1)

search_button = st.sidebar.button("Search")

# MAIN UI
st.divider()
st.header("📤 Search by Image Upload")
uploaded_file = st.file_uploader("Upload an image to find similar items...", type=["jpg", "jpeg", "png"])

upload_ranked_results = []
detected_unique_classes = []

if uploaded_file is not None:
    # 1. Process Uploaded Image
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    
    # 2. YOLO Detection
    yolo_results = yolo_model.predict(img_np, conf=0.3, verbose=False)[0]
    detected_unique_classes = sorted(list(set([yolo_model.names[int(box.cls)] for box in yolo_results.boxes])))
    
    st.subheader("Detected Objects in Upload")
    if detected_unique_classes:
        st.success(f"Found: {', '.join(detected_unique_classes)}")
        st.image(yolo_results.plot(), caption="Uploaded Image Detections", width=400)
        
        # 3. Get ResNet Embedding & Search Visual Index
        query_vec = get_image_embedding(pil_img)
        
        # 4. Filter by Labels (YOLO)
        filtered_ids = search_images(detected_unique_classes, mode="AND", count_thresholds={c: 1 for c in detected_unique_classes})
        
        # 5. Rank by Visual Similarity (FAISS)
        if filtered_ids:
            # Get distances for ALL images in dataset from FAISS
            D, I = visual_index.search(query_vec.reshape(1, -1), len(image_mapping))
            
            # Create a lookup for similarity rank
            dist_map = {image_mapping[idx]: dist for dist, idx in zip(D[0], I[0])}
            
            # Sort the YOLO-filtered results by their ResNet distance
            upload_ranked_results = sorted(filtered_ids, key=lambda x: dist_map.get(x, float('inf')))
            
            # Limit to top 10
            upload_ranked_results = upload_ranked_results[:10]
    else:
        st.warning("No known objects detected in the uploaded image.")

# RESULTS DISPLAY
show_boxes = st.checkbox("Show Bounding Boxes on Results", value=True)

# 1. Similar Images Section (Hybrid)
if uploaded_file and upload_ranked_results:
    st.divider()
    st.header("Similar Images from Dataset (Ranked by Visual Style)")
    
    u_cols = st.columns(4)
    for u_idx, u_id in enumerate(upload_ranked_results):
        u_item = metadata[u_id]
        u_img = Image.open(u_item["image_path"]).convert("RGB")
        u_img_np = np.array(u_img)
        
        if show_boxes:
            for d in u_item["detections"]:
                if d["class_name"] in detected_unique_classes:
                    cv2.rectangle(u_img_np, (int(d["bbox"][0]), int(d["bbox"][1])), (int(d["bbox"][2]), int(d["bbox"][3])), (255,0,0), 2)
        
        with u_cols[u_idx % 4]:
            st.image(u_img_np, use_container_width=True)
            st.caption(f"{u_id}")

# 2. Manual Search Results 
if search_button:
    manual_results = search_images(selected_classes, mode, count_thresholds)
    if manual_results:
        st.divider()
        st.header("Manual Search Results")
        m_cols = st.columns(4)
        for m_idx, m_id in enumerate(manual_results):
            m_item = metadata[m_id]
            m_img = Image.open(m_item["image_path"]).convert("RGB")
            m_img_np = np.array(m_img)
            
            if show_boxes:
                for d in m_item["detections"]:
                    if d["class_name"] in selected_classes:
                        cv2.rectangle(m_img_np, (int(d["bbox"][0]), int(d["bbox"][1])), (int(d["bbox"][2]), int(d["bbox"][3])), (255,0,0), 2)
            
            with m_cols[m_idx % 4]:
                st.image(m_img_np, use_container_width=True)
                st.caption(m_id)
    else:
        st.warning("No matching images found.")