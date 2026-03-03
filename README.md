# YOLO Hybrid Image Search Engine
A hybrid image retrieval system that combines **YOLOv11 object-aware filtering** with **ResNet50 global visual embeddings** and **FAISS similarity search** to enable intelligent, scene-level and object-level image search.

---

## Dataset
This project uses a subset of the **COCO 2017 Validation Dataset** for object detection and image retrieval experiments.

* **Total Images:** 500.
* **Selection Criteria:** Images were selected to represent a diverse range of overlapping object categories (e.g., people, vehicles, furniture) to test the robustness of the hybrid ranking logic.
* **Format:** JPEG images stored in `data/raw/`.
* **Ground Truth:** Labels and bounding boxes were derived from the official COCO annotations and verified using YOLO11 detection during the indexing phase.



### Data Structure
During the `index` phase, the system generates a structured representation of the dataset:
* **`metadata.json`**: Stores object counts and bounding box coordinates for semantic filtering.
* **`visual_index.faiss`**: A high-dimensional vector index containing L2-normalized ResNet50 embeddings for every image.
* **`image_mapping.npy`**: A NumPy map linking vector indices to original filenames for instant retrieval.

---

## Overview

This project implements a **Hybrid Content-Based Image Retrieval (CBIR) System** with two complementary search mechanisms:

1. **Object-Aware Logical Search (YOLOv11)**
   - Detects objects in images
   - Supports AND / OR filtering
   - Supports count thresholds
   - Enables upload-based object matching

2. **Visual Similarity Search (ResNet50 + FAISS)**
   - Extracts 2048-d global feature embeddings
   - Uses cosine similarity for ranking
   - Returns Top-K visually similar images

The system combines both approaches to produce **semantically relevant and visually similar results**.

---

## Architecture

                     ┌────────────────────┐
                     │   Uploaded Image   │
                     └─────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
            ┌───────▼────────┐   ┌────────▼────────┐
            │   YOLOv11      │   │    ResNet50     │
            │Object Detection│   │ Feature Extract │
            └───────┬────────┘   └────────┬────────┘
                    │                     │
            ┌───────▼────────┐   ┌────────▼────────┐
            │ Logical Filter │   │ FAISS Index     │
            │ (metadata.json)│   │ (Cosine Search) │
            └────────┬───────┘   └────────┬────────┘
                     └──────────┬──────────┘
                                ▼
                      Hybrid Ranked Results

The system operates in a three-stage pipeline:
1.  **Object Detection (YOLO11):** Identifies objects in the query image to create a "Semantic Candidate Set".
2.  **Feature Extraction (ResNet50):** Generates a 2048-dimensional vector representing the global style and composition.
3.  **Vector Retrieval (FAISS):** Performs an L2-normalized similarity search to rank candidates from most to least visually similar.

---

## Performance & Evaluation
Evaluated on a subset of the COCO dataset (500 images), the system achieved the following benchmarks:

| Metric | Result |
| :--- | :--- |
| **mAP (Mean Average Precision)** | **0.9074** |
| **Mean Recall @ 10** | **58.83%** |
| **Mean Precision @ 10** | **48.50%** |
| **Avg Visual Distance** | **0.7834** |

*A **mAP of 0.90** indicates that the ResNet50 visual ranking highly correlates with the ground-truth object categories detected by YOLO11.*

---

## Project Structure
```text
IMAGE SEARCH APP/
├── data/
│   └── processed/          # FAISS index and metadata.json
├── models/                 # Pre-trained YOLO11 weights
├── src/                    # Core logic
│   ├── generate_metadata.py
│   ├── genrate_embeddings.py
│   ├── search_metadata.py
│   └── eval.py
├── app.py                  # Streamlit UI
├── main.py                 # Project Orchestrator
└── requirements.txt        # Dependencies
```

---

## Installation & Usage

### Prerequisites

- Python 3.9+

### Setup

```bash
git clone https://github.com/your-username/yolo-resnet-search.git
cd yolo-resnet-search
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Usage

The project includes a central `main.py` to orchestrate all tasks.

###  Initialize the Index  
Processes images and builds the vector database.

```bash
python main.py index
```

This runs:
- YOLO metadata generation
- ResNet50 embedding extraction
- FAISS index building

### Launch the App  
Starts the Streamlit web interface.

```bash
python main.py run
```

Or directly:

```bash
streamlit run app.py
```

### Run Evaluation  
Generates the performance metrics report.

```bash
python main.py eval
```

Outputs:
- Mean Precision@K
- Mean Recall@K
- mAP
- Average similarity distance

---

## License
This project is open source and available under the MIT License.

---

## Contact
For questions or collaborations, feel free to open an issue or reach out!

---
