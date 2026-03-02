import os 
import json
from ultralytics import YOLO

RAW_DATA_DIR = "data/raw/coco-val-2017-500"
PROCESSED_DIR = "data/processed"
OUTPUT_JSON = os.path.join(PROCESSED_DIR, "metadata.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

model = YOLO("yolo11n.pt")

def generate_metadata():
    metadata = {}

    image_files = [
        f for f in os.listdir(RAW_DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {len(image_files)} images...")

    for index, img_name in enumerate(image_files):
        img_path = os.path.join(RAW_DATA_DIR, img_name)
        
        results = model.predict(source=img_path, device='cpu', conf=0.5, verbose=False)[0]
        
        detections = []
        class_counts = {}
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            conf = round(float(box.conf[0]), 2)
            bbox = [round(x, 1) for x in box.xyxy[0].tolist()]  # bbox format: [x1, y1, x2, y2]
            
            # Add to detections list
            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": bbox
            })
            
            # Update class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
            
        metadata[img_name] = {
            "image_path": img_path,
            "detections": detections,
            "class_counts": class_counts,
            "total_objects": len(detections)
        }

        #Progress bar
        print(f"[{index+1}/{len(image_files)}] Processing {img_name}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Success! Metadata for {len(image_files)} images saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_metadata()