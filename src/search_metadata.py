import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "metadata.json")

def load_metadata():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Exclude images with zero detections
    filtered_metadata = {
        image_id: data
        for image_id, data in metadata.items()
        if data.get("total_objects", 0) > 0
    }

    return filtered_metadata

def search_images(selected_classes, mode="AND", count_thresholds=None):
    if count_thresholds is None:
        count_thresholds = {}

    metadata = load_metadata()

    if not selected_classes:
        return []   # If no classes selected, return empty

    results = []

    for image_id, data in metadata.items():

        class_counts = data.get("class_counts", {})

        # Logical condition
        if mode == "AND":
            logical_match = all(
                class_counts.get(cls, 0) >= 1
                for cls in selected_classes
            )

        elif mode == "OR":
            logical_match = any(
                class_counts.get(cls, 0) >= 1
                for cls in selected_classes
            )

        else:
            raise ValueError("Mode must be 'AND' or 'OR'")

        if not logical_match:
            continue

        # Count thresholds
        if mode == "AND":
            # All thresholds must be satisfied
            threshold_match = all(
                class_counts.get(cls, 0) >= count_thresholds.get(cls, 1)
                for cls in selected_classes
            )

            if not threshold_match:
                continue

        elif mode == "OR":
            # At least one selected class must satisfy its threshold
            threshold_match = any(
                class_counts.get(cls, 0) >= count_thresholds.get(cls, 1)
                for cls in selected_classes
            )

            if not threshold_match:
                continue

        results.append(image_id)

    # Sort alphabetically
    results = sorted(results)

    return results