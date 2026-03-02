import numpy as np
import faiss
import random
from tqdm import tqdm
from search_metadata import load_metadata, search_images


# Load Data
metadata = load_metadata()
image_mapping = np.load("data/processed/image_mapping.npy")
visual_index = faiss.read_index("data/processed/visual_index.faiss")


# Average Precision
def calculate_ap(relevance_mask):
    if sum(relevance_mask) == 0:
        return 0

    precision_values = []
    correct = 0

    for i, rel in enumerate(relevance_mask):
        if rel == 1:
            correct += 1
            precision_at_i = correct / (i + 1)
            precision_values.append(precision_at_i)

    return np.mean(precision_values)

# Full Evaluation
def evaluate_full_metrics(num_samples=20, k=10):

    all_ids = list(metadata.keys())
    samples = random.sample(all_ids, min(num_samples, len(all_ids)))

    metrics = {
        "precision": [],
        "recall": [],
        "ap": [],
        "distances": []
    }

    for query_id in tqdm(samples):

        # Ground Truth (Metadata Filter)
        q_item = metadata[query_id]
        q_classes = list(q_item.get("class_counts", {}).keys())

        gt_ids = search_images(
            q_classes,
            mode="AND",
            count_thresholds={c: 1 for c in q_classes}
        )

        total_relevant = len(gt_ids)

        if total_relevant == 0:
            continue

        # Get Query Vector from FAISS
        idx_positions = np.where(image_mapping == query_id)[0]
        if len(idx_positions) == 0:
            continue

        idx_pos = idx_positions[0]

        query_vec = visual_index.reconstruct(int(idx_pos)).reshape(1, -1)

        # Run FAISS Search
        D, I = visual_index.search(query_vec, k)

        top_k_indices = I[0]
        top_k_ids = [image_mapping[idx] for idx in top_k_indices]

        # Compute Metrics
        relevance_mask = [1 if img_id in gt_ids else 0 for img_id in top_k_ids]
        hits = sum(relevance_mask)

        precision = hits / k
        recall = hits / total_relevant
        ap = calculate_ap(relevance_mask)

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["ap"].append(ap)
        metrics["distances"].append(np.mean(D[0]))

    # Final Report
    print("\n" + "═" * 40)
    print(f"SYSTEM PERFORMANCE (K={k})")
    print(f"Mean Precision@{k}: {np.mean(metrics['precision']) * 100:.2f}%")
    print(f"Mean Recall@{k}:    {np.mean(metrics['recall']) * 100:.2f}%")
    print(f"mAP (Mean Avg Precision): {np.mean(metrics['ap']):.4f}")
    print(f"Avg Visual Distance:      {np.mean(metrics['distances']):.4f}")
    print("═" * 40)

if __name__ == "__main__":
    evaluate_full_metrics(num_samples=20, k=10)