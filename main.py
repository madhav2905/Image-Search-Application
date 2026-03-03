import sys
import subprocess

def run_indexing():
    """Runs the full pipeline to prepare the search engine."""
    print("Starting Full Indexing Pipeline...")
    
    # 1. Generate Metadata
    print("\n--- Step 1: Generating Metadata (YOLO) ---")
    subprocess.run([sys.executable, "src/generate_metadata.py"], check=True)
    
    # 2. Generate Embeddings
    print("\n--- Step 2: Generating Visual Index (ResNet50) ---")
    subprocess.run([sys.executable, "src/generate_embeddings.py"], check=True)
    
    print("\n Indexing Complete! Data is ready in data/processed/")

def run_app():
    """Launches the Streamlit Web Application."""
    print("Launching Search Engine UI...")
    subprocess.run(["streamlit", "run", "app.py"], check=True)

def run_eval():
    """Runs the performance evaluation script."""
    print("Running System Evaluation...")
    subprocess.run([sys.executable, "src/eval.py"], check=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [index | run | eval]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "index":
        run_indexing()
    elif command == "run":
        run_app()
    elif command == "eval":
        run_eval()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: index, run, eval")