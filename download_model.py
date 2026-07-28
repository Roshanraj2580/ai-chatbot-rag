"""
Script to pre-download the nomic-embed-text-v1 model.
Run this before starting the application to ensure the model is cached.
"""
import os
import sys
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
# Define path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nomic-embed-text-v1")

def download_model():
    """
    Download the embedding model to the local models directory.
    """
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Saving to: {MODEL_PATH}")
    print("This may take a few minutes (~100-500MB)...")
    print()
    
    try:
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # Download and save to specific directory
        # We use the parent 'models' dir as cache_folder so it organizes correctly, 
        # but we want to ensure it ends up in MODEL_PATH.
        # SentenceTransformer.save() is the most reliable way to store it for local loading.
        
        print("Downloading from HuggingFace...")
        model = SentenceTransformer(MODEL_NAME, cache_folder=os.path.join(BASE_DIR, "models"), trust_remote_code=True)
        
        print(f"Saving model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        
        print(f"\n✅ Model downloaded and saved to: {MODEL_PATH}")
        
        # Test the model
        print("\nTesting model...")
        test_embedding = model.encode(["test sentence"], normalize_embeddings=True)
        print(f"✅ Model works! Embedding dimension: {test_embedding.shape[1]}")
        
        print("\n🎉 Model ready to use!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have ~500MB free disk space")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Nomic Embed Text v1 Model Downloader")
    print("=" * 60)
    print()
    
    print("Press Ctrl+C to cancel")
    print()
    
    success = download_model()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
