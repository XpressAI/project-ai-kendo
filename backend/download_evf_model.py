import os
from huggingface_hub import snapshot_download

def download_model():
    # Define the repo ID and preset path for model checkpoints
    repo_id = "YxZhang/evf-sam2"
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to backend folder
    model_dir = os.path.abspath(os.path.join(current_dir, "../EVF-SAM/checkpoints/evf_sam2"))

    # Ensure the checkpoints directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Download the model
    print(f"Downloading model {repo_id} to {model_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
    print(f"Model successfully downloaded to {model_dir}")

if __name__ == "__main__":
    download_model()
