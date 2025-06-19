import os
import argparse
import importlib.util
from huggingface_hub import snapshot_download
from tqdm import tqdm

def get_package_path():
    spec = importlib.util.find_spec('large_v1')
    return os.path.dirname(spec.origin) if spec else os.getcwd()

def download_models(path):
    os.makedirs(path, exist_ok=True)
    repo_id = "westlakehang/LARGE"
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=path,
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm,
            ignore_patterns=[".*", "*.md"]
        )
        print(f"\nModels downloaded successfully to {path}")
    except Exception as e:
        print(f"Error downloading models: {e}")

def main():
    package_path = get_package_path()
    default_model_path = os.path.join(package_path, "model")
    parser = argparse.ArgumentParser(description="Download LARGE models")
    parser.add_argument(
        "-p", 
        "--path", 
        default=default_model_path, 
        help=f"Model download path (default: {default_model_path})"
    )
    
    args = parser.parse_args()
    download_models(args.path)

if __name__ == "__main__":
    main()