from huggingface_hub import snapshot_download
import os
import sys # Import sys để thoát chương trình khi có lỗi

REPOS_TO_DOWNLOAD = [
    "jinaai/jina-embeddings-v3",
    "jinaai/xlm-roberta-flash-implementation"
]

print("--- Starting download of all required repositories ---")

for repo_id in REPOS_TO_DOWNLOAD:
    print(f"\nDownloading {repo_id}...")
    try:
        # force_download=True để đảm bảo tải lại hoàn toàn
        # local_dir_use_symlinks=False để tránh symlinks gây nhầm lẫn trong Docker
        # Lấy đường dẫn nơi model được tải xuống
        local_path = snapshot_download(repo_id=repo_id, force_download=True, local_dir_use_symlinks=False)
        print(f"Successfully downloaded {repo_id} to: {local_path}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}", file=sys.stderr) # In lỗi ra stderr
        print("Please check your internet connection and try again.", file=sys.stderr)
        sys.exit(1) # Thoát ngay lập tức với mã lỗi

print("\n--- All repositories downloaded successfully to Hugging Face cache ---")

hf_cache_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
print(f"Expected Hugging Face cache location: {hf_cache_home}")
print("You can verify the contents there.")