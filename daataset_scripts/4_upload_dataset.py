import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi

api = HfApi()

# data_dir = "dataset_hf"
# repo_id = "OrcinusOrca/YouTube-Cantonese"
data_dir = "dataset_hf_en"
repo_id = "OrcinusOrca/YouTube-English"

try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
    )
except Exception as e:
    pass

# api.delete_files(
#     repo_id=repo_id,
#     repo_type="dataset",
#     delete_patterns=["*.tar"],
# )
api.upload_large_folder(
    folder_path=data_dir,
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["*.tar", "README.md"],
)

# huggingface-cli upload-large-folder OrcinusOrca/YouTube-Cantonese data --repo-type=dataset
