from huggingface_hub import create_repo, upload_folder, login
from dotenv import load_dotenv
import os

_logged_in = False


def upload_to_hf(repo_id, folder_path, commit_msg):
    global _logged_in

    if not _logged_in:
        load_dotenv()
        token = os.getenv("HF_ACCESS_TOKEN")

        if not token:
            raise ValueError("HF_ACCESS_TOKEN not found in environment")

        login(token=token)
        _logged_in = True

    create_repo(repo_id, repo_type="model", exist_ok=True)

    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_msg
    )
