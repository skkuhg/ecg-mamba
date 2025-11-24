#!/usr/bin/env python3
"""Upload ECG-Mamba project to Hugging Face."""

from huggingface_hub import HfApi, create_repo, login
import os
import sys

# You'll need to provide your token
print("Please authenticate with Hugging Face.")
print("You can get your token from: https://huggingface.co/settings/tokens")
token = input("Enter your Hugging Face token (or press Enter to use cached credentials): ").strip()

if token:
    login(token=token)
else:
    # Try to use cached credentials
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"Authenticated as: {whoami['name']}")
    except Exception as e:
        print(f"Error: Not authenticated. Please provide a token.")
        print(f"Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

# Get username
api = HfApi()
whoami = api.whoami()
username = whoami['name']

# Create repository with full repo_id
repo_name = f"{username}/ecg-mamba"
print(f"\nCreating repository: {repo_name}")

try:
    repo_url = create_repo(
        repo_id=repo_name,
        repo_type="model",
        private=False,
        exist_ok=True
    )
    print(f"Repository created/exists: {repo_url}")
except Exception as e:
    print(f"Error creating repository: {e}")
    sys.exit(1)

# Upload files
files_to_upload = [
    "ECG_Mamba_Colab_Test.ipynb",
    "README.md",
    "LICENSE"
]

print("\nUploading files...")
for file in files_to_upload:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_name,
                repo_type="model",
            )
            print(f"[OK] {file} uploaded successfully")
        except Exception as e:
            print(f"[ERROR] Error uploading {file}: {e}")
    else:
        print(f"[ERROR] File not found: {file}")

print(f"\nAll done! Visit your repository at: https://huggingface.co/{repo_name}")
