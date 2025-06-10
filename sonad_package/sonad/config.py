import os
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".sonad_package_config.json"

def configure_token():
    token = input("Enter your GitHub token: ")
    config = {"github_token": token}
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f)
    print("Token configured successfully!")

def get_token():
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return config.get("github_token")