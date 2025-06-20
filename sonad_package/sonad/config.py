import os
import json
from pathlib import Path
import subprocess

CONFIG_PATH = Path.home() / ".sonad_package_config.json"

def configure_token():
    # 1) Get & save GitHub token
    token = input("Enter your GitHub token: ")
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({"github_token": token}, f, indent=2)
    print("GitHub token configured successfully.")

    # 2) Configure SOMEF CLI
    # SOMEF asks 6 questions in order:
    #   1. Authorization []
    #   2. Documentation classifier model file [...]
    #   3. Invocation classifier model file [...]
    #   4. Installation classifier model file [...]
    #   5. Citation classifier model file [...]
    #   6. Base URI for RDF generation [...]
    #
    # We supply the token to #1, then blank answers (just Enter) to the rest.
    answers = [token] + [""] * 5
    stdin_data = "\n".join(answers) + "\n"

    try:
        subprocess.run(
            ["somef", "configure"],
            input=stdin_data,
            text=True,       # send/receive strings instead of bytes
            check=True       # raise if exit code != 0
        )
        print("SOMEF CLI configured successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error configuring SOMEF CLI (exit code {e.returncode}): {e}")

def get_token():
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return config.get("github_token")