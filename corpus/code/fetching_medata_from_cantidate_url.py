import subprocess
import json
import tempfile
import os

def extract_somef_metadata(repo_url: str, somef_path: str = "D:\MASTER\TMF\somef") -> dict:
    # Create a temp file to store the output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        output_path = tmp_file.name

    try:
        # Run SOMEF with poetry from its own directory
        subprocess.run([
            "poetry", "run", "somef", "describe",
            "-r", repo_url,
            "-o", output_path,
            "-t", "0.8",
            "-m"
        ], cwd=somef_path, check=True)

        # Load the JSON output into Python
        with open(output_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Extract only the fields you care about
        relevant = {
            "name": metadata.get("name", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", []),
            "authors": []
        }

        # Extract relevant author info
        for author in metadata.get("authors", []):
            relevant["authors"].append({
                "name": author.get("name", ""),
                "givenName": author.get("givenName", ""),
                "familyName": author.get("familyName", "")
            })

        return relevant

    except subprocess.CalledProcessError as e:
        print(f"Failed to extract metadata for {repo_url}: {e}")
        return {}

    finally:
        # Optional: delete temp file
        os.remove(output_path)

# ✅ Example: loop over many GitHub URLs
github_repos = [
    "https://github.com/dgarijo/Widoco/"
]

for url in github_repos:
    metadata = extract_somef_metadata(url)
    print(f"\n📦 Metadata for: {url}\n{json.dumps(metadata, indent=2)}")
