import subprocess
import json
import tempfile
import os
import requests
import re
from urllib.parse import urlparse

#Function that based on the github username, fetches the full name of the user
# from the GitHub API. If the user is not found, it returns the username itself.
def get_github_user_data(username: str) -> str:
    url = f"https://api.github.com/users/{username}"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            full_name = data.get("name", "")
            return  full_name or username
            
 
    except Exception as e:
        print(f"Failed to fetch GitHub user data for {username}: {e}")

    # fallback
    return username
#Function that retrieves the metadata from a GitHub repository using somef
def extract_somef_metadata(repo_url: str, somef_path: str = r"D:/MASTER/TMF/somef") -> dict:
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

        def get_first_value(key):
            return metadata.get(key, [{}])[0].get("result", {}).get("value", "")

        # Split keywords string into list
        raw_keywords = get_first_value("keywords")
        keywords = [kw.strip() for kw in raw_keywords.split(",")] if raw_keywords else []

        # "owner" is treated as author (GitHub username)
        owner = get_first_value("owner")
        #gets only first description, could be multiple
        return {
            "name": get_first_value("name"),
            "description": get_first_value("description"),
            "keywords": keywords,
            "authors": [{"name": get_github_user_data(owner)}] if owner else []
        }


    except subprocess.CalledProcessError as e:
        print(f"Failed to extract metadata for {repo_url}: {e}")
        return {}

    finally:
        # delete temp file
        os.remove(output_path)
#Function that retireves the metadata from CRAN
def extract_cran_metadata(url: str) -> dict:
    
    return {}
#Function that handles the generic website metadata extraction
def extract_website_metadata(url: str) -> dict:
    return {}

#Function that retrieves the metadata from any link
def get_metadata(url: str) -> dict:
    if not isinstance(url, str) or not url.strip():
        return {"error": "Invalid URL"}

    url = url.strip()
    domain = urlparse(url).netloc.lower()

    # GitHub repo
    if "github.com" in domain:
        return extract_somef_metadata(url)

    # CRAN package (common formats: cran.r-project.org or pkg.go.dev/r)
    if "cran.r-project.org" in domain or re.search(r"(r-project\.org|cran)\b", domain):
        return extract_cran_metadata(url)

    # Generic website fallback
    return extract_website_metadata(url)
    
github_repos = [
    "https://github.com/dgarijo/Widoco/"
]

   
