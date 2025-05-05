import subprocess
import json
import tempfile
import os
import requests
import re
from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List
import re
from bs4 import BeautifulSoup


def extract_pypi_metadata(url: str) -> Dict[str, Any]:
    """
    Fetch metadata for a PyPI package given its project URL.

    This function:
      1. Parses the package name from the URL.
      2. Calls the PyPI JSON API to retrieve package info.
      3. Extracts the package name, summary description, keywords (from JSON or classifiers),
         and authors (author + maintainer fields).
      4. Derives fallback keywords from Trove classifiers if none are provided.

    Args:
        url: The URL of the PyPI project page (e.g. "https://pypi.org/project/foo").

    Returns:
        A dict with keys:
          - "name"        (str): the package name
          - "description" (str): the package summary
          - "keywords"    (List[str]): list of keywords or derived classifier tags
          - "authors"     (List[str]): list of author/maintainer names
        Or, on error, {"error": "..."}.
    """
    # 1) extract package name
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in ("project", "simple"):
        pkg = parts[1]
    else:
        pkg = parts[0] if parts else None

    if not pkg:
        return {"error": f"Cannot parse PyPI package name from URL: {url}"}

    # 2) fetch JSON metadata
    api_url = f"https://pypi.org/pypi/{pkg}/json"
    resp    = requests.get(api_url)
    if resp.status_code == 404:
        return {"error": f"Package '{pkg}' not found on PyPI"}
    resp.raise_for_status()
    info = resp.json().get("info", {})

    # 3) parse authors
    def split_authors(raw: str) -> List[str]:
        clean = re.sub(r"<[^>]+>", "", raw or "")
        parts = re.split(r",| and |;", clean)
        return [p.strip() for p in parts if p.strip()]

    authors = split_authors(info.get("author", ""))
    for m in split_authors(info.get("maintainer", "")):
        if m not in authors:
            authors.append(m)

    # 4) summary & description
    summary     = info.get("summary", "").strip()

    # 5) JSON keywords (almost always empty)
    raw_kw   = info.get("keywords") or ""
    kw_parts = re.split(r"[,\s]+", raw_kw.strip())
    keywords = [w for w in kw_parts if w]

    # 6) Trove classifiers (always present)
    classifiers = info.get("classifiers", [])

    # 7) derive fallback keywords from classifiers if JSON was empty
    if not keywords and classifiers:
        # e.g. split "Topic :: Internet :: WWW/HTTP :: Dynamic Content"
        derived = []
        for c in classifiers:
            parts = [part.strip() for part in c.split("::")]
            # skip the top-level category (e.g. "Topic", "License", etc.)
            derived.extend(parts[1:])
        # de-dup and filter
        seen = set()
        keywords = []
        for tag in derived:
            if tag and tag not in seen:
                seen.add(tag)
                keywords.append(tag)

    return {
        "name"        : info.get("name", pkg),
        "description" : summary,
        "keywords"    : keywords,
        "authors"     : authors,
    }

def parse_authors_r(authors_r: str) -> List[str]:
    """
    Parse an R Authors@R field into a list of author names.

    This function:
      1. Finds all `person(...)` blocks in the Authors@R string.
      2. Extracts the quoted names inside each block.
      3. Joins given and family names into "Given Family" format,
         and includes single-quoted organization entries.

    Args:
        authors_r: The raw Authors@R string from a CRAN DESCRIPTION.

    Returns:
        A list of author names (e.g. ["First Last", "OrgName"]).
    """
    blocks = re.findall(r'person\((.*?)\)', authors_r, flags=re.DOTALL)
    out = []
    for block in blocks:
        names = re.findall(r'"([^"]+)"', block)
        if len(names) >= 2:
            out.append(f"{names[0]} {names[1]}")
        elif len(names) == 1:
            out.append(names[0])
    return out

def extract_cran_metadata(url: str) -> Dict[str, Any]:
    """
    Fetch metadata for a CRAN R package given its documentation URL.

    This function:
      1. Parses the package name from the URL query or path.
      2. Calls the CRANDB JSON API to retrieve package info.
      3. Extracts the package name, DESCRIPTION field, and authors
         from Authors@R or Author fields, with HTML fallback scraping.
      4. Extracts keywords from DESCRIPTION or Task Views, with HTML fallback.

    Args:
        url: The URL of the CRAN package page
             (e.g. "https://cran.r-project.org/web/packages/pkg/index.html" or "?package=pkg").

    Returns:
        A dict with keys:
          - "name"        (str): package name
          - "description" (str): DESCRIPTION text
          - "keywords"    (List[str]): list of Keywords or Task View tags
          - "authors"     (List[str]): list of author names
    """
    # 1) extract pkg name
    parsed = urlparse(url)
    qs     = parse_qs(parsed.query)
    if "package" in qs:
        pkg = qs["package"][0]
    else:
        m = re.search(r"/package=([^/]+)", parsed.path)
        if m:
            pkg = m.group(1)
        else:
            parts = parsed.path.strip("/").split("/")
            if "packages" in parts:
                pkg = parts[parts.index("packages") + 1]
            else:
                raise ValueError(f"Cannot parse package name from URL: {url}")

    # 2) fetch JSON from CRANDB
    api_url = f"https://crandb.r-pkg.org/{pkg}"
    resp    = requests.get(api_url)
    resp.raise_for_status()
    data    = resp.json()

    name        = data.get("Package", pkg)
    description = data.get("Description", "")

    # 3) AUTHORS: JSON Authors@R → DESCRIPTION Author → HTML fallback
    authors: List[str] = []
    if data.get("Authors@R"):
        authors = parse_authors_r(data["Authors@R"])
    elif data.get("Author"):
        raw = data["Author"]
        # strip out any <email> and [role,…] bits
        raw = re.sub(r'<[^>]+>', '', raw)
        raw = re.sub(r'\[.*?\]', '', raw)
        # split on commas, semicolons or " and "
        parts = re.split(r',|;| and ', raw)
        authors = [p.strip() for p in parts if p.strip()]

    if not authors:
        # HTML fallback: scrape the <dt>Author:</dt> block
        html = requests.get(f"https://cran.r-project.org/web/packages/{pkg}/index.html").text
        soup = BeautifulSoup(html, "html.parser")
        dt = soup.find("dt", string=re.compile(r"Author:", re.IGNORECASE))
        if dt:
            txt   = dt.find_next_sibling("dd").get_text()
            txt   = re.sub(r'<[^>]+>', '', txt)
            txt   = re.sub(r'\[.*?\]', '', txt)
            parts = re.split(r',|;| and ', txt)
            authors = [p.strip() for p in parts if p.strip()]

    # 4) KEYWORDS: DESCRIPTION Keywords → Task Views HTML fallback
    raw_kw = data.get("Keywords") or ""
    kws    = [w.strip() for w in re.split(r"[,\s]+", raw_kw) if w.strip()]
    if not kws:
        html = requests.get(f"https://cran.r-project.org/web/packages/{pkg}/index.html").text
        soup = BeautifulSoup(html, "html.parser")
        dt   = next((d for d in soup.find_all("dt")
                     if "views" in d.get_text().lower()), None)
        if dt:
            dd_text = dt.find_next_sibling("dd").get_text()
            kws = [v.strip() for v in dd_text.split(",") if v.strip()]

    return {
        "name"       : name,
        "description": description,
        "keywords"   : kws,
        "authors"    : authors,
    }



def get_github_user_data(username: str) -> str:
    """
    Retrieve the full name for a GitHub user via the GitHub API.

    This function:
      1. Sends an authenticated request to GitHub’s /users/{username} endpoint
         if a GITHUB_TOKEN is set, or unauthenticated otherwise.
      2. Returns the “name” field if present, else falls back to the login.

    Args:
        username: The GitHub login name (e.g. "octocat").

    Returns:
        The user’s full name (str), or the original username if not found or on error.
    """
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


def extract_somef_metadata(repo_url: str, somef_path: str = r"D:/MASTER/TMF/somef") -> dict:
    """
    Run the SOMEF tool on a GitHub repository to extract metadata.

    This function:
      1. Creates a temporary JSON file.
      2. Invokes `poetry run somef describe -r {repo_url}` directing output to the temp file.
      3. Loads the SOMEF JSON output and extracts fields "name", "description",
         "keywords", and "owner" (mapped to authors via GitHub lookup).
      4. Cleans up the temporary file.

    Args:
        repo_url: URL of the GitHub repository to analyze.
        somef_path: Path to the SOMEF project directory (where `poetry run somef` is available).

    Returns:
        A dict with keys:
          - "name"        (str)
          - "description" (str)
          - "keywords"    (List[str])
          - "authors"     (List[str])
        Or an empty dict on failure.
    """
    # Create a temp file to store the output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        output_path = tmp_file.name

    try:
        # Run SOMEF with poetry from its own directory
        subprocess.run([
            "poetry", "run", "somef", "describe",
            "-r", repo_url,
            "-o", output_path,
            "-t", "0.93",
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
            "authors": [get_github_user_data(owner)] if owner else []
        }


    except subprocess.CalledProcessError as e:
        print(f"Failed to extract metadata for {repo_url}: {e}")
        return {}

    finally:
        # delete temp file
        os.remove(output_path)

#Function that handles the generic website metadata extraction
def extract_website_metadata(url: str) -> dict:
    """
    Placeholder for extracting generic website metadata.

    Args:
        url: The URL of the website.

    Returns:
        A dict of extracted metadata (implementation-specific).
    """
    return {}

#Function that retrieves the metadata from any link
def get_metadata(url: str) -> dict:
    """
    Dispatch metadata extraction based on URL domain.

    This function inspects the URL’s domain and routes to:
      - GitHub repositories (extract_somef_metadata)
      - CRAN packages (extract_cran_metadata)
      - PyPI packages (extract_pypi_metadata)
      - Generic websites (extract_website_metadata)

    Args:
        url: The URL from which to extract metadata.

    Returns:
        A metadata dict as produced by one of the specialized extractors,
        or {"error": "..."} on invalid input or failure.
    """
    if not isinstance(url, str) or not url.strip():
        return {"error": "Invalid URL"}

    url = url.strip()
    parsed = urlparse(url)
    domain = urlparse(url).netloc.lower()
    path   = parsed.path or ""

    # GitHub repo
    if "github.com" in domain:
        return extract_somef_metadata(url)

    # CRAN package (common formats: cran.r-project.org or pkg.go.dev/r)
    if domain == "cran.r-project.org" and (
        path.startswith("/web/packages/") or
        path.startswith("/package=")
    ):        return extract_cran_metadata(url)
    ## PyPI package (common formats: pypi.org or pypi.python.org)
    if "pypi.org" in domain or "pypi.python.org" in domain:
        return extract_pypi_metadata(url)
    # Generic website fallback
    return extract_website_metadata(url)
    
