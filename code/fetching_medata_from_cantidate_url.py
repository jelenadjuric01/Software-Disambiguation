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
from rake_nltk import Rake

BLACKLIST = {"Development Status", "License", "Programming Language", "Topic", "Framework"}
MIN_LEN   = 3

def extract_pypi_metadata(url: str) -> Dict[str, Any]:
    """Fetch metadata for a PyPI package given its project URL.

    Parses the package name from the URL, retrieves info from the PyPI JSON API,
    and extracts:
      - name       : package name
      - description: summary string
      - keywords   : list from JSON or derived from Trove classifiers
      - authors    : combined author + maintainer names
      - language   : always "Python"

    Args:
        url: PyPI project URL (e.g. "https://pypi.org/project/foo").

    Returns:
        A dict with keys:
          name (str), description (str), keywords (List[str]),
          authors (List[str]), language (str).
        On error, returns {"error": "..."}.
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
        derived = []
        for c in classifiers:
            # keep only Topic-related classifiers
            if not c.startswith("Topic ::"):
                continue
            tag = c.split("::")[-1].strip()
            # length filter + blacklist
            if len(tag) < MIN_LEN or tag in BLACKLIST:
                continue
            derived.append(tag)
        # dedupe while preserving order
        seen = set()
        keywords = [t for t in derived if not (t in seen or seen.add(t))]

    return {
        "name"        : info.get("name", pkg),
        "description" : summary,
        "keywords"    : keywords,
        "authors"     : authors,
        "language"  : "Python"
    }

def parse_authors_r(authors_r: str) -> List[str]:
    """Parse an R Authors@R DESCRIPTION field into author names.

    Finds all `person(...)` blocks in the string, extracts quoted tokens,
    and joins given + family names into "Given Family" format.  Single-quoted
    entries (organizations) are included as-is.

    Args:
        authors_r: Raw Authors@R field from a CRAN DESCRIPTION file.

    Returns:
        A list of author or organization names (e.g. ["First Last", "OrgName"]).
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
    """Fetch metadata for a CRAN R package given its documentation URL.

    Parses the package name from the URL or query, queries the CRANDB API,
    and extracts:
      - name       : package name
      - description: DESCRIPTION text
      - keywords   : from DESCRIPTION, Task Views, or RAKE fallback
      - authors    : from Authors@R, Author field, or HTML fallback
      - language   : always "R"

    Args:
        url: CRAN package URL, e.g.
             "https://cran.r-project.org/web/packages/pkg/index.html"
             or "?package=pkg".

    Returns:
        A dict with keys:
          name (str), description (str), keywords (List[str]), authors (List[str]), language (str).
        Raises ValueError if the package name cannot be parsed.
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
        r = Rake(min_length=2, max_length=3)
        r.extract_keywords_from_text(description)
        kws = r.get_ranked_phrases()[:5]

        # 4c) clean & filter
        cleaned = []
        for kw in kws:
            # strip stray punctuation/quotes and lowercase
            tag = kw.strip(' "\'.,').lower()
            # keep only multi-word, alphanumeric phrases
            if len(tag.split()) > 1 and re.match(r'^[\w\s]+$', tag):
                cleaned.append(tag)
        # dedupe
        seen = set()
        kws = [t for t in cleaned if not (t in seen or seen.add(t))]


    return {
        "name"       : name,
        "description": description,
        "keywords"   : kws,
        "authors"    : authors,
        "language"   : "R"
    }



def get_github_user_data(username: str) -> str:
    """Retrieve a GitHub user’s display name via the GitHub API.

    Sends an authenticated request if GITHUB_TOKEN is set; otherwise unauthenticated.
    Returns the “name” field from the API, falling back to the login on error or if blank.

    Args:
        username: GitHub login (e.g. "octocat").

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
    """Run the SOMEF tool on a GitHub repository to extract metadata.

    Invokes `poetry run somef describe` in a temp file, then reads JSON to extract:
      - name        : project name
      - description : text description
      - keywords    : list of keywords
      - authors     : list containing the GitHub repo owner’s display name
      - language    : primary programming language by code size

    Args:
        repo_url:   URL of the GitHub repository.
        somef_path: Path to the SOMEF project directory where `poetry run somef` is available.

    Returns:
        A dict with keys:
          name (str), description (str), keywords (List[str]), authors (List[str]), language (str).
        Returns an empty dict on failure.
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
        #get language
        langs = metadata.get("programming_languages", [])
        primary_language = ""
        if langs:
            # pick the entry with the largest "size" under result
            primary = max(
                langs,
                key=lambda x: x.get("result", {}).get("size", 0)
            )
            primary_language = primary.get("result", {}).get("value", "")
        #gets only first description, could be multiple
        return {
            "name": get_first_value("name"),
            "description": get_first_value("description"),
            "keywords": keywords,
            "authors": [get_github_user_data(owner)] if owner else [],
            "language": primary_language

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
    """Dispatch metadata extraction based on the URL’s domain.

    Routes to the appropriate extractor:
      - GitHub repos      → extract_somef_metadata
      - CRAN packages     → extract_cran_metadata
      - PyPI packages     → extract_pypi_metadata
      - Other websites    → extract_website_metadata

    Args:
        url: The URL from which to extract metadata.

    Returns:
        A metadata dict as returned by one of the specialized extractors,
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
    
if __name__ == "__main__":
    # Example usage
    url = "https://github.com/scikit-learn/scikit-learn"
    metadata = get_metadata(url)
    print(metadata)
