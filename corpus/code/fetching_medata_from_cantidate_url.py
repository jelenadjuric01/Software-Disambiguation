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
from urllib.parse import urlparse
from typing import Dict, Any, List
from bs4 import BeautifulSoup


def extract_pypi_metadata(url: str) -> Dict[str, Any]:
    """
    Given a PyPI project URL, returns:
      - name        : str
      - description : str
      - keywords    : List[str]   # JSON keywords or derived from classifiers
      - classifiers : List[str]   # raw Trove classifiers
      - authors     : List[str]
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
    """Extract a list of 'Given Family' from an R Authors@R string."""
    folks = re.findall(r'person\((.*?)\)', authors_r, flags=re.DOTALL)
    out = []
    for block in folks:
        given  = re.search(r'given\s*=\s*"([^"]+)"', block)
        family = re.search(r'family\s*=\s*"([^"]+)"', block)
        if given and family:
            out.append(f"{given.group(1).strip()} {family.group(1).strip()}")
    return out

def extract_cran_metadata(url: str) -> Dict[str, Any]:
    """
    Given a CRAN package page URL, extract:
      - name        : str
      - description : str
      - keywords    : List[str]   # DESCRIPTION Keywords or Task Views
      - authors     : List[str]   # from Authors@R, DESCRIPTION Author, or HTML
    """
    # 1) pull out the pkg name
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

    # 2) try CRANDB JSON
    api_url = f"https://crandb.r-pkg.org/{pkg}"
    resp    = requests.get(api_url)
    resp.raise_for_status()
    data    = resp.json()

    name        = data.get("Package")
    description = data.get("Description", "")

    # 3) AUTHORS: JSON Authors@R → DESCRIPTION Author → HTML fallback
    authors: List[str] = []
    if data.get("Authors@R"):
        authors = parse_authors_r(data["Authors@R"])
    elif data.get("Author"):
        # grab everything before each [role,…]
        raw = data["Author"]
        authors = [a.strip() for a in re.findall(r'([^,\[]+?)(?=\s*\[)', raw)]
    if not authors:
        # HTML scrape of the <dt>Author:</dt> line
        html = requests.get(f"https://cran.r-project.org/web/packages/{pkg}/index.html").text
        soup = BeautifulSoup(html, "html.parser")
        dt = soup.find("dt", string=re.compile(r"Author:", re.IGNORECASE))
        if dt:
            txt = dt.find_next_sibling("dd").get_text()
            authors = [a.strip() for a in re.findall(r'([^,\[]+?)(?=\s*\[)', txt)]

    # 4) KEYWORDS: DESCRIPTION Keywords → Task Views HTML fallback
    raw_kw = data.get("Keywords") or ""
    kws    = [w.strip() for w in re.split(r"[,\s]+", raw_kw) if w.strip()]

    if not kws:
        # look for the “In views:” line on the CRAN page
        html = requests.get(f"https://cran.r-project.org/web/packages/{pkg}/index.html").text
        soup = BeautifulSoup(html, "html.parser")
        dt   = soup.find("dt", string=re.compile(r"In\s*views:", re.IGNORECASE))
        if dt:
            txt = dt.find_next_sibling("dd").get_text()
            kws = [v.strip() for v in txt.split(",") if v.strip()]

    return {
        "name"       : name,
        "description": description,
        "keywords"   : kws,
        "authors"    : authors,
    }


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
    ## PyPI package (common formats: pypi.org or pypi.python.org)
    if "pypi.org" in domain or "pypi.python.org" in domain:
        return extract_pypi_metadata(url)
    # Generic website fallback
    return extract_website_metadata(url)
    
if __name__ == "__main__":
    url  = "https://cran.r-project.org/package=dplyr"
    meta = extract_cran_metadata(url)
    print(meta)
    # → {
    #      'name': 'dplyr',
    #      'description': 'A grammar of data manipulation, providing a consistent set of verbs ...',
    #      'keywords': ['Data Import, Tidy Data, Data Transformation, Data Aggregation'], 
    #      'authors': ['Hadley Wickham', ...]
    #    }