import subprocess
import json
import sys
import tempfile
import os
import pandas as pd
import requests
import re
from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Tuple
import re
from bs4 import BeautifulSoup
from rake_nltk import Rake
import shutil


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

def extract_pypi_metadata_Rake_after(url: str) -> Dict[str, Any]:
    """
        Extract metadata for a PyPI package with layered keyword fallback.

        This function retrieves package metadata and attempts to extract keywords
        in a two-step fallback strategy. It first checks for JSON-defined keywords.
        If none are found, it tries to derive keywords from Trove classifiers.
        If classifiers also fail to provide valid keywords, it applies RAKE
        to extract keyword phrases from the summary text.

        Args:
            url (str): A PyPI project URL (e.g. "https://pypi.org/project/example").

        Returns:
            dict: A dictionary containing:
                - name (str): Package name
                - description (str): Summary description
                - keywords (List[str]): Extracted, derived, or RAKE-generated keywords
                - authors (List[str]): Combined author and maintainer names
                - language (str): Always "Python"
            If extraction fails, returns {"error": "..."}.
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
    if not keywords:
        r = Rake(min_length=2, max_length=3)
        r.extract_keywords_from_text(summary)
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
        keywords = kws
        

    return {
        "name"        : info.get("name", pkg),
        "description" : summary,
        "keywords"    : keywords,
        "authors"     : authors,
        "language"  : "Python"
    }

def extract_pypi_metadata_RAKE(url: str) -> Dict[str, Any]:
    """
    Extract metadata for a PyPI package using RAKE for keyword extraction.

    This function ignores Trove classifiers and uses RAKE to generate keywords
    directly from the summary text if the JSON 'keywords' field is empty.
    This is useful when classifier-derived tags are insufficient or undesired.

    Args:
        url (str): A PyPI project URL (e.g. "https://pypi.org/project/example").

    Returns:
        dict: A dictionary containing:
            - name (str): Package name
            - description (str): Summary description
            - keywords (List[str]): JSON-defined or RAKE-generated keyword phrases
            - authors (List[str]): Combined author and maintainer names
            - language (str): Always "Python"
        If extraction fails, returns {"error": "..."}.
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
    if not keywords:
        r = Rake(min_length=2, max_length=3)
        r.extract_keywords_from_text(summary)
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
        keywords = kws
    
    
    return {
        "name"        : info.get("name", pkg),
        "description" : summary,
        "keywords"    : keywords,
        "authors"     : authors,
        "language"  : "Python"
    }

def extract_pypi_metadata_RAKE_class(url: str) -> Dict[str, Any]:
    """
Extract metadata for a PyPI package with fallback to RAKE and classifiers.

This function attempts to generate keywords by first checking the JSON 'keywords'
field. If empty, it applies RAKE to extract keywords from the summary. If RAKE
also produces no valid keywords, it falls back to using topic-based Trove classifiers.

Args:
    url (str): A PyPI project URL (e.g. "https://pypi.org/project/example").

Returns:
    dict: A dictionary containing:
        - name (str): Package name
        - description (str): Summary description
        - keywords (List[str]): From JSON, RAKE, or Trove classifiers (in that order)
        - authors (List[str]): Combined author and maintainer names
        - language (str): Always "Python"
    If extraction fails, returns {"error": "..."}.
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
    if not keywords:
        r = Rake(min_length=2, max_length=3)
        r.extract_keywords_from_text(summary)
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
        keywords = kws
        if not kws:
            classifiers = info.get("classifiers", [])

    # 7) derive fallback keywords from classifiers if JSON was empty
            if classifiers:
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
    return data
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



def extract_somef_metadata(repo_url: str, somef_path: str = "D:\\MASTER\\TMF\\somef") -> dict:
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
    

    # Prepare the two command variants: with and without -kt
    # Note: keep -t and -m in both
    temp_dir = os.path.join(somef_path, "temp")
    output_path = os.path.join(temp_dir, "metadata.json")
    os.makedirs(temp_dir, exist_ok=True)
    kt_path = temp_dir
    if sys.platform == "win32":
        # Windows long‐path prefix
        kt_path = "\\\\?\\" + temp_dir

    base_cmd = [
        "poetry", "run", "somef", "describe",
        "-r", repo_url,
        "-o", output_path,
        "-t", "0.93",
        "-m"
    ]
    cmds = [
        base_cmd + ["-kt", kt_path],  # first attempt
        base_cmd                     # retry without -kt
    ]

    try:
        # Try each command in turn
        for cmd in cmds:
            try:
                subprocess.run(cmd, cwd=somef_path, check=True)
                break
            except subprocess.CalledProcessError as e:
                print(f"Command failed ({' '.join(cmd)}): {e}")
        else:
            # both attempts failed
            print(f"All SOMEF attempts failed for {repo_url}")
            return {}

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

        # Determine primary language by largest code size
        langs = metadata.get("programming_languages", [])
        primary_language = ""
        if langs:
            primary = max(langs, key=lambda x: x.get("result", {}).get("size", 0))
            primary_language = primary.get("result", {}).get("value", "")

        return {
            "name": get_first_value("name"),
            "description": get_first_value("description"),
            "keywords": keywords,
            "authors": [get_github_user_data(owner)] if owner else [],
            "language": primary_language
        }

    finally:
        # Cleanup: delete temp JSON and clear out the temp directory
        try:
            os.remove(output_path)
        except OSError:
            pass

        if os.path.isdir(temp_dir):
            for entry in os.scandir(temp_dir):
                path = entry.path
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def extract_somef_metadata_with_RAKE(repo_url: str, somef_path: str = r"D:/MASTER/TMF/somef") -> dict:
    """
    Extract metadata from a GitHub repository using SOMEF with RAKE fallback for keywords.

    This function runs the SOMEF tool on the given repository and parses
    metadata from the resulting JSON. If the extracted keywords field is empty,
    it applies RAKE to extract up to 5 multi-word keywords from the description text.

    Args:
        repo_url (str): URL of the GitHub repository.
        somef_path (str): Path to the SOMEF project directory where `poetry run somef` is available.

    Returns:
        dict: A dictionary containing:
            - name (str)
            - description (str)
            - keywords (List[str]) — extracted from SOMEF or generated via RAKE
            - authors (List[str]) — GitHub owner's name
            - language (str) — most dominant programming language in the repo
        Returns an empty dictionary on failure.
"""

    # Create a temp file to store the output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        output_path = tmp_file.name

    try:
        # Run SOMEF with poetry from its own directory
        path = "D:\\MASTER\\TMF\\somef\\temp"
        os.makedirs(path, exist_ok=True)
        if sys.platform == "win32":
        # note: in a Python string literal this is "\\\\?\\"
            path = "\\\\?\\" + path
        subprocess.run([
            "poetry","run", "somef", "describe",
            "-r", repo_url,
            "-o", output_path,
            "-t", "0.93",
            "-m",
            "-kt", path
        ], cwd=somef_path, check=True)

        # Load the JSON output into Python
        with open(output_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        def get_first_value(key):
            return metadata.get(key, [{}])[0].get("result", {}).get("value", "")

        # Split keywords string into list
        raw_keywords = get_first_value("keywords")
        text = get_first_value("description")
        keywords = [kw.strip() for kw in raw_keywords.split(",")] if raw_keywords else []
        if len(keywords)==0:
            kws = []
            if not pd.isna(text) and text:
                r = Rake(min_length=2, max_length=3)
                r.extract_keywords_from_text(text)
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
                keywords = kws
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
        #path = "D:\\MASTER\\TMF\\somef\\temp"
        for entry in os.scandir(path):
            entry_path = entry.path
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry_path) 
            else:
                os.remove(entry_path)


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
        return extract_somef_metadata_with_RAKE(url)

    # CRAN package (common formats: cran.r-project.org or pkg.go.dev/r)
    if domain == "cran.r-project.org" and (
        path.startswith("/web/packages/") or
        path.startswith("/package=")
    ):        return extract_cran_metadata(url)
    ## PyPI package (common formats: pypi.org or pypi.python.org)
    if "pypi.org" in domain or "pypi.python.org" in domain:
        return extract_pypi_metadata_Rake_after(url)
    # Generic website fallback
    return extract_website_metadata(url)
def _clean_github_url(raw_url: str) -> str:
    """
    Given a GitHub URL that may include extra path segments (e.g. "/tree/master", "/issues", "/tarball/v1.0"),
    normalize it to "https://github.com/{owner}/{repo}". If the URL isn’t a valid GitHub repo URL,
    returns an empty string.
    """
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if "github.com" not in host:
        return ""
    # Remove any ".git" suffix from path temporarily
    path = parsed.path.rstrip("/").lstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]

    segments = [seg for seg in path.split("/") if seg]
    if len(segments) < 2:
        return ""
    owner, repo = segments[0], segments[1]
    # Reconstruct the clean repository URL
    return f"https://github.com/{owner}/{repo}"

def get_github_link_from_pypi(url: str) -> Tuple[str,int]:
    """
    Given a PyPI project URL (e.g. "https://pypi.org/project/example"), fetches the package's JSON
    metadata and returns the first GitHub repository URL found (in project_urls or home_page).
    If no GitHub link is present, returns an empty string.

    Args:
        url (str): A PyPI project URL.

    Returns:
        str: The GitHub URL if found, otherwise "".
    """
    # 1) Extract package name
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in ("project", "simple"):
        pkg = parts[1]
    else:
        pkg = parts[0] if parts else None

    if not pkg:
        return ""

    # 2) Fetch JSON metadata
    api_url = f"https://pypi.org/pypi/{pkg}/json"
    resp = requests.get(api_url)
    if resp.status_code != 200:
        return ""
    info: Dict[str, Any] = resp.json().get("info", {})
    full_description = (info.get("description") or "").strip()
    description_length = len(full_description)  
    project_urls = info.get("project_urls") or {}
    for link in project_urls.values():
        if link and "github.com" in link.lower():
            clean = _clean_github_url(link.strip())
            if clean:
                return clean, description_length

    # 4) Fallback: check home_page
    home_page = info.get("home_page", "") or ""
    if "github.com" in home_page.lower():
        clean = _clean_github_url(home_page.strip())
        if clean:
            return clean, description_length

    # 6) No GitHub link found
    return "", description_length

def extract_somef_metadata_with_RAKE_readme(repo_url: str, somef_path: str = r"D:/MASTER/TMF/somef") -> dict:
    """
    Extract metadata from a GitHub repository using SOMEF with RAKE fallback for keywords,
    and determine whether the README file is empty.

    This function runs the SOMEF tool on the given repository and parses
    metadata from the resulting JSON. If the extracted keywords field is empty,
    it applies RAKE to extract up to 5 multi-word keywords from the description text.
    It also fetches the README (if a readme_url exists) and checks if its content is empty.

    Args:
        repo_url (str): URL of the GitHub repository.
        somef_path (str): Path to the SOMEF project directory where `poetry run somef` is available.

    Returns:
        dict: A dictionary containing:
            - name (str)
            - description (str)
            - keywords (List[str]) — extracted from SOMEF or generated via RAKE
            - authors (List[str]) — GitHub owner's name
            - language (str) — most dominant programming language in the repo
            - readme_empty (bool) — True if README is missing or empty, False otherwise
        Returns an empty dictionary on failure.
    """

    # Create a temp file to store SOMEF output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        output_path = tmp_file.name

    try:
        # Ensure SOMEF temp directory exists (for -kt)
        path = os.path.join("D:", os.sep, "MASTER", "TMF", "somef", "temp")
        os.makedirs(path, exist_ok=True)
        if sys.platform == "win32":
            # Prefix with \\?\ to allow long Windows paths
            path = "\\\\?\\" + path

        # Run SOMEF with poetry
        subprocess.run([
            "poetry", "run", "somef", "describe",
            "-r", repo_url,
            "-o", output_path,
            "-t", "0.93",
            "-m"
            #"-kt", path
        ], cwd=somef_path, check=True)

        # Load the JSON output into Python
        with open(output_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        def get_first_value(key: str) -> str:
            entries = metadata.get(key, [{}])
            if not entries or not isinstance(entries, list):
                return ""
            return entries[0].get("result", {}).get("value", "") or ""

        # 1) Extract name, description, owner, and programming languages
        name = get_first_value("name")
        text_description = get_first_value("description")
        owner = get_first_value("owner")

        # 2) Extract or derive keywords
        raw_keywords = get_first_value("keywords")
        keywords = [kw.strip() for kw in raw_keywords.split(",")] if raw_keywords else []
        if not keywords:
            # If SOMEF gave no keywords, fall back to RAKE on the description text
            if text_description and not pd.isna(text_description):
                r = Rake(min_length=2, max_length=3)
                r.extract_keywords_from_text(text_description)
                top_phrases = r.get_ranked_phrases()[:5]

                cleaned = []
                for phrase in top_phrases:
                    tag = phrase.strip(' "\'.,').lower()
                    if len(tag.split()) > 1 and re.match(r'^[\w\s]+$', tag):
                        cleaned.append(tag)
                seen = set()
                keywords = [t for t in cleaned if not (t in seen or seen.add(t))]

        # 3) Primary programming language
        langs = metadata.get("programming_languages", [])
        primary_language = ""
        if langs:
            primary = max(langs, key=lambda x: x.get("result", {}).get("size", 0))
            primary_language = primary.get("result", {}).get("value", "")

        # 4) Check if README is empty
        readme_empty = True
        readme_urls = metadata.get("readme_url", [])

        if isinstance(readme_urls, list):
            for entry in readme_urls:
                # Extract the URL string, whether entry is a dict or a plain string
                url = ""
                if isinstance(entry, dict):
                    url = entry.get("result", {}).get("value", "") or ""
                else:
                    url = entry or ""

                if not url:
                    continue

                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        content = resp.text.strip()
                        if content:
                            # Found a non-empty README, no need to check further
                            readme_empty = False
                            break
                    # If status_code != 200 or content is empty, try the next URL
                except Exception:
                    # On request failure, just move on to the next URL
                    continue


        # 5) Prepare return dictionary
        return {
            "name": name,
            "description": text_description,
            "keywords": keywords,
            "authors": [owner] if owner else [],
            "language": primary_language,
            "readme_empty": readme_empty
        }

    except Exception:
        return {}
    finally:
        # Clean up the temporary SOMEF output file
        try:
            os.remove(output_path)
        #path = "D:\\MASTER\\TMF\\somef\\temp"
            for entry in os.scandir(path):
                entry_path = entry.path
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry_path) 
                else:
                    os.remove(entry_path)
        except OSError:
            pass

if __name__ == "__main__":
    # Example usage
    
    corpus = pd.read_csv("binary_llm_results_qwen.csv")      # assumes there's a column named "name" and a column named "url"
    czi_test = pd.read_csv("binary_llm_results_qwen_new.csv")           # assumes there's a column named "name" in here as well

    # 2) Merge them on "name" (left‐join so that every row in CZI_test is preserved)
    #    We'll pull in corpus["url"] and call it "ground truth"
    corpus.dropna(subset=["predicted_label"], inplace=True)  # drop rows with NaN in "predicted_label"
    df_stacked = pd.concat([corpus, czi_test], axis=0, ignore_index=True)
    df_stacked.to_csv("binary_llm_results_qwen_stacked.csv", index=False)
