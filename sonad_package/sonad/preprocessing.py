from pathlib import Path
from SPARQLWrapper import JSON, SPARQLWrapper
import pandas as pd
import json
from typing import Any, List, Set, Tuple, Dict, Optional
import os
import requests
import re
import csv
import numpy as np
import difflib
import time
import xmlrpc.client
from functools import lru_cache
import subprocess
import sys
import tempfile
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import shutil

from sentence_transformers import SentenceTransformer, util

import textdistance

# 1. module import
#    - A lightweight BERT for paragtraph similarity
_BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')



def paragraph_description_similarity_BERT(text1: str, text2: str) -> float:
    """
Compute semantic similarity between two texts using SBERT embeddings.

This function encodes both input texts using a lightweight SBERT model
(`all-MiniLM-L6-v2`) and calculates the cosine similarity between them.

Args:
    text1 (str): First input string (e.g., a paragraph from a paper).
    text2 (str): Second input string (e.g., software description).

Returns:
    float: Cosine similarity score in [0.0, 1.0], or `np.nan` if either input is invalid.
"""

    # validate inputs
    if not text1 or pd.isna(text1) or not (text1 := text1.strip()):
        return np.nan
    if not text2 or pd.isna(text2) or not (text2 := text2.strip()):
        return np.nan

    # encode both texts to BERT embeddings (as PyTorch tensors)
    embeddings = _BERT_MODEL.encode([text1, text2], convert_to_tensor=True)
    # compute cosine similarity
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # .item() to get Python float, ensure non-negative [0,1] range
    return float(sim.item())



# Define common software affixes to strip out
COMMON_AFFIXES = {
    "pro", "enterprise", "suite", "edition", "cloud", "desktop",
    "online", "server", "client", "portable", "lite"
}

def normalize_software_name(name: str) -> str:
    """
Normalize a software name string by removing noise and standardizing format.

Steps:
  1. Lowercase and strip.
  2. Remove version numbers (e.g., "v2.0", "2022").
  3. Remove punctuation.
  4. Remove known affixes (e.g., "Pro", "Suite").
  5. Collapse whitespace.

Args:
    name (str): Raw software name string.

Returns:
    str: Cleaned and normalized software name.
"""

    s = name.lower().strip()
    # 1. Remove version-like tokens (v1, 2.0, 2022)
    s = re.sub(r"\b(v?\d+(\.\d+)*|\d{4})\b", " ", s)
    # 2. Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # 3. Tokenize and remove common affixes
    tokens = [tok for tok in s.split() if tok not in COMMON_AFFIXES]
    # 4. Re-join and collapse whitespace
    return " ".join(tokens)

def software_name_similarity(name1: str, name2: str) -> float:
    """Measure Jaro–Winkler similarity between two software names.

    Both names are first normalized via `normalize_software_name`.

    Args:
        name1 (str): First software name.
        name2 (str): Second software name.

    Returns:
        float: Jaro–Winkler similarity ∈ [0.0, 1.0].
    """
    n1 = normalize_software_name(name1)
    n2 = normalize_software_name(name2)
    return textdistance.jaro_winkler(n1, n2)


def synonym_name_similarity(name1: str, names: str) -> float:
    """
Compute Jaro-Wrinkler similarity between a software name and a list of synonyms.

The input synonyms string is comma-separated. Each synonym and the main name
are normalized before comparison. The final similarity score is the average
over all pairwise comparisons.

Args:
    name1 (str): Software name to compare.
    names (str): Comma-separated synonyms.

Returns:
    float: Average similarity ∈ [0.0, 1.0], or np.nan on invalid input.
"""

    if not names or pd.isna(names) or not name1 or pd.isna(name1):
        return np.nan
    # After stripping, if either is empty
    names = names.split(",")

    # Normalize the first name
    n1 = normalize_software_name(name1)
    # Normalize the list of synonyms
    n2 = [normalize_software_name(name) for name in names]
    # Compute Jaro-Winkler similarity for each synonym
    similarities = [textdistance.jaro_winkler(n1, name) for name in n2]
    # Return the average similarity
    return np.mean(similarities)
def programming_language_similarity(lang1: Optional[str],
                                    lang2: Optional[str]) -> float:
    """
Compare two programming language names using Jaro–Winkler similarity.

Returns:
- 1.0 if both normalized names are equal (case-insensitive)
- 0.0 if both are present but differ
- np.nan if either value is missing or empty

Args:
    lang1 (Optional[str]): First language name.
    lang2 (Optional[str]): Second language name.

Returns:
    float: Similarity score ∈ [0.0, 1.0], or np.nan if data is missing.
"""

    # 1) Missing → nan
    if pd.isna(lang1) or pd.isna(lang2):
        return np.nan

    # 2) Normalize to lowercase strings
    l1 = str(lang1).strip().lower()
    l2 = str(lang2).strip().lower()

    # 3) Empty after stripping → nan
    if not l1 or not l2:
        return np.nan

    return textdistance.jaro_winkler(l1, l2)


def normalize_author_name(name: str) -> str:
    """Normalize an author name to lowercase, first-last order, letters only.

    Steps:
      1. Flip "Last, First" → "First Last".
      2. Expand initials (e.g., "J.K." → "J K").
      3. Lowercase.
      4. Remove non-letter characters.
      5. Collapse spaces.

    Args:
        name (str): Raw author name.

    Returns:
        str: Normalized author name.
    """
    s = name.strip()
    # 1. Flip "Last, First" to "First Last"
    if ',' in s:
        last, first = [p.strip() for p in s.split(',', 1)]
        s = f"{first} {last}"
    # 2. Expand initials with dots into space‑separated letters
    #    e.g. "J.K." → "J K"
    s = re.sub(r'\b([A-Za-z])\.\s*', r'\1 ', s)
    # 3. Lowercase
    s = s.lower()
    # 4. Remove anything except letters and spaces
    s = re.sub(r'[^a-z\s]', ' ', s)
    # 5. Collapse whitespace into a single space
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def author_name_similarity(name1: str, name2: str) -> float:
    """Compute Jaro–Winkler similarity between two normalized author names.

    If either input is missing or blank, returns `np.nan` immediately.

    Args:
        name1 (str): First author name.
        name2 (str): Second author name.

    Returns:
        float: Similarity ∈ [0.0, 1.0], or `np.nan` if either name is missing.
    """
    # Treat None as missing
    if not name1 or not name2 or pd.isna(name1) or pd.isna(name2):
        return np.nan
    # After stripping, if either is empty
    if not name1.strip() or not name2.strip():
        return np.nan
    n1 = normalize_author_name(name1)
    n2 = normalize_author_name(name2)
    return textdistance.jaro_winkler(n1, n2)


def compute_similarity_test(df: pd.DataFrame,output_path:str = None) -> pd.DataFrame:
    """
Compute multiple similarity metrics between software mentions and candidate metadata.

Metrics include:
- `name_metric`: Jaro-Winkler similarity on normalized names.
- `author_metric`: Similarity between author names.
- `paragraph_metric`: SBERT-based semantic similarity.
- `language_metric`: Similarity between programming language mentions.
- `synonym_metric`: Similarity to known synonyms.

Args:
    df (pd.DataFrame): Input DataFrame with paper/software/metadata columns.
    output_path (str, optional): If provided, saves resulting DataFrame as CSV.

Returns:
    pd.DataFrame: DataFrame with similarity scores.
"""
    print("🔍 Computing similarity metrics…")
    for col in ['name_metric','author_metric','paragraph_metric',"language_metric",'synonym_metric']:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Mask of all rows that have valid metadata_name
    valid = (
        df['metadata_name'].notna() &
        df['metadata_name'].astype(str).str.strip().astype(bool)
    )
    print("Computing name metric...")
    # 2) name_metric: only for valid rows where name_metric is still NaN
    nm = valid & df['name_metric'].isna()
    df.loc[nm, 'name_metric'] = df.loc[nm].apply(
        lambda r: software_name_similarity(r['name'], r['metadata_name']),
        axis=1
    )
    print("Computing author metric...")
    # 3) author_metric:
    am = valid & df['author_metric'].isna()
    df.loc[am, 'author_metric'] = df.loc[am].apply(
        lambda r: author_name_similarity(r['authors'], r['metadata_authors']),
        axis=1
    )
    print("Computing paragraph metric...")
    # 4) paragraph_metric: only for rows with metadata_description & NaN
    pm = valid & df['paragraph_metric'].isna()
    df.loc[pm, 'paragraph_metric'] = df.loc[pm].apply(
        lambda r: paragraph_description_similarity_BERT(
            r['paragraph'], r['metadata_description']
        ),
        axis=1
    )
    print("Computing language metric...")
    lm = valid & df['language_metric'].isna()
    df.loc[lm, 'language_metric'] = df.loc[lm].apply(
        lambda r: programming_language_similarity(
            r['language'],
            r['metadata_language']
        ),
        axis=1
    )
    print("Computing synonym metric...")
    sm = valid & df['synonym_metric'].isna()
    df.loc[sm, 'synonym_metric'] = df.loc[sm].apply(
        lambda r: synonym_name_similarity(
            r['metadata_name'],
            r['synonyms']
        ),
        axis=1
    )
    # 6) Build the “sub” DataFrame you originally returned
    cols = [
        'name','doi','paragraph','authors','language','candidate_urls','synonyms',
        'metadata_name','metadata_authors','metadata_description','metadata_language',
        'name_metric','author_metric','paragraph_metric','language_metric','synonym_metric'
    ]
    
    sub = df.loc[valid, cols].copy()

    # 7) Optionally save
    if output_path:
        sub.to_csv(output_path, index=False)
        print(f"📄 Similarity metrics saved to {output_path}")

    return sub




def extract_pypi_metadata(url: str) -> Dict[str, Any]:
    """
        Extract metadata for a PyPI package given its project URL.
        Args:
            url (str): A PyPI project URL (e.g. "https://pypi.org/project/example").

        Returns:
            dict: A dictionary containing:
                - name (str): Package name
                - description (str): Summary description
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
    summary = (info.get("summary") or "").strip()

    
    return {
        "name"        : info.get("name", pkg),
        "description" : summary,
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
    

    return {
        "name"       : name,
        "description": description,
        "authors"    : authors,
        "language"   : "R"
    }



def extract_github_url_from_cran_package(url: str) -> List[str]:
    """
    Given a CRAN package URL (any format), extract all GitHub URLs from CRANDB metadata.

    Args:
        cran_url (str): Any CRAN package URL.

    Returns:
        List[str]: Unique GitHub repository URLs (e.g., https://github.com/user/repo).
    """
    # Step 1: Parse package name
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


    # Step 3: Gather possible GitHub URLs
    url_fields = []
    if "URL" in data:
        url_fields.extend(re.split(r",|\\s+", data["URL"]))
    if "BugReports" in data:
        url_fields.append(data["BugReports"].strip())

    github_urls = set()
    for u in url_fields:
        if "github.com" in u.lower():
            clean = _clean_github_url(u.strip())
            if clean:
                github_urls.add(clean)

    return sorted(github_urls)


def get_github_user_data(username: str, github_token: str) -> str:
    """Retrieve a GitHub user’s display name via the GitHub API.

    Sends an authenticated request if GITHUB_TOKEN is set; otherwise unauthenticated.
    Returns the “name” field from the API, falling back to the login on error or if blank.

    Args:
        username: GitHub login (e.g. "octocat").

    Returns:
        The user’s full name (str), or the original username if not found or on error.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}

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



def extract_somef_metadata(repo_url: str, github_token: str = None) -> dict:
    """
    Run the SOMEF tool on a GitHub repository to extract metadata.
    Calls SOMEF directly as a package (no poetry, no external somef directory).

    Args:
        repo_url: URL of the GitHub repository.

    Returns:
        A dict with:
          name (str), description (str), authors (List[str]), language (str), readme_empty (bool).
        Returns empty dict on failure.
    """
    
    # Create a temp file to store SOMEF output

    package_dir = Path(__file__).parent
    temp_path = package_dir / "someftemp"
    temp_path.mkdir(exist_ok=True, parents=True)

    # prepare a temp file for SOMEF output
    with tempfile.NamedTemporaryFile(dir=str(temp_path), delete=False, suffix=".json") as tmp_file:
        output_path = tmp_file.name

    try:
        if sys.platform == "win32":
            # allow long Windows paths
            temp_path = "\\\\?\\" + str(temp_path)

        base_cmd = [
            "somef", "describe",
            "-r", repo_url,
            "-o", output_path,
            "-t", "0.93",
            "-m"
        ]

        cmds = [
            base_cmd,                      # first attempt
            base_cmd + ["-kt", temp_path]  # retry with -kt
        ]

        # Try each command until we see a non-empty JSON
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"Command raised exception ({' '.join(cmd)}): {e}")

            # Check if output JSON exists and is non-empty
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and data:
                        # Got valid metadata → stop trying further cmds
                        break
                except (ValueError, json.JSONDecodeError):
                    # invalid or empty JSON → try next cmd
                    pass
        else:
            # all attempts failed to produce metadata
            print(f"All SOMEF attempts failed for {repo_url}")
            return {}

        # Load the final JSON
        with open(output_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        def get_first_value(key):
            return metadata.get(key, [{}])[0].get("result", {}).get("value", "")

        owner = get_first_value("owner")

        # primary language = largest code size
        langs = metadata.get("programming_languages", [])
        primary_language = ""
        if langs:
            primary = max(langs, key=lambda x: x.get("result", {}).get("size", 0))
            primary_language = primary.get("result", {}).get("value", "")

        # check for non-empty README
        readme_empty = True
        for entry in metadata.get("readme_url", []):
            url = entry.get("result", {}).get("value", "") if isinstance(entry, dict) else entry
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200 and resp.text.strip():
                    readme_empty = False
                    break
            except Exception:
                continue

        return {
            "name": get_first_value("name"),
            "description": get_first_value("description"),
            "authors": [get_github_user_data(owner, github_token)] if owner else [],
            "language": primary_language,
            "readme_empty": readme_empty
        }

    finally:
        # Cleanup: delete temp JSON and clear out the temp directory
        try:
            os.remove(output_path)
        except OSError:
            pass

        # Clean up someftemp directory
        temp_path = package_dir / "someftemp"
        if temp_path.exists():
            for entry in temp_path.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()
                except OSError:
                    continue


#Function that retrieves the metadata from any link
def get_metadata(url: str, github_token: str = None) -> dict:
    """Dispatch metadata extraction based on the URL’s domain.

    Routes to the appropriate extractor:
      - GitHub repos      → extract_somef_metadata
      - CRAN packages     → extract_cran_metadata
      - PyPI packages     → extract_pypi_metadata

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
        metadata = extract_somef_metadata(url, github_token)
        if metadata and metadata.get("readme_empty") and 'cran' not in metadata.get("authors", []):
            return {"readme_empty": True}
        return metadata

    # CRAN package (common formats: cran.r-project.org or pkg.go.dev/r)
    if domain == "cran.r-project.org" and (
        path.startswith("/web/packages/") or
        path.startswith("/package=")
    ):        return extract_cran_metadata(url)
    ## PyPI package (common formats: pypi.org or pypi.python.org)
    if "pypi.org" in domain or "pypi.python.org" in domain:
        return extract_pypi_metadata(url)
    
    



GITHUB_API_URL = "https://api.github.com/search/repositories"

COMMON_GITHUB_HEADERS = {
    "Accept":     "application/vnd.github.v3+json",
    "User-Agent": "my-software-disambiguator"
}

def fetch_github_urls(
    name: str,
    per_page: int = 5,
    max_retries: int = 3,
    github_token: str = None
) -> List[str]:
    """
Query GitHub's Search API for repositories matching a given name.

Handles rate limiting using retry logic.

Args:
    name (str): Name of the software or keyword to search.
    per_page (int): Max number of repositories to fetch.
    max_retries (int): Number of retry attempts in case of rate limiting.

Returns:
    List[str]: List of GitHub repository URLs.
"""
    print(f"🔍 Searching GitHub for '{name}'…")
    params = {
        "q":        f"{name} in:name",
        "sort":     "stars",
        "order":    "desc",
        "per_page": per_page
    }

    for attempt in range(1, max_retries + 1):
        headers = COMMON_GITHUB_HEADERS.copy()
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        resp = requests.get(GITHUB_API_URL, params=params, headers=headers, timeout=10)
        # 403 could be a rate-limit on the Search API
        if resp.status_code == 403:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_ts - time.time()-1, 1)
            print(f"[Attempt {attempt}] Rate limited. Sleeping {int(wait)}s until reset…")
            time.sleep(wait)
            continue

        # a 401 means bad token, 404 would be weird, anything else we raise
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["html_url"] for item in items]

    # If we exhausted retries
    raise RuntimeError(f"GitHub search for '{name}' failed after {max_retries} attempts (last status: {resp.status_code})")


PYPI_JSON_URL    = "https://pypi.org/pypi/{pkg}/json"
PYPI_PROJECT_URL = "https://pypi.org/project/{pkg}/"

@lru_cache(maxsize=512)
def _get_pypi_info(pkg: str, timeout: float = 10.0) -> Dict:
    """
    Retrieve the PyPI JSON 'info' block for a given package name.

    Performs a GET to `https://pypi.org/pypi/{pkg}/json`. Cached in memory
    to avoid repeated network calls.

    Parameters:
        pkg (str):
            Name of the PyPI package.
        timeout (float):
            HTTP request timeout in seconds.

    Returns:
        Dict[str, Any]:
            The 'info' sub-dictionary from the PyPI JSON response, or `{}` on error.
    """
    try:
        r = requests.get(PYPI_JSON_URL.format(pkg=pkg), timeout=timeout)
        if r.status_code == 200:
            return r.json().get("info", {})
    except requests.RequestException:
        pass
    return {}

@lru_cache(maxsize=256)
def fetch_pypi_urls(
    pkg_name: str,
    max_results: int = 5,
    timeout: float = 10.0
) -> List[str]:
    """
    Find PyPI project page URLs for a package name.

    1. Tries exact JSON API lookup (`_get_pypi_info`) to get `package_url` or `project_url`.
    2. If fewer than `max_results`, falls back to an XML-RPC name search and API lookups.

    Parameters:
        pkg_name (str):
            Name of the PyPI package to search for.
        max_results (int):
            Maximum number of URLs to return.
        timeout (float):
            HTTP request timeout in seconds.

    Returns:
        List[str]:
            List of PyPI project page URLs, up to `max_results`.
    """
    urls: List[str] = []
    print(f"🔍 Searching PyPI for '{pkg_name}'…")
    # 1) Exact match
    info = _get_pypi_info(pkg_name, timeout)
    if info:
        url = info.get("package_url") or info.get("project_url")
        if url:
            urls.append(url)

    if len(urls) >= max_results:
        return urls[:max_results]

    # 2) Fuzzy search
    try:
        client = xmlrpc.client.ServerProxy("https://pypi.org/pypi")
        hits = client.search({"name": pkg_name}, "or")
        seen = set(pkg_name.lower())

        for hit in hits:
            name = hit.get("name")
            key  = name.lower() if name else None
            if not key or key in seen:
                continue
            seen.add(key)

            # pull its JSON info to get the true URL
            info = _get_pypi_info(name, timeout)
            if info:
                url = info.get("package_url") or info.get("project_url")
                if url:
                    urls.append(url)
                    if len(urls) >= max_results:
                        break
                    continue

            # fallback (should rarely be needed)
            urls.append(PYPI_PROJECT_URL.format(pkg=name))
            if len(urls) >= max_results:
                break

    except Exception:
        pass

    return urls[:max_results]


CRAN_PACKAGES_URL = "https://cran.r-project.org/src/contrib/PACKAGES"
CRAN_BASE_URL     = "https://cran.r-project.org/web/packages/{pkg}/index.html"
CRAN_SHORT_URL    = "https://cran.r-project.org/package={pkg}"

@lru_cache(maxsize=1)
def _load_cran_packages(timeout: float = 10.0) -> List[str]:
    """
    Fetch and parse the CRAN PACKAGES index into a list of package names.

    Downloads `https://cran.r-project.org/src/contrib/PACKAGES` and extracts
    every "Package: X" line. Cached in memory so it's only downloaded once.

    Parameters:
        timeout (float):
            HTTP request timeout in seconds.

    Returns:
        List[str]:
            All package names available on CRAN.

    Raises:
        requests.exceptions.RequestException:
            If the download fails.
    """
    resp = requests.get(CRAN_PACKAGES_URL, timeout=timeout)
    resp.raise_for_status()
    pkgs = []
    for line in resp.text.splitlines():
        if line.startswith("Package:"):
            pkgs.append(line.split(":", 1)[1].strip())
    return pkgs

@lru_cache(maxsize=256)
def fetch_cran_urls(
    name: str,
    max_results: int = 5,
    timeout: float = 10.0
) -> List[str]:
    """
    Return CRAN package page URLs whose names best match the query.

    1. If an exact name match exists, returns that.
    2. Otherwise includes substring matches.
    3. Finally uses `difflib.get_close_matches` for fuzzy matching.

    Parameters:
        name (str):
            Query string for package names.
        max_results (int):
            Maximum number of URLs to return.
        timeout (float):
            HTTP request timeout for loading the index.

    Returns:
        List[str]:
            CRAN package page URLs (e.g. `https://cran.r-project.org/package=<pkg>`).
    """
    print(f"🔍 Searching CRAN for '{name}'…")
    pkgs = _load_cran_packages(timeout)
    urls: List[str] = []
    name_lower = name.lower()

    # 1) Exact
    if name in pkgs:
        urls.append(CRAN_SHORT_URL.format(pkg=name))

    # 2) Substring
    if len(urls) < max_results:
        subs = [p for p in pkgs if name_lower in p.lower() and p != name]
        for p in subs:
            if len(urls) >= max_results:
                break
            urls.append(CRAN_SHORT_URL.format(pkg=p))

    # 3) Fuzzy
    if len(urls) < max_results:
        # cutoff=0.6 is a sensible default; tweak as needed
        fuzzy = difflib.get_close_matches(name, pkgs, n=max_results, cutoff=0.6)
        for p in fuzzy:
            if len(urls) >= max_results:
                break
            if p not in [u.split("/")[-2] for u in urls]:
                urls.append(CRAN_SHORT_URL.format(pkg=p))

    return urls[:max_results]


def fetch_candidate_urls(name: str, github_token: str = None) -> set[str]:
    """
    Aggregate candidate software URLs from GitHub, PyPI, and CRAN.

    1. Queries GitHub repos via `fetch_github_urls`.
    2. Looks up PyPI project pages via `fetch_pypi_urls`.
    3. Finds CRAN package pages via `fetch_cran_urls`.
    4. Combines all results into a set to remove duplicates.

    Parameters:
        name (str):
            Software or package name to search for.
        github_token (Optional[str]):
            GitHub personal access token for authenticated searches.

    Returns:
        Set[str]:
            Unique candidate URLs discovered for `name`.
    """
    results = []

    # GitHub
    try:
        results += fetch_github_urls(name, github_token=github_token)
        print(f"[+] Found {(results)} GitHub URLs for '{name}'")
    except Exception as e:
        print(f"[!] GitHub fetch failed for '{name}': {e}")

    # PyPI
    try:
        pypi_results= fetch_pypi_urls(name)
        print(f"[+] Found {(pypi_results)} PyPI URLs for '{name}'")
        results += pypi_results
    except Exception as e:
        print(f"[!] PyPI fetch failed for '{name}': {e}")

    # CRAN
    try:
        cran_results = fetch_cran_urls(name)
        print(f"[+] Found {(cran_results)} CRAN URLs for '{name}'")
        results += cran_results
        

    except Exception as e:
        print(f"[!] CRAN check failed for '{name}': {e}")

    # dedupe, preserve order
    return set(results)

def load_candidates(path: str) -> Dict[str, Set[str]]:
    """
    Load a JSON cache of candidate URLs from disk.

    Parameters:
        path (str):
            Path to a JSON file containing `{name: [url1, url2, …]}`.

    Returns:
        Dict[str, Set[str]]:
            Mapping from each `name` to a set of its URLs.

    Raises:
        FileNotFoundError:
            If the file doesn't exist.
        json.JSONDecodeError:
            If the file contains invalid JSON.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: corrupt JSON cache; starting fresh.")
                data = {}
    else:
        data = {}

    # convert lists→sets
    return {name: set(urls) for name, urls in data.items()}

def save_candidates(candidates: Dict[str, Set[str]], path: str):
    """
    Persist candidate URL cache to disk as pretty-printed JSON.

    Converts each set of URLs into a sorted list before writing.

    Parameters:
        candidates (Dict[str, Set[str]]):
            Mapping from names to URL sets.
        path (str):
            Path to write the JSON file.

    Raises:
        IOError:
            If writing to disk fails.
    """
    serializable = {name: sorted(list(urls)) for name, urls in candidates.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

def update_candidate_cache(
    corpus: pd.DataFrame,
    fetcher,                # your fetch_candidate_urls(name) function
    cache_path: str,
    github_token: str = None
) -> Dict[str, Set[str]]:
    """
    Load, augment, and persist a cache of candidate URLs for each software name.

    1. Loads existing JSON cache from `cache_path` (or initializes empty).
    2. For each unique `name` in `corpus['name']`:
       a. Ensures an entry exists in the cache.
       b. Adds any URLs already in `corpus['candidate_urls']`.
       c. Fetches fresh URLs via `fetcher(name, github_token)` and merges them.
    3. Saves the updated cache back to `cache_path`.

    Parameters:
        corpus (pd.DataFrame):
            DataFrame containing at least a `'name'` column and
            optionally a `'candidate_urls'` column of comma-separated URLs.
        fetcher (Callable[[str, Optional[str]], Iterable[str]]):
            Function to fetch new URLs for a given software name.
        cache_path (str):
            Filesystem path to the JSON cache file.
        github_token (Optional[str]):
            GitHub token passed through to the fetcher for authenticated requests.

    Returns:
        Dict[str, Set[str]]:
            Mapping from each software name to its updated set of candidate URLs.

    Raises:
        IOError:
            If reading from or writing to `cache_path` fails.
    """
    # 1) load existing
    candidates = load_candidates(cache_path)

    # 2) iterate unique names
    for name in corpus['name'].unique():
        # initialize if needed
        if name not in candidates:
            candidates[name] = set()

        # 3) add any pre-existing URLs from your dataframe
        if 'candidate_urls' in corpus.columns:
            urls_cell = corpus.loc[corpus['name'] == name, 'candidate_urls'].dropna().astype(str)
            for cell in urls_cell:
                for u in cell.split(','):
                    u = u.strip()
                    if u:
                        candidates[name].add(u)
        
        # 4) fetch & add new ones
        new = set(fetcher(name, github_token=github_token))
        # only do the network hit if there’s something new to add
        if not new.issubset(candidates[name]):
            candidates[name].update(new)

    # 5) persist back to JSON
    save_candidates(candidates, cache_path)
    return candidates

from urllib.parse import urlparse, urlunparse
from typing import Dict, Iterable, List

def normalize_url(u: str) -> str:
    """
    Convert a URL to a normalized, canonical form.

    - Forces the 'https' scheme.
    - Lowercases the network location.
    - Strips any trailing slash from the path.
    - Drops params, query strings, and fragments.

    Parameters:
        u (str):
            The original URL.

    Returns:
        str:
            The normalized URL, suitable for de-duplication.
    """
    p = urlparse(u)
    scheme = "https"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")
    # drop params, query, fragment
    return urlunparse((scheme, netloc, path, "", "", ""))

def dedupe_candidates(candidates: Dict[str, Iterable[str]]) -> None:
    """
    Normalize and deduplicate URLs in-place for each key.

    - Normalizes via `normalize_url`.
    - Prefers `https` over `http` when both exist.

    Parameters:
        candidates (Dict[str, Iterable[str]]):
            Mapping from names to lists/sets of raw URLs.

    Side Effects:
        Modifies `candidates` in place, replacing each value with a list
        of deduplicated, normalized URLs.
    """
    for key, urls in candidates.items():
        seen: Dict[str, str] = {}
        for u in urls:
            norm = normalize_url(u)
            if norm not in seen:
                # first time we see this normalized URL,
                # store the original
                seen[norm] = u
            else:
                # if we already have an http version, but now see an https one, upgrade it
                if u.startswith("https") and not seen[norm].startswith("https"):
                    seen[norm] = u
        # replace with de-duplicated list
        candidates[key] = list(seen.values())

from concurrent.futures import ThreadPoolExecutor, as_completed

# 1) Blacklist of extensions to skip
DISALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.rar', '.tar', '.gz', '.7z',
    '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.json', '.xml', '.csv', '.txt'
}

# 2) Create one Session for connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('http://', adapter)
session.mount('https://', adapter)

def is_website_url(url, timeout=5):
    """
    Return True if the URL likely points to an HTML page.

    - Rejects common non-HTML file extensions.
    - Issues an HTTP HEAD and checks for 'text/html' in Content-Type.

    Parameters:
        url (str):
            URL to test.
        timeout (float):
            HTTP request timeout in seconds.

    Returns:
        bool:
            True if the URL returns an HTML page, False otherwise.
    """
    path = urlparse(url).path.lower()
    if os.path.splitext(path)[1] in DISALLOWED_EXTENSIONS:
        return False
    try:
        resp = session.head(url, allow_redirects=True, timeout=timeout)
        return 'text/html' in resp.headers.get('Content-Type', '')
    except requests.RequestException:
        return False

def filter_url_dict_parallel(url_dict, keys, max_workers=20):
    """
    In-place filter of `url_dict` to keep only URLs that return HTML pages.

    1. Gathers all URLs under the given `keys`.
    2. Concurrently issues HEAD requests (via `is_website_url`) to check for 'text/html'.
    3. Replaces each `url_dict[key]` with only those URLs that passed.

    Parameters:
        url_dict (Dict[str, List[str]]):
            Mapping from software names to lists of URL strings.
        keys (Iterable[str]):
            Subset of keys in `url_dict` to check.
        max_workers (int):
            Number of threads for parallel HTTP checks.

    Side Effects:
        Modifies `url_dict` in place.
    """
    # 1) Collect only the URLs under the specified keys
    all_urls = {u for k in keys if k in url_dict for u in url_dict[k]}
    
    # 2) Check them in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(is_website_url, u): u for u in all_urls}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                results[u] = fut.result()
            except Exception:
                results[u] = False
    
    # 3) Rewrite in place, filtering each requested key
    for k in keys:
        url_dict[k] = [u for u in url_dict[k] if results.get(u, False)]


PAT = re.compile(
    r'^https?://search\.r-project\.org/CRAN/refmans/[^/]+/help/[^/]+\.html$'
)
match = PAT.match  # local reference to speed up lookups

def filter_cran_refs(url_dict):
    """
    Remove CRAN reference-manual URLs from each entry in the cache, in place.

    Parameters:
        url_dict (Dict[str, List[str]]):
            Mapping from names to lists of URL strings.

    Side Effects:
        Modifies `url_dict` in place by filtering out any URLs
        matching CRAN reference-manual patterns (e.g. '/manual/' in path).
    """
    for software, urls in url_dict.items():
        # build a new list only once, using the local `match`
        filtered = [u for u in urls if not match(u)]
        url_dict[software] = filtered
def _clean_github_url(raw_url: str) -> str:
    """
    Normalize a GitHub repo URL to the form 'https://github.com/{owner}/{repo}'.

    Removes any trailing '.git', extra path segments (e.g. '/tree/...'),
    query parameters, and fragments. Returns an empty string if the input
    is not recognized as a GitHub repository URL.

    Parameters:
        raw_url (str):
            The original GitHub URL.

    Returns:
        str:
            Canonical repo URL or '' if invalid.
    """
    pattern = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[A-Za-z0-9_-]+)"
    r"/"
    r"(?P<repo>[A-Za-z0-9._-]+)"
    r"(?=(?:[^A-Za-z0-9._-]|$))",
    re.IGNORECASE,
)
    m = pattern.search(raw_url)
    if not m:
        return ""
    owner = m.group("owner")
    repo  = m.group("repo")
    # 2) strip a trailing ".git" (case-insensitive), if it snuck in
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
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
        return "",0

    # 2) Fetch JSON metadata
    api_url = f"https://pypi.org/pypi/{pkg}/json"
    resp = requests.get(api_url)
    if resp.status_code != 200:
        return "",0
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

def cran_to_github_url(cran_url: str) -> str:
    """
    Converts a CRAN URL to a GitHub URL where author is 'cran'
    and repository is the package name.

    Args:
        cran_url (str): URL pointing to a CRAN package.

    Returns:
        str: GitHub-style URL like https://github.com/cran/<package>
             or None if package name couldn't be extracted.
    """
    if not isinstance(cran_url, str):
        return None

    # Regex to capture CRAN package name from various formats
    patterns = [
        r"cran\.r-project\.org/package=([^&/#?]+)",                      # ?package= or package=
        r"cran\.r-project\.org/\?package=([^&/#?]+)",
        r"cran\.r-project\.org/web/packages/([^/]+)/?",                 # /web/packages/<pkg>/
    ]

    for pattern in patterns:
        match = re.search(pattern, cran_url, re.IGNORECASE)
        if match:
            package = match.group(1)
            return f"https://github.com/cran/{package}"

    return None  # If no pattern matched

def get_candidate_urls(
    input: pd.DataFrame,
    cache_path: str = "candidate_urls.json",
    fetcher=fetch_candidate_urls,
    github_token: str = None
) -> pd.DataFrame:
    """
    Update a DataFrame with cleaned, deduplicated candidate URLs per name.

    1. Loads or updates the URL cache via `update_candidate_cache`.
    2. Normalizes & deduplicates all URLs.
    3. Filters out non-HTML & CRAN-refman links in parallel.
    4. Converts CRAN/PyPI URLs to GitHub when possible.
    5. Saves the final cache and writes a `'candidate_urls'` column
       back into `input`.

    Parameters:
        input (pd.DataFrame):
            DataFrame with a `'name'` column (and optional old `'candidate_urls'`).
        cache_path (str):
            Path to the JSON cache file for persistence.
        fetcher (callable):
            Function to fetch fresh candidate URLs for a given name.
        github_token (Optional[str]):
            GitHub token for API calls.

    Returns:
        pd.DataFrame:
            The same DataFrame, now with an up-to-date `'candidate_urls'` column.
    """
    print("🔍 Starting candidate URL extraction...")
    # 1) Update or load existing candidates
    candidates = update_candidate_cache(input, fetcher, cache_path, github_token)
    print("Deduplicating and normalizing URLs...")
    # 2) Normalize and deduplicate URLs
    dedupe_candidates(candidates)
    print("Filtering non-website URLs in parallel...")
    keys = input["name"].unique()
    # 3) Filter out non-website URLs in parallel

    filter_url_dict_parallel(candidates, keys)
    print("Filtering CRAN refman links...")
    # 4) Filter out CRAN refman links
    filter_cran_refs(candidates)
    # 5) Save candidates
    
    for key in keys:
        urls = candidates.get(key, [])
        # Work on a copy of the original list so we can modify freely.
        updated_urls = list(urls)
        print(f"Processing {key}")
        for u in urls:
            url = u.strip()
            parsed = urlparse(url)
            domain = urlparse(url).netloc.lower()
            path   = parsed.path or ""

            # CRAN package (common formats: cran.r-project.org or pkg.go.dev/r)
            if domain == "cran.r-project.org" and (
                path.startswith("/web/packages/") or
                path.startswith("/package=")
            ):
                github = cran_to_github_url(url)
                if github:
                    # If we have a valid GitHub link, replace the CRAN URL with it
                    if github not in updated_urls:
                        updated_urls.append(github)
                        print(f"Added GitHub URL: {github} from CRAN URL: {u}")
                githubs = extract_github_url_from_cran_package(url)
                for github in githubs:
                    if github and github not in updated_urls:
                        updated_urls.append(github)
                        print(f"Added GitHub URL: {github} from CRAN URL: {u}")
                
            elif "pypi.org" in domain or "pypi.python.org" in domain:
                github, description_length = get_github_link_from_pypi(u)
                if not github and description_length < 400:
                    updated_urls.remove(u)
                    print(f"Removed PyPI URL: {u} (no GitHub link found and description too short)")
                    
                else:
                    # If we have a valid GitHub link, replace the PyPI URL with it
                    if github and github not in updated_urls:
                        updated_urls.append(github)
                        print(f"Added GitHub URL: {github} from PyPI URL: {u}")
            elif "github.com" in domain:
                # If it's already a GitHub URL, clean it up
                if url == 'https://github.com/QianMo/Real-Time-Rendering-4th-Bibliography-Collection' or url == 'https://github.com/TapXWorld/ChinaTextbook':
                    updated_urls.remove(u)
                    continue  # remove the original URL
                cleaned = _clean_github_url(url)
                if cleaned and cleaned not in updated_urls:
                    updated_urls.append(cleaned)
                    updated_urls.remove(u)  # remove the original URL
                    print(f"Added cleaned GitHub URL: {cleaned} from original URL: {u}")
            
        # Replace the old list with the updated one
        candidates[key] = updated_urls
    cleaned_dict: dict[str, list[str]] = {}
    for key, url_list in candidates.items():
        cleaned_list = []
        for u in url_list:
            cleaned_list.append(_normalize_url_final(u))
        cleaned_dict[key] = cleaned_list
    print("Final deduplication of candidate URLs...")
    dedupe_candidates(cleaned_dict)
    save_candidates(cleaned_dict, cache_path)
    candidates = cleaned_dict
    if 'candidate_urls' not in input.columns:
        input['candidate_urls'] = np.nan
    input['candidate_urls'] = input['name'].map(candidates).astype(str)
    input['candidate_urls'] = input['candidate_urls'].str.replace("{", "").str.replace("}", "").str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace('"', '').str.replace(",", ",").str.replace(" ", "") # remove unwanted characters
    input['candidate_urls'] = input['candidate_urls'].str.replace("'", "").str.replace('"', '').str.replace(",", ",").str.replace(" ", "")
    return input


def dictionary_with_candidate_metadata(df:pd.DataFrame, output_json_path: str = "metadata_cache.json", github_token: str = None) -> Dict[str, dict]:
    """Extract and cache metadata for all unique candidate URLs in a DataFrame.

    This function:
      1. Gathers every non-empty URL from the `candidate_urls` column.
      2. Loads an existing JSON cache from `output_json_path`, or starts a new one.
      3. For each URL not already cached (or with empty metadata), calls `get_metadata(url)`
         and updates the cache.
      4. Writes the updated cache back to `output_json_path`.

    Args:
        df (pd.DataFrame): DataFrame with a `candidate_urls` column containing
            comma-separated URL strings.
        output_json_path (str): Path to the JSON file used for caching
            URL → metadata mappings.

    Returns:
        Dict[str, dict]: A mapping from each URL (str) to its metadata dict.
    """
    print("🔍 Starting metadata extraction for candidate URLs...")
    # Step 1: Extract unique, non-empty URLs
    url_set = set()
    for cell in df["candidate_urls"].dropna():
        if isinstance(cell, str):
            urls = [url.strip() for url in cell.split(",") if url.strip()]
            url_set.update(urls)

    # Step 2: Load existing cache or initialize empty one
    if os.path.exists(output_json_path) and os.path.getsize(output_json_path) > 0:
        with open(output_json_path, "r", encoding="utf-8") as f:
            try:
                metadata_cache = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Could not decode existing JSON. Starting with empty cache.")
                metadata_cache = {}
    else:
        metadata_cache = {}

    # Step 3: Fetch and update missing metadata
    try:
        identifier = 0
        num_url = len(url_set)
        num_dict = len(metadata_cache)
        for url in url_set:
            if url not in metadata_cache or metadata_cache[url] in [None, {}]:
                #print(f"🔍 Processing: {identifier}/{num_url-num_dict}")
                print(f"🔍 Processing: {url}")
                metadata_cache[url] = get_metadata(url, github_token=github_token)
            identifier += 1
    except KeyboardInterrupt:
        print("🔍 Process interrupted.")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_cache, f, indent=2, ensure_ascii=False)
        print(f"⚠️ Error at {url!r}: {e!r}  → cache saved to {output_json_path}")
        raise
    except Exception as e:
        # On first error: save and then re-raise
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_cache, f, indent=2, ensure_ascii=False)
        print(f"⚠️ Error at {url!r}: {e!r}  → cache saved to {output_json_path}")
        raise

    else:
        # If we got here with no exceptions, save normally
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_cache, f, indent=2, ensure_ascii=False)
        print(f"✅ All done — cache saved to {output_json_path}")

    return metadata_cache

def sanitize_text_for_csv(text: str) -> str:
    """Prepare a text string for safe CSV export.

    Replaces control characters with spaces, escapes internal quotes,
    and trims whitespace.

    Args:
        text: Raw input string.

    Returns:
        A cleaned string with no control characters and RFC-4180-compliant quotes.
    """
    # 1) Replace control chars (U+0000–U+001F, U+007F) with space
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    # 2) Escape any internal double‑quotes per RFC 4180: " → ""
    text = text.replace('"', '""')
    # 3) Trim leading/trailing whitespace
    return text.strip()

def add_metadata(df: pd.DataFrame, metadata: dict, output_path: str = None) -> None:
    """Populate a DataFrame in place with metadata for each candidate URL.

    Ensures the columns
    `metadata_name`, `metadata_authors`,
    `metadata_description`, and `metadata_language` exist. Then for each row
    missing `metadata_name`:
      1. Looks up its URL in the `metadata` dict.
      2. Sanitizes each field via `sanitize_text_for_csv`.
      3. Writes the values into the DataFrame.
    Optionally saves the updated DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame with `candidate_urls` and optional
            metadata columns to fill.
        metadata (Dict[str, dict]): Mapping URLs (str) → metadata dicts with keys
            `"name"`, `"authors"`, `"description"`, `"language"`.
        output_path (str, optional): If provided, path to write the updated
            DataFrame as CSV using minimal quoting.

    Returns:
        None
    """
    # Ensure metadata columns exist
    for col in ["metadata_name", "metadata_authors", "metadata_description","metadata_language"]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        # Skip rows where metadata_name is already present
        name_cell = row.get("metadata_name", "")
        if pd.notna(name_cell) and str(name_cell).strip():
            continue

        url = row.get("candidate_urls", "")
        if not isinstance(url, str) or not url.strip():
            print(f"Skipping row {idx}: missing or invalid URL")
            continue

        meta = metadata.get(url, {})
        if not meta:
            continue

        # 1) Name
        raw_name = meta.get("name", "") or ""
        df.at[idx, "metadata_name"] = sanitize_text_for_csv(raw_name)

        # 2) Authors (list → comma‑sep string)
        authors = meta.get("authors") or []
        authors_str = ", ".join(authors) if isinstance(authors, list) else ""
        df.at[idx, "metadata_authors"] = sanitize_text_for_csv(authors_str)

        # 4) Description
        raw_desc = meta.get("description", "") or ""
        df.at[idx, "metadata_description"] = sanitize_text_for_csv(raw_desc)

        raw_lang = meta.get("language", "") or ""
        df.at[idx, "metadata_language"] = sanitize_text_for_csv(raw_lang)

        #print(f"Processed row {idx} for URL: {url}")

    # Save to CSV if requested, using minimal quoting (fields with commas/quotes will be wrapped & escaped)
    if output_path:
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"📄 Updated CSV file saved to {output_path}")

def make_pairs(df:pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Explode comma-separated candidate URLs into one row per (mention, URL) pair.

    1. Splits `candidate_urls` into lists.
    2. Uses `DataFrame.explode` to produce one row per URL.
    3. Resets the index.
    4. Saves the exploded DataFrame to `output_path`.

    Parameters:
        df (pd.DataFrame):
            DataFrame with a `candidate_urls` column of comma-separated URLs.
        output_path (str):
            CSV path where the exploded table will be written.

    Returns:
        pd.DataFrame:
            The exploded DataFrame.
    """
    df["candidate_urls"] = df["candidate_urls"].fillna('').apply(
        lambda x: [url.strip() for url in str(x).split(',') if url.strip()]
    )
    df_exploded = df.explode("candidate_urls").reset_index(drop=True)
    
    # Assign new unique ID
    #df_exploded["id"] = range(1, len(df_exploded) + 1)
    df_exploded.to_csv(output_path, index=False)  # Save the DataFrame to a temporary CSV file

    return df_exploded





# Reuse or customize these lists/mappings
COMMON_LANGUAGES = [
    "Python", "R", "Java", "C\\+\\+", "C#", "C", "JavaScript",
    "TypeScript", "Ruby", "Go", "Rust", "Scala", "Haskell",
    "MATLAB", "PHP", "Perl", "Swift", "Kotlin", "Dart", "Julia"
]

IDE_MAPPING = {
    "rstudio": "R",
    "pycharm": "Python",
    "jupyter": "Python",
    "spyder": "Python",
    "eclipse": "Java",
    "intellij": "Java",
    "visual studio": "C#",
    "netbeans": "Java",
    "android studio": "Java",
    # add more IDE→language pairs as needed
}

def get_language_positions(
    text: str
) -> List[Tuple[str, int, int]]:
    """Detect programming language and IDE mentions in text with character spans.

    Scans `text` for known language names and IDE keywords, recording
    each match’s start and end indices.

    Args:
        text (str): Input document string.

    Returns:
        List[Tuple[str, int, int]]: A list of tuples
        `(language, start_index, end_index)` for each mention.
    """
    
    languages = COMMON_LANGUAGES
    ide_mapping = IDE_MAPPING

    positions: List[Tuple[str, int, int]] = []

    # Detect explicit language names
    for lang in languages:
        pattern = rf"\b{lang}\b"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            name = lang.replace("\\+\\+", "++")
            positions.append((name, m.start(), m.end()))

    # Detect IDE mentions and map back to language
    for ide, lang in ide_mapping.items():
        pattern = rf"\b{re.escape(ide)}\b"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            positions.append((lang, m.start(), m.end()))

    return positions

def find_nearest_language_for_softwares(
    text: str,
    software_names: str
) -> Optional[str]: 
    """
Identify the programming language mentioned closest to a software name.

Uses character-level proximity to find which programming language or IDE
is mentioned nearest to the given software name within the provided text.

Args:
    text (str): Full paragraph of text to search.
    software_names (str): The name of the software to find.

Returns:
    Optional[str]: Name of the closest language (e.g. "Python"), or None
    if no valid software or language is found.
"""

    languages = COMMON_LANGUAGES
    ide_mapping = IDE_MAPPING
    lang_positions = get_language_positions(text)

  
    # find first occurrence of software mention
    match = re.search(rf"\b{re.escape(software_names)}\b", text, flags=re.IGNORECASE)
    if not match:
        return None
        

    center = (match.start() + match.end()) // 2

        # pick the language with minimum distance to the software
    nearest = min(
        lang_positions,
        key=lambda lp: abs(((lp[1] + lp[2]) // 2) - center),
        default=None
    )
    result = nearest[0] if nearest else None

    return result



def get_authors(doi):
    """
    Retrieve author display names for a work via the OpenAlex API.

    Fetches `https://api.openalex.org/works/https://doi.org/{doi}` and
    extracts each author’s `display_name`.

    Parameters:
        doi (str):
            Digital Object Identifier of the work.

    Returns:
        Dict[str, Any]:
            `{ "doi": doi, "authors": [<display_name>, …] }`.

    Raises:
        requests.exceptions.RequestException:
            If the HTTP request fails.
        KeyError:
            If the expected 'authorships' field is missing.
    """
    # Set the URL for the OpenAlex API
    url = "https://api.openalex.org/works/https://doi.org/"
    # Set the headers
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    # Set the parameters for the query
    #params = {
    #    'query': 'your_query_here',  # Replace 'your_query_here' with your actual query
    #    'apikey': 'your_api_key_here',  # Replace 'your_api_key_here' with your actual API key
    #}
    response = requests.get(url+str(doi), headers=headers)
    json_response = response.json()
    return_value = {"doi":doi}
    if(json_response["authorships"] is not None):
        return_authors = []
        for author in json_response["authorships"]:
            if(author["author"]):
                a = author["author"]
                return_authors.append(a["display_name"])
    return_value["authors"] = return_authors
    return return_value

def get_synonyms_from_CZI(df, dictionary ,keys):
    """
    Populate `dictionary[key]` with CZI-provided synonyms from a DataFrame.

    For each `key`:
     - If `dictionary[key]` is empty, finds rows in `df` where
       `software_mention`.str.lower() == `key` and adds `synonym` values.

    Parameters:
        df (pd.DataFrame):
            Must have columns `software_mention` and `synonym`.
        dictionary (Dict[str, Set[str]]):
            Mapping from lowercase software name to a set of synonyms.
        keys (Iterable[str]):
            Software names (lowercase) to update.

    Side Effects:
        Updates `dictionary` in place.
    """
    for key in keys:
        if dictionary[key] != set():
            continue
        # Find matching rows in synonyms_df where the software mention matches the dictionary key
        matches = df[df["software_mention"].str.lower() == key]["synonym"].tolist()
        # Store synonyms as a list
        dictionary[key].update(matches)

def get_synonyms_from_SoftwareKG(dictionary, keys):
    """
    Populate synonyms for keys by querying the SoftwareKG SPARQL endpoint.

    For each `key` whose `dictionary[key]` is empty, runs a SPARQL query
    on GESIS SomeSci to retrieve alternative spellings (`nif:anchorOf`)
    linked to the same software entity.

    Parameters:
        dictionary (Dict[str, Set[str]]):
            Mapping from lowercase software name to a set of synonyms.
        keys (Iterable[str]):
            Software names (lowercase) to query.

    Side Effects:
        Updates `dictionary` in place.
    """
    # Define the SPARQL endpoint
    sparql = SPARQLWrapper("https://data.gesis.org/somesci/sparql")
    # Execute the query
    for key in keys:
        if dictionary[key] != set():
            continue
        query = f"""
    PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
PREFIX sms: <http://data.gesis.org/somesci/>
PREFIX its: <http://www.w3.org/2005/11/its/rdf#>

SELECT DISTINCT ?synonym
WHERE {{
    # Find the software entity associated with the given spelling
    ?sw_phrase a nif:Phrase ;
               its:taClassRef [ rdfs:subClassOf sms:Software ] ;
               its:taIdentRef ?sw_identity ;
               nif:anchorOf "{key}" .  # Replace "Excel" with the desired software name

    # Retrieve other spellings linked to the same software identity
    ?other_phrase its:taIdentRef ?sw_identity ;
                  nif:anchorOf ?synonym .
    
    FILTER (?synonym != "{key}")  # Exclude the original input spelling from results
}}
ORDER BY ?synonym
    """
        try:
            # Set query and return format
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            # Process results
            for result in results["results"]["bindings"]:
                synonym = result.get("synonym", {}).get("value")
                if synonym:
                    dictionary[key].add(synonym)

        except Exception as e:
            print(f"Error retrieving synonyms for {key}: {e}")
def get_synonyms(dictionary, keys, CZI = 1, SoftwareKG = 1, CZI_df: pd.DataFrame = None) -> Dict[str, set]:
    """
    Retrieve synonyms for software names using CZI and/or SoftwareKG sources.

    Depending on flags, calls `get_synonyms_from_CZI` and/or
    `get_synonyms_from_SoftwareKG`, then converts each set into a list.

    Parameters:
        dictionary (Dict[str, Set[str]]):
            Initial mapping from lowercase software name to a set of synonyms.
        keys (Iterable[str]):
            Names (lowercase) for which to fetch synonyms.
        CZI (bool):
            Whether to use the CZI DataFrame source.
        SoftwareKG (bool):
            Whether to use the SPARQL SoftwareKG source.
        CZI_df (Optional[pd.DataFrame]):
            DataFrame for CZI source lookups.

    Returns:
        Dict[str, List[str]]:
            Mapping from software name to list of synonyms.
    """
    if CZI == 1:
        get_synonyms_from_CZI(CZI_df, dictionary, keys)
    if SoftwareKG == 1:
        get_synonyms_from_SoftwareKG(dictionary, keys)
    dictionary = {key: list(value) for key, value in dictionary.items()}
    return dictionary

def get_synonyms_from_file(synonym_file_location: str, benchmark_df: pd.DataFrame, CZI = 1, SoftwareKG = 1, CZI_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load, augment, and persist a synonyms dictionary, then map into the DataFrame.

    1. If `synonym_file_location` exists and is non-empty, load it as JSON to a dict of sets.
       Otherwise initialize an empty set for each unique `benchmark_df["name"]`.
    2. Ensure every lowercase name from `benchmark_df` has an entry in the dict.
    3. Call `get_synonyms(...)` with CZI and/or SoftwareKG sources to populate missing synonyms.
    4. Write the updated dictionary back to `synonym_file_location` as JSON.
    5. Add or overwrite `benchmark_df["synonyms"]` by lowercasing each name, looking up its
       synonyms (as a list), and joining them with commas.

    Parameters:
        synonym_file_location (str):
            Filesystem path to the JSON file used for caching software-name → synonyms.
        benchmark_df (pd.DataFrame):
            Input DataFrame containing at least a `"name"` column of software mentions.
        CZI (bool):
            Whether to fetch synonyms from the CZI CSV source.
        SoftwareKG (bool):
            Whether to fetch synonyms via the SoftwareKG SPARQL endpoint.
        CZI_df (Optional[pd.DataFrame]):
            DataFrame loaded from the CZI synonyms CSV (if `CZI` is True).

    Returns:
        pd.DataFrame:
            The same `benchmark_df`, now guaranteed to have a `"synonyms"` column
            of comma-separated synonym strings.

    Side Effects:
        - Reads from and writes to `synonym_file_location`.
        - Prints status messages to stdout.

    Raises:
        IOError:
            If writing the JSON file fails.
    """
    print("Fetching synonyms...")
    if os.path.exists(synonym_file_location) and os.path.getsize(synonym_file_location) > 0:
        with open(synonym_file_location, "r", encoding="utf-8") as f:
            try:
                benchmark_dictionary = json.load(f)
                benchmark_dictionary = {k: set(v) for k, v in benchmark_dictionary.items()}
                names = benchmark_df["name"].str.lower().unique()
                # Ensure all names in benchmark_df are present in the dictionary
                for name in names:
                    if name not in benchmark_dictionary:
                        benchmark_dictionary[name] = set()

            except json.JSONDecodeError:
                print("⚠️ Warning: Could not decode existing JSON. Starting with empty cache.")
                benchmark_dictionary = {name.lower(): set() for name in benchmark_df["name"].unique()}
    else:
            benchmark_dictionary = {name.lower(): set() for name in benchmark_df["name"].unique()}
    names = benchmark_df["name"].str.lower().unique()
    benchmark_dictionary = get_synonyms(benchmark_dictionary, CZI=CZI, SoftwareKG=SoftwareKG, keys=names, CZI_df=CZI_df)
    # Save the updated dictionary to a JSON file
    with open(synonym_file_location, "w", encoding="utf-8") as f:
        json.dump(benchmark_dictionary, f, ensure_ascii=False, indent=4)
    benchmark_df["synonyms"] = (benchmark_df["name"]
    .str.lower()
    .map(benchmark_dictionary)
    .str.join(",")
)
    return benchmark_df
def aggregate_group(subdf):
    """
    Aggregate a group of (mention, URL, prediction) rows into summary fields.

    For the subgroup `subdf`, returns a Series with:
      - 'synonyms': all unique `subdf["synonyms"]` values, joined by ", ".
      - 'language': all unique `subdf["language"]` values, joined by ", ".
      - 'authors':  all unique `subdf["authors"]` values, joined by ", ".
      - 'urls':     all `subdf["candidate_urls"]` where `prediction == 1`, joined by ", ".
      - 'not_urls': all `subdf["candidate_urls"]` where `prediction == 0`, joined by ", ".

    Parameters:
        subdf (pd.DataFrame):
            A slice of the full DataFrame for one (name, paragraph, DOI) group,
            containing at minimum the columns
            `['synonyms','language','authors','candidate_urls','prediction']`.

    Returns:
        pd.Series:
            A one-row summary of that group, suitable for reassembly into a grouped DataFrame.
    """
    return pd.Series({
        'synonyms': ', '.join(subdf['synonyms'].dropna().astype(str).unique()),
        'language': ', '.join(subdf['language'].dropna().astype(str).unique()),
        'authors': ', '.join(subdf['authors'].dropna().astype(str).unique()),
        'urls': ', '.join(subdf.loc[subdf['prediction'] == 1, 'candidate_urls'].dropna().astype(str)),
        'not_urls': ', '.join(subdf.loc[subdf['prediction'] == 0, 'candidate_urls'].dropna().astype(str)),
    })
def _normalize_url_final(url: str) -> str:
    """
    Domain-specific final normalization of a single URL.

    - If the host ends in "github.com" or "cran.r-project.org": remove any trailing slash.
    - If the host ends in "pypi.org": ensure exactly one trailing slash.
    - Otherwise leave the path untouched.

    Parameters:
        url (str):
            The original URL to normalize.

    Returns:
        str:
            The rebuilt URL with adjusted trailing slash rules.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path or ""
    
    if domain.endswith("github.com") or domain.endswith("cran.r-project.org"):
        # remove any trailing slash from the path
        path = path.rstrip("/")
    elif domain.endswith("pypi.org"):
        # ensure there IS exactly one trailing slash
        if not path.endswith("/"):
            path = path + "/"
    # rebuild URL with the (possibly) modified path
    cleaned = parsed._replace(path=path)
    return urlunparse(cleaned)
if __name__ == "__main__":
    url = "https://github.com/ec-europa/widoco"
    data = extract_somef_metadata(url)
    print(data)
