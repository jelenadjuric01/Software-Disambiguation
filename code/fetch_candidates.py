import numpy as np
import requests
from typing import List, Dict, Set
import os
import pandas as pd
import difflib
import time
import json

import re
from urllib.parse import urljoin
import xmlrpc.client
from functools import lru_cache


GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
    "User-Agent":    "my-software-disambiguator"  # any non-empty string
}

def fetch_github_urls(
    name: str,
    per_page: int = 5,
    max_retries: int = 3
) -> List[str]:
    """
    Return up to `per_page` GitHub repo URLs matching `name`, handling rate limits.
    """
    params = {
        "q":        f"{name} in:name",
        "sort":     "stars",
        "order":    "desc",
        "per_page": per_page
    }

    for attempt in range(1, max_retries + 1):
        resp = requests.get(GITHUB_API_URL, params=params, headers=HEADERS, timeout=10)
        # 403 could be a rate-limit on the Search API
        if resp.status_code == 403:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_ts - time.time()+1, 1)
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
    Fetches the JSON info block for `pkg`, or returns {} on error.
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
    1) Exact lookup via JSON API → returns info['package_url'] (or info['project_url'])
    2) Fuzzy lookup via XML‐RPC + JSON API per hit
    """
    urls: List[str] = []

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
    Cached in memory so we only download it once.
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
    Return up to `max_results` canonical CRAN URLs for packages matching `name`:
      1) exact match
      2) substring match
      3) fuzzy match via difflib
    """
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


def fetch_candidate_urls(name: str) -> set[str]:
    """
    For each software name, fetch candidate URLs in this order:
      1. GitHub
      2. PyPI
      3. CRAN
      4. General Google search (excluding above domains)
    """
    results = []

    # GitHub
    try:
        results += fetch_github_urls(name)
    except Exception as e:
        print(f"[!] GitHub fetch failed for '{name}': {e}")

    # PyPI
    try:
        results += fetch_pypi_urls(name)
    except Exception as e:
        print(f"[!] PyPI fetch failed for '{name}': {e}")

    # CRAN
    try:
        results += fetch_cran_urls(name)
    except Exception as e:
        print(f"[!] CRAN check failed for '{name}': {e}")

    # dedupe, preserve order
    return set(results)

def load_candidates(path: str) -> Dict[str, Set[str]]:
    """Load a JSON cache of {name: [urls…]}, return {name: set(urls)…}."""
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
    """Convert sets→lists and write out a pretty JSON file."""
    serializable = {name: sorted(list(urls)) for name, urls in candidates.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

def update_candidate_cache(
    corpus: pd.DataFrame,
    fetcher,                # your fetch_candidate_urls(name) function
    cache_path: str
) -> Dict[str, Set[str]]:
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
        new = set(fetcher(name))
        # only do the network hit if there’s something new to add
        if not new.issubset(candidates[name]):
            candidates[name].update(new)

    # 5) persist back to JSON
    save_candidates(candidates, cache_path)
    return candidates

from urllib.parse import urlparse, urlunparse
from typing import Dict, Iterable, List

def normalize_url(u: str) -> str:
    p = urlparse(u)
    scheme = "https"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")
    # drop params, query, fragment
    return urlunparse((scheme, netloc, path, "", "", ""))

def dedupe_candidates(candidates: Dict[str, Iterable[str]]) -> None:
    """
    For each key in `candidates`, normalize its URLs and drop duplicates,
    preferring the https version when http & https both appear.
    Modifies `candidates` in place, replacing each value with a List[str].
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
    """Return True if url passes extension + HEAD-test for HTML."""
    path = urlparse(url).path.lower()
    if os.path.splitext(path)[1] in DISALLOWED_EXTENSIONS:
        return False
    try:
        resp = session.head(url, allow_redirects=True, timeout=timeout)
        return 'text/html' in resp.headers.get('Content-Type', '')
    except requests.RequestException:
        return False

def filter_url_dict_parallel(url_dict, max_workers=20):
    # Flatten all URLs and dedupe
    all_urls = {u for urls in url_dict.values() for u in urls}
    
    # Fire off checks in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(is_website_url, u): u for u in all_urls}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                results[u] = fut.result()
            except Exception:
                results[u] = False
    
    # Rebuild filtered dict
    return {
        key: [u for u in urls if results.get(u)]
        for key, urls in url_dict.items()
    }


PAT = re.compile(
    r'^https?://search\.r-project\.org/CRAN/refmans/[^/]+/help/[^/]+\.html$'
)
match = PAT.match  # local reference to speed up lookups

def filter_cran_refs(url_dict):
    """
    In-place filter of url_dict[software] lists,
    removing any URL matching our CRAN-refman pattern.
    """
    for software, urls in url_dict.items():
        # build a new list only once, using the local `match`
        filtered = [u for u in urls if not match(u)]
        url_dict[software] = filtered

def get_candidate_urls(
    input: pd.DataFrame,
    cache_path: str = "candidate_urls.json",
    fetcher=fetch_candidate_urls
) -> pd.DataFrame:
    """
    Main entry point to update candidate URLs for each software in the corpus.
    Returns a dictionary of {software_name: set(urls)}.
    """
    # 1) Update or load existing candidates
    candidates = update_candidate_cache(input, fetcher, cache_path)

    # 2) Normalize and deduplicate URLs
    dedupe_candidates(candidates)

    # 3) Filter out non-website URLs in parallel
    candidates = filter_url_dict_parallel(candidates)

    # 4) Filter out CRAN refman links
    filter_cran_refs(candidates)
    # 5) Save candidates
    save_candidates(candidates, cache_path)
    if 'candidate_urls' not in input.columns:
        input['candidate_urls'] = np.nan
    input['candidate_urls'] = input['name'].map(candidates).astype(str)
    input['candidate_urls'] = input['candidate_urls'].str.replace("{", "").str.replace("}", "").str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace('"', '').str.replace(",", ",").str.replace(" ", "") # remove unwanted characters
    input['candidate_urls'] = input['candidate_urls'].str.replace("'", "").str.replace('"', '').str.replace(",", ",").str.replace(" ", "")
    return input