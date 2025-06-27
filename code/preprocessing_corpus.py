import json
from urllib.parse import urlparse
import pandas as pd
import json
from typing import List, Tuple, Dict, Optional
import os
from fetching_medata_from_cantidate_url import get_metadata  
import re
import csv
from similarity_metrics import compute_similarity_df, compute_similarity_test
from rake_nltk import Rake
import string

    
def dictionary_with_candidate_metadata(df:pd.DataFrame, output_json_path: str = "metadata_cache.json") -> Dict[str, dict]:
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
    Raises:
        Exception:
            Any exception raised by `get_metadata` will be re-raised after the
            cache is saved to `output_json_path`.
    """
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
                metadata_cache[url] = get_metadata(url)
            identifier += 1

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
    """
    Clean a text string so it can be safely embedded in a CSV.

    This performs:
      - Replacement of control characters (U+0000–U+001F, U+007F) with spaces.
      - Doubling of any internal double-quotes (`" → ""`) per RFC-4180.
      - Trimming of leading and trailing whitespace.

    Args:
        text (str):
            Raw input string (may contain control chars, quotes, etc.).

    Returns:
        str:
            The cleaned string, ready for CSV export.
    """
    # 1) Replace control chars (U+0000–U+001F, U+007F) with space
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    # 2) Escape any internal double‑quotes per RFC 4180: " → ""
    text = text.replace('"', '""')
    # 3) Trim leading/trailing whitespace
    return text.strip()

def add_metadata(df: pd.DataFrame, metadata: dict, output_path: str = None):
    """
    Populate `df` in-place with metadata fields for each candidate URL.

    Ensures columns
      `metadata_name`, `metadata_authors`, `metadata_keywords`,
      `metadata_description`, and `metadata_language`
    exist, then for every row whose `metadata_name` is blank:
      1. Looks up the URL in `metadata[url]`.
      2. Applies `sanitize_text_for_csv` to each field.
      3. Writes the cleaned values back into `df`.

    If `output_path` is provided, saves the updated DataFrame as CSV with
    minimal quoting.

    Args:
        df (pd.DataFrame):
            Must contain `"candidate_urls"` and (optionally) existing metadata cols.
        metadata (Dict[str, dict]):
            URL→dict mapping with keys `"name"`, `"authors"`, `"keywords"`,
            `"description"`, and `"language"`.
        output_path (str, optional):
            Path to write the CSV. If `None`, no file is written.

    Returns:
        None

    Raises:
        IOError:
            If writing to `output_path` fails.
    """
    # Ensure metadata columns exist
    for col in ["metadata_name", "metadata_authors", "metadata_keywords", "metadata_description","metadata_language"]:
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

        # 3) Keywords (list → comma‑sep string)
        keywords = meta.get("keywords") or []
        kw_str = ", ".join(keywords) if isinstance(keywords, list) else ""
        df.at[idx, "metadata_keywords"] = sanitize_text_for_csv(kw_str)

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
    Explode each comma-separated URL into its own row and assign a unique ID.

    Transforms:
      - Splits `df["candidate_urls"]` into lists.
      - Uses `DataFrame.explode` to get one (row, URL) pair per line.
      - Adds a 1-based integer column `"id"`.
      - Saves the result to `output_path` as CSV.

    Args:
        df (pd.DataFrame):
            Must contain `"candidate_urls"` of comma-separated URL lists.
        output_path (str):
            File path where the exploded table will be written.

    Returns:
        pd.DataFrame:
            The exploded DataFrame with a new `"id"` column.

    Raises:
        IOError:
            If saving to `output_path` fails.
    """
    df["candidate_urls"] = df["candidate_urls"].fillna('').apply(
        lambda x: [url.strip() for url in str(x).split(',') if url.strip()]
    )
    df_exploded = df.explode("candidate_urls").reset_index(drop=True)
    
    # Assign new unique ID
    df_exploded["id"] = range(1, len(df_exploded) + 1)
    df_exploded.to_csv(output_path, index=False)  # Save the DataFrame to a temporary CSV file

    return df_exploded


def even_out_dataframes(df_full: pd.DataFrame, df_metrics: pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Ensure every row in `df_full` appears in `df_metrics`, filling missing metrics with NaN.

    1. Performs a left-merge on ['name','doi','paragraph','candidate_urls'].
    2. Identifies rows only in `df_full` and reindexes them to match `df_metrics` columns.
    3. Appends those rows to `df_metrics`.
    4. Drops any rows still missing `metadata_name`.
    5. Optionally writes the combined DataFrame to CSV.

    Args:
        df_full (pd.DataFrame):
            Master DataFrame of all expected entries.
        df_metrics (pd.DataFrame):
            DataFrame containing some subset of metric columns.
        output_path (str, optional):
            If provided, path to save the combined CSV.

    Returns:
        pd.DataFrame:
            A DataFrame containing all rows, with missing metrics as NaN.

    Raises:
        IOError:
            If saving to `output_path` fails.
    """
    key_cols = ['name', 'doi', 'paragraph',"candidate_urls"]

    # 1) Merge left-only to find rows in full but not in metrics
    merged = df_full.merge(
        df_metrics[key_cols],
        on=key_cols,
        how='left',
        indicator=True
    )

    missing = merged[merged['_merge'] == 'left_only'].copy()

    # 2) Build a DataFrame with exactly the same columns as df_metrics
    #    Reindex will:
    #      - pick the columns in df_metrics.columns order
    #      - fill missing metric-cols with NaN automatically
    to_append = missing.reindex(columns=df_metrics.columns)

    # 3) Concatenate
    result = pd.concat([df_metrics, to_append], ignore_index=True, sort=False)
    result = result.dropna(subset='metadata_name', how='all')
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"📄 Updated CSV file saved to {output_path}")
    return result


# Reuse or customize these lists/mappings
COMMON_LANGUAGES = [
    "ABAP",
    "Ada",
    "ALGOL",
    "APL",
    "AppleScript",
    "Assembly",
    "AWK",
    "Bash",
    "Batch",
    "C",
    "C#",
    "C\\+\\+",            # escaped for regex
    "Clojure",
    "COBOL",
    "Crystal",
    "D",
    "Dart",
    "Delphi",
    "Erlang",
    "Elixir",
    "Elm",
    "F#",
    "Fortran",
    "Go",
    "Groovy",
    "Haskell",
    "HTML",
    "Java",
    "JavaScript",
    "Julia",
    "Kotlin",
    "LabVIEW",
    "Lisp",
    "Lua",
    "MATLAB",
    "Objective-C",
    "OCaml",
    "Pascal",
    "Perl",
    "PHP",
    "PowerShell",
    "Prolog",
    "Python",
    "R",
    "Racket",
    "Rexx",
    "Ruby",
    "Rust",
    "Scala",
    "Scheme",
    "Shell",
    "SQL",
    "Swift",
    "Tcl",
    "TypeScript",
    "VBScript",
    "VBA",
    "Visual Basic",
    "Visual Basic .NET",
    "WebAssembly",
    "Wolfram",
    "Zig",
]

IDE_MAPPING = {
    # Python
    "pycharm": "Python",
    "jupyter": "Python",
    "spyder": "Python",
    "vscode": "Python",
    "atom": "Python",
    "sublime text": "Python",
    "thonny": "Python",

    # R
    "rstudio": "R",

    # Java
    "intellij": "Java",
    "eclipse": "Java",
    "netbeans": "Java",
    "android studio": "Java",

    # C/C++
    "visual studio": "C#",
    "clion": "C++",
    "qt creator": "C++",
    "code::blocks": "C++",
    "xcode": "C",
    "dev c++": "C++",

    # C#
    "visual studio": "C#",
    "sharpdevelop": "C#",

    # JavaScript / TypeScript
    "vscode": "JavaScript",
    "webstorm": "JavaScript",
    "atom": "JavaScript",
    "sublime text": "JavaScript",

    # Go
    "goland": "Go",

    # Rust
    "intellij": "Rust",
    "vscode": "Rust",

    # Scala
    "intellij": "Scala",
    "ensime": "Scala",

    # Haskell
    "haskell ide": "Haskell",
    "intellij": "Haskell",

    # MATLAB
    "matlab": "MATLAB",

    # PHP
    "phpstorm": "PHP",
    "netbeans": "PHP",

    # Perl
    "padre": "Perl",

    # Swift / Objective-C
    "xcode": "Swift",
    "xcode": "Objective-C",

    # Kotlin
    "intellij": "Kotlin",

    # Dart / Flutter
    "android studio": "Dart",
    "vscode": "Dart",

    # Julia
    "julia studio": "Julia",
    "vscode": "Julia",

    # Ruby
    "ruby mine": "Ruby",
    "vscode": "Ruby",

    # Erlang / Elixir
    "intellij": "Erlang",
    "intellij": "Elixir",

    # F#
    "visual studio": "F#",
}

def get_language_positions(
    text: str
) -> List[Tuple[str, int, int]]:
    """
    Find character spans for known programming languages and IDEs in `text`.

    Scans for each entry in `COMMON_LANGUAGES` (as regex `\bLang\b`) and each
    key in `IDE_MAPPING`, mapping IDE name back to its language.

    Args:
        text (str):
            The document in which to search.

    Returns:
        List[Tuple[str, int, int]]:
            Tuples of (language, start_index, end_index) for each match.
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
    Identify which programming language mention lies closest to a software name.

    1. Finds the first case-insensitive occurrence of `software_names` in `text`.
    2. Computes midpoints of all language/IDE spans (from `get_language_positions`).
    3. Returns the language whose midpoint is closest to that of the software.

    Args:
        text (str):
            The paragraph to search.
        software_names (str):
            The software name to locate.

    Returns:
        Optional[str]:
            The nearest language (e.g. "Python") or `None` if no match found.
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


def select_rows_below_threshold(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float = 0.3
) -> pd.DataFrame:
    """Filter rows where any specified column’s value is below a threshold.

    Treats NaNs as not below threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (List[str]): List of column names to check.
        threshold (float): Threshold value; defaults to 0.3.

    Returns:
        pd.DataFrame: New DataFrame containing only rows where any of
        `cols` has a value < `threshold`.
    """
    # Boolean mask: True for values < threshold, NaN → False
    below = df[cols].lt(threshold).fillna(False).any(axis=1)
    # Select only those rows
    return df.loc[below].reset_index(drop=True)
def keywords_from_paper_rake(text: str, top_n: int = 5) -> str:
    """
Extract keyword phrases from input text using RAKE after punctuation removal.

This function cleans the input by removing all punctuation, then applies
the RAKE (Rapid Automatic Keyword Extraction) algorithm to identify key
multi-word phrases. It returns the top-ranked phrases joined as a
comma-separated string.

Args:
    text (str): Input text from which to extract keywords.
    top_n (int): Number of top-ranked phrases to return. Default is 5.

Returns:
    str: A comma-separated string of up to `top_n` keyword phrases.
         If no keywords are found or input is empty, returns an empty string.
"""

    # 1) Guard against None
    raw = text or ""
    # 2) Remove punctuation
    clean = raw.translate(str.maketrans("", "", string.punctuation))
    # 3) Run RAKE
    rake = Rake()  # you can pass min_length, max_length if you like
    rake.extract_keywords_from_text(clean)
    # 4) Grab top_n phrases
    phrases = rake.get_ranked_phrases()[:top_n]
    # 5) Join and return
    return ",".join(phrases)
def missing_github_RAKE(metadata:dict):
    """
Enrich GitHub repository metadata with keywords using RAKE if missing.

For each GitHub URL in the metadata dictionary:
- Checks if the `keywords` field is missing or empty
- If so, applies RAKE to the description to generate up to 5 keywords
- Updates the `metadata[url]["keywords"]` field in place

Args:
    metadata (dict): A dictionary mapping URLs to metadata dictionaries.

Returns:
    None: This function modifies the metadata dictionary in place.
"""

    for url in metadata.keys():
        if "github.com" in url:
            # Extract the text from the metadata
            text = metadata[url].get("description", "")
            keywords = metadata[url].get("keywords", "")
            if len(keywords)==0:
                kws = []
                if not kws and not pd.isna(text) and text:
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
                    metadata[url]["keywords"] = kws
                    print(f"Extracted keywords from GitHub URL: {url}, {kws}")
            
def is_pypi_url(url: str) -> bool:
    """
    Check whether `url` is a PyPI project page.

    Considers hosts containing "pypi.org" or "python.org" and a path
    starting with "/project/".

    Args:
        url (str):
            The URL to test.

    Returns:
        bool:
            `True` if it matches a PyPI project pattern, else `False`.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        ("pypi.org" in host or "python.org" in host)  # cover pypi.org (and edge cases)
        and path.startswith("/project/")
    )
if __name__ == "__main__":
    
    excel_path = "research/corpus/corpus_v3_14.xlsx"
    output_json_path = "research/corpus/temp/metadata_caches/metadata_cache_v3_13.json"
    output_path = "research/corpus/temp/v3.17/updated_with_metadata.csv"
    output_path_similarities = "research/corpus/temp/v3.17/similarities.csv"
    output_path_pairs = "research/corpus/temp/v3.17/pairs.csv"
    model_input_path = "research/corpus/temp/v3.17/model_input.csv"
    #df = pd.read_excel(excel_path)
    df = pd.read_csv(output_path)
    df['language'] = df.apply(
    lambda row: find_nearest_language_for_softwares(
        text=row['paragraph'],
        software_names=row['name']

    ),
    axis=1
)
    df['language'] = df['language'].fillna('')
    metadata_cache = dictionary_with_candidate_metadata(df, output_json_path)
    df = make_pairs(df,output_path_pairs)

    add_metadata(df,metadata_cache, output_path)
    df = compute_similarity_df(df,output_path_similarities)

    sim = compute_similarity_test(df, output_path_similarities)
    model_input = sim[['name_metric', 'paragraph_metric','language_metric','synonym_metric','author_metric','true_label']].copy()
    model_input.to_csv(model_input_path, index=False)
    