from urllib.parse import urlparse
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import os
from fetch_candidates import load_candidates, save_candidates
from fetching_medata_from_cantidate_url import extract_pypi_metadata_RAKE, extract_somef_metadata_with_RAKE_readme,get_github_link_from_pypi, extract_pypi_metadata_RAKE_class, extract_pypi_metadata_Rake_after, get_metadata  
import re
import csv
from similarity_metrics import compute_similarity_df, compute_similarity_test, keyword_similarity_with_fallback, keyword_similarity_with_fallback_SBERT, paragraph_description_similarity_BERT, software_name_similarity_levenshtein, synonym_name_similarity, synonym_name_similarity_levenshtein
from evaluation import  evaluation
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

def add_metadata(df: pd.DataFrame, metadata: dict, output_path: str = None):
    """Populate a DataFrame in place with metadata for each candidate URL.

    Ensures the columns
    `metadata_name`, `metadata_authors`, `metadata_keywords`,
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
            `"name"`, `"authors"`, `"keywords"`, `"description"`, `"language"`.
        output_path (str, optional): If provided, path to write the updated
            DataFrame as CSV using minimal quoting.

    Returns:
        None
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
    """Explode candidate URLs into one row per (mention, URL) pair and save to CSV.

    1. Splits the `candidate_urls` column on commas and explodes each URL
       into its own row.
    2. Assigns a new unique integer `id` to each row.
    3. Computes `probability (ground truth)` = 1 if the URL appears in
       `url (ground truth)`, else 0.
    4. Saves the exploded DataFrame to `output_path` and returns it.

    Args:
        df (pd.DataFrame): DataFrame with columns
            `candidate_urls` (comma-separated URLs) and
            `url (ground truth)`.
        output_path (str): File path to save the exploded CSV.

    Returns:
        pd.DataFrame: Exploded DataFrame with new `id` and
        `probability (ground truth)` columns.
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
    """Append missing rows from `df_full` into `df_metrics`, filling metrics with NaN.

    Ensures that every (`name`, `doi`, `paragraph`, `candidate_urls`) in
    `df_full` is present in `df_metrics`. Any rows missing in `df_metrics`
    are appended with metric columns set to NaN.

    Args:
        df_full (pd.DataFrame): The master DataFrame containing all expected rows.
        df_metrics (pd.DataFrame): DataFrame with metric columns.
        output_path (str, optional): If provided, saves the result to CSV.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all rows from both inputs,
        with missing metrics as NaN.
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
    Returns True if the given URL points to a PyPI project page (e.g. "https://pypi.org/project/foo/").
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        ("pypi.org" in host or "python.org" in host)  # cover pypi.org (and edge cases)
        and path.startswith("/project/")
    )
if __name__ == "__main__":
    # Taking corpus, extracting metadata from candidate urls, cumputing similarities and saving the updated file version 1
    """'''
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus_v1.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/updated_with_metadata_file_v1.csv"
    output_path_similarities = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/similarities_version_1.csv"
    # Build metadata cache from Excel

    # Load the DataFrame again to add metadata
    df = pd.read_excel(excel_path)
    metadata_cache = dictionary_with_candidate_metadata(df, output_json_path)
    print(metadata_cache)
    df = make_pairs(df)

    add_metadata(df,metadata_cache, output_path)
    df = compute_similarity_df(df,output_path_similarities)
    output_path_calculated_version_1 = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/calculated_version_1.csv"
    # Load the DataFrame again to see the results
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/similarities_version_1.csv")
    # Get the average, min, and max for each metric
    get_average_min_max(df, output_path_calculated_version_1)
    

    # Taking corpus, extracting metadata from candidate urls, cumputing similarities and saving the updated file version 2
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus_v2.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/updated_with_metadata_file.csv"
    output_path_similarities = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/similarities.csv"
    output_path_pairs = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/pairs.csv"
    output_path_calculated = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/calculated.csv"

    # Build metadata cache from Excel
    
    # Load the DataFrame again to add metadata
    df = pd.read_excel(excel_path)
    metadata_cache = dictionary_with_candidate_metadata(df, output_json_path)
    print(metadata_cache)
    df = make_pairs(df,output_path_pairs)

    add_metadata(df,metadata_cache, output_path)
    df = compute_similarity_df(df,output_path_similarities)
    # Load the DataFrame again to see the results
    df = pd.read_csv(output_path_similarities)
    # Get the average, min, and max for each metric
    get_average_min_max(df, output_path_calculated)
    outputh_avg_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/average_ranked.csv"
    outputh_min_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/min_ranked.csv"
    outputh_max_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/max_ranked.csv"
    #Get ranked candidates and save the updated file version 2
    df = pd.read_csv(output_path_calculated)
    df_avg, df_min, df_max = split_by_avg_min_max(df)
    df_avg = group_by_candidates(df_avg, outputh_avg_ranked)
    df_min = group_by_candidates(df_min, outputh_min_ranked)
    df_max = group_by_candidates(df_max, outputh_max_ranked)
    print("Evaluation  of average")
    evaluation(df_avg)
    print("Evaluation  of min")
    evaluation(df_min)
    print("Evaluation  of max")
    evaluation(df_max)
    #Version 3
    # Taking corpus, extracting metadata from candidate urls, cumputing similarities and saving the updated file version 3
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus_v3.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/updated_with_metadata_file.csv"
    output_path_similarities = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/similarities.csv"
    output_path_pairs = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/pairs.csv"
    output_path_calculated = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/calculated.csv"
    df = pd.read_excel(excel_path)
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
    # Load the DataFrame again to see the results
    df = pd.read_csv(output_path_similarities)
    # Get the average, min, and max for each metric
    get_average_min_max(df, output_path_calculated)
    outputh_avg_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/average_ranked.csv"
    outputh_min_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/min_ranked.csv"
    outputh_max_ranked = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/max_ranked.csv"
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/calculated.csv")
    df_avg, df_min, df_max = split_by_summary(df)
    print("Evaluation  of average")
    evaluation(df_avg)
    print("Evaluation  of min")
    evaluation(df_min)
    print("Evaluation  of max")
    evaluation(df_max)
    df_avg.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/average_ranked.csv")
    df_min.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/min_ranked.csv")
    df_max.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/max_ranked.csv")
   
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/calculated_positives.csv")
    filtered = select_rows_below_threshold(df,['name_metric','keywords_metric','paragraph_metric','language_metric'],0.1)
    filtered.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/low_quality.csv")"""
    #Version 3.1 adding synonyms similarity
    '''similarities_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.1/similarities.csv"
    updated_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.1/updated_with_metadata_file.csv"
    df = pd.read_csv(similarities_path)
    df_updated = pd.read_csv(updated_path)
    df_updated = df_updated.dropna(subset=['metadata_name']).copy()
    df['synonyms']=df_updated["synonyms"]
    df['synonym_metric'] = df.apply(
        lambda row: synonym_name_similarity(
            row['metadata_name'],
            row['synonyms']
        ),
        axis=1
    )
    df.to_csv(similarities_path, index=False)
    model_input = df[['name_metric', 'keywords_metric', 'paragraph_metric', 'author_metric','language_metric','synonym_metric','true_label']].copy()
    model_input.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.1/model_input.csv", index=False)'''
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus_v3_14.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache_v3_13.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.17/updated_with_metadata.csv"
    output_path_similarities = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.17/similarities.csv"
    output_path_pairs = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.17/pairs.csv"
    model_input_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.17/model_input.csv"
    #df = pd.read_excel(excel_path)
    df = pd.read_csv(output_path)
    

    sim = compute_similarity_test(df, output_path_similarities)
    model_input = sim[['name_metric', 'paragraph_metric','language_metric','synonym_metric','author_metric','true_label']].copy()
    model_input.to_csv(model_input_path, index=False)
    '''candidates = load_candidates("D:/MASTER/TMF/Software-Disambiguation/corpus/candidate_urls.json")
    candidates_13 = load_candidates("D:/MASTER/TMF/Software-Disambiguation/corpus/candidate_urls_v3_14.json")
    for key in set(candidates.keys()) | set(candidates_13.keys()):
        urls1 = set(candidates.get(key, []))
        urls2 = set(candidates_13.get(key, []))

        only_in_1 = urls1 - urls2
        only_in_2 = urls2 - urls1

        if only_in_1 or only_in_2:
            print(f"Entry: {key}")
            for u in only_in_1:
                print(f"  → only in first dict:  {u}")
            for u in only_in_2:
                print(f"  → only in second dict: {u}")
            print()  # blank line for readability'''

