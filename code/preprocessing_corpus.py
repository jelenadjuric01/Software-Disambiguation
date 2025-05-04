import pandas as pd
from pathlib import Path
import json
from typing import Dict
import os
from fetching_medata_from_cantidate_url import get_metadata  
import re
import csv
from similarity_metrics import compute_similarity_df, get_average_min_max
    
def dictionary_with_candidate_metadata(df:pd.DataFrame, output_json_path: str = "metadata_cache.json") -> Dict[str, dict]:
    """
    Extract and cache metadata for all unique candidate URLs in a DataFrame.

    This function:
      1. Gathers every non-empty URL from the `candidate_urls` column.
      2. Loads an existing JSON cache from `output_json_path`, or initializes a new one.
      3. For each URL not already cached (or with empty metadata), calls `get_metadata(url)`
         and updates the cache.
      4. Writes the updated cache back to `output_json_path`.

    Args:
        df: A pandas DataFrame containing a “candidate_urls” column where each cell
            is a comma-separated string of URLs.
        output_json_path: Path to the JSON file used for caching URL→metadata mappings.

    Returns:
        A dict mapping each URL (str) to its metadata (dict).
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
                print(f"🔍 Processing: {identifier}/{num_url-num_dict}")
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
    Clean a text string so it can safely be written as a CSV field.

    This replaces control characters (newlines, tabs, nulls, etc.) with spaces,
    escapes any internal double-quotes by doubling them, and trims whitespace.

    Args:
        text: The raw string to sanitize.

    Returns:
        A cleaned string with no control characters and properly escaped quotes.
    """
    # 1) Replace control chars (U+0000–U+001F, U+007F) with space
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    # 2) Escape any internal double‑quotes per RFC 4180: " → ""
    text = text.replace('"', '""')
    # 3) Trim leading/trailing whitespace
    return text.strip()

def add_metadata(df: pd.DataFrame, metadata: dict, output_path: str = None):
    """
    Populate a DataFrame with metadata fields for each candidate URL, in place.

    This function:
      1. Ensures the columns “metadata_name”, “metadata_authors”,
         “metadata_keywords”, and “metadata_description” exist.
      2. For each row lacking `metadata_name` and with a valid URL in
         `candidate_urls`, looks up its metadata in the provided `metadata` dict.
      3. Sanitizes each metadata field via `sanitize_text_for_csv` and writes
         it into the DataFrame.
      4. If `output_path` is given, saves the updated DataFrame to CSV using
         minimal quoting.

    Args:
        df: A pandas DataFrame with a “candidate_urls” column and optional
            metadata columns to fill.
        metadata: A dict mapping URLs (str) to metadata dicts with keys
                  "name", "authors", "keywords", and "description".
        output_path: Optional path to write the updated DataFrame as a CSV file.

    Returns:
        None. The DataFrame `df` is modified in place.
    """
    # Ensure metadata columns exist
    for col in ["metadata_name", "metadata_authors", "metadata_keywords", "metadata_description"]:
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

        #print(f"Processed row {idx} for URL: {url}")

    # Save to CSV if requested, using minimal quoting (fields with commas/quotes will be wrapped & escaped)
    if output_path:
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"📄 Updated CSV file saved to {output_path}")

def make_pairs(df:pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Explode comma-separated candidate URLs into one row per (mention, URL) pair.

    This function:
      1. Splits the “candidate_urls” column on commas and explodes the list
         into individual rows.
      2. Assigns a new unique integer “id” to each row.
      3. Computes a ground-truth flag “probability (ground truth)” which is 1
         if the candidate URL appears in the “url (ground truth)” column, else 0.
      4. Saves the pairwise result to a temporary CSV and returns the exploded DataFrame.

    Args:
        df: A pandas DataFrame containing at least the columns
            “candidate_urls” (comma-separated URLs) and “url (ground truth)”.

    Returns:
        A new DataFrame with columns:
          - id: unique integer per (mention, URL) pair
          - candidate_urls: one URL per row
          - url (ground truth): original ground-truth URL list
          - probability (ground truth): 1 or 0
          plus any other original columns repeated per exploded row.
    """
    df["candidate_urls"] = df["candidate_urls"].fillna('').apply(
        lambda x: [url.strip() for url in str(x).split(',') if url.strip()]
    )
    df_exploded = df.explode("candidate_urls").reset_index(drop=True)
    
    # Assign new unique ID
    df_exploded["id"] = range(1, len(df_exploded) + 1)
    df_exploded["probability (ground truth)"] = (
        df_exploded.apply(
            lambda row: int(row["candidate_urls"] in row["url (ground truth)"]),
            axis=1
        )
    )
    df_exploded.to_csv(output_path, index=False)  # Save the DataFrame to a temporary CSV file

    return df_exploded

    
    
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
    """

    # Taking corpus, extracting metadata from candidate urls, cumputing similarities and saving the updated file version 2
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus_v2.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/updated_with_metadata_file.csv"
    output_path_similarities = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/similarities.csv"
    output_path_pairs = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/pairs.csv"
    # Build metadata cache from Excel
    
    # Load the DataFrame again to add metadata
    df = pd.read_excel(excel_path)
    metadata_cache = dictionary_with_candidate_metadata(df, output_json_path)
    print(metadata_cache)
    df = make_pairs(df,output_path_pairs)

    add_metadata(df,metadata_cache, output_path)
    df = compute_similarity_df(df,output_path_similarities)
    output_path_calculated_version_1 = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/calculated.csv"
    # Load the DataFrame again to see the results
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v2/similarities.csv")
    # Get the average, min, and max for each metric
    get_average_min_max(df, output_path_calculated_version_1)