import pandas as pd
from pathlib import Path
import json
from typing import Dict
import os
from fetching_medata_from_cantidate_url import get_metadata  
import re
import csv
    
def dictionary_with_candidate_metadata(df:pd.DataFrame, output_json_path: str = "metadata_cache.json") -> Dict[str, dict]:
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
    for url in url_set: #Add that is also fetches again if the metadata is empty
        if url not in metadata_cache or metadata_cache[url] in [None, {}]:
            print(f"🔍 Processing: {url}")
            metadata = get_metadata(url)
            metadata_cache[url] = metadata

    # Step 4: Save updated metadata
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f, indent=2, ensure_ascii=False)
        print(f"📦 Metadata cache saved to: {output_json_path}")

    return metadata_cache

def sanitize_text_for_csv(text: str) -> str:
    """
    Replace control characters (incl. newlines, tabs, nulls) with spaces,
    escape internal double-quotes by doubling them, and trim.
    """
    # 1) Replace control chars (U+0000–U+001F, U+007F) with space
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    # 2) Escape any internal double‑quotes per RFC 4180: " → ""
    text = text.replace('"', '""')
    # 3) Trim leading/trailing whitespace
    return text.strip()

def add_metadata(df: pd.DataFrame, metadata: dict, output_path: str = None):
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
            print(f"Skipping row {idx}: no metadata found for URL {url}")
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

        print(f"Processed row {idx} for URL: {url}")

    # Save to CSV if requested, using minimal quoting (fields with commas/quotes will be wrapped & escaped)
    if output_path:
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"📄 Updated CSV file saved to {output_path}")

#Makes pairs of candidate urls and metadata and sets their probabilities, saves the temp file
def make_pairs(df:pd.DataFrame) -> pd.DataFrame:
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
    df_exploded.to_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/pairwise_temp.csv", index=False)  # Save the DataFrame to a temporary CSV file

    return df_exploded
    
if __name__ == "__main__":
    # Example usage
    excel_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/corpus.xlsx"
    output_json_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/metadata_cache.json"
    output_path = "D:/MASTER/TMF/Software-Disambiguation/corpus/temp/updated_with_metadata_file.csv"

    # Build metadata cache from Excel

    # Load the DataFrame again to add metadata
    df = pd.read_excel(excel_path)
    metadata_cache = dictionary_with_candidate_metadata(df, output_json_path)
    print(metadata_cache)
    df = make_pairs(df)

    add_metadata(df,metadata_cache, output_path)