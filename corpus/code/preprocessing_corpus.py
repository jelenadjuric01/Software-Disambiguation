import pandas as pd
from pathlib import Path
import json
from typing import Dict

# 👇 assumes this function is already defined
from fetching_medata_from_cantidate_url import get_metadata  # replace with actual import


def dictionary_with_cantidate_metadata(df,output_json_path: str = "metadata_cache.json") -> Dict[str, dict]:
    

    # Extract unique, non-empty URLs
    url_set = set()
    for cell in df["candidate_urls"].dropna():
        if isinstance(cell, str):
            urls = [url.strip() for url in cell.split(",") if url.strip()]
            url_set.update(urls)

    # Initialize dictionary
    metadata_cache = {url: None for url in url_set}
    # Fill metadata where missing
    for url in metadata_cache:
        print(f"🔍 Processing: {url}")
        metadata = get_metadata(url)
        metadata_cache[url] = metadata

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f, indent=2, ensure_ascii=False)
        print(f"📦 Metadata cache saved to: {output_json_path}")

    return metadata_cache


def add_metadata(df, metadata:dict,output_path: str = None):


    # Columns to ensure exist
    metadata_columns = [
        "metadata_name",
        "metadata_authors",
        "metadata_keywords",
        "metadata_description"
    ]

    # Add missing columns
    for col in metadata_columns:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, "metadata_name"]) and str(df.at[idx, "metadata_name"]).strip() != "":
            continue  # Skip if metadata_name already exists
        url = row.get("candidate_urls")
        if not isinstance(url, str) or url.strip() == "":
            print(f"Skipping row {idx}: missing or invalid URL")
            continue
        metadata_data = metadata.get(url, {})
        if not metadata_data:
            print(f"Skipping row {idx}: no metadata found for URL {url}")
            continue

        df.at[idx, "metadata_name"] = metadata_data.get("name", "")

        # Convert authors and keywords to JSON strings
        authors = metadata_data.get("authors", [])
        df.at[idx, "metadata_authors"] = json.dumps(authors, ensure_ascii=False)

        keywords = metadata_data.get("keywords", [])
        df.at[idx, "metadata_keywords"] = json.dumps(keywords, ensure_ascii=False)

        df.at[idx, "metadata_description"] = metadata_data.get("description", "")
        print(f"Processed row {idx} for URL: {url}")

    # Save to Excel if output path is provided
    if output_path:
        df.to_excel(output_path, index=False)
        print(f"📄 Updated Excel file saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    excel_path = "path_to_your_excel_file.xlsx"
    output_json_path = "metadata_cache.json"
    output_excel_path = "updated_excel_file.xlsx"

    # Build metadata cache from Excel

    # Load the DataFrame again to add metadata
    df = pd.read_excel(excel_path)
    metadata_cache = dictionary_with_cantidate_metadata(excel_path, output_json_path)

    add_metadata(df,metadata_cache, output_excel_path)