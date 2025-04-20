import pandas as pd
from pathlib import Path
import json
from typing import Dict
import os
from fetching_medata_from_cantidate_url import get_metadata  

 
    
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


def add_metadata(df:pd.DataFrame, metadata:dict,output_path: str = None):


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
        authors = metadata_data.get("authors") or []
        if isinstance(authors, list) and authors:
            df.at[idx, "metadata_authors"] = ", ".join(authors)
        else:
            df.at[idx, "metadata_authors"] = ""

        # Keywords: same treatment
        keywords = metadata_data.get("keywords") or []
        if isinstance(keywords, list) and keywords:
            df.at[idx, "metadata_keywords"] = ", ".join(keywords)
        else:
            df.at[idx, "metadata_keywords"] = ""

        df.at[idx, "metadata_description"] = metadata_data.get("description", "")
        print(f"Processed row {idx} for URL: {url}")

    # Save to Excel if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"📄 Updated Excel file saved to {output_path}")
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