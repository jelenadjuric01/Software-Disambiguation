import re
import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import numpy as np
import pandas as pd
import textdistance
import csv

# 1. Load your models once (at module import)
#    - A lightweight BERT for keyword-vs-keyword
#    - A larger RoBERTa for fallback (keywords vs. description)
_BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
_ROBERTA_MODEL = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

def _encode_and_cosine(model: SentenceTransformer, texts1, texts2) -> float:
    """
    Helper: encode two lists of texts and return mean cosine similarity
    between corresponding embeddings.
    """
    emb1 = model.encode(texts1, convert_to_tensor=False)
    emb2 = model.encode(texts2, convert_to_tensor=False)
    # If lists are length 1, this is just cosine(emb1[0], emb2[0])
    sims = cosine_similarity(emb1, emb2)
    return float(np.mean(np.diag(sims)))

def keyword_similarity_with_fallback(
    paper_keywords: Optional[str],
    site_keywords: Optional[str],
    software_description: str
) -> float:
    """
    Compute a similarity score between paper‐keywords and site‐keywords,
    with a RoBERTa fallback to software_description if site_keywords is missing.

    Args:
        paper_keywords: comma‐separated keywords from the paper.
        site_keywords: comma‐separated keywords from the software site.
        software_description: free‐text description of the software.

    Returns:
        A float in [-1.0, 1.0]:
          - 0.0 if paper_keywords is empty or None.
          - If site_keywords is empty/None: cosine similarity between
            paper_keywords and software_description using RoBERTa.
          - Otherwise: cosine similarity between paper_keywords and
            site_keywords using BERT.
    """
    # 1. Guard: if paper_keywords missing → 0.0
    if not paper_keywords or not paper_keywords.strip():
        return 0.0

    # Normalize by splitting on commas, lowercasing, stripping
    def _normalize_list(s: str) -> list[str]:
        return [kw.strip().lower() for kw in s.split(',') if kw.strip()]

    pk_list = _normalize_list(paper_keywords)

    # 2. If site_keywords missing → fallback to description
    if not site_keywords or not site_keywords.strip():
        # treat the entire keywords list as a single "document"
        if not software_description or not software_description.strip():
            return 0.0
        # If software_description is empty, return 0.0
        text1 = [" ".join(pk_list)]
        text2 = [software_description.strip()]
        return _encode_and_cosine(_ROBERTA_MODEL, text1, text2)

    # 3. Both present → compare keyword‐lists via BERT
    sk_list = _normalize_list(site_keywords)
    text1 = [" ".join(pk_list)]
    text2 = [" ".join(sk_list)]
    return _encode_and_cosine(_BERT_MODEL, text1, text2)

def paragraph_description_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using RoBERTa sentence embeddings.
    
    Args:
        text1: e.g. a paragraph from a paper.
        text2: e.g. the software's description.
    
    Returns:
        A float in [-1.0, 1.0], where higher means more semantically similar.
    """
    if not text1 or not (text1 := text1.strip()):
        return 0.0
    if not text2 or not (text2 := text2.strip()):
        return 0.0
    # 1. Encode both texts (batch is faster than one by one)
    embeddings = _ROBERTA_MODEL.encode([text1, text2], convert_to_tensor=False)
    # 2. Compute cosine similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim)

# Define common software affixes to strip out
COMMON_AFFIXES = {
    "pro", "enterprise", "suite", "edition", "cloud", "desktop",
    "online", "server", "client", "portable", "lite"
}

def normalize_software_name(name: str) -> str:
    """
    Normalize a software name by:
      1. Lowercasing
      2. Removing version numbers (e.g., v2.0, 2021)
      3. Stripping common affixes (e.g., Pro, Suite)
      4. Removing punctuation
      5. Collapsing whitespace
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
    """
    Compute Jaro–Winkler similarity between two software names
    after normalization.
    
    Returns:
        A float between 0.0 and 1.0.
    """
    n1 = normalize_software_name(name1)
    n2 = normalize_software_name(name2)
    return textdistance.jaro_winkler(n1, n2)



def normalize_author_name(name: str) -> str:
    """
    Normalize an author name by:
      1. Flipping "Last, First Middle" → "First Middle Last"
      2. Expanding initials (e.g. "J.K." → "J K")
      3. Lowercasing
      4. Removing anything but letters and spaces
      5. Collapsing multiple spaces
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
    """
    Compute the Jaro–Winkler similarity between two author names
    after normalization.

    Returns:
        A float between 0.0 and 1.0 (1.0 = exact match).
    If either name is None or empty (after stripping), returns 0.0 immediately.
    """
    # Treat None as missing
    if not name1 or not name2:
        return 0.0
    # After stripping, if either is empty
    if not name1.strip() or not name2.strip():
        return 0.0
    n1 = normalize_author_name(name1)
    n2 = normalize_author_name(name2)
    return textdistance.jaro_winkler(n1, n2)


def compute_similarity_df(df: pd.DataFrame,output_path:str = None) -> pd.DataFrame:
    """
    Given a DataFrame with columns
      id, name, doi, paragraph, authors, field/topic/keywords, url,
      candidate_urls, probability, metadata_name, metadata_authors,
      metadata_keywords, metadata_description

    Returns a new DataFrame containing only rows where metadata_name is non-empty,
    copying all those source columns and adding:

      - name_metric       (software_name_similarity) Jaro Winkler similarity
      - author_metric     (author_name_similarity) Jaro Winkler similarity
      - paragraph_metric  (paragraph_description_similarity) RoBERTa cosine similarity
      - keywords_metric   (keyword_similarity_with_fallback) BERT cosine similarity
    """
    # 1) Filter to rows with a non-empty metadata_name
    mask = df['metadata_name'].notna() & df['metadata_name'].str.strip().astype(bool)
    sub = df.loc[mask].copy()

    # 2) Compute each metric
    sub['name_metric'] = sub.apply(
        lambda r: software_name_similarity(r['name'], r['metadata_name']), axis=1
    )
    sub['author_metric'] = sub.apply(
        lambda r: author_name_similarity(r['authors'], r['metadata_authors']), axis=1
    )
    sub['paragraph_metric'] = sub.apply(
        lambda r: paragraph_description_similarity(r['paragraph'], r['metadata_description']), axis=1
    )
    sub['keywords_metric'] = sub.apply(
        lambda r: keyword_similarity_with_fallback(
            r['field/topic/keywords'],
            r['metadata_keywords'],
            r['metadata_description']
        ),
        axis=1
    )

    # 3) Select and return the requested columns
    cols = [
        'id','name','doi','paragraph','authors','field/topic/keywords',
        'url (ground truth)','candidate_urls','probability (ground truth)',
        'metadata_name','metadata_authors','metadata_keywords','metadata_description',
        'name_metric','author_metric','paragraph_metric','keywords_metric'
    ]
    if output_path:
        # Save to CSV if requested, 
        sub[cols].to_csv(output_path, index=False)
        print(f"📄 Similarity metrics saved to {output_path}")
    
    return sub[cols]
def get_average_min_max(df: pd.DataFrame, output_path: str = None) -> None:
    """
    Compute per-row and overall summary statistics for the similarity metrics in a DataFrame.

    This function:
      1. Adds three new columns to `df`:
         - "average": the row-wise mean of ["name_metric", "author_metric", "paragraph_metric", "keywords_metric"]
         - "min":     the row-wise minimum of those same four metric columns
         - "max":     the row-wise maximum of those same four metric columns
      2. Prints the overall mean, min, and max for each of these seven columns:
         ["name_metric", "author_metric", "paragraph_metric", "keywords_metric",
          "average", "min", "max"]
      3. Optionally writes the augmented DataFrame to a CSV file if `output_path` is provided.

    Args:
        df: A pandas DataFrame containing at least the columns
            "name_metric", "author_metric", "paragraph_metric", and "keywords_metric".
        output_path: Path to save the updated DataFrame as CSV. If None, no file is written.

    Returns:
        None: The DataFrame `df` is modified in place and summary statistics are printed.
    """
    # Calculate average, min, and max for each candidate URL
    df['average'] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric"]].mean(axis=1)
    df["min"] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric"]].min(axis=1)
    df["max"] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric"]].max(axis=1)
    metrics = ["name_metric","author_metric",'paragraph_metric','keywords_metric', "average","min","max"]
# 2) Or, if you prefer one‐by‐one formatting:
    for m in metrics:
        avg = df[m].mean()
        mn  = df[m].min()
        mx  = df[m].max()
        print(f"{m:15s} →  avg: {avg:.4f}   min: {mn:.4f}   max: {mx:.4f}")
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"📄 Average, min, and max metrics saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'id': [1, 2],
        'name': ['Software A', 'Software B'],
        'doi': ['10.1000/xyz123', '10.1000/xyz456'],
        'paragraph': ['This is a description of Software A.', 'This is a description of Software B.'],
        'authors': ['Alice Smith', 'Bob Johnson'],
        'field/topic/keywords': ['AI, ML', 'Data Science'],
        'url': ['http://example.com/a', 'http://example.com/b'],
        'candidate_urls': ['http://example.com/c', 'http://example.com/d'],
        'probability': [1, 0],
        'metadata_name': ['Software A', 'Software B'],
        'metadata_authors': ['Alice Smith', 'Bob Johnson'],
        'metadata_keywords': ['AI, ML', 'Data Science'],
        'metadata_description': ['A software for AI and ML.', 'A software for Data Science.']
    })

    compute_similarity_df(df)