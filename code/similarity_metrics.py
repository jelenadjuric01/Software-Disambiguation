import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import numpy as np
import pandas as pd
import textdistance

# 1. module import
#    - A lightweight BERT for keyword-vs-keyword
#    - A larger RoBERTa for fallback (keywords vs. description) and description vs. paragraph
_BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
_ROBERTA_MODEL = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

def _encode_and_cosine(model: SentenceTransformer, texts1, texts2) -> float:
    """Encode two lists of texts and return the mean cosine similarity.

    This helper function uses `model` to embed `texts1` and `texts2`,
    computes pairwise cosine similarities, and returns the average
    of the diagonal (i.e., corresponding pairs).

    Args:
        model (SentenceTransformer): Pre-loaded sentence transformer.
        texts1 (List[str]): First batch of text inputs.
        texts2 (List[str]): Second batch of text inputs.

    Returns:
        float: Mean cosine similarity of corresponding embeddings.
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
    """Compute a keyword-based similarity with a description fallback.

    If `site_keywords` is present, compares comma-separated
    `paper_keywords` vs. `site_keywords` using a lightweight BERT model.
    Otherwise falls back to comparing `paper_keywords` vs.
    `software_description` using a larger RoBERTa model.

    Args:
        paper_keywords (Optional[str]): Comma-separated keywords from the paper.
        site_keywords (Optional[str]): Comma-separated keywords from the software site.
        software_description (str): Full text description of the software.

    Returns:
        float:
          - `np.nan` if `paper_keywords` is missing or empty.
          - Cosine similarity ∈ [–1.0, 1.0] computed with the chosen model.
    """
    if pd.isna(paper_keywords):
        return np.nan

    # 2) Coerce to string and strip ALL whitespace
    pk_str = str(paper_keywords).strip()
    # 3) If empty after stripping → 0.0
    if not pk_str:
        return np.nan

    # Normalize by splitting on commas, lowercasing, stripping
    def _normalize_list(s: str) -> list[str]:
        return [kw.strip().lower() for kw in s.split(',') if kw.strip()]

    pk_list = _normalize_list(paper_keywords)

    # 2. If site_keywords missing → fallback to description
    if not site_keywords or pd.isna(site_keywords) or not site_keywords.strip():
        # treat the entire keywords list as a single "document"
        if not software_description or not software_description.strip():
            return np.nan
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
    """Compute semantic similarity between two texts via RoBERTa embeddings.

    Strips and validates `text1` and `text2`, encodes both with the
    RoBERTa model, and returns their cosine similarity.

    Args:
        text1 (str): Arbitrary text (e.g., a paper paragraph).
        text2 (str): Another text (e.g., software description).

    Returns:
        float: Cosine similarity ∈ [–1.0, 1.0], or `np.nan` if either input is empty.
    """
    if not text1 or pd.isna(text1) or not (text1 := text1.strip()):
        return np.nan
    if not text2 or pd.isna(text2) or not (text2 := text2.strip()):
        return np.nan
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
    """Normalize a software name to a canonical lowercase token string.

    Steps:
      1. Lowercase and trim.
      2. Remove version tokens (e.g., "v2.0", "2021").
      3. Strip punctuation.
      4. Remove common affixes (e.g., "Pro", "Suite").
      5. Collapse whitespace.

    Args:
        name (str): Original software name.

    Returns:
        str: Normalized name.
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
    """Compute Jaro–Winkler similarity between software name and list of synonyms.


    Args:
        name1 (str): First software name.
        names List(str): List of synonyms.

    Returns:
        float: Jaro–Winkler similarity ∈ [0.0, 1.0] that is average of the similarities.
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
    """Compute a simple equality-based similarity between two languages.

    Returns 1.0 if both non-empty names match case-insensitively,
    0.0 if both present but differ, or `np.nan` if either is missing.

    Args:
        lang1 (Optional[str]): First language name.
        lang2 (Optional[str]): Second language name.

    Returns:
        float: Similarity score or `np.nan` for missing data.
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


def compute_similarity_df(df: pd.DataFrame,output_path:str = None) -> pd.DataFrame:
    """Compute and collect multiple similarity metrics for each candidate.

    For rows with valid `metadata_name`, computes:
      - `name_metric`       via `software_name_similarity`
      - `author_metric`     via `author_name_similarity`
      - `paragraph_metric`  via `paragraph_description_similarity`
      - `keywords_metric`   via `keyword_similarity_with_fallback`
      - `language_metric`   via `programming_language_similarity`

    Args:
        df (pd.DataFrame): Input with metadata and source columns.
        output_path (str, optional): If given, CSV filepath to save results.

    Returns:
        pd.DataFrame: Subset DataFrame with original columns plus the five metrics.
    """
    for col in ['name_metric','author_metric','paragraph_metric','keywords_metric',"language_metric"]:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Mask of all rows that have valid metadata_name
    valid = (
        df['metadata_name'].notna() &
        df['metadata_name'].astype(str).str.strip().astype(bool)
    )

    # 2) name_metric: only for valid rows where name_metric is still NaN
    nm = valid & df['name_metric'].isna()
    df.loc[nm, 'name_metric'] = df.loc[nm].apply(
        lambda r: software_name_similarity(r['name'], r['metadata_name']),
        axis=1
    )

    # 3) author_metric:
    am = valid & df['author_metric'].isna()
    df.loc[am, 'author_metric'] = df.loc[am].apply(
        lambda r: author_name_similarity(r['authors'], r['metadata_authors']),
        axis=1
    )

    # 4) paragraph_metric: only for rows with metadata_description & NaN
    pm = valid & df['paragraph_metric'].isna()
    df.loc[pm, 'paragraph_metric'] = df.loc[pm].apply(
        lambda r: paragraph_description_similarity(
            r['paragraph'], r['metadata_description']
        ),
        axis=1
    )

    # 5) keywords_metric: only for rows with both metadata_keywords & metadata_description & NaN
   
    km = valid & df['keywords_metric'].isna()
    df.loc[km, 'keywords_metric'] = df.loc[km].apply(
        lambda r: keyword_similarity_with_fallback(
            r['field/topic/keywords'],
            r['metadata_keywords'],
            r['metadata_description']
        ),
        axis=1
    )
    lm = valid & df['language_metric'].isna()
    df.loc[lm, 'language_metric'] = df.loc[lm].apply(
        lambda r: programming_language_similarity(
            r['language'],
            r['metadata_language']
        ),
        axis=1
    )
    # 6) Build the “sub” DataFrame you originally returned
    cols = [
        'id','name','doi','paragraph','authors','field/topic/keywords','language',
        'url (ground truth)','candidate_urls',
        'metadata_name','metadata_authors','metadata_keywords','metadata_description','metadata_language',
        'name_metric','author_metric','paragraph_metric','keywords_metric','language_metric','true_label'
    ]
    # 7) Add the true_label column
    df['true_label'] = [
    int(c in [u.strip() for u in g.split(',')])
    for c, g in zip(df['candidate_urls'], df['url (ground truth)'])
    ] 
    sub = df.loc[valid, cols].copy()

    # 7) Optionally save
    if output_path:
        sub.to_csv(output_path, index=False)
        print(f"📄 Similarity metrics saved to {output_path}")

    return sub
def get_average_min_max(df: pd.DataFrame, output_path: str = None) -> None:
    """Add per-row and overall average/min/max of similarity metrics.

    1. Adds columns `average`, `min`, and `max` across
       ['name_metric', 'author_metric', 'paragraph_metric',
        'keywords_metric', 'language_metric'].
    2. Prints overall statistics for each metric.
    3. Optionally saves the augmented DataFrame to `output_path`.

    Args:
        df (pd.DataFrame): Must contain the five metric columns.
        output_path (str, optional): CSV filepath to write the updated DataFrame.

    Returns:
        None
    """
    # Calculate average, min, and max for each candidate URL
    df['average'] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric",'language_metric']].mean(axis=1,skipna=True)
    df["min"] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric",'language_metric']].min(axis=1)
    df["max"] = df[['name_metric', 'author_metric', 'paragraph_metric',"keywords_metric","language_metric"]].max(axis=1)
    metrics = ["name_metric","author_metric",'paragraph_metric','keywords_metric', "language_metric","average","min","max"]
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
        'language': ['Python', 'R'],
        'url (ground truth)': ['http://example.com/a', 'http://example.com/b'],
        'candidate_urls': ['http://example.com/c', 'http://example.com/d'],
        'probability (ground truth)': [1, 0],
        'metadata_name': ['Software A', 'Software B'],
        'metadata_authors': ['Alice Smith', 'Bob Johnson'],
        'metadata_keywords': ['AI, ML', 'Data Science'],
        'metadata_description': ['A software for AI and ML.', 'A software for Data Science.'],
        'metadata_language': ['Python', 'R'],
    })

    print(compute_similarity_df(df))