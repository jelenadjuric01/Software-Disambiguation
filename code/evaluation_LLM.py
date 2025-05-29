import os
import re
import time
import json
from typing import List, Tuple

import pandas as pd
from groq import Groq

# -----------------------------------------------------------------------------
# 1. Load your data
# -----------------------------------------------------------------------------
EXCEL_PATH = "D:\MASTER\TMF\Software-Disambiguation\corpus\corpus_v3_sampled_LLM.xlsx"
df = pd.read_excel(EXCEL_PATH)
df["ground_truth"] = (
    df["url (ground truth)"]
    .fillna("")
    .apply(lambda s: [u.strip() for u in s.split(",") if u.strip()])
)

# -----------------------------------------------------------------------------
# 2. Initialize Groq client
# -----------------------------------------------------------------------------
api_key = os.getenv("EVAL_KEY")
if not api_key:
    raise RuntimeError("Please set EVAL_KEY in your environment")
client = Groq(api_key=api_key)

# -----------------------------------------------------------------------------
# 3. Groq wrapper (JSON-forced) - FIXED VERSION
# -----------------------------------------------------------------------------
def predict_with_groq(name: str, doi: str, context: str,max_retries: int = 2) -> List[str]:
    prompt = f"""
Given the software name, DOI of a paper in which it was mentioned, and the surrounding paragraph, find *all* URLs the software refers to.

Software: {name}
DOI: {doi}
Context: {context}

Instructions:
- Include only URLs starting with http:// or https://
- Return only URLs from GitHub, CRAN and PyPI
- If no URLs found, return: {{ "urls": [] }}
- Do not include any explanation or additional text
- Only return the JSON object
""".strip()

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="compound-beta",  # More reliable model for JSON
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=1024,   # Shorter to discourage explanations
                response_format={"type": "json_object"},
                stop=None,
            )
            
            raw_content = completion.choices[0].message.content
            
            if not raw_content or raw_content.strip() == "":
                print(f"  Attempt {attempt + 1}: Empty response")
                continue
            
            # Try to parse JSON
            try:
                parsed = json.loads(raw_content)
                
                # Extract URLs from various possible formats
                urls = []
                if isinstance(parsed, dict):
                    # Look for URLs in common keys
                    for key in ['urls', 'links', 'websites', 'references', 'data', 'result']:
                        if key in parsed and isinstance(parsed[key], list):
                            urls = [str(url).strip() for url in parsed[key] if url]
                            break
                    
                    # If no recognized key, try any list value
                    if not urls:
                        for value in parsed.values():
                            if isinstance(value, list):
                                urls = [str(url).strip() for url in value if url]
                                break
                elif isinstance(parsed, list):
                    urls = [str(url).strip() for url in parsed if url]
                
                # Filter to valid URLs and remove duplicates
                valid_urls = []
                seen = set()
                for url in urls:
                    # Clean up common issues
                    url = url.rstrip('.,;')  # Remove trailing punctuation
                    if url.startswith(('http://', 'https://')) and url not in seen:
                        valid_urls.append(url)
                        seen.add(url)
                
                print(f"  Attempt {attempt + 1}: Success - found {len(valid_urls)} URLs")
                return valid_urls
                
            except json.JSONDecodeError as e:
                print(f"  Attempt {attempt + 1}: JSON error - {e}")
                
                # Fallback: extract URLs using regex
                url_pattern = r'https?://[^\s\'"<>\]\},]+'
                urls = re.findall(url_pattern, raw_content)
                if urls:
                    # Clean and deduplicate
                    clean_urls = []
                    seen = set()
                    for url in urls:
                        url = url.rstrip('.,;')
                        if url not in seen:
                            clean_urls.append(url)
                            seen.add(url)
                    print(f"  Attempt {attempt + 1}: Fallback extracted {len(clean_urls)} URLs")
                    return clean_urls
                
                if attempt < max_retries:
                    print(f"  Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: API error - {e}")
            if attempt < max_retries:
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
                continue
    
    print(f"  All attempts failed")
    return []

# -----------------------------------------------------------------------------
# 4. Precision/Recall/F1 with better URL matching
# -----------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Normalize URL for comparison"""
    url = url.lower().strip()
    # Remove trailing slashes and common suffixes
    url = re.sub(r'[/\.]*$', '', url)
    # Remove www prefix for comparison
    url = re.sub(r'^https?://(www\.)?', 'https://', url)
    return url

def compute_prf(pred: List[str], truth: List[str]) -> Tuple[float, float, float]:
    # Normalize URLs for comparison
    pred_normalized = {normalize_url(url) for url in pred}
    truth_normalized = {normalize_url(url) for url in truth}
    
    tp = len(pred_normalized & truth_normalized)
    fp = len(pred_normalized - truth_normalized)
    fn = len(truth_normalized - pred_normalized)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    return precision, recall, f1

def normalize_cran_urls(urls):
    """
    Given a list of URLs, normalize any CRAN package URLs in various forms to
    the canonical `https://cran.r-project.org/package=<pkg>` form, then dedupe
    while preserving order.
    
    Handles:
      - https://cran.r-project.org/web/packages/<pkg>/index.html
      - https://cran.r-project.org/web/packages/<pkg>/
      - https://cran.r-project.org/package=<pkg>  (or with leading `/?package=`)
    
    Args:
        urls (list of str): input list of URLs.
        
    Returns:
        list of str: normalized and deduplicated URLs.
    """
    normalized = []
    seen = set()
    
    # match /web/packages/<pkg>/index.html  OR  /web/packages/<pkg>/
    webpkg_re = re.compile(
        r'^https?://cran\.r-project\.org/web/packages/([^/]+)(?:/(?:index\.html)?)?$',
        re.IGNORECASE
    )
    # match both /package=<pkg> and /?package=<pkg>
    pack_re = re.compile(
        r'^https?://cran\.r-project\.org/(?:\?package=|package=)([^&/]+)',
        re.IGNORECASE
    )
    
    for url in urls:
        u = url.strip()
        
        m = webpkg_re.match(u)
        if m:
            # e.g. /web/packages/psych/  or .../index.html
            pkg = m.group(1)
            canon = f"https://cran.r-project.org/package={pkg}"
        else:
            m2 = pack_re.match(u)
            if m2:
                # already in package= form
                pkg = m2.group(1)
                canon = f"https://cran.r-project.org/package={pkg}"
            else:
                # other URLs left unchanged
                canon = u
        
        if canon not in seen:
            seen.add(canon)
            normalized.append(canon)
    
    return normalized
# -----------------------------------------------------------------------------
# 5. Run evaluation loop with progress tracking
# -----------------------------------------------------------------------------
def run_evaluation():
    records = []
    total_rows = len(df)
    print(f"Starting evaluation of {total_rows} software entries...")
    print("=" * 80)

    for idx, row in df.iterrows():
        name, doi, ctx = row["name"], row["doi"], row["paragraph"]
        truth = row["ground_truth"]

        print(f"\n[{idx+1}/{total_rows}] Processing: {name}")
        print(f"  DOI: {doi}")
        print(f"  Ground truth: {truth}")

        try:
            pred = predict_with_groq(name, doi, ctx)
            if pred:
                # Normalize CRAN URLs if any are found
                pred = normalize_cran_urls(pred)
            print(f"  Predicted: {pred}")
            
            p, r, f1 = compute_prf(pred, truth)
            print(f"  Metrics: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

            records.append({
                "row": idx,
                "software": name, 
                "doi": doi,
                "precision": p, 
                "recall": r, 
                "f1": f1, 
                "predicted": pred,
                "ground_truth": truth,
                "predicted_count": len(pred),
                "truth_count": len(truth)
            })
            
        except Exception as e:
            print(f"  Unexpected error: {e}")
            records.append({
                "row": idx,
                "software": name, 
                "doi": doi,
                "precision": 0.0, 
                "recall": 0.0, 
                "f1": 0.0, 
                "predicted": [],
                "ground_truth": truth,
                "predicted_count": 0,
                "truth_count": len(truth)
            })

        # Rate limiting - reduced from 4 to 2 seconds
        if idx < total_rows - 1:  # Don't wait after last item
            print("  Waiting 2 seconds...")
            time.sleep(2)

    return records

# -----------------------------------------------------------------------------
# 6. Enhanced results analysis and saving
# -----------------------------------------------------------------------------
def analyze_and_save_results(records):
    if not records:
        print("No successful evaluations to summarize.")
        return

    results_df = pd.DataFrame(records)
    
    # Calculate averages
    summary = results_df.agg({
        "precision": "mean",
        "recall": "mean",
        "f1": "mean"
    })
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total entries evaluated: {len(results_df)}")
    print(f"Average Precision: {summary['precision']:.3f}")
    print(f"Average Recall:    {summary['recall']:.3f}")
    print(f"Average F1 Score:  {summary['f1']:.3f}")
    
    # Additional statistics
    perfect_matches = len(results_df[results_df['f1'] == 1.0])
    zero_scores = len(results_df[results_df['f1'] == 0.0])
    
    print(f"\nPerfect matches (F1=1.0): {perfect_matches}")
    print(f"Zero scores (F1=0.0):     {zero_scores}")
    print(f"Partial matches:          {len(results_df) - perfect_matches - zero_scores}")
    
    # Save detailed results
    output_path = f"evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Print per-software summary (top 10 and bottom 10 by F1)
    sorted_results = results_df.sort_values('f1', ascending=False)
    
    print(f"\nTOP 10 PERFORMERS:")
    print("-" * 70)
    print(f"{'Software':<25} | {'P':<5} | {'R':<5} | {'F1':<5} | {'Pred':<4} | {'Truth':<5}")
    print("-" * 70)
    for _, record in sorted_results.head(10).iterrows():
        print(f"{record['software'][:24]:<25} | {record['precision']:.3f} | {record['recall']:.3f} | {record['f1']:.3f} | {record['predicted_count']:<4} | {record['truth_count']:<5}")
    
    if len(sorted_results) > 10:
        print(f"\nBOTTOM 10 PERFORMERS:")
        print("-" * 70)
        print(f"{'Software':<25} | {'P':<5} | {'R':<5} | {'F1':<5} | {'Pred':<4} | {'Truth':<5}")
        print("-" * 70)
        for _, record in sorted_results.tail(10).iterrows():
            print(f"{record['software'][:24]:<25} | {record['precision']:.3f} | {record['recall']:.3f} | {record['f1']:.3f} | {record['predicted_count']:<4} | {record['truth_count']:<5}")

# -----------------------------------------------------------------------------
# 7. Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        records = run_evaluation()
        analyze_and_save_results(records)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()