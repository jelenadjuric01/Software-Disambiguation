import os
import time
import json
import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# 1. Load your data
# -----------------------------------------------------------------------------
test_df = pd.read_csv("code/test_LLM.csv")
prev_df = pd.read_csv("binary_llm_results_gemma.csv")


# 3) Merge on the composite key ["name", "doi", "paragraph"].
merged = test_df.merge(
    prev_df[["name", "doi", "candidate_urls", "predicted_label"]],
    on=["name", "doi", "candidate_urls"],
    how="left"
)

# 4) Filter to only those rows where predicted_label is still NaN
df = merged[merged["predicted_label"].isna()].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. Initialize Groq client
# -----------------------------------------------------------------------------
api_key = os.getenv("EVAL_KEY")
if not api_key:
    raise RuntimeError("Please set EVAL_KEY in your environment")
client = Groq(api_key=api_key)

# -----------------------------------------------------------------------------
# 3. Binary prediction function
# -----------------------------------------------------------------------------
def binary_match_with_llm(row, max_retries=2):
    prompt = f"""
You are given information about a software mention in a scientific paper and a candidate URL.

Software: {row['name']}
DOI: {row['doi']}
Context: {row['paragraph']}
Paper authors: {row['authors']}
Programming language from context: {row['language']}
Synonyms: {row['synonyms']}

Candidate URL: {row['candidate_urls']}
→ URL name: {row['metadata_name']}
→ URL authors: {row['metadata_authors']}
→ URL description: {row['metadata_description']}
→ URL programming language: {row['metadata_language']}

Task:
Determine if the candidate URL refers to the same software mentioned in the paper.

Return only a single digit:
- 1 if the URL corresponds to the software
- 0 if it does not
""".strip()

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=1024,
                stop=None,
            )
            reply = response.choices[0].message.content.strip()
            if "1" in reply:
                return 1
            elif "0" in reply:
                return 0
            else:
                print(f"Unrecognized response: {reply}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None  # if all attempts fail

# -----------------------------------------------------------------------------
# 4. Evaluation loop
# -----------------------------------------------------------------------------
def run_binary_evaluation():
    preds = []
    total = len(df)
    print(f"Starting binary evaluation on {total} rows...\n" + "="*60)

    for i, row in df.iterrows():
        print(f"[{i+1}/{total}] Evaluating: {row['name']} | {row['candidate_urls']}")
        pred = binary_match_with_llm(row)
        label = row['true_label']
        preds.append({
            "software": row["name"],
            "doi": row["doi"],
            "candidate_url": row["candidate_urls"],
            "true_label": label,
            "predicted_label": pred,
            "correct": int(pred == label)
        })
        time.sleep(1.5)  # slight rate limiting

    return pd.DataFrame(preds)

# -----------------------------------------------------------------------------
# 5. Compute and print metrics
# -----------------------------------------------------------------------------
def compute_binary_metrics(df_results):
    y_true = df_results["true_label"]
    y_pred = df_results["predicted_label"]

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nEVALUATION METRICS")
    print("=" * 40)
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")

# -----------------------------------------------------------------------------
# 6. Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        results = run_binary_evaluation()
        results.to_csv("binary_llm_results_gemma_new.csv", index=False)
        compute_binary_metrics(results)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during evaluation: {e}")
