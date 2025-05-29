# Software-Disambiguation

Tackling the problem of software disambiguation in texts using Python and semi-supervised machine learning.

The idea is to create a program that will perform software name disambiguation based on metadata beyond just the name itself. This addresses two core challenges:
- **Homonyms**: Software with the same name that refer to different tools
- **Synonyms**: Software with slightly different names or typos that refer to the same tool

---

## 🔍 Goal

Given a CSV with software mentions and context (name, paragraph, DOI, etc.), the tool will:
- Retrieve synonyms and candidate URLs
- Fetch metadata for each URL
- Compute similarity scores
- Use a trained model to classify which URLs are referred to in the paper
- Return a final CSV indicating which candidate URLs are relevant

---

## 📁 Repository Structure

This repository includes both the **final program** and the **research pipeline** used to build it.

### ✔️ Final Program

- `demo.ipynb` — main notebook that runs the entire disambiguation process
- `demo.zip` — archive containing:
  - `model.pkl`, `models.py`, `preprocessing.py`
  - Input/output sample files
  - `CZI/synonyms_matrix.csv`
  - Optional `json/` cache folder
- donwloading `demo.zip`


### 📊 Research and Feature Engineering

These folders were used to test and evaluate different steps of the pipeline:
- `corpus/` — raw data preparation and cleaning
- `code/` — full pipeline scripts used outside the notebook

You can explore the progression from feature design to the final tool in the notebook.

---

## ⚙️ Setup Instructions

### 1. Python Environment

Make sure you have **Python 3.10+** installed.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
pandas
numpy
scikit-learn
transformers
torch
requests
openai
tqdm
textdistance
cloudpickle
sentence-transformers
notebook
ipython
```

---

## 🔐 GitHub Token Required

To search for candidate repositories on GitHub, you must set a GitHub personal access token.

See how to generate one: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

**Set your token like this:**

- **macOS/Linux:**
  ```bash
  export GITHUB_TOKEN=your_token_here
  ```

- **Windows:**
  ```cmd
  set GITHUB_TOKEN=your_token_here
  ```

---

## 🧩 SOMEF Dependency

To extract repository metadata, clone and set up [SOMEF](https://github.com/KnowledgeCaptureAndDiscovery/somef).

```bash
git clone https://github.com/KnowledgeCaptureAndDiscovery/somef.git
```

In the notebook, set `somef_path` to the local path of the SOMEF repo.

---

## 🚀 Running the Notebook

Use either:
1. A CSV file with `name`, `doi`, `paragraph`, and optional `candidate_urls`
2. Manual input prompts in the notebook

In the **first code cell**, configure:
- `input_file`, `model_path`, `model_input_path`
- `output_path_aggregated_groups`, `somef_path`
- Optional: paths for intermediate steps (set to `None` to disable)

⚠️ If the `json/` folder (with cached synonyms, candidates, metadata) is not present, the notebook will regenerate it and **you must provide writable paths** to store those JSON files.

---

## ✅ Output

Final results are saved to:
- `aggregated_groups.csv`: input + predicted relevant (`url`) and irrelevant (`not url`) URLs
