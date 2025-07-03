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

# SONAD: Software Name Disambiguation

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

**SONAD** (Software Name Disambiguation) is a command-line tool and Python package that links software mentions in scientific papers to their corresponding repository URLs. It leverages NLP, third-party tools like SOMEF, and metadata to resolve software names. It is limited to fetching URLs from GitHub, PyPI and CRAN. Take into account that this is not 100% accurate as it uses a machine learning model trained on data, but it did outperform models llama-3.1-8b-instant, qwen-qwq-32b, gemma2-9b-it and deepseek-r1-distill-llama-70b.

---

## Installation

Install using pip:

```
pip install sonad
```


---

### 🛠 Development Installation

To install the package in development mode using [Poetry](https://python-poetry.org/), follow these steps:

1. **Clone the repository** and navigate to the `sonad_package` folder:

    ```bash
    git clone https://github.com/jelenadjuric01/Software-Disambiguation.git
    cd Software-Disambiguation/sonad_package
    ```

2. **Install the package with Poetry**:

    ```bash
    poetry install
    ```

> **Note:** To ensure the package works correctly in development mode, you must manually download the file `synonym_matrix.csv`, which is available from Zenodo in folder `sonad_package/sonad/CZI/synonym_matrix.csv`:  
> [https://zenodo.org/records/15765003](https://zenodo.org/records/15765003)  
> Once downloaded, place the file inside the following path:  
> `sonad_package/sonad/CZI/synonym_matrix.csv`

---

## Initial Configuration

Before using SONAD, you **must install and configure SOMEF**  
(https://github.com/KnowledgeCaptureAndDiscovery/somef/?tab=readme-ov-file),  
which is used for software metadata extraction.

Follow their installation instructions to make sure `somef` runs correctly on your system.

It is also **strongly recommended to provide a GitHub API token** to avoid rate limits when querying GitHub. You can configure this once using:

```
sonad configure
```

Your token will be saved for future runs.

---

## Requirements

SONAD requires Python 3.10. All dependencies are installed automatically.

Some key libraries:
- pandas
- scikit-learn
- xgboost
- sentence-transformers
- beautifulsoup4
- requests
- SPARQLWrapper
- somef
- textdistance
- lxml
- cloudpickle

---

## Usage

After installation, you can run the main command:

```
sonad process -i <input_file.csv> -o <output_file.csv> [-t <temp_folder>] 
```

### Parameters

- `-i`, `--input` (required): Path to the input CSV file.
- `-o`, `--output` (required): Path where the output CSV will be saved.
- `-t`, `--temp` (optional): Folder where temporary files will be written it the folder is provided.

---

## Input Format

The input CSV must contain the following columns:

- `name`: The software name mentioned in the paper.
- `doi`: The DOI of the paper.
- `paragraph`: The paragraph in which the software is mentioned.

Optionally, it can include:

- `candidate_urls`: A comma-separated list of candidate software URLs that might correspond to the software.

### Example:

```
name,doi,paragraph,candidate_urls
Scikit-learn,10.1000/xyz123,"We used Scikit-learn for classification.","https://github.com/scikit-learn/scikit-learn"
```

---
## Output Format

The output CSV will contain one row for each input mention, with the following columns:
- `name`: The software name from the input.
- `paragraph`: The paragraph where the software was mentioned.
- `doi`: The DOI of the paper in which the software was mentioned.
- `synonyms`: Alternative names or variants of the software name identified during processing.
- `language`: The inferred programming language(s) used by the software, if available.
- `authors`: Names of authors of the paper it they can be fetched from OpenAlex tool.
- `urls`: A comma-separated list of predicted repository or project URLs (e.g., GitHub, PyPI, CRAN).
- `not_urls`: URLs that were considered but rejected during disambiguation (e.g., due to low confidence or irrelevance).

---

### Example:

```
name,paragraph,doi,synonyms,language,authors,urls,not_urls
Scikit-learn,"We used Scikit-learn for classification.",10.1000/xyz123,"scikit learn;sklearn",Python,"Pedregosa et al.","https://github.com/scikit-learn/scikit-learn","https://pypi.org/project/sklearn/"
```

---

## License

MIT License © Jelena Djuric  
https://github.com/jelenadjuric01

---

## Contributions

Feel free to submit issues or pull requests on the GitHub repository:  
https://github.com/jelenadjuric01/Software-Disambiguation
