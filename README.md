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

### ✔️ Final Program

install using
pip install sonad

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


## 📁 Repository Structure

code/ - Research pipeline implementation containing the complete machine learning workflow: data preprocessing (preprocessing_corpus.py), candidate URL fetching (fetch_candidates.py, fetching_metadata_from_candidate_url.py), feature engineering (feature_extraction.py), similarity calculations (similarity_metrics.py), model training and evaluation (models.py, evaluation.py, evaluation_LLM.py), hyperparameter optimization (fine_tuning.py), and deployment scripts (deployment.py).

demo/ - Testing environment with sample data and temporary processing folders (CZI/, json/, somef_temp/, temp/) used for validating the final SONAD tool, including test input files (input.csv), trained model artifacts (model.pkl), core processing logic (core.py), data preprocessing (preprocessing.py), model implementations (models.py), and aggregated results (aggregated_groups.csv).

research/code/ - Data preparation and corpus development scripts including DOI integration (adding_doi.ipynb), corpus expansion (appending_corpus.ipynb), metadata collection for author and keyword extraction (fetching_authors_and_keywords.ipynb), synonym generation (fetching_synonyms.ipynb), candidate URL discovery (fetching_url_cantidates.ipynb), website filtering (removing_websites.ipynb), dataset sampling strategies (sampling_dataset.ipynb), and synonym dictionary creation (synonym_dictionary.json).

research/corpus/ - Evolution of the training dataset across multiple versions, starting from initial benchmark data (corpus_v1.xlsx) through progressive refinements incorporating additional data sources, improved ground truth validation, enhanced candidate URL fetching, and methodological improvements (corpus_v2.xlsx through corpus_v3_16.xlsx), with each version documenting specific changes in data processing, similarity calculations, and model training approaches as detailed in the versions file.

research/CZI_sampled/ - Curated samples from the CZI dataset organized by software repository type:

CRAN/ - R package samples including complete cleaned CZI extracts (cran_from_CZI.csv), top 10 R packages (cran_sampled_top_10.csv), and general samples (cran_sampled.csv)
GitHub/ - GitHub repository samples with original CZI data (github_from_CZI.csv), cleaned samples (github_sample_cleaned.csv), and processed samples (github_sampled.csv)
PyPI/ - Python package samples including CZI cleaned extracts (pypi_from_CZI.csv), top 10 Python packages (pypi_sampled_top_10.csv), and general samples (pypi_sampled.csv)
testing/ - Validation datasets with large-scale test samples (CZI_sampled_big.csv), model predictions across different versions (CZI_test_predictions_v18.csv, CZI_test_predictions_v19_light.csv), and comprehensive test results (CZI_test_big_predictions.csv, CZI_test.csv)

research/evaluation_llm/ - Comparative evaluation results against large language models, containing binary classification performance of SONAD versus state-of-the-art LLMs including DeepSeek (binary_llm_results_deepseek_stacked.csv), Gemma (binary_llm_results_gemma_stacked.csv), Llama (binary_llm_results_llama_stacked.csv), and Qwen (binary_llm_results_qwen_stacked.csv), along with the test dataset used for LLM benchmarking (test_LLM.csv).

research/temp/ - Temporary processing files and caches used during model development:

candidate_urls/ - Cached candidate URL discovery results across different corpus versions (candidate_urls_v3_13.json through candidate_urls_v3_16.json and candidate_urls.json) to avoid redundant web scraping during iterative development
metadata_caches/ - Cached software metadata extraction results from SOMEF and other tools across corpus versions (metadata_cache_v3_3.json through metadata_cache_v3_13.json and metadata_cache.json) to optimize processing time during feature engineering and model training -v3/ through v3.16/ - Systematic progression of model development across 17 versions, each containing:
model_input.csv - Core training dataset for machine learning model (present in every version)
pairs.csv - Software mention and candidate URL pairs (optional, available in some versions)
similarities.csv - Computed similarity metrics between mentions and candidates (optional, version-dependent)
updated_with_metadata_file.csv - Enhanced dataset with extracted software metadata (optional, available in select versions)
model_input_no_keywords.csv - Alternative training dataset excluding keyword features (optional)

Each version represents iterative improvements in data processing, feature engineering, similarity calculations, and model architecture as documented in the versions file. The consistent presence of model_input.csv across all versions enables reproducible comparison of model performance, while optional files reflect methodological experiments and refinements specific to each development iteration.

research/results/ - Model evaluation and analysis outputs containing:

dense_correlation_matrix.png - Visualization of feature correlations showing relationships between extracted similarity metrics and metadata features
feature_extraction.txt - Log file documenting the feature engineering process and extracted characteristics from software mentions and candidate URLs
statistics_by_versions.xlsx - Comprehensive performance metrics and statistical analysis across all model versions, enabling comparison of different methodological approaches
tuning_results.csv - Hyperparameter optimization results showing the best parameter configurations for machine learning models used in software disambiguation

sonad_package/ - Production-ready Python package distribution containing:

sonad/ - Main package source code with core modules:
__init__.py - Package initialization and version management
cli.py - Command-line interface for the sonad tool
config.py - Configuration management including GitHub API token setup
core.py - Core disambiguation logic and processing pipeline
model.pkl - Pre-trained machine learning model for software disambiguation
models.py - Model loading and prediction functionality
preprocessing.py - Data preprocessing and feature extraction utilities
Support folders (pycache/, CZI/, json/, someftemp/) - Runtime caches and temporary processing directories
tests/ - Unit tests and package validation (test_import.py)
LICENSE - MIT license for package distribution
MANIFEST.in - Package manifest specifying included files
poetry.lock - Dependency lock file ensuring reproducible installations
pyproject.toml - Modern Python package configuration and dependencies
README.md - Package documentation and usage instructions

LICENSE - MIT license file granting permission to use, modify, and distribute the SONAD software package.

README.md - Main documentation file describing SONAD (Software Name Disambiguation), a Python tool that uses machine learning to link software mentions in scientific papers to their corresponding GitHub, PyPI, or CRAN repository URLs.

input.csv - Sample input file demonstrating the required CSV format with columns for software name, DOI, paragraph context, and optional candidate URLs.

input_not_CZI.csv - Alternative sample input file showing the standard format without ground truth column, used for testing the disambiguation tool.

output.csv - Example output file showing the tool's results including identified synonyms, programming language, authors, confirmed URLs, and rejected candidate URLs.

versions - Development log documenting the evolution of the corpus and methodology across 19 versions, detailing changes in data sources, feature engineering, similarity calculations, and model improvements from initial benchmark data through various refinements of the machine learning pipeline.

---

## License

MIT License © Jelena Djuric  
https://github.com/jelenadjuric01

---

## Contributions

Feel free to submit issues or pull requests on the GitHub repository:  
https://github.com/jelenadjuric01/Software-Disambiguation
