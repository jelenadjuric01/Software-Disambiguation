# SONAD: Software Name Disambiguation

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

**SONAD** (Software Name Disambiguation) is a command-line tool and Python package that links software mentions in scientific papers to their corresponding metadata and repository URLs. It leverages NLP, third-party tools like SOMEF, and GitHub metadata to resolve software names accurately.

---

## Installation

Clone the repository and install using pip:

```
pip install sonad
```

For development mode (auto-refreshes when you edit the code):

```
pip install -e .
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

SONAD requires Python 3.9. All dependencies are installed automatically.

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


## License

MIT License © Jelena Djuric  
https://github.com/jelenadjuric01

---

## Contributions

Feel free to submit issues or pull requests on the GitHub repository:  
https://github.com/jelenadjuric01/Software-Disambiguation
