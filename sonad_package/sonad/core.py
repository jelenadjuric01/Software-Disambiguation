import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
from .preprocessing import (
    find_nearest_language_for_softwares,
    get_authors,
    get_synonyms_from_file,
    make_pairs,
    dictionary_with_candidate_metadata,
    add_metadata,
    aggregate_group,
    get_candidate_urls,
    compute_similarity_test
)
from .models import make_model, get_preprocessing_pipeline
import pkg_resources

class PackageResources:
    @staticmethod
    def get_model_path():
        """
        Locate the serialized model file within package resources.

        Attempts to return the path to "model.pkl" bundled in the "sonad" package.
        Falls back to a copy in the same directory as this module if pkg_resources
        lookup fails.

        Returns:
            Union[str, Path]:
                Filesystem path to the model pickle file.
        """
        try:
            return pkg_resources.resource_filename('sonad', 'model.pkl')
        except:
            return Path(__file__).parent / 'model.pkl'

    @staticmethod
    def get_czi_path():
        """
        Locate the CZI synonyms CSV within package resources.

        Attempts to return the path to "CZI/synonyms_matrix.csv" bundled in the
        "sonad" package. Falls back to a copy in the module directory.

        Returns:
            Union[str, Path]:
                Filesystem path to the CZI synonyms CSV.
        """
        try:
            return pkg_resources.resource_filename('sonad', 'CZI/synonyms_matrix.csv')
        except:
            return Path(__file__).parent / 'CZI/synonyms_matrix.csv'

    @staticmethod
    def get_synonyms_path():
        """
        Locate the precomputed synonym dictionary JSON within package resources.

        Attempts to return the path to "json/synonym_dictionary.json" in the
        "sonad" package. Falls back to the local module directory if needed.

        Returns:
            Union[str, Path]:
                Filesystem path to the synonym dictionary JSON.
        """
        try:
            return pkg_resources.resource_filename('sonad', 'json/synonym_dictionary.json')
        except:
            return Path(__file__).parent / 'json/synonym_dictionary.json'
    @staticmethod
    def get_metadata_path():
        """
        Locate the metadata cache JSON within package resources.

        Attempts to return the path to "json/metadata_cache.json" bundled in the
        "sonad" package. Falls back to the local module directory.

        Returns:
            Union[str, Path]:
                Filesystem path to the metadata cache file.
        """
        try:
            return pkg_resources.resource_filename('sonad', 'json/metadata_cache.json')
        except:
            return Path(__file__).parent / 'json/metadata_cache.json'
    @staticmethod
    def get_candidates_cache_path():
        """
        Locate the candidate URLs cache JSON within package resources.

        Attempts to return the path to "json/candidate_urls.json" in the
        "sonad" package. Falls back to the module directory if lookup fails.

        Returns:
            Union[str, Path]:
                Filesystem path to the candidate URLs cache.
        """
        try:
            return pkg_resources.resource_filename('sonad', 'json/candidate_urls.json')
        except:
            return Path(__file__).parent / 'json/candidate_urls.json'

def process_files(input_path, output_path, folder_path=None, github_token=None):
    """
    Run the full end-to-end pipeline on an input CSV of paper‐software mentions.

    1. Reads the input CSV into a DataFrame.
    2. Loads CZI synonym data.
    3. Populates synonyms, nearest programming language, and paper authors.
    4. Fetches and caches candidate software URLs.
    5. Extracts metadata for each candidate URL.
    6. Explodes to (paper, URL) pairs and adds metadata fields.
    7. Computes similarity metrics and writes intermediate CSVs if requested.
    8. Loads a scikit‐learn model from disk, makes predictions, and appends them.
    9. Aggregates per‐paper results and writes the final output CSV.

    Parameters:
        input_path (Union[str, Path]):
            Path to the input CSV file containing columns like 'name', 'paragraph', 'doi'.
        output_path (Union[str, Path]):
            Path where the final aggregated output CSV will be saved.
        folder_path (Optional[Union[str, Path]]):
            If provided, intermediate CSVs and JSON caches will be written under this
            directory in a "temp" subfolder; otherwise, no intermediates are persisted.
        github_token (Optional[str]):
            Personal access token to pass to GitHub API calls (for rate-limit safety).

    Returns:
        None

    Side Effects:
        - Prints progress messages to stdout.
        - Reads and writes multiple CSV and JSON files:
          • CZI synonyms, candidate caches, metadata caches
          • Exploded pairs, intermediate similarity tables
        - Loads a pickled model via cloudpickle and writes final predictions.

    Raises:
        FileNotFoundError:
            If the input CSV does not exist.
        pd.errors.EmptyDataError:
            If the input CSV is malformed or empty.
        Exception:
            Propagates any errors from downstream calls (e.g., HTTP failures,
            JSON parsing errors, subprocess errors in metadata extraction).
    """
    # Set up paths
    package_dir = Path(__file__).parent
    model_path = PackageResources.get_model_path()
    czi_path = PackageResources.get_czi_path()
    synonyms_file = PackageResources.get_synonyms_path()
    candidates_cache_file = PackageResources.get_candidates_cache_path()
    metadata_cache_file = PackageResources.get_metadata_path()
    # Initialize paths for temp files (only used if folder_path is provided)
    output_file_corpus = None
    output_path_pairs = None
    output_path_updated_with_metadata = None
    output_path_similarities = None
    output_path_model_input = None
    
    if folder_path is not None:
        folder_path = Path(folder_path)
        folder_path.mkdir(exist_ok=True)
        
        # Paths for intermediate files
        temp_dir = folder_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        json_dir = folder_path / "json"
        json_dir.mkdir(exist_ok=True)
        
        output_file_corpus = temp_dir / "corpus_with_candidates.csv"
        output_path_pairs = temp_dir / "pairs.csv"
        output_path_updated_with_metadata = temp_dir / "updated_with_metadata.csv"
        output_path_similarities = temp_dir / "similarities.csv"
        output_path_model_input = temp_dir / "model_input.csv"
    
    # Load input data
    input_dataframe = pd.read_csv(input_path)
    
    print("Loading CZI data for synonym extraction...")
    CZI = pd.read_csv(czi_path)
    
    # Processing pipeline
    print("Processing  input data...")
    get_synonyms_from_file(synonyms_file, input_dataframe, CZI_df=CZI)

    print("Finding nearest language for each software from the surrounding paragraph...")
    input_dataframe['language'] = input_dataframe.apply(
        lambda row: find_nearest_language_for_softwares(row['paragraph'], row['name']), 
        axis=1
    )
    
    print("Getting authors for each paper using openAlex tool...")
    results = input_dataframe['doi'].apply(get_authors)
    input_dataframe['authors'] = results.apply(
        lambda x: ','.join(x.get('authors', [])) if isinstance(x, dict) else ''
    )
    
    input_dataframe = get_candidate_urls(
        input_dataframe, 
        candidates_cache_file,
        github_token=github_token
    )
    input_dataframe.fillna(value=np.nan, inplace=True)
    input_dataframe = input_dataframe.infer_objects(copy=False)

    if output_file_corpus is not None:
        input_dataframe.to_csv(output_file_corpus, index=False)
    
    metadata_cache = dictionary_with_candidate_metadata(
        input_dataframe, 
        metadata_cache_file,
        github_token=github_token
    )
    
    input_dataframe = make_pairs(
        input_dataframe, 
        output_path_pairs if folder_path is not None else None
    )
    
    add_metadata(
        input_dataframe, 
        metadata_cache, 
        output_path_updated_with_metadata if folder_path is not None else None
    )
    
    input_dataframe = compute_similarity_test(
        input_dataframe, 
        output_path_similarities if folder_path is not None else None
    )
    
    # Prepare model input (in memory only)
    model_input = input_dataframe[['name_metric', 'paragraph_metric', 'language_metric', 
                                 'synonym_metric', 'author_metric']].copy()
    if folder_path is not None:
        model_input.to_csv(output_path_model_input, index=False)
    print("Predicting with the model...")
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    
    predictions = model.predict(model_input)
    input_dataframe['prediction'] = predictions
    
    if output_path_similarities is not None:
        input_dataframe.to_csv(output_path_similarities, index=False)
    
    grouped = input_dataframe.groupby(['name', 'paragraph', 'doi']).apply(
        aggregate_group, include_groups=True
    ).reset_index()
    
    grouped.to_csv(output_path, index=False)
    print(f"Processing complete. Output saved to {output_path}")