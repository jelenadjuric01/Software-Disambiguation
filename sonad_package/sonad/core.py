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
        try:
            return pkg_resources.resource_filename('sonad', 'model.pkl')
        except:
            return Path(__file__).parent / 'model.pkl'

    @staticmethod
    def get_czi_path():
        try:
            return pkg_resources.resource_filename('sonad', 'CZI/synonyms_matrix.csv')
        except:
            return Path(__file__).parent / 'CZI/synonyms_matrix.csv'

    @staticmethod
    def get_synonyms_path():
        try:
            return pkg_resources.resource_filename('sonad', 'json/synonym_dictionary.json')
        except:
            return Path(__file__).parent / 'json/synonym_dictionary.json'
    @staticmethod
    def get_metadata_path():
        try:
            return pkg_resources.resource_filename('sonad', 'json/metadata_cache.json')
        except:
            return Path(__file__).parent / 'json/metadata_cache.json'
    @staticmethod
    def get_candidates_cache_path():
        try:
            return pkg_resources.resource_filename('sonad', 'json/candidate_urls.json')
        except:
            return Path(__file__).parent / 'json/candidate_urls.json'

def process_files(input_path, output_path, folder_path=None, github_token=None):
    """
    Main processing function that handles the entire pipeline
    Args:
        input_path: Path to input CSV file
        output_path: Path where aggregated_groups.csv will be saved
        folder_path: Optional folder path for temporary files (if None, no temp files are saved)
        github_token: GitHub token for API calls
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
    
    print("Loading CZI data...")
    CZI = pd.read_csv(czi_path)
    
    # Processing pipeline
    print("Processing data...")
    get_synonyms_from_file(synonyms_file, input_dataframe, CZI_df=CZI)
    
    print("Finding nearest language for each software...")
    input_dataframe['language'] = input_dataframe.apply(
        lambda row: find_nearest_language_for_softwares(row['paragraph'], row['name']), 
        axis=1
    )
    
    print("Getting authors for each paper...")
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
        metadata_cache_file
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