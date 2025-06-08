import sys
import pandas as pd

import numpy as np
import cloudpickle


from preprocessing import find_nearest_language_for_softwares,get_authors,get_synonyms_from_file, make_pairs, dictionary_with_candidate_metadata, add_metadata,aggregate_group,get_candidate_urls,compute_similarity_test
from models import make_model, get_preprocessing_pipeline




#Add the path to the input file (optional)
input_file = "./input.csv"
if input_file is None or input_file == "":
    name = input("Enter the software mention: ")
    if name == "":
        print("No software mention provided. Exiting.")
        sys.exit(1)
    paragraph = input("Enter the paragraph: ")
    if paragraph == "":
        print("No paragraph provided. Exiting.")
        sys.exit(1)
    doi = input("Enter the DOI: ")
    if doi == "":
        print("No DOI provided. Exiting.")
        sys.exit(1)
    candidate_urls = input("Enter the candidate URLs (comma-separated, optional): ")
    input_dataframe = pd.DataFrame({
        'name': [name],
        'paragraph': [paragraph],
        'doi': [doi],
        'candidate_urls': [candidate_urls]
    })
else:
    input_dataframe = pd.read_csv(input_file, delimiter = ';')
# Add the path to the output file for file with added languages, synonyms, authors and candidate URLs (optional)
output_file_corpus = './temp/corpus_with_candidates.csv'
# Add the path to the output file for file with pairs of software names with candidate URLs (optional)
output_path_pairs = "./temp/pairs.csv"
# Add the path to the output file for file with added metadata (optional)
output_path_updated_with_metadata = "./temp/updated_with_metadata.csv"
# Add the path to the output file for file with calculated similarities (optional)
output_path_similarities = "./temp/similarities.csv"
#Add the path to the model
model_path = "./model.pkl"
if model_path is None or model_path == "":
    model_path = "./model.pkl"
# Add the path to the output file for file with model input
model_input_path = "./model_input.csv"
if model_input_path is None or model_input_path == "":
    model_input_path = "./model_input.csv"
# Add the path to the output file with predictions (optional)
output_path_predictions = "./temp/predictions.csv"
# Add the path to the output file with aggregated groups)
output_path_aggregated_groups = "./aggregated_groups.csv"
if output_path_aggregated_groups is None or output_path_aggregated_groups == "":
    output_path_aggregated_groups = "./aggregated_groups.csv"



candidates_cache_file = "./json/candidate_urls.json"
synonyms_file = "./json/synonym_dictionary.json"
metadata_cache_file = "./json/metadata_cache.json"

print("Loading CZI data...")
CZI = pd.read_csv("./CZI/synonyms_matrix.csv")



# Get the synonyms from the file
get_synonyms_from_file(synonyms_file, input_dataframe,CZI_df=CZI)
# Find the nearest language for each software
print("Finding nearest language for each software...")
input_dataframe['language'] = input_dataframe.apply(
    lambda row: find_nearest_language_for_softwares(row['paragraph'], row['name']), axis=1
)
# Get authors for each DOI
print("Getting authors for each paper...")
results = input_dataframe['doi'].apply(get_authors)
input_dataframe['authors'] = results.apply(lambda x: ','.join(x.get('authors', [])) if isinstance(x, dict) else '')
# Get candidate URLs for each software
input_dataframe=get_candidate_urls(input_dataframe, candidates_cache_file)
#Fill all missing values with Nan
input_dataframe.fillna(value=np.nan, inplace=True)
# Save the updated DataFrame to a new CSV file (optional)
if output_file_corpus is not None and output_file_corpus != "":
    input_dataframe.to_csv(output_file_corpus, index=False)

# Create a preprocessing pipeline
metadata_cache = dictionary_with_candidate_metadata(input_dataframe, metadata_cache_file)
input_dataframe= make_pairs(input_dataframe,output_path_pairs)

add_metadata(input_dataframe,metadata_cache, output_path_updated_with_metadata)
input_dataframe= compute_similarity_test(input_dataframe,output_path_similarities)

model_input = input_dataframe[['name_metric', 'paragraph_metric','language_metric','synonym_metric','author_metric']].copy()
model_input.to_csv(model_input_path, index=False)


input_dataframe = pd.read_csv(output_path_similarities)
model_input = pd.read_csv(model_input_path)
#Loading model
print("Predicting with the model...")
with open(model_path, "rb") as f:
    model = cloudpickle.load(f)
predictions = model.predict(model_input)
# Add predictions to the input DataFrame``
input_dataframe['prediction'] = predictions
# Save the final DataFrame with predictions to a new CSV file
if output_path_similarities is not None:
    input_dataframe.to_csv(output_path_similarities, index=False)
grouped = input_dataframe.groupby(['name', 'paragraph', 'doi']).apply(aggregate_group).reset_index()
grouped.to_csv(output_path_aggregated_groups, index=False)
print("Processing complete. Output files generated.")

