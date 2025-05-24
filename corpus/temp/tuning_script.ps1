# run_tuning.ps1
# PowerShell script to run hyperparameter tuning for specified versions
# Ensure this script is saved and run from its directory in your VS Code terminal.

# Path to Python interpreter and tuning script
$python = 'C:/Users/Jelena/AppData/Local/Programs/Python/Python312/python.exe'
$script = 'd:/MASTER/TMF/Software-Disambiguation/code/fine_tuning.py'

# Columns to impute when using Random Forest\ `$metrics = 'paragraph_metric author_metric language_metric synonym_metric'

Write-Host "Starting hyperparameter tuning runs..."

# v3.8 → Random Forest only
Write-Host "`nRunning v3.8 (Random Forest)"
& $python $script `
    --data "D:\MASTER\TMF\Software-Disambiguation\corpus\temp\v3.8\model_input_no_keywords.csv" `
    --models "Random Forest" `
    --cols-to-impute paragraph_metric author_metric language_metric synonym_metric

# v3.9 → XGBoost only
Write-Host "`nRunning v3.9 (XGBoost)"
& $python $script `
    --data "D:\MASTER\TMF\Software-Disambiguation\corpus\temp\v3.9\model_input_no_keywords.csv" `
    --models XGBoost

# v3.12 → RF, XGBoost, LightGBM
Write-Host "`nRunning v3.12 (Random Forest, XGBoost, LightGBM)"
& $python $script `
    --data "D:\MASTER\TMF\Software-Disambiguation\corpus\temp\v3.12\model_input_no_keywords.csv" `
    --models "Random Forest" XGBoost LightGBM `
    --cols-to-impute paragraph_metric author_metric language_metric synonym_metric

Write-Host "`nAll runs completed. Results appended to tuning_results.csv."
