conda create -n alpaca-eval python=3.11.11
conda activate alpaca-eval
pip install 'alpaca-eval[all]'

hf download tonyshelby/processed_data --repo-type dataset --local-dir ./processed_data
