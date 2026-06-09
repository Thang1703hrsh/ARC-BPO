# Create and activate conda environment
conda create -n diversity-metrics python=3.11 -y
conda activate diversity-metrics

# Install dependencies
pip install vllm transformers sacrebleu tqdm

# Run generation
#   --model: path to your merged model dir, or HF repo id (e.g. tonyshelby/llama-QTBPO-merged)
#   --out:   diversity_metrics/<RUN_NAME>/diversity_generations.jsonl
python diversity_metrics/generation_vllm.py \
  --model <MODEL_PATH_OR_HF_REPO> \
  --prompts processed_data/diversity_prompts.jsonl \
  --out diversity_metrics/<RUN_NAME>/diversity_generations.jsonl \
  --k 5 \
  --tensor_parallel_size 2 \
  --batch_size 64 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_p 0.95 \
  --seed 1234

# Compute diversity metrics
python diversity_metrics/compute_diversity.py \
  --infile diversity_metrics/<RUN_NAME>/diversity_generations.jsonl \
  --out    diversity_metrics/<RUN_NAME>/diversity_metrics.json
