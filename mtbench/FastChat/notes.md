conda create -n mtbench python=3.11 -y
conda activate mtbench
pip install -e ".[model_worker,llm_judge]"
pip install vllm

# serve 2 model with vLLM
CUDA_VISIBLE_DEVICES=0 vllm serve modelA \
  --port 8000 \
  --served-model-name Q_models

CUDA_VISIBLE_DEVICES=0 vllm serve modelB \
  --port 8001 \
  --served-model-name tisdpo

# Generate MT-Bench answers for both models
python3 fastchat/llm_judge/gen_api_answer.py \
  --model Q_models \
  --openai-api-base http://localhost:8000/v1 \
  --parallel 16 \
  --answer-file data/mt_bench/model_answer/Q_models.jsonl \
  --force-temperature 0

python3 fastchat/llm_judge/gen_api_answer.py \
  --model tisdpo \
  --openai-api-base http://localhost:8001/v1 \
  --parallel 16 \
  --answer-file data/mt_bench/model_answer/tisdpo.jsonl \
  --force-temperature 0

# win rate
export OPENAI_API_BASE=
export OPENAI_API_KEY=
python3 fastchat/llm_judge/gen_judgment.py \
  --mode pairwise-all \
  --model-list Q_models tisdpo \
  --judge-model meta/llama3-70b-instruct \
  --parallel 16

# print result
pip install pandas
python3 fastchat/llm_judge/show_result.py \
  --mode pairwise-all \
  --model-list Q_models tisdpo \
  --judge-model meta/llama3-70b-instruct
