# Chạy ARC-BPO Sensitivity trên server Linux 2xA100 bằng uv

Tài liệu này dành riêng cho cấu hình:

```text
Backbone: RLHFlow/LLaMA3-SFT-v2
Dataset: princeton-nlp/llama3-ultrafeedback-armorm
Training seed: 0 (một seed cho mọi setting)
Nominal examples/run: 10,000
Global batch size: 64
Gradient accumulation: 8
GPU: 2x NVIDIA A100
Per-GPU microbatch: 4
Fine-tuning: LoRA
New training runs: 14
```

So với hướng dẫn 4 GPU, accumulation được tăng từ 4 lên 8. Nhờ đó global
batch vẫn là 64 và micro-batch vẫn là 4/GPU:

```text
64 / (8 accumulation x 2 GPU) = 4 examples/GPU/micro-step
```

Không chạy cấu hình 2 GPU với accumulation 4 nếu chưa kiểm tra bộ nhớ, vì khi
đó micro-batch tăng thành 8/GPU và có nguy cơ CUDA OOM.

`noise_seed=2026` chỉ cố định cùng một tập preference pairs bị đảo nhãn trong
các run noise 20%; nó không phải training seed thứ hai.

> **Lưu ý về protocol:** launcher mặc định dùng `EXCLUDE_DEFAULT_POINTS=true`,
> nên chạy 14 setting mới và khi tổng hợp sẽ dùng default anchors đã công bố từ
> cấu hình 4 GPU/accumulation 4. Cách này phù hợp để chạy grid hiện tại nhưng
> không phải exact-match 2 GPU/accumulation 8. Một bảng controlled sensitivity
> nghiêm ngặt cần thêm clean default và noise-20 default được train bằng chính
> cấu hình 2 GPU/accumulation 8 (tổng cộng 16 unique runs).

## 1. Yêu cầu server

Khuyến nghị:

```text
Linux x86_64
2x A100; A100-80GB được khuyến nghị, A100-40GB có thể thiếu bộ nhớ
NVIDIA driver hỗ trợ CUDA 12.4
RAM hệ thống tối thiểu khoảng 128GB; 192-256GB an toàn hơn
Ít nhất 150-200GB dung lượng trống cho model, dataset, cache và checkpoint
tmux
uv
```

Kiểm tra GPU và driver:

```bash
nvidia-smi
nvidia-smi -L
```

## 2. Clone hoặc cập nhật code

Clone mới:

```bash
git clone --branch main --single-branch \
  https://github.com/Thang1703hrsh/ARC-BPO.git
cd ARC-BPO
```

Nếu server đã có repo:

```bash
cd ARC-BPO
git fetch origin
git switch main
git pull --ff-only origin main
```

Xác nhận các file cần thiết:

```bash
git status --short
ls run_sensitivity.py \
  sensitivity/common.py \
  sensitivity/requirements_uv.txt \
  script/train/arc_bpo_sensitivity.sh
```

## 3. Tạo môi trường uv

Repo dùng pip-compatible interface của uv; không chạy `uv sync` vì thư mục gốc
không có `pyproject.toml`/`uv.lock`.

```bash
uv --version
uv venv .venv --python 3.11
source .venv/bin/activate
```

Cài PyTorch CUDA 12.4 trước:

```bash
uv pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Cài bộ dependency đã pin giống Modal image:

```bash
uv pip install -r sensitivity/requirements_uv.txt
uv pip check
```

## 4. Kiểm tra môi trường và đúng 2xA100

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -c '
import torch
import transformers
import datasets
import peft
import vllm
import lm_eval
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert all("A100" in torch.cuda.get_device_name(i) for i in range(2))
'
```

## 5. Lưu Hugging Face token ngoài repository

Không cần chạy `hf auth login`. Token phải có quyền ghi model repository và
được lưu một lần ở ngoài thư mục Git. Nhập token theo cách ẩn:

```bash
install -d -m 700 "$HOME/.config/arc-bpo"
read -rsp "Hugging Face token: " HF_TOKEN
echo
printf '%s\n' "$HF_TOKEN" > "$HOME/.config/arc-bpo/hf_token"
unset HF_TOKEN
chmod 600 "$HOME/.config/arc-bpo/hf_token"
test -s "$HOME/.config/arc-bpo/hf_token" && echo "HF token file is ready"
```

Launcher tự đọc `~/.config/arc-bpo/hf_token` và chỉ export token trong process
train. Token không được in ra terminal/log và không nằm trong repository.

Dùng repo riêng để không trộn checkpoint với run 4 GPU/accumulation 4:

```bash
export HF_REPO_ID=ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64-2xa100-ga8
export HF_PRIVATE=false
export HF_UPLOAD_ADAPTER_ONLY=true
```

## 6. Chạy bằng Bash launcher

`script/train/arc_bpo_sensitivity.sh` là wrapper Bash cho toàn bộ workflow.
Bạn không cần gõ trực tiếp tên các file Python. Các mode hỗ trợ gồm `audit`,
`smoke`, `full`, `evaluate` và `summarize`.

Trước tiên, chạy audit và một smoke run trong `tmux`:

```bash
tmux new -s arc-sensitivity-2gpu
bash script/train/arc_bpo_sensitivity.sh audit
bash script/train/arc_bpo_sensitivity.sh smoke
```

Chỉ chạy full sau khi smoke run hoàn thành và không bị CUDA OOM. Ví dụ theo
cùng kiểu truyền biến môi trường của các launcher train:

```bash
GPU_IDS=0,1 \
N_EXAMPLES=10000 \
BATCH_SIZE=64 \
GRAD_ACCUM=8 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
HF_REPO_ID=ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64-2xa100-ga8 \
HF_PRIVATE=false \
HF_UPLOAD_ADAPTER_ONLY=true \
SAVE_EVERY_EXAMPLES=10000 \
bash script/train/arc_bpo_sensitivity.sh full
```

Nếu file mặc định chưa tồn tại và chưa export `HF_TOKEN`, launcher sẽ dừng ở
prompt `Hugging Face token:`. Paste token rồi nhấn Enter; token chỉ tồn tại trong
process hiện tại. Có thể chỉ định một token file khác đã `chmod 600`:

```bash
HF_TOKEN_FILE="$HOME/.config/arc-bpo/hf_token" \
bash script/train/arc_bpo_sensitivity.sh full
```

Các lệnh rút gọn khác:

```bash
bash script/train/arc_bpo_sensitivity.sh audit
bash script/train/arc_bpo_sensitivity.sh smoke
bash script/train/arc_bpo_sensitivity.sh evaluate
bash script/train/arc_bpo_sensitivity.sh summarize
```

Các phần dưới đây giữ lại lệnh Python tương đương để debug khi cần.

## 7. Audit-only trước khi train

Lệnh này sinh và audit 14 config nhưng chưa dùng GPU để train:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --gradient_accumulation_steps 8 \
  --output_root outputs/sensitivity/llama3-10k-bs64-2xa100-ga8 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 2 \
  --expected_gpu_name A100 \
  --exclude_default_points
```

Kiểm tra phải có 14 run rows và đúng một seed `0`:

```bash
wc -l outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/run_manifest.csv

tail -n +2 outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/run_manifest.csv \
  | cut -d, -f6 \
  | sort -u
```

Kết quả mong đợi:

```text
15 outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/run_manifest.csv
0
```

## 8. Chạy thử một setting

Nên chạy trong tmux để job không dừng khi SSH mất kết nối:

```bash
tmux new -s arc-sensitivity-2gpu
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -u run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --gradient_accumulation_steps 8 \
  --output_root outputs/sensitivity/llama3-10k-bs64-2xa100-ga8 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 2 \
  --expected_gpu_name A100 \
  --exclude_default_points \
  --max_runs 1 \
  --execute \
  2>&1 | tee outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/first-run.log
```

Detach tmux bằng `Ctrl+B`, sau đó `D`. Quay lại bằng:

```bash
tmux attach -t arc-sensitivity-2gpu
```

Nếu smoke run CUDA OOM trên A100-40GB, không chạy full grid. Hai GPU chứa cả
policy và reference model FSDP nên model-state memory vẫn cao dù micro-batch đã
giữ ở mức 4/GPU.

## 9. Chạy toàn bộ 14 settings

Sau khi run đầu thành công:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -u run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --gradient_accumulation_steps 8 \
  --output_root outputs/sensitivity/llama3-10k-bs64-2xa100-ga8 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 2 \
  --expected_gpu_name A100 \
  --exclude_default_points \
  --execute \
  2>&1 | tee outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/training.log
```

Mỗi setting dùng cả 2 A100; 14 settings chạy tuần tự. Với cùng micro-batch/GPU,
training thường chậm gần gấp đôi cấu hình 4 GPU. Nên dự phòng khoảng 35-70 giờ,
sau đó cộng thêm thời gian evaluation.

Sau mỗi run thành công, launcher kiểm tra `LATEST` rồi upload LoRA adapter và
audited config lên repo HF đã đặt ở bước 5. Nếu process dừng, chạy lại nguyên
command; checkpoint hoàn chỉnh được bỏ qua và HF upload được thử lại. Không dùng
`--force` khi resume bình thường.

## 10. Evaluation trên 2 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python evaluate_sensitivity.py \
  --manifest outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/run_manifest.csv \
  --only_status trained,checkpoint_exists \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 4096 \
  --batch_size auto:4 \
  --merge_device auto \
  --merge_dtype bfloat16
```

Tổng hợp kết quả:

```bash
.venv/bin/python summarize_sensitivity.py \
  --manifest outputs/sensitivity/llama3-10k-bs64-2xa100-ga8/run_manifest.csv \
  --published_anchors sensitivity/published_anchors.json \
  --main_result sensitivity/published_main_result.json
```

## 11. Lưu ý và xử lý lỗi

- `Expected 2 visible CUDA GPUs`: kiểm tra `CUDA_VISIBLE_DEVICES=0,1` và scheduler.
- `mismatched devices`: GPU được cấp không phải A100.
- `must be divisible`: xác nhận batch 64, accumulation 8 và 2 GPU.
- CUDA OOM: ưu tiên A100-80GB; không giảm global batch nếu cần so sánh trực tiếp
  với setting bs64 mà chưa ghi nhận thay đổi protocol.
- HF 401/403: thay nội dung `~/.config/arc-bpo/hf_token` bằng token mới có write
  permission và giữ permission của file là `600`.
- Dependency conflict: chạy `uv pip check`; nếu cần, tạo lại `.venv` và cài từ
  `sensitivity/requirements_uv.txt`.
- Setting ghi `n_examples=10000`, nhưng iterator dùng full batch nên thực tế xử
  lý 157 x 64 = 10,048 examples/run, giống hành vi 4-GPU trước đó.
