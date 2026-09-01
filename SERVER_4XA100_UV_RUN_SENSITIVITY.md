# Chạy ARC-BPO Sensitivity trên server Linux 4xA100 bằng uv

Tài liệu này dành riêng cho cấu hình:

```text
Backbone: RLHFlow/LLaMA3-SFT-v2
Dataset: princeton-nlp/llama3-ultrafeedback-armorm
Training seed: 0 (một seed cho mọi setting)
Nominal examples/run: 10,000
Global batch size: 64
Gradient accumulation: 4
GPU: 4x NVIDIA A100
Fine-tuning: LoRA
New training runs: 14
```

`noise_seed=2026` chỉ cố định cùng một tập preference pairs bị đảo nhãn trong
các run noise 20%; nó không phải training seed thứ hai.

## 1. Yêu cầu server

Khuyến nghị:

```text
Linux x86_64
4x A100 (40GB hoặc 80GB; 80GB an toàn hơn)
NVIDIA driver hỗ trợ CUDA 12.4
RAM hệ thống khoảng 256GB
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
git pull --ff-only origin main
```

Xác nhận launcher tồn tại:

```bash
git status --short
ls run_sensitivity.py sensitivity/common.py sensitivity/requirements_uv.txt
```

## 3. Tạo môi trường uv

Repo dùng pip-compatible interface của uv; không chạy `uv sync` vì thư mục gốc
không có `pyproject.toml`/`uv.lock`.

```bash
uv --version
uv venv .venv --python 3.11
source .venv/bin/activate
```

Cài đúng PyTorch CUDA 12.4 trước:

```bash
uv pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Cài bộ dependency đã pin giống Modal image:

```bash
uv pip install -r sensitivity/requirements_uv.txt
uv pip check
```

Không dùng `uv pip install -r requirements.txt` thay cho file trên trong run
chính thức, vì các giới hạn version rộng có thể lấy Transformers/PEFT mới hơn
bộ đã được kiểm tra.

## 4. Kiểm tra môi trường và đúng 4xA100

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -c '
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
assert torch.cuda.device_count() == 4
assert all("A100" in torch.cuda.get_device_name(i) for i in range(4))
'
```

Launcher cũng lặp lại các kiểm tra này trước khi train. Nó sẽ dừng nếu không có
đúng 4 GPU A100 hoặc nếu global batch không chia hết cho
`gradient_accumulation x GPU count`.

Với setting hiện tại:

```text
per-GPU microbatch = 64 / (4 accumulation x 4 GPU) = 4
optimizer steps/run = ceil(10000 / 64) = 157
```

Iterator chỉ phát full batches, nên setting được đặt tên 10k nhưng thực tế xử lý
157 x 64 = 10,048 examples. Giữ hành vi này để nhất quán với các run
`10k-bs64` trước đó.

## 5. Đăng nhập Hugging Face

Token phải có quyền ghi model repository:

```bash
hf auth login
hf auth whoami
```

Đặt repo checkpoint dễ nhận biết:

```bash
export HF_REPO_ID=ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64
export HF_PRIVATE=false
export HF_UPLOAD_ADAPTER_ONLY=true
```

Không ghi trực tiếp token vào source code hoặc command được lưu trong Git.

## 6. Audit-only trước khi dùng GPU

Lệnh này sinh và audit config nhưng chưa train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --output_root outputs/sensitivity/llama3-10k-bs64 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 4 \
  --expected_gpu_name A100 \
  --exclude_default_points
```

Kiểm tra phải có 14 run rows và đúng một seed `0`:

```bash
wc -l outputs/sensitivity/llama3-10k-bs64/run_manifest.csv

tail -n +2 outputs/sensitivity/llama3-10k-bs64/run_manifest.csv \
  | cut -d, -f6 \
  | sort -u
```

Kết quả mong đợi:

```text
15 outputs/sensitivity/llama3-10k-bs64/run_manifest.csv
0
```

`15` gồm một header và 14 training runs. `--exclude_default_points` tái sử dụng
published default anchors và tránh train lặp cùng một clean default nhiều lần.

## 7. Chạy thử một setting để đo tốc độ

```bash
tmux new -s arc-sensitivity
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python -u run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --output_root outputs/sensitivity/llama3-10k-bs64 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 4 \
  --expected_gpu_name A100 \
  --exclude_default_points \
  --max_runs 1 \
  --execute \
  2>&1 | tee outputs/sensitivity/llama3-10k-bs64/first-run.log
```

Detach tmux bằng `Ctrl+B`, sau đó `D`. Quay lại bằng:

```bash
tmux attach -t arc-sensitivity
```

Run đầu tiên được giữ lại; full command ở bước tiếp theo sẽ resume và không
train lại checkpoint hoàn chỉnh.

## 8. Chạy toàn bộ 14 settings

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python -u run_sensitivity.py \
  --preset llama3-10k-bs64 \
  --output_root outputs/sensitivity/llama3-10k-bs64 \
  --seeds 0 \
  --noise_rate 0.20 \
  --noise_seed 2026 \
  --expected_gpus 4 \
  --expected_gpu_name A100 \
  --exclude_default_points \
  --execute \
  2>&1 | tee outputs/sensitivity/llama3-10k-bs64/training.log
```

Mỗi setting dùng cả 4 A100; 14 settings chạy tuần tự. Thời gian training thường
nằm trong khoảng 15-30 giờ, nhưng nên dự phòng 24-36 giờ cho tải dữ liệu/model,
save và upload.

Sau mỗi run thành công, launcher kiểm tra `LATEST` rồi upload adapter và config:

```text
checkpoints/sens_T_4_clean_seed0/
checkpoints/sens_kappa_3_noise20_seed0/
checkpoints/sens_delta0_0.5_clean_seed0/
checkpoints/sens_lambda_2_clean_seed0/
...
```

Nếu SSH ngắt nhưng tmux còn chạy thì job không bị ảnh hưởng. Nếu process dừng,
chạy lại nguyên command; checkpoint hoàn chỉnh được bỏ qua và HF upload được thử
lại.

Không dùng `--force` khi resume bình thường vì cờ đó train lại checkpoint đã có.

## 9. Evaluation sau training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python evaluate_sensitivity.py \
  --manifest outputs/sensitivity/llama3-10k-bs64/run_manifest.csv \
  --only_status trained,checkpoint_exists \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 4096 \
  --batch_size auto:4 \
  --merge_device auto \
  --merge_dtype bfloat16
```

Tổng hợp CSV/JSON/PDF/LaTeX/Markdown:

```bash
.venv/bin/python summarize_sensitivity.py \
  --manifest outputs/sensitivity/llama3-10k-bs64/run_manifest.csv \
  --published_anchors sensitivity/published_anchors.json \
  --main_result sensitivity/published_main_result.json
```

## 10. Kiểm tra tiến độ và lỗi thường gặp

Xem run status:

```bash
column -s, -t outputs/sensitivity/llama3-10k-bs64/run_manifest.csv | less -S
```

Xem log throughput:

```bash
grep "examples_per_second" outputs/sensitivity/llama3-10k-bs64/training.log | tail
```

Các lỗi cần xử lý trước khi retry:

- `Expected 4 visible CUDA GPUs`: kiểm tra `CUDA_VISIBLE_DEVICES` và scheduler.
- `mismatched devices`: GPU được cấp không phải A100.
- `must be divisible`: batch size không tương thích GPU count/accumulation.
- CUDA OOM: xác nhận server thực sự cấp đủ bốn GPU cho cùng process.
- HF 401/403: chạy lại `hf auth login` với token có write permission.
- Dependency conflict: chạy `uv pip check`; nếu môi trường bị thay đổi, xóa
  `.venv`, tạo lại và cài từ `sensitivity/requirements_uv.txt`.
