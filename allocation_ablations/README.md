# Allocation ablations: Mistral-7B-v0.1 LoRA, 16k, one seed

This launcher pins the requested setting:

- base SFT model: `HuggingFaceH4/mistral-7b-sft-alpha` (Mistral-7B-v0.1);
- LoRA enabled, initialized fresh for every new run;
- dataset: `HuggingFaceH4/ultrafeedback_binarized`, `train_prefs` split;
- 16,000 training examples;
- global batch size 64;
- exactly one matched seed, default `0`;
- four GPUs and gradient accumulation 4 by default, matching the public
  bs64 launch command.

The dataset's `score_chosen` and `score_rejected` columns supply the detached
advantage proxy. Advantage variants use strict mode, so a missing score stops
the run instead of silently falling back to uniform allocation.

Every variant starts from a fresh LoRA adapter on the common SFT base. A
trained uniform adapter is a result reference only and must **not** initialize
the other ablations.

## Standalone Advantage allocation definition

The standalone `advantage` row uses the canonical quadratic base-Bregman
generator `h(r)=0.5*r^2`, giving
`D_h(target, model)=0.5*(target-model)^2`. `uniform` uses the same quadratic
objective, so uniform versus advantage changes allocation only. The
`advantage_sba_no_winsor` row then changes only the divergence from quadratic
to SBA. No TBPO objective or value head is used.

Audit the complete requested set without launching training:

```bash
python run_allocation_ablations.py
```

This writes `outputs/ablations/run_manifest.csv`, `config_audit.json`, and
pairwise config diffs. All three requested rows are runnable.

## Run all three variants

Start all three from the common SFT base:

```bash
python run_allocation_ablations.py \
  --variants uniform,advantage,advantage_sba_no_winsor \
  --seed 0 \
  --gpu_ids 0,1,2,3 \
  --grad_accum 4 \
  --execute
```

To reuse the supplied uniform adapter as an existing result and train only the
no-winsor run:

```bash
python run_allocation_ablations.py \
  --variants uniform,advantage_sba_no_winsor \
  --seed 0 \
  --reuse_uniform_checkpoint \
  --execute
```

The reused row remains marked `external_checkpoint_unverified`: an adapter-only
HF upload proves its base model and LoRA structure, but not the original seed,
dataset order, batch configuration, or training/evaluation protocol. Locate
the original run's resolved `config.yaml` before using that row in a paper.

If only one GPU is available, global bs64 can be retained with, for example,
`--gpu_ids 0 --grad_accum 16`. This changes gradient accumulation relative to
the public four-GPU run and therefore is not the strictest comparison to that
checkpoint.

## Run on Modal

[`modal_allocation_ablations.py`](../modal_allocation_ablations.py) runs the
controlled setting on one Modal `H100-80GB`. It preserves global batch 64 with
gradient accumulation 64 and per-GPU microbatch 1, which keeps the FP32 policy,
FP16 reference model, and activations within a conservative memory envelope.
The container requests 4 CPU cores and 64 GiB system memory; this reduces the
former 16-CPU/256-GiB allocation while retaining minimum practical headroom for
low-memory model loading and FSDP full-state checkpoint gathering.
The full job uses exactly one matched seed (`0`) and 16,000 requested training
examples. Periodic 5k/10k/15k snapshots are disabled: each run saves only the
final `LATEST` LoRA checkpoint after 16k, then uploads that adapter to Hugging
Face when `--hf-repo-id` is provided.

The default Modal variant is `advantage_sba_no_winsor`. Every Modal variant is
submitted independently and starts from a fresh LoRA adapter on the SFT base.
Modal supports `uniform`, `advantage`, and `advantage_sba_no_winsor` as
separate jobs.

### Install and authenticate

From the repository root in PowerShell:

```powershell
py -m pip install -r allocation_ablations/requirements_modal.txt
modal setup
```

Create a Modal secret named `huggingface-secret` with a key named `HF_TOKEN`.
Using the Modal dashboard avoids putting the token into shell history. If the
token is already in the local environment:

```powershell
modal secret create huggingface-secret HF_TOKEN="$env:HF_TOKEN"
```

### Smoke test

Smoke mode trains 64 examples, which exercises one complete global-bs64
optimizer update on one H100, score extraction, strict
advantage allocation, no-winsorization, checkpoint saving, and Volume commits:

```powershell
modal run modal_allocation_ablations.py --mode smoke
```

Results are persisted in the Modal Volume
`arc-bpo-mistral-allocation-ablation-results` under:

```text
allocation_ablations/mistral7b-advantage_sba_no_winsor-lora-64examples-bs64-seed0/smoke/
```

Inspect the manifest and logs:

```powershell
modal volume get arc-bpo-mistral-allocation-ablation-results allocation_ablations/mistral7b-advantage_sba_no_winsor-lora-64examples-bs64-seed0/smoke .\allocation-smoke
```

### Full 16k run

After the smoke checkpoint is valid:

```powershell
modal run modal_allocation_ablations.py --mode full --variant advantage
```

When `--output-name` is omitted, the launcher derives a unique name from the
variant, example count, batch size, and seed. The run above is saved under:

```text
allocation_ablations/mistral7b-advantage-lora-16k-bs64-seed0/full/
```

Download the result tree, including the LoRA adapter, resolved training config,
config audit, manifest, and logs:

```powershell
modal volume get arc-bpo-mistral-allocation-ablation-results allocation_ablations/mistral7b-advantage-lora-16k-bs64-seed0/full .\allocation-full
```

The job resolves current Hugging Face commit SHAs and passes those exact
revisions into model, tokenizer, and dataset loading. The resolved values are
saved in `modal_run_config.json`.

### Optional Hugging Face upload

No external checkpoint is uploaded by default. To upload the completed LoRA
adapter plus its resolved configs to a private repository:

```powershell
modal run modal_allocation_ablations.py --mode full --variant advantage --hf-repo-id ducthang1703/mistral7b-arc-bpo-advantage-quadratic-lora-16k-bs64-seed0
```

For a true fire-and-forget submission that does not wait for the remote result,
use the launcher's asynchronous entrypoint together with Modal detach mode:

```powershell
modal run --detach --quiet modal_allocation_ablations.py --background --mode full --variant advantage --n-examples 16000 --global-batch-size 64 --gradient-accumulation-steps 64 --hf-repo-id ducthang1703/mistral7b-arc-bpo-advantage-quadratic-lora-16k-bs64-seed0 *> $null
```

The FunctionCall id is hidden by the redirection; monitor the run in the Modal
dashboard and inspect the persisted Volume after completion.

Use `--no-hf-private` only when you explicitly want the repository to be
public. Rerunning the same configuration resumes/skips a complete checkpoint;
`--force` explicitly retrains it. If any scientific setting differs, choose a
new `--output-name` rather than mixing runs in one directory.
