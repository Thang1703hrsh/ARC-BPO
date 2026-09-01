# Allocation ablations: Llama-3-8B, one seed

This launcher pins the requested setting:

- base SFT model: `RLHFlow/LLaMA3-SFT-v2`;
- LoRA enabled, initialized fresh for every new run;
- dataset: `princeton-nlp/llama3-ultrafeedback-armorm`, train split;
- 10,000 training examples;
- global batch size 64;
- exactly one matched seed, default `0`;
- four GPUs and gradient accumulation 4 by default, matching the public
  bs64 launch command.

The supplied adapter
`ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64` is a trained uniform
result. It must **not** initialize the other ablations, since that would give
them a 10k-example head start and invalidate the controlled comparison.

## Important blocker in the plan

The current `arc_bpo_pair_loss` always uses the SBA generator. The repository
does not contain a distinct pre-SBA/base-Bregman loss for the reviewer row
named `advantage`. `A_tbpo` and `BPO_SBA` are not valid substitutes because
they change the objective and/or chunk semantics. Consequently, the full
three-row request is deliberately blocked until the intended loss is supplied.

Audit the complete requested set without launching training:

```bash
python run_allocation_ablations.py
```

This writes `outputs/ablations/run_manifest.csv`, `config_audit.json`, and
pairwise config diffs. The `advantage` row is marked
`blocked_missing_loss_definition`.

## Run the two scientifically defined variants

Start both from the common SFT base:

```bash
python run_allocation_ablations.py \
  --variants uniform,advantage_sba_no_winsor \
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
controlled setting on one Modal container with four `A100-80GB` GPUs. This is
intentional: it preserves the public run's global batch 64, gradient
accumulation 4, and per-GPU microbatch 4. The full job uses exactly one matched
seed (`0`) and 10,000 requested training examples.

The default Modal variant is `advantage_sba_no_winsor`. The existing uniform
adapter is recorded as the reference row and is never loaded as training
initialization. The standalone `advantage` variant remains blocked for the
scientific reason described above.

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
optimizer update, FSDP across all four GPUs, score extraction, strict
advantage allocation, no-winsorization, checkpoint saving, and Volume commits:

```powershell
modal run modal_allocation_ablations.py --mode smoke
```

Results are persisted in the Modal Volume
`arc-bpo-allocation-ablation-results` under:

```text
allocation_ablations/llama3-allocation-ablation-bs64-seed0/smoke/
```

Inspect the manifest and logs:

```powershell
modal volume get arc-bpo-allocation-ablation-results allocation_ablations/llama3-allocation-ablation-bs64-seed0/smoke .\allocation-smoke
```

### Full 10k run

After the smoke checkpoint is valid:

```powershell
modal run modal_allocation_ablations.py --mode full
```

Download the result tree, including the LoRA adapter, resolved training config,
config audit, manifest, and logs:

```powershell
modal volume get arc-bpo-allocation-ablation-results allocation_ablations/llama3-allocation-ablation-bs64-seed0/full .\allocation-full
```

The job resolves current Hugging Face commit SHAs and passes those exact
revisions into model, tokenizer, and dataset loading. The resolved values are
saved in `modal_run_config.json`.

### Optional Hugging Face upload

No external checkpoint is uploaded by default. To upload the completed LoRA
adapter plus its resolved configs to a private repository:

```powershell
modal run modal_allocation_ablations.py --mode full --hf-repo-id ducthang1703/llama3-arc-bpo-adv-sba-no-winsor-lora-10k-bs64-seed0
```

Use `--no-hf-private` only when you explicitly want the repository to be
public. Rerunning the same configuration resumes/skips a complete checkpoint;
`--force` explicitly retrains it. If any scientific setting differs, choose a
new `--output-name` rather than mixing runs in one directory.
