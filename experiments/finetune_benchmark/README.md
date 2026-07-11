# Quick LoRA benchmark

This benchmark reuses the project's existing finetuning implementation:

- dataset: `backend.finetune_service.build_finetune_dataset`
- model loading: `backend.lora_finetune.load_model_and_tokenizer`
- LoRA injection: `backend.lora_finetune.apply_lora`
- adapter saving: `backend.lora_finetune.save_lora`

It runs three automatic comparisons on a fixed event split:

1. Qwen2.5-1.5B-Instruct without LoRA;
2. Qwen2.5-1.5B-Instruct with the newly trained LoRA adapter;
3. Qwen2.5-0.5B-Instruct as a weak, low-resource baseline.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\finetune_benchmark\run_quick_benchmark.py
```

The script writes raw predictions, metrics, environment information and a
Markdown/CSV comparison table under `outputs/finetune_benchmark/<run_id>`.
The automatic scores measure report grounding on the existing template-based
dataset; they are not an independent expert assessment of root-cause quality.

The checked-in quick configuration uses FP16. On this Windows host,
bitsandbytes 4-bit inference fell back to a non-Triton path and was materially
slower, while FP16 kept the 1.5B model below 3 GB of inference VRAM.
