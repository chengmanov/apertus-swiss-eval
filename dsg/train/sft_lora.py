"""LoRA / QLoRA fine-tuning of Apertus-8B-Instruct on the FINMA SFT set.

Follows the swiss-ai/apertus-finetuning-recipes approach (TRL SFTTrainer).
Runs on the tuning machine (NVIDIA GPU) -- NOT on the Vulkan eval box.

  # 24 GB GPU (4-bit QLoRA):
  python train/sft_lora.py --quant 4bit --batch 1 --grad-accum 16
  # 40 GB+ GPU (bf16 LoRA):
  python train/sft_lora.py --quant none --batch 4 --grad-accum 4

Outputs the adapter to train/out/adapter and (with --merge) a merged fp16
model to train/out/merged for GGUF conversion.

Requires (install on the tuning machine, not pinned in this repo's uv env):
  pip install "transformers>=4.56" "trl>=0.12" "peft>=0.13" "datasets>=3.0" \
              "bitsandbytes>=0.44" "accelerate>=1.0"
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--out", default=str(ROOT / "train" / "out"))
    ap.add_argument("--merge", action="store_true", help="also write a merged fp16 model for GGUF")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "adapter").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    # Force the whole model onto a single GPU. device_map="auto" lets accelerate
    # offload layers to the meta/CPU device, which breaks the LoRA backward pass
    # ("expected device meta but got cuda:0") on some accelerate versions.
    device_map = {"": 0} if quant_cfg is None else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_cfg,
        dtype=torch.bfloat16, device_map=device_map)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files={
        "train": str(ROOT / "tasks" / "sft_train.jsonl"),
        "val": str(ROOT / "tasks" / "sft_val.jsonl"),
    })

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        bf16=True, max_length=args.max_seq_len, packing=False,
        gradient_checkpointing=True, report_to=[], seed=20260719,
        # Standard cross-entropy. trl's default 'chunked_nll' memory
        # optimization crashes patching this model's functools.partial forward
        # (trl 1.8.0); 'nll' is mathematically identical and the 128 GB GB10
        # has ample memory. trl's own error message recommends exactly this.
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model, args=cfg, peft_config=peft_cfg,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    print(f"adapter saved -> {out / 'adapter'}")

    if args.merge:
        from peft import AutoPeftModelForCausalLM
        merged = AutoPeftModelForCausalLM.from_pretrained(
            str(out / "adapter"), dtype=torch.float16, device_map="cpu")
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(out / "merged"), safe_serialization=True)
        tok.save_pretrained(str(out / "merged"))
        print(f"merged fp16 model -> {out / 'merged'}")
        print("Convert to GGUF on the eval box:\n"
              "  python llama.cpp/convert_hf_to_gguf.py train/out/merged "
              "--outfile models/Apertus-8B-finma-lora.gguf --outtype q8_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
