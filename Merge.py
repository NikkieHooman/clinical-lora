import argparse
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_save(adapter_path: str, output_path: str,
                   push_to_hub: str | None = None):
    adapter_path = Path(adapter_path)
    output_path  = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading base model (bfloat16) ...")
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",       
        token=os.getenv("HF_TOKEN"),
    )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter into base weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model → {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(adapter_path)
    tok.save_pretrained(output_path)

    if push_to_hub:
        print(f"Pushing to Hub → {push_to_hub}")
        model.push_to_hub(push_to_hub, private=True,
                          token=os.getenv("HF_TOKEN"))
        tok.push_to_hub(push_to_hub, token=os.getenv("HF_TOKEN"))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",      required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--push_to_hub",  default=None)
    args = parser.parse_args()
    merge_and_save(args.adapter, args.output, args.push_to_hub)
