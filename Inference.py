import argparse
import json
import sys
from pathlib import Path
from typing import Literal
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dataset import format_prompt, TEMPLATES


Task = Literal["icd_coding", "summarization", "ner"]

GENERATION_KWARGS = {
    "icd_coding": dict(
        do_sample=False,      
        max_new_tokens=256,
        repetition_penalty=1.1,
    ),
    "summarization": dict(
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        max_new_tokens=512,
        repetition_penalty=1.15,
    ),
    "ner": dict(
        do_sample=False,
        max_new_tokens=512,
        repetition_penalty=1.1,
    ),
}


class ClinicalLoRA:
    def __init__(self, adapter_path: str, task: Task,
                 device: str = "auto", merge: bool = True):
        self.task = task
        adapter_path = Path(adapter_path)

        print(f"Loading base model + adapter ({adapter_path}) ...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        if merge:
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            model = peft_model.merge_and_unload()
        else:
            model = PeftModel.from_pretrained(base_model, adapter_path)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,   
        )
        self.tokenizer = tokenizer
        print("Ready.")

  
    def predict(self, **kwargs) -> dict | list | str:
        prompt   = format_prompt(self.task, kwargs, include_response=False)
        gen_kw   = GENERATION_KWARGS[self.task]
        raw_out  = self.pipe(prompt, **gen_kw)[0]["generated_text"].strip()
        return self._parse(raw_out)

    def predict_batch(self, samples: list[dict],
                      batch_size: int = 8) -> list:
        results = []
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            prompts = [format_prompt(self.task, s, include_response=False)
                       for s in chunk]
            gen_kw  = GENERATION_KWARGS[self.task]
            outs    = self.pipe(prompts, **gen_kw, batch_size=batch_size)
            for o in outs:
                raw = o[0]["generated_text"].strip()
                results.append(self._parse(raw))
            print(f"  processed {min(i+batch_size, len(samples))}/{len(samples)}")
        return results


    def _parse(self, raw: str) -> dict | list | str:
        if self.task == "summarization":
            return raw

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return {"raw_output": raw, "parse_error": True}


def cli():
    parser = argparse.ArgumentParser(description="Clinical LoRA inference")
    parser.add_argument("--task",     required=True,
                        choices=["icd_coding", "summarization", "ner"])
    parser.add_argument("--adapter",  required=True,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--input",    default=None,
                        help="Path to .txt file with clinical note / sentence")
    parser.add_argument("--text",     default=None,
                        help="Inline text input (alternative to --input)")
    parser.add_argument("--batch",    default=None,
                        help="CSV file for batch inference (column: 'text' or 'sentence')")
    parser.add_argument("--output",   default=None,
                        help="Save predictions to this JSON file")
    args = parser.parse_args()

    model = ClinicalLoRA(adapter_path=args.adapter, task=args.task)

    if args.batch:
        import pandas as pd
        df  = pd.read_csv(args.batch)
        key = TEMPLATES[args.task]["input_key"]
        samples = [{key: row[key]} for _, row in df.iterrows()]
        preds   = model.predict_batch(samples)
        if args.output:
            Path(args.output).write_text(json.dumps(preds, indent=2))
            print(f"Saved {len(preds)} predictions → {args.output}")
        else:
            print(json.dumps(preds[:3], indent=2), "... (showing first 3)")

    else:
        if args.input:
            text = Path(args.input).read_text().strip()
        elif args.text:
            text = args.text
        else:
            print("Provide --input or --text"); sys.exit(1)

        key    = TEMPLATES[args.task]["input_key"]
        result = model.predict(**{key: text})
        output = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result
        print("\n── Prediction ──────────────────────────────────────────")
        print(output)
        if args.output:
            Path(args.output).write_text(output)


if __name__ == "__main__":
    cli()
