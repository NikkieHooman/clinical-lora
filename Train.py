import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from dataset import (
    build_icd_dataset,
    build_ner_dataset,
    build_summarization_dataset,
    tokenize_dataset,
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   
    "gate_proj", "up_proj", "down_proj",       
]

TASK_DATASETS = {
    "icd_coding":    build_icd_dataset,
    "summarization": build_summarization_dataset,
    "ner":           build_ner_dataset,
}


def load_cfg(path: str = "configs/lora_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_base_model(cfg: dict):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    model.config.use_cache = False           
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    return model


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.getenv("HF_TOKEN")
    )
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"    
    return tok


def apply_lora(model, cfg: dict):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["rank"],               
        lora_alpha=cfg["lora"]["alpha"],     
        lora_dropout=cfg["lora"]["dropout"], 
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train(task: str, cfg: dict,
          mimic_root: str | None = None,
          n2c2_root:  str | None = None):

    output_dir = Path(cfg["training"]["output_dir"]) / task
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding {task} dataset ...")
    if task == "ner":
        if not n2c2_root:
            raise ValueError("--n2c2_root required for ner task")
        dataset: DatasetDict = build_ner_dataset(n2c2_root)
    else:
        if not mimic_root:
            raise ValueError("--mimic_root required for icd_coding / summarization")
        dataset: DatasetDict = TASK_DATASETS[task](mimic_root)

    print(f"  train: {len(dataset['train']):,}  "
          f"val: {len(dataset['val']):,}  "
          f"test: {len(dataset['test']):,}")

    tokenizer = load_tokenizer()
    model     = load_base_model(cfg)
    model     = apply_lora(model, cfg)

    t = cfg["training"]
    sft_cfg = SFTConfig(
        output_dir              = str(output_dir),
        num_train_epochs        = t["epochs"],
        per_device_train_batch_size = t["batch_size"],
        per_device_eval_batch_size  = t["batch_size"],
        gradient_accumulation_steps = t["grad_accum"],
        learning_rate           = t["lr"],
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.03,
        weight_decay            = 0.01,
        optim                   = "paged_adamw_8bit",  
        fp16                    = False,
        bf16                    = True,
        gradient_checkpointing  = True,
        max_seq_length          = t["max_seq_len"],
        packing                 = True,          
        dataset_text_field      = "prompt",
        logging_steps           = 25,
        eval_strategy           = "steps",
        eval_steps              = 200,
        save_strategy           = "steps",
        save_steps              = 200,
        save_total_limit        = 3,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        report_to               = "wandb" if t.get("use_wandb") else "none",
        run_name                = f"clinical-lora-{task}",
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["val"],
        args            = sft_cfg,
    )

    print(f"\nTraining {task} ...")
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved → {adapter_path}")
    print("Merge with base model for deployment:")
    print(f"  python src/merge.py --adapter {adapter_path} --output models/{task}_merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       required=True,
                        choices=["icd_coding", "summarization", "ner"])
    parser.add_argument("--config",     default="configs/lora_config.yaml")
    parser.add_argument("--mimic_root", default=None)
    parser.add_argument("--n2c2_root",  default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    train(args.task, cfg,
          mimic_root=args.mimic_root,
          n2c2_root=args.n2c2_root)
