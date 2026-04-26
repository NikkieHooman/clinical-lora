import json
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = (
    "You are a clinical NLP assistant trained on medical records. "
    "Follow the instructions carefully and respond only with the requested output."
)

TEMPLATES = {
    "icd_coding": {
        "instruction": (
            "Given the following hospital discharge note, predict the most relevant."
            "ICD-9 diagnosis codes. Return a JSON list of up to 5 codes with short descriptions."
        ),
        "input_key":  "text",
        "output_key": "icd_codes",  
    },
    "summarization": {
        "instruction": (
            "Summarize the following discharge note into a concise, structured summary."
            "covering: (1) Chief complaint, (2) Key findings, (3) Diagnoses, "
            "(4) Procedures, (5) Discharge condition and follow-up plan."
        ),
        "input_key":  "text",
        "output_key": "summary",
    },
    "ner": {
        "instruction": (
            "Extract all clinical named entities from the sentence below. "
            "Return a JSON list where each item has 'entity', 'type' "
            "(one of: PROBLEM, TREATMENT, TEST), and 'span'."
        ),
        "input_key":  "sentence",
        "output_key": "entities",    
    },
}


def format_prompt(task: str, sample: dict, include_response: bool = True) -> str:
    tmpl    = TEMPLATES[task]
    input_  = sample[tmpl["input_key"]]
    output_ = sample.get(tmpl["output_key"], "")

    if isinstance(output_, (list, dict)):
        output_ = json.dumps(output_, ensure_ascii=False)

    prompt = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n{tmpl['instruction']}\n\n"
        f"### Input:\n{input_}\n\n"
        f"### Response:\n"
    )
    if include_response:
        prompt += output_ + "\n"
    return prompt


def _load_mimic_notes(mimic_root: str | Path) -> pd.DataFrame:
    path = Path(mimic_root) / "NOTEEVENTS.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"NOTEEVENTS.csv not found at {path}.\n"
            "Download MIMIC-III from https://physionet.org/content/mimiciii/1.4/"
        )
    df = pd.read_csv(path, usecols=["HADM_ID", "CATEGORY", "TEXT"])
    df = df[df["CATEGORY"] == "Discharge summary"].dropna(subset=["TEXT"])
    df["TEXT"] = df["TEXT"].str.strip()
    return df.reset_index(drop=True)


def _load_mimic_icd(mimic_root: str | Path) -> pd.DataFrame:
    diag_path = Path(mimic_root) / "DIAGNOSES_ICD.csv"
    desc_path = Path(mimic_root) / "D_ICD_DIAGNOSES.csv"
    diag = pd.read_csv(diag_path, usecols=["HADM_ID", "ICD9_CODE", "SEQ_NUM"])
    desc = pd.read_csv(desc_path, usecols=["ICD9_CODE", "SHORT_TITLE"])
    merged = diag.merge(desc, on="ICD9_CODE", how="left")
    top5 = (merged.sort_values("SEQ_NUM")
                  .groupby("HADM_ID")
                  .head(5)
                  .groupby("HADM_ID")
                  .apply(lambda g: [
                      {"code": row.ICD9_CODE, "description": row.SHORT_TITLE}
                      for _, row in g.iterrows()
                  ])
                  .reset_index(name="icd_codes"))
    return top5


def build_icd_dataset(mimic_root: str | Path,
                      max_note_chars: int = 3000,
                      seed: int = 42) -> DatasetDict:

    notes = _load_mimic_notes(mimic_root)
    icd   = _load_mimic_icd(mimic_root)
    df    = notes.merge(icd, on="HADM_ID").dropna()
    df["text"] = df["TEXT"].str[:max_note_chars]
    df = df[["text", "icd_codes"]].drop_duplicates().reset_index(drop=True)
    return _split_and_format(df, task="icd_coding", seed=seed)


def build_summarization_dataset(mimic_root: str | Path,
                                max_note_chars: int = 6000,
                                seed: int = 42) -> DatasetDict:

    notes = _load_mimic_notes(mimic_root)
    records = []
    for _, row in notes.iterrows():
        text = row["TEXT"]
        summary = _extract_section(text, "brief hospital course")
        if summary and len(summary) > 100:
            records.append({
                "text":    text[:max_note_chars],
                "summary": summary.strip(),
            })
    df = pd.DataFrame(records)
    return _split_and_format(df, task="summarization", seed=seed)


def build_ner_dataset(n2c2_root: str | Path, seed: int = 42) -> DatasetDict:
    records = []
    ann_dir = Path(n2c2_root) / "concept_assertion_relation_training_data" / "beth" / "ann"
    txt_dir = Path(n2c2_root) / "concept_assertion_relation_training_data" / "beth" / "txt"

    for ann_file in sorted(ann_dir.glob("*.con")):
        txt_file = txt_dir / ann_file.with_suffix(".txt").name
        if not txt_file.exists():
            continue
        sentences = txt_file.read_text().splitlines()
        entities_by_line: dict[int, list] = {}

        for line in ann_file.read_text().splitlines():
            if not line.startswith("c="):
                continue
            try:
                parts   = line.split("||")
                ent_str = parts[0]          
                typ_str = parts[1]          
                entity  = ent_str.split('"')[1]
                loc     = ent_str.split('"')[2].strip()
                lineno  = int(loc.split(":")[0]) - 1
                span    = list(map(int, loc.split(":")[1].split("-")))
                etype   = typ_str.split('"')[1].upper()
                if etype not in ("PROBLEM", "TREATMENT", "TEST"):
                    continue
                entities_by_line.setdefault(lineno, []).append(
                    {"entity": entity, "type": etype, "span": span}
                )
            except (IndexError, ValueError):
                continue

        for lineno, ents in entities_by_line.items():
            if lineno < len(sentences) and sentences[lineno].strip():
                records.append({
                    "sentence": sentences[lineno].strip(),
                    "entities": ents,
                })

    df = pd.DataFrame(records)
    return _split_and_format(df, task="ner", seed=seed)


def _extract_section(text: str, section_name: str) -> str:
    import re
    pattern = rf"{re.escape(section_name)}[:\s]*(.*?)(?=\n[A-Z][A-Za-z\s]+:|\Z)"
    match   = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _split_and_format(df: pd.DataFrame, task: str,
                      train_frac=0.80, val_frac=0.10,
                      seed: int = 42) -> DatasetDict:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n       = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        "train": df.iloc[:n_train],
        "val":   df.iloc[n_train:n_train + n_val],
        "test":  df.iloc[n_train + n_val:],
    }

    def to_hf(split_df, include_response):
        records = []
        for _, row in split_df.iterrows():
            records.append({
                "prompt":   format_prompt(task, row.to_dict(), include_response),
                "raw":      row.to_dict(),  
            })
        return Dataset.from_list(records)

    return DatasetDict({
        "train": to_hf(splits["train"], include_response=True),
        "val":   to_hf(splits["val"],   include_response=False),
        "test":  to_hf(splits["test"],  include_response=False),
    })


def tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizer,
                     max_length: int = 1024) -> DatasetDict:
    response_tag = "### Response:\n"

    def tokenize(example):
        full   = tokenizer(example["prompt"], truncation=True,
                           max_length=max_length, padding=False)
        labels = full["input_ids"].copy()
        prompt_only = tokenizer.encode(
            example["prompt"].split(response_tag)[0] + response_tag,
            add_special_tokens=False
        )
        mask_len = len(prompt_only)
        labels[:mask_len] = [-100] * mask_len

        full["labels"] = labels
        return full

    return dataset.map(tokenize, remove_columns=["prompt", "raw"],
                       num_proc=4, desc="Tokenising")
