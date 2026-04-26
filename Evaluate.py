import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataset import build_icd_dataset, build_ner_dataset, build_summarization_dataset
from inference import ClinicalLoRA

def evaluate_icd(model: ClinicalLoRA, test_dataset,
                 n_samples: int = 500) -> dict:
    tp_total = fp_total = fn_total = 0

    for i, sample in enumerate(tqdm(test_dataset.select(range(n_samples)),
                                    desc="ICD eval")):
        raw      = sample["raw"]
        gold_set = {c["code"] for c in raw["icd_codes"]}
        pred     = model.predict(text=raw["text"])

        if isinstance(pred, list):
            pred_set = {p["code"] for p in pred if isinstance(p, dict) and "code" in p}
        else:
            pred_set = set()

        tp_total += len(gold_set & pred_set)
        fp_total += len(pred_set - gold_set)
        fn_total += len(gold_set - pred_set)

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall    = tp_total / (tp_total + fn_total + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    results = {
        "precision_at_5": round(precision, 4),
        "recall_at_5":    round(recall, 4),
        "micro_f1":       round(f1, 4),
        "n_samples":      n_samples,
    }
    _print_results("ICD Coding", results)
    return results


def evaluate_summarization(model: ClinicalLoRA, test_dataset,
                           n_samples: int = 200) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)

    r1s, r2s, rLs = [], [], []
    preds, refs   = [], []

    for sample in tqdm(test_dataset.select(range(n_samples)), desc="ROUGE"):
        raw       = sample["raw"]
        reference = raw["summary"]
        pred      = model.predict(text=raw["text"])

        scores = scorer.score(reference, pred)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rLs.append(scores["rougeL"].fmeasure)
        preds.append(pred)
        refs.append(reference)

    # BERTScore 
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            preds, refs,
            model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            lang="en", verbose=False
        )
        bs_f1 = float(F1.mean())
    except ImportError:
        bs_f1 = None

    results = {
        "rouge1":      round(float(np.mean(r1s)), 4),
        "rouge2":      round(float(np.mean(r2s)), 4),
        "rougeL":      round(float(np.mean(rLs)), 4),
        "bertscore_f1": round(bs_f1, 4) if bs_f1 else "install bert-score",
        "n_samples":   n_samples,
    }
    _print_results("Summarization", results)
    return results


# NER 

def evaluate_ner(model: ClinicalLoRA, test_dataset,
                 n_samples: int = 500) -> dict:
    entity_types = ["PROBLEM", "TREATMENT", "TEST"]
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in entity_types}
    stats["overall"] = {"tp": 0, "fp": 0, "fn": 0}

    for sample in tqdm(test_dataset.select(range(n_samples)), desc="NER eval"):
        raw   = sample["raw"]
        gold  = {(e["entity"].lower(), e["type"]) for e in raw["entities"]}
        pred_raw = model.predict(sentence=raw["sentence"])

        if isinstance(pred_raw, list):
            pred = {(p["entity"].lower(), p["type"])
                    for p in pred_raw
                    if isinstance(p, dict) and "entity" in p and "type" in p}
        else:
            pred = set()

        for etype in entity_types:
            g = {e for e in gold if e[1] == etype}
            p = {e for e in pred if e[1] == etype}
            stats[etype]["tp"] += len(g & p)
            stats[etype]["fp"] += len(p - g)
            stats[etype]["fn"] += len(g - p)

        stats["overall"]["tp"] += len(gold & pred)
        stats["overall"]["fp"] += len(pred - gold)
        stats["overall"]["fn"] += len(gold - pred)

    results = {}
    for key, s in stats.items():
        p  = s["tp"] / (s["tp"] + s["fp"] + 1e-9)
        r  = s["tp"] / (s["tp"] + s["fn"] + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        results[key] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        }

    _print_results("NER", results)
    return results


def _print_results(task: str, results: dict):
    print(f"\n{'─'*50}")
    print(f"  {task} results")
    print(f"{'─'*50}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:<15} {vv}")
        else:
            print(f"  {k:<20} {v}")


# CLI 
EVAL_FNS   = {
    "icd_coding":    evaluate_icd,
    "summarization": evaluate_summarization,
    "ner":           evaluate_ner,
}
BUILD_FNS  = {
    "icd_coding":    build_icd_dataset,
    "summarization": build_summarization_dataset,
    "ner":           build_ner_dataset,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       required=True,
                        choices=list(EVAL_FNS))
    parser.add_argument("--adapter",    required=True)
    parser.add_argument("--mimic_root", default=None)
    parser.add_argument("--n2c2_root",  default=None)
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    model = ClinicalLoRA(adapter_path=args.adapter, task=args.task)

    root   = args.n2c2_root if args.task == "ner" else args.mimic_root
    ds     = BUILD_FNS[args.task](root)
    result = EVAL_FNS[args.task](model, ds["test"], n_samples=args.n_samples)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nSaved → {args.output}")
