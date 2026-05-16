"""
Evaluate VADER and FinBERT on the Financial PhraseBank dataset.
Uses the 'sentences_allagree' split (highest quality annotations).

HuggingFace: https://huggingface.co/datasets/financial_phrasebank

Usage:
    python evaluate_phrasebank.py
    python evaluate_phrasebank.py --split sentences_agree
    python evaluate_phrasebank.py --sample     faster, random subset
"""
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(, str(Path(__file__).parent))
from config import PROCESSED_DIR


  Label Mapping 
LABEL_MAP = {: "negative", : "neutral", : "positive"}
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_phrasebank(split: str = "sentences_allagree") -> pd.DataFrame:
    """
    Load Financial PhraseBank from local data.
    
    Splits available:
      - sentences_allagree   (~, sentences, all annotators agree)
      - sentences_agree    (~, sentences, % agreement)
      - sentences_agree    (~, sentences, % agreement)
      - sentences_agree    (~, sentences, % agreement)
    """
    print(f"\n Loading Financial PhraseBank ({split})...")
    
     Map split name to filename
    file_map = {
        "sentences_allagree": "Sentences_AllAgree.txt",
        "sentences_agree": "Sentences_Agree.txt",
        "sentences_agree": "Sentences_Agree.txt",
        "sentences_agree": "Sentences_Agree.txt",
    }
    filename = file_map.get(split, "Sentences_AllAgree.txt")
    
     Try local data directory first
    local_paths = [
        Path(__file__).parent / "data" / "phrasebank" / "FinancialPhraseBank-v." / filename,
        Path(__file__).parent / "data" / "raw" / filename,
    ]
    
    filepath = None
    for p in local_paths:
        if p.exists():
            filepath = p
            break
    
    if filepath is None:
         Download from HuggingFace Hub
        print("   Downloading from HuggingFace Hub...")
        from huggingface_hub import hf_hub_download
        import zipfile, tempfile, shutil
        
        zip_path = hf_hub_download(
            'takala/financial_phrasebank',
            'data/FinancialPhraseBank-v..zip',
            repo_type='dataset'
        )
        extract_dir = Path(__file__).parent / "data" / "phrasebank"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        filepath = extract_dir / "FinancialPhraseBank-v." / filename
    
     Parse the file: format is "sentence@label\r\n" encoded in latin-
    rows = []
    with open(filepath, 'r', encoding='latin-') as f:
        for line in f:
            line = line.strip()
            if not line or '@' not in line:
                continue
             Split on last @ (some sentences contain @)
            sentence, label = line.rsplit('@', )
            rows.append({
                "sentence": sentence.strip(),
                "label_text": label.strip(),
            })
    
    df = pd.DataFrame(rows)
    
     Map text labels to integers
    label_to_int = {"negative": , "neutral": , "positive": }
    df["label"] = df["label_text"].str.lower().map(label_to_int)
    df["true_label"] = df["label_text"].str.lower()
    
     Drop any unmapped
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    
    print(f"   Loaded {len(df)} sentences from {filepath.name}")
    print(f"  Distribution: {df['true_label'].value_counts().to_dict()}")
    
    return df


def evaluate_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Run VADER on PhraseBank sentences."""
    from src.sentiment.vader_analyzer import VaderAnalyzer
    
    print(f"\n{'='}")
    print(f"  VADER Evaluation on {len(df)} sentences")
    print(f"{'='}")
    
    vader = VaderAnalyzer()
    
    results = []
    for _, row in df.iterrows():
        scores = vader.analyzer.polarity_scores(row["sentence"])
        compound = scores["compound"]
        
        if compound >= .:
            label = "positive"
        elif compound <= -.:
            label = "negative"
        else:
            label = "neutral"
        
        results.append({
            "sentence": row["sentence"],
            "true_label": row["true_label"],
            "pred_label": label,
            "compound": compound,
            "pos_score": scores["pos"],
            "neg_score": scores["neg"],
            "neu_score": scores["neu"],
        })
    
    return pd.DataFrame(results)


def evaluate_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT on PhraseBank sentences."""
    from src.sentiment.finbert_analyzer import FinBERTAnalyzer
    
    print(f"\n{'='}")
    print(f"  FinBERT Evaluation on {len(df)} sentences")
    print(f"{'='}")
    
    finbert = FinBERTAnalyzer()
    
     Batch inference
    texts = df["sentence"].tolist()
    all_results = []
    batch_size = 
    
    for i in range(, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = finbert._analyze_batch_internal(batch)
        all_results.extend(batch_results)
    
    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        r = all_results[idx]
        results.append({
            "sentence": row["sentence"],
            "true_label": row["true_label"],
            "pred_label": r["label"],
            "compound": r["compound_score"],
            "pos_score": r["score_positive"],
            "neg_score": r["score_negative"],
            "neu_score": r["score_neutral"],
        })
    
    return pd.DataFrame(results)


def compute_metrics(results_df: pd.DataFrame, model_name: str) -> dict:
    """Compute classification metrics."""
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_NAMES, average="weighted"
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    
     Per-class metrics
    report = classification_report(
        y_true, y_pred, labels=LABEL_NAMES,
        output_dict=True, zero_division=
    )
    
    metrics = {
        "model": model_name,
        "n_samples": len(results_df),
        "accuracy": round(accuracy, ),
        "precision_weighted": round(precision, ),
        "recall_weighted": round(recall, ),
        "f_weighted": round(f, ),
        "cohen_kappa": round(kappa, ),
        "per_class": {},
    }
    
    for label in LABEL_NAMES:
        if label in report:
            metrics["per_class"][label] = {
                "precision": round(report[label]["precision"], ),
                "recall": round(report[label]["recall"], ),
                "f": round(report[label]["f-score"], ),
                "support": report[label]["support"],
            }
    
    return metrics


def plot_confusion_matrix(results_df: pd.DataFrame, model_name: str,
                          save_path: Path) -> plt.Figure:
    """Plot and save confusion matrix."""
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]
    
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    cm_pct = cm.astype("float") / cm.sum(axis=)[:, np.newaxis]  
    
     Annotate with count + percentage
    annotations = np.array([
        [f"{cm[i][j]}\n({cm_pct[i][j]:.f}%)" for j in range()]
        for i in range()
    ])
    
    fig, ax = plt.subplots(figsize=(, ))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
                linewidths=.)
    ax.set_xlabel("Predicted Label", fontsize=)
    ax.set_ylabel("True Label", fontsize=)
    ax.set_title(f"{model_name}  Confusion Matrix (Financial PhraseBank)", fontsize=)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=, bbox_inches="tight")
    print(f"   Saved: {save_path}")
    return fig


def plot_comparison(vader_metrics: dict, finbert_metrics: dict,
                    save_path: Path) -> plt.Figure:
    """Plot side-by-side comparison of VADER vs FinBERT."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F", "Cohen's Î"]
    vader_vals = [
        vader_metrics["accuracy"],
        vader_metrics["precision_weighted"],
        vader_metrics["recall_weighted"],
        vader_metrics["f_weighted"],
        vader_metrics["cohen_kappa"],
    ]
    finbert_vals = [
        finbert_metrics["accuracy"],
        finbert_metrics["precision_weighted"],
        finbert_metrics["recall_weighted"],
        finbert_metrics["f_weighted"],
        finbert_metrics["cohen_kappa"],
    ]
    
    x = np.arange(len(metrics_names))
    width = .
    
    fig, ax = plt.subplots(figsize=(, ))
    bars = ax.bar(x - width/, vader_vals, width, label="VADER",
                   color="db", alpha=.)
    bars = ax.bar(x + width/, finbert_vals, width, label="FinBERT",
                   color="bb", alpha=.)
    
    ax.set_ylabel("Score", fontsize=)
    ax.set_title("VADER vs FinBERT  Financial PhraseBank Evaluation", fontsize=)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=)
    ax.set_ylim(, .)
    ax.legend(fontsize=)
    ax.axhline(y=., color="gray", linestyle="--", alpha=.)
    
     Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/., height,
                f'{height:.f}', ha='center', va='bottom', fontsize=)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/., height,
                f'{height:.f}', ha='center', va='bottom', fontsize=)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=, bbox_inches="tight")
    print(f"   Saved: {save_path}")
    return fig


def plot_per_class_f(vader_metrics: dict, finbert_metrics: dict,
                      save_path: Path) -> plt.Figure:
    """Plot per-class F comparison."""
    labels = LABEL_NAMES
    vader_f = [vader_metrics["per_class"].get(l, {}).get("f", ) for l in labels]
    finbert_f = [finbert_metrics["per_class"].get(l, {}).get("f", ) for l in labels]
    
    x = np.arange(len(labels))
    width = .
    
    fig, ax = plt.subplots(figsize=(, ))
    ax.bar(x - width/, vader_f, width, label="VADER", color="db", alpha=.)
    ax.bar(x + width/, finbert_f, width, label="FinBERT", color="bb", alpha=.)
    
    ax.set_ylabel("F Score", fontsize=)
    ax.set_title("Per-Class F Score  VADER vs FinBERT", fontsize=)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels], fontsize=)
    ax.set_ylim(, .)
    ax.legend(fontsize=)
    
    for i, (vf, ff) in enumerate(zip(vader_f, finbert_f)):
        ax.text(i - width/, vf + ., f'{vf:.f}', ha='center', fontsize=)
        ax.text(i + width/, ff + ., f'{ff:.f}', ha='center', fontsize=)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=, bbox_inches="tight")
    print(f"   Saved: {save_path}")
    return fig


def print_report(metrics: dict):
    """Print formatted evaluation report."""
    print(f"\n{''}")
    print(f"  {metrics['model']} Results")
    print(f"{''}")
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  Accuracy:       {metrics['accuracy']:.f} ({metrics['accuracy']:.f}%)")
    print(f"  Precision (w):  {metrics['precision_weighted']:.f}")
    print(f"  Recall (w):     {metrics['recall_weighted']:.f}")
    print(f"  F Score (w):   {metrics['f_weighted']:.f}")
    print(f"  Cohen's Î:      {metrics['cohen_kappa']:.f}")
    print(f"\n  Per-Class:")
    for label in LABEL_NAMES:
        cls = metrics["per_class"].get(label, {})
        print(f"    {label:>}: P={cls.get('precision',):.f}  "
              f"R={cls.get('recall',):.f}  "
              f"F={cls.get('f',):.f}  "
              f"(n={cls.get('support',)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Financial PhraseBank")
    parser.add_argument("--split", default="sentences_allagree",
                       choices=["sentences_allagree", "sentences_agree",
                                "sentences_agree", "sentences_agree"],
                       help="PhraseBank split (default: sentences_allagree)")
    parser.add_argument("--sample", type=int, default=None,
                       help="Random sample N sentences (faster testing)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=")
    print("  FinSent Evaluation  Financial PhraseBank")
    print("=")
    
     Load dataset
    df = load_phrasebank(args.split)
    
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=).reset_index(drop=True)
        print(f"\n   Sampled {args.sample} sentences")
    
      Evaluate VADER 
    vader_results = evaluate_vader(df)
    vader_metrics = compute_metrics(vader_results, "VADER")
    print_report(vader_metrics)
    
      Evaluate FinBERT 
    finbert_results = evaluate_finbert(df)
    finbert_metrics = compute_metrics(finbert_results, "FinBERT")
    print_report(finbert_metrics)
    
      Confusion Matrices 
    plot_confusion_matrix(vader_results, "VADER",
                          output_dir / "vader_confusion_matrix.png")
    plot_confusion_matrix(finbert_results, "FinBERT",
                          output_dir / "finbert_confusion_matrix.png")
    
      Comparison Charts 
    plot_comparison(vader_metrics, finbert_metrics,
                    output_dir / "model_comparison_metrics.png")
    plot_per_class_f(vader_metrics, finbert_metrics,
                      output_dir / "per_class_f_comparison.png")
    
      Save Results 
    all_metrics = {
        "dataset": "Financial PhraseBank",
        "split": args.split,
        "vader": vader_metrics,
        "finbert": finbert_metrics,
    }
    
    metrics_path = output_dir / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=)
    print(f"\n Metrics saved: {metrics_path}")
    
     Save prediction CSVs
    vader_results.to_csv(output_dir / "vader_predictions.csv", index=False)
    finbert_results.to_csv(output_dir / "finbert_predictions.csv", index=False)
    print(f" Predictions saved: {output_dir}")
    
      Summary 
    print("\n" + "=")
    print("  EVALUATION SUMMARY")
    print("=")
    print(f"\n  Dataset: Financial PhraseBank ({args.split})")
    print(f"  Samples: {len(df)}")
    print(f"\n  {'Metric':<} {'VADER':>} {'FinBERT':>} {'Î':>}")
    print(f"  {''}")
    
    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision (w)"),
        ("recall_weighted", "Recall (w)"),
        ("f_weighted", "F (w)"),
        ("cohen_kappa", "Cohen's Î"),
    ]:
        vv = vader_metrics[metric_key]
        fv = finbert_metrics[metric_key]
        delta = fv - vv
        sign = "+" if delta >  else ""
        print(f"  {metric_label:<} {vv:>.f} {fv:>.f} {sign}{delta:>.f}")
    
    winner = "FinBERT" if finbert_metrics["accuracy"] > vader_metrics["accuracy"] else "VADER"
    margin = abs(finbert_metrics["accuracy"] - vader_metrics["accuracy"])  
    print(f"\n   {winner} wins by {margin:.f}% accuracy margin")
    print()


if __name__ == "__main__":
    main()
