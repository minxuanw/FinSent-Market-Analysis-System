"""
Evaluate VADER and FinBERT on the Financial PhraseBank dataset.
Uses the 'sentences_allagree' split (highest quality annotations).

HuggingFace: https://huggingface.co/datasets/financial_phrasebank

Usage:
    python evaluate_phrasebank.py
    python evaluate_phrasebank.py --split sentences_75agree
    python evaluate_phrasebank.py --sample 500     # faster, random subset
"""
import argparse
import pandas as pd
import numpy as np
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

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR


# Label Mapping
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_phrasebank(split: str = "sentences_allagree") -> pd.DataFrame:
    """
    Load Financial PhraseBank from local data.

    Splits available:
      - sentences_allagree   (~2245 sentences, all annotators agree)
      - sentences_75agree    (~3452 sentences, 75% agreement)
      - sentences_66agree    (~4211 sentences, 66% agreement)
      - sentences_50agree    (~4840 sentences, 50% agreement)
    """
    print(f"\n  Loading Financial PhraseBank ({split})...")

    # Map split name to filename
    file_map = {
        "sentences_allagree": "Sentences_AllAgree.txt",
        "sentences_75agree": "Sentences_75Agree.txt",
        "sentences_66agree": "Sentences_66Agree.txt",
        "sentences_50agree": "Sentences_50Agree.txt",
    }
    filename = file_map.get(split, "Sentences_AllAgree.txt")

    # Try local data directory first
    local_paths = [
        Path(__file__).parent / "data" / "phrasebank" / "FinancialPhraseBank-v1.0" / filename,
        Path(__file__).parent / "data" / "raw" / filename,
    ]

    filepath = None
    for p in local_paths:
        if p.exists():
            filepath = p
            break

    if filepath is None:
        # Download from HuggingFace Hub
        print("    Downloading from HuggingFace Hub...")
        from huggingface_hub import hf_hub_download
        import zipfile, tempfile, shutil

        zip_path = hf_hub_download(
            'takala/financial_phrasebank',
            'data/FinancialPhraseBank-v1.0.zip',
            repo_type='dataset'
        )
        extract_dir = Path(__file__).parent / "data" / "phrasebank"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        filepath = extract_dir / "FinancialPhraseBank-v1.0" / filename

    # Parse the file: format is "sentence@label\r\n" encoded in latin-1
    rows = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or '@' not in line:
                continue
            # Split on last @ (some sentences contain @)
            sentence, label = line.rsplit('@', 1)
            rows.append({
                "sentence": sentence.strip(),
                "label_text": label.strip(),
            })

    df = pd.DataFrame(rows)

    # Map text labels to integers
    label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label_text"].str.lower().map(label_to_int)
    df["true_label"] = df["label_text"].str.lower()

    # Drop any unmapped
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"    Loaded {len(df)} sentences from {filepath.name}")
    print(f"   Distribution: {df['true_label'].value_counts().to_dict()}")

    return df


def evaluate_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Run VADER on PhraseBank sentences."""
    from src.sentiment.vader_analyzer import VaderAnalyzer

    print(f"\n{'=' * 50}")
    print(f"  VADER Evaluation on {len(df)} sentences")
    print(f"{'=' * 50}")

    vader = VaderAnalyzer()

    results = []
    for _, row in df.iterrows():
        scores = vader.analyzer.polarity_scores(row["sentence"])
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
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

    print(f"\n{'=' * 50}")
    print(f"  FinBERT Evaluation on {len(df)} sentences")
    print(f"{'=' * 50}")

    finbert = FinBERTAnalyzer()

    # Batch inference
    texts = df["sentence"].tolist()
    all_results = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_NAMES, average="weighted"
    )
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    report = classification_report(
        y_true, y_pred, labels=LABEL_NAMES,
        output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_name,
        "n_samples": len(results_df),
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": {},
    }

    for label in LABEL_NAMES:
        if label in report:
            metrics["per_class"][label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"],
            }

    return metrics


def plot_confusion_matrix(results_df: pd.DataFrame, model_name: str,
                          save_path: Path) -> plt.Figure:
    """Plot and save confusion matrix."""
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Annotate with count + percentage
    annotations = np.array([
        [f"{cm[i][j]}\n({cm_pct[i][j]:.1f}%)" for j in range(3)]
        for i in range(3)
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
                linewidths=0.5)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix (Financial PhraseBank)", fontsize=12)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    return fig


def plot_comparison(vader_metrics: dict, finbert_metrics: dict,
                    save_path: Path) -> plt.Figure:
    """Plot side-by-side comparison of VADER vs FinBERT."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "Cohen's K"]
    vader_vals = [
        vader_metrics["accuracy"],
        vader_metrics["precision_weighted"],
        vader_metrics["recall_weighted"],
        vader_metrics["f1_weighted"],
        vader_metrics["cohen_kappa"],
    ]
    finbert_vals = [
        finbert_metrics["accuracy"],
        finbert_metrics["precision_weighted"],
        finbert_metrics["recall_weighted"],
        finbert_metrics["f1_weighted"],
        finbert_metrics["cohen_kappa"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, vader_vals, width, label="VADER",
            color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, finbert_vals, width, label="FinBERT",
                    color="#9b59b6", alpha=0.85)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("VADER vs FinBERT — Financial PhraseBank Evaluation", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Add value labels
    for bar in list(ax.containers[0]) + list(ax.containers[1]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    return fig


def plot_per_class_f1(vader_metrics: dict, finbert_metrics: dict,
                      save_path: Path) -> plt.Figure:
    """Plot per-class F1 comparison."""
    labels = LABEL_NAMES
    vader_f1 = [vader_metrics["per_class"].get(l, {}).get("f1", 0) for l in labels]
    finbert_f1 = [finbert_metrics["per_class"].get(l, {}).get("f1", 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, vader_f1, width, label="VADER", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, finbert_f1, width, label="FinBERT", color="#9b59b6", alpha=0.85)

    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score — VADER vs FinBERT", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)

    for i, (vf, ff) in enumerate(zip(vader_f1, finbert_f1)):
        ax.text(i - width / 2, vf + 0.01, f'{vf:.3f}', ha='center', fontsize=9)
        ax.text(i + width / 2, ff + 0.01, f'{ff:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    return fig


def print_report(metrics: dict):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 50}")
    print(f"  {metrics['model']} Results")
    print(f"{'=' * 50}")
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"  Precision (w):  {metrics['precision_weighted']:.4f}")
    print(f"  Recall (w):     {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (w):   {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:  {metrics['cohen_kappa']:.4f}")
    print(f"\n  Per-Class:")
    for label in LABEL_NAMES:
        cls = metrics["per_class"].get(label, {})
        print(f"    {label:>10}: P={cls.get('precision', 0):.4f}  "
              f"R={cls.get('recall', 0):.4f}  "
              f"F1={cls.get('f1', 0):.4f}  "
              f"(n={cls.get('support', 0)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Financial PhraseBank")
    parser.add_argument("--split", default="sentences_allagree",
                        choices=["sentences_allagree", "sentences_75agree",
                                 "sentences_66agree", "sentences_50agree"],
                        help="PhraseBank split (default: sentences_allagree)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N sentences (faster testing)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FinSent Evaluation — Financial PhraseBank")
    print("=" * 60)

    # Load dataset
    df = load_phrasebank(args.split)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"\n  ⚡ Sampled {args.sample} sentences")

    # --- Evaluate VADER ---
    vader_results = evaluate_vader(df)
    vader_metrics = compute_metrics(vader_results, "VADER")
    print_report(vader_metrics)

    # --- Evaluate FinBERT ---
    finbert_results = evaluate_finbert(df)
    finbert_metrics = compute_metrics(finbert_results, "FinBERT")
    print_report(finbert_metrics)

    # --- Confusion Matrices ---
    plot_confusion_matrix(vader_results, "VADER",
                          output_dir / "vader_confusion_matrix.png")
    plot_confusion_matrix(finbert_results, "FinBERT",
                          output_dir / "finbert_confusion_matrix.png")

    # --- Comparison Charts ---
    plot_comparison(vader_metrics, finbert_metrics,
                    output_dir / "model_comparison_metrics.png")
    plot_per_class_f1(vader_metrics, finbert_metrics,
                      output_dir / "per_class_f1_comparison.png")

    # --- Save Results ---
    all_metrics = {
        "dataset": "Financial PhraseBank",
        "split": args.split,
        "vader": vader_metrics,
        "finbert": finbert_metrics,
    }

    metrics_path = output_dir / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # Save prediction CSVs
    vader_results.to_csv(output_dir / "vader_predictions.csv", index=False)
    finbert_results.to_csv(output_dir / "finbert_predictions.csv", index=False)
    print(f"  Predictions saved: {output_dir}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n  Dataset: Financial PhraseBank ({args.split})")
    print(f"  Samples: {len(df)}")
    print(f"\n  {'Metric':<20} {'VADER':>8} {'FinBERT':>8} {'Δ':>8}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")

    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision (w)"),
        ("recall_weighted", "Recall (w)"),
        ("f1_weighted", "F1 (w)"),
        ("cohen_kappa", "Cohen's Kappa"),
    ]:
        vv = vader_metrics[metric_key]
        fv = finbert_metrics[metric_key]
        delta = fv - vv
        sign = "+" if delta > 0 else ""
        print(f"  {metric_label:<20} {vv:>8.4f} {fv:>8.4f} {sign}{delta:>7.4f}")

    winner = "FinBERT" if finbert_metrics["accuracy"] > vader_metrics["accuracy"] else "VADER"
    margin = abs(finbert_metrics["accuracy"] - vader_metrics["accuracy"]) * 100
    print(f"\n  → {winner} wins by {margin:.1f}% accuracy margin")
    print()


if __name__ == "__main__":
    main()
