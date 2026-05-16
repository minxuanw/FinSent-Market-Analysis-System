"""
FinBERT Sentiment Analyzer — Advanced model.
Domain-specific transformer model for financial sentiment classification.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import sys
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)


class FinBERTAnalyzer:
    """FinBERT-based sentiment analyzer for financial text."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name_id = model_name or config.FINBERT_MODEL
        self.model_label = "FinBERT"
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f"  Loading FinBERT model: {self.model_name_id}")
        print(f"    Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_id)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping from FinBERT
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
        # Note: FinBERT label order may vary by model version; we'll use the model's config
        if hasattr(self.model.config, "id2label"):
            self.label_map = {int(k): v for k, v in self.model.config.id2label.items()}

        print(f"  ✓ Model loaded. Labels: {self.label_map}")

    def analyze_single(self, text: str) -> dict:
        """Analyze sentiment of a single text using FinBERT."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        label = self.label_map.get(predicted_class, "neutral")
        prob_values = probs.cpu().numpy()[0]

        # Map probability values to standard field names
        # FinBERT outputs: positive, negative, neutral probabilities
        result = {
            "model": self.model_label,
            "label": label,
            "compound_score": 0.0,
            "score_positive": 0.0,
            "score_negative": 0.0,
            "score_neutral": 0.0,
        }

        for idx, prob in enumerate(prob_values):
            lbl = self.label_map.get(idx, "")
            if lbl == "positive":
                result["score_positive"] = float(prob)
            elif lbl == "negative":
                result["score_negative"] = float(prob)
            elif lbl == "neutral":
                result["score_neutral"] = float(prob)

        # Compute compound score: positive - negative (normalized)
        result["compound_score"] = result["score_positive"] - result["score_negative"]

        return result

    def analyze_batch(self, df: pd.DataFrame, text_col: str = "headline_clean",
                      batch_size: int = 32) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts using batched inference.

        Args:
            df: DataFrame with text column
            text_col: Name of the text column
            batch_size: Batch size for inference

        Returns:
            Original DataFrame with sentiment columns appended
        """
        print(f"\n  FinBERT: Analyzing {len(df)} headlines...")

        texts = df[text_col].fillna("").tolist()
        all_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT batches"):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._analyze_batch_internal(batch_texts)
            all_results.extend(batch_results)

        sentiment_df = pd.DataFrame(all_results)
        combined = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

        # Print summary
        label_counts = combined["label"].value_counts()
        print(f"    Results: ", end="")
        for label, count in label_counts.items():
            print(f"{label}={count} ({count/len(combined):.1f}%)  ", end="")
        print()

        return combined

    def _analyze_batch_internal(self, texts: list) -> list:
        """Process a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1).cpu().numpy()
            prob_values = probs.cpu().numpy()

        results = []
        for idx in range(len(texts)):
            pred_class = int(predicted_classes[idx])
            label = self.label_map.get(pred_class, "neutral")

            result = {
                "model": self.model_label,
                "label": label,
                "compound_score": 0.0,
                "score_positive": 0.0,
                "score_negative": 0.0,
                "score_neutral": 0.0,
            }

            for cls_idx, prob in enumerate(prob_values[idx]):
                lbl = self.label_map.get(cls_idx, "")
                if lbl == "positive":
                    result["score_positive"] = float(prob)
                elif lbl == "negative":
                    result["score_negative"] = float(prob)
                elif lbl == "neutral":
                    result["score_neutral"] = float(prob)

            result["compound_score"] = result["score_positive"] - result["score_negative"]
            results.append(result)

        return results

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment to daily level by ticker."""
        daily = df.groupby(["ticker", "date"]).agg(
            avg_compound=("compound_score", "mean"),
            avg_positive=("score_positive", "mean"),
            avg_negative=("score_negative", "mean"),
            avg_neutral=("score_neutral", "mean"),
            article_count=("headline", "count"),
        ).reset_index()
        daily["model"] = self.model_label
        return daily


if __name__ == "__main__":
    analyzer = FinBERTAnalyzer()

    # Test with sample headlines
    test_headlines = [
        "Apple reports record quarterly earnings, beating analyst expectations",
        "Market crashes as inflation fears grow among investors",
        "Federal Reserve holds interest rates steady amid economic uncertainty",
        "Tesla stock surges after announcing new model lineup",
        "Bank of America warns of potential recession in 2026",
    ]

    print("\n  FinBERT Sentiment Analysis Test:\n")
    for headline in test_headlines:
        result = analyzer.analyze_single(headline)
        print(f"  [{result['label'].upper():>8}] (pos={result['score_positive']:.3f}, neg={result['score_negative']:.3f})  {headline}")
