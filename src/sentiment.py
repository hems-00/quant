from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

class FinBERTSentiment:
    def __init__(self, model_path="C:/hf_models/finbert", device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.labels = self.model.config.id2label

        self.pos_idx = next(
            k for k, v in self.labels.items()
            if "positive" in v.lower()
        )

        self.neg_idx = next(
            k for k, v in self.labels.items()
            if "negative" in v.lower()
        )

        print(f"FinBERT loaded on {self.device}")

    def predict_batch(self, texts, batch_size=32):
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)

            pos_idx = [k for k,v in self.labels.items()
                    if "positive" in v.lower()][0]

            neg_idx = [k for k,v in self.labels.items()
                    if "negative" in v.lower()][0]

            sentiment = probs[:, pos_idx] - probs[:, neg_idx]

            scores.extend(sentiment.cpu().numpy())

        return scores

    def process_headlines(self, df):
        df = df.copy()

        df["sentiment_score"] = self.predict_batch(
            df["headline"].astype(str).tolist()
        )

        daily = df.groupby("date").agg(
            sent_mean=("sentiment_score", "mean"),
            sent_variance=("sentiment_score", "var"),
            headline_volume=("sentiment_score", "count")
        )

        daily = daily.fillna(0)

        return daily

