import pandas as pd
import torch
from transformers import pipeline

class TransformationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run_integrity_check(self) -> dict:
        return {
            "Total Rows": len(self.df),
            "Null Values": self.df.isnull().sum().to_dict(),
            "Duplicate Rows": int(self.df.duplicated().sum()),
            "Schema": self.df.dtypes.astype(str).to_dict()
        }

    def apply_business_rules(self, mapping: dict) -> pd.DataFrame:
        """
        mapping format:
        {
            "purchase_count": "col_name or None",
            "avg_order_value": "col_name or None",
            "retention_days": "col_name or None"
        }
        """

        def get_col(col_name):
            selected = mapping.get(col_name)
            if selected and selected in self.df.columns:
                return self.df[selected]
            return 0  # fallback

        self.df["Health_Score"] = (
            get_col("purchase_count") * 0.4 +
            get_col("avg_order_value") * 0.4 +
            get_col("retention_days") * 0.2
        ).clip(0, 100)

        self.df["Risk_Level"] = self.df["Health_Score"].apply(
            lambda x: "High" if x < 30 else ("Medium" if x < 70 else "Low")
        )

        if "avg_order_value" in mapping.values():
            col = mapping["avg_order_value"]
            if col in self.df.columns:
                self.df["Segment"] = pd.qcut(
                    self.df[col],
                    3,
                    labels=["Budget", "Mid-Tier", "VIP"]
                )
            else:
                self.df["Segment"] = "Unknown"
        else:
            self.df["Segment"] = "Unknown"

        return self.df

class AIAgent:
    def __init__(self):
        # Optimized for Streamlit Cloud (CPU-friendly)[cite: 2]
        self.model_id = "distilgpt2"
        self.pipe = pipeline(
            "text-generation", 
            model=self.model_id,
            device = -1
        )

    def generate_response(self, prompt: str) -> str:
        """General function to query the local LLM[cite: 2]."""
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages, max_new_tokens=150, temperature=0.7,do_sample=True)
        return output[0]['generated_text']
