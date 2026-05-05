import pandas as pd
import torch
from transformers import pipeline

class TransformationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_integrity_check(self) -> dict:
        """Identifies nulls, duplicates, and schema issues."""
        return {
            "Total Rows": len(self.df),
            "Null Values": self.df.isnull().sum().to_dict(),
            "Duplicate Rows": int(self.df.duplicated().sum()),
            "Schema": self.df.dtypes.apply(lambda x: str(x)).to_dict()
        }

    def apply_business_rules(self) -> pd.DataFrame:
        """Applies complex multi-column health and risk logic."""
        # Rule 1: Customer Health Score
        self.df['Health_Score'] = (
            (self.df['purchase_count'] * 0.4) + 
            (self.df['avg_order_value'] * 0.4) + 
            (self.df['retention_days'] * 0.2)
        ).clip(0, 100)

        # Rule 2: Risk Classification
        self.df['Risk_Level'] = self.df['Health_Score'].apply(
            lambda x: 'High' if x < 30 else ('Medium' if x < 70 else 'Low')
        )

        # Rule 3: Tiering
        self.df['Segment'] = pd.qcut(self.df['avg_order_value'], 3, labels=["Budget", "Mid-Tier", "VIP"])
        return self.df

class AIAgent:
    def __init__(self):
        # Optimized for Streamlit Cloud (CPU-friendly)[cite: 2]
        self.model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.pipe = pipeline(
            "text-generation", 
            model=self.model_id, 
            device_map="auto",
            torch_dtype="auto"
        )

    def generate_response(self, prompt: str) -> str:
        """General function to query the local LLM[cite: 2]."""
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages, max_new_tokens=400, temperature=0.7)
        return output[0]['generated_text']