import streamlit as st
import pandas as pd
from pipeline_engine import TransformationEngine, AIAgent

st.set_page_config(page_title="Data Governance Pipeline", layout="wide")
st.title("🚀 AI-Powered Data Governance & Insight Pipeline")

# Initialize AI Agent once and cache it[cite: 2]
@st.cache_resource
def get_ai_agent():
    return AIAgent()

ai_agent = get_ai_agent()

uploaded_file = st.file_uploader("Upload Retail/Financial Dataset (CSV)", type=["csv", "xlsx"])

if uploaded_file:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    else:
        st.error("Unsupported file format")
        st.stop()
    engine = TransformationEngine(df)
    
    tab1, tab2, tab3 = st.tabs(["Governance Report", "Business Transformation", "AI Strategy"])
    
    with tab1:
        st.subheader("📋 Data Integrity Report")
        st.json(engine.run_integrity_check())
        
    with tab2:
        st.subheader("⚙️ Transformed Data")
        processed_df = engine.apply_business_rules()
        st.dataframe(processed_df.head(10))
        
    with tab3:
        if st.button("Generate Strategic Analysis"):
            with st.spinner("AI analyzing metrics..."):
                summary = {
                            "avg_order_value_mean": processed_df["avg_order_value"].mean(),
                            "high_risk_pct": (processed_df["Risk_Level"] == "High").mean(),
                            "vip_customers": (processed_df["Segment"] == "VIP").sum()
                        }
                prompt = f"As a consultant, provide 3 strategic recommendations for a CEO based on this data summary: {stats}"
                recommendations = ai_agent.generate_response(prompt)
                st.success(recommendations)
