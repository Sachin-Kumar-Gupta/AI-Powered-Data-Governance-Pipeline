import streamlit as st
import pandas as pd
from pipeline_engine import TransformationEngine

st.set_page_config(page_title="Data Governance Pipeline", layout="wide")
st.title("🚀 AI-Powered Data Governance & Insight Pipeline")

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "Upload Dataset (CSV or XLSX)",
    type=["csv", "xlsx"]
)

df = None

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

    tab1, tab2, tab3 = st.tabs([
        "Governance Report",
        "Schema Mapping",
        "Business Insights"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("📋 Data Integrity Report")
        st.json(engine.run_integrity_check())

        st.write("Preview Data")
        st.dataframe(df.head())

    # ---------------- TAB 2 (Schema Mapping) ----------------
    with tab2:
        st.subheader("🔗 Schema Mapping Layer")

        columns = ["None"] + df.columns.tolist()

        purchase_col = st.selectbox("Map Purchase Count", columns)
        avg_order_col = st.selectbox("Map Avg Order Value", columns)
        retention_col = st.selectbox("Map Retention Days", columns)

        mapping = {
            "purchase_count": purchase_col,
            "avg_order_value": avg_order_col,
            "retention_days": retention_col
        }

        st.session_state["mapping"] = mapping

        st.info("Mapping saved. Go to Business Insights tab.")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("⚙️ AI Business Transformation")

        if "mapping" not in st.session_state:
            st.warning("Please configure schema mapping first.")
            st.stop()

        if st.button("Generate Insights"):
            processed_df = engine.apply_business_rules(
                st.session_state["mapping"]
            )

            st.dataframe(processed_df.head(20))
