import streamlit as st
import pandas as pd
from pathlib import Path
import io
from typing import Dict, Any, Optional
import logging
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration and API client (original patterns!)
try:
    from frontend.config import config
    from frontend.utils.api_client import api_client
except ImportError:
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from frontend.config import config
    from frontend.utils.api_client import api_client

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- PREMIUM AI REMEMBERED CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%); font-family: 'Inter',sans-serif;}
#MainMenu, footer, header {visibility:hidden;}
.ai-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:2.5rem 2rem; border-radius:1.5rem;margin-bottom:2rem;box-shadow:0 16px 40px #667eea45;text-shadow:0 2px 22px #000a;}
.ai-header h1 {font-size:2.5rem; font-weight:700; color:white; margin:0;}
.ai-header p {font-size:1.2rem; color:rgba(255,255,255,0.96); margin:0.5rem 0 0;}
.metric-card, .glass-section, .success-box, .error-box {background:rgba(255,255,255,0.065);backdrop-filter:blur(6px);border-radius:0.9rem;padding:1.2rem;margin:0.6rem 0;box-shadow:0 2px 16px #2222;}
.success-box {border:1.5px solid #43e97bb7;}
.error-box {border:1.5px solid #e74c3cbb;}
.stButton>button {border-radius:0.6rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;font-weight:600;}
.stButton>button:hover {transform:translateY(-2px);box-shadow:0 8px 28px #667eeaba;}
.stat-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:1rem;padding:1.2rem 1.5rem;color:white;margin:0.5em 0;box-shadow:0 6px 18px #667eea22;}
.stat-card h3 {font-weight:700; font-size:1.7rem;}
.stMetric label {color:#a0aec0 !important;}
.stMetric [data-testid="stMetricValue"] {font-weight:700; color:white !important;}
.stDataFrame {background: rgba(255,255,255,0.04)!important; border-radius:1rem;}
select, .stSelectbox>div>div>div {background:rgba(255,255,255,0.08)!important;color:white!important;}
</style>
""", unsafe_allow_html=True)
# -------- END CSS --------

# ---- STATE INIT ----
if "file_id" not in st.session_state: st.session_state.file_id = None
if "uploaded_file" not in st.session_state: st.session_state.uploaded_file = None
if "validation_result" not in st.session_state: st.session_state.validation_result = None
if "results" not in st.session_state: st.session_state.results = None
if "selected_sku" not in st.session_state: st.session_state.selected_sku = None

# ---- HEADER ----
st.markdown("""
<div class="ai-header">
    <h1>🤖 DropSmart</h1>
    <p>Product & Price Intelligence for Dropshipping Sellers</p>
</div>
""", unsafe_allow_html=True)

# ---- TOP NAVIGATION ----
tab = st.tabs(["🏠 Home / Upload", "📊 Dashboard", "🔍 Product Detail", "📥 Export CSV"])

with tab[0]:
    # Home / Upload tab
    st.header("📤 Upload Product Data")
    st.markdown("Upload your supplier Excel file to get started with product analysis.")

    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Upload a file with product data including SKU, product_name, cost, price, shipping_cost, lead_time_days, and availability"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Type", uploaded_file.type or "application/vnd.ms-excel")

        if st.button("📤 Upload to Server", type="primary"):
            with st.spinner("Uploading file..."):
                try:
                    file_bytes = uploaded_file.read()
                    upload_response = api_client.upload_file(file_bytes, uploaded_file.name)
                    st.session_state.file_id = upload_response["file_id"]
                    st.success(f"✅ File uploaded successfully!")
                    st.info(f"**File ID:** {st.session_state.file_id}")
                    st.info(f"**Total Rows:** {upload_response['total_rows']}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    logger.error(f"Upload error: {e}", exc_info=True)

    if st.session_state.file_id:
        st.markdown("---")
        st.subheader("🔍 Schema Validation")
        if st.button("✅ Validate Schema", type="primary"):
            with st.spinner("Validating schema..."):
                try:
                    validation_result = api_client.validate_schema(st.session_state.file_id)
                    st.session_state.validation_result = validation_result
                    if validation_result["is_valid"]:
                        st.success("✅ Schema is valid!")
                        if validation_result.get("warnings"):
                            st.warning(f"⚠️ {len(validation_result['warnings'])} warnings found")
                    else:
                        st.error(f"❌ Schema validation failed with {len(validation_result['errors'])} errors")
                    with st.expander("📋 Validation Details", expanded=not validation_result["is_valid"]):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Summary**")
                            st.write(f"- Total Rows: {validation_result['total_rows']}")
                            st.write(f"- Total Columns: {validation_result['total_columns']}")
                            st.write(f"- Valid: {'✅ Yes' if validation_result['is_valid'] else '❌ No'}")
                        with col2:
                            st.write("**Missing Fields**")
                            if validation_result.get("missing_required_fields"):
                                st.error("Required:")
                                for field in validation_result["missing_required_fields"]:
                                    st.write(f"  - {field}")
                            if validation_result.get("missing_optional_fields"):
                                st.warning("Optional:")
                                for field in validation_result["missing_optional_fields"]:
                                    st.write(f"  - {field}")
                        if validation_result.get("errors"):
                            st.write("**Errors**")
                            for error in validation_result["errors"]:
                                st.error(f"- {error.get('field', 'Unknown')}: {error.get('message', 'Unknown error')}")
                        if validation_result.get("warnings"):
                            st.write("**Warnings**")
                            for warning in validation_result["warnings"]:
                                st.warning(f"- {warning}")
                    if validation_result["is_valid"]:
                        st.markdown("---")
                        if st.button("🚀 Process Products", type="primary", key="process_products_main"):
                            with st.spinner("Processing products with ML models... This may take a moment."):
                                try:
                                    results = api_client.get_results(st.session_state.file_id)
                                    st.session_state.results = results
                                    st.success(f"✅ Processed {results['total_products']} products successfully!")
                                    st.balloons()
                                    st.info("👉 Navigate to **Dashboard** to view results")
                                except Exception as e:
                                    st.error(f"❌ Processing failed: {str(e)}")
                                    logger.error(f"Processing error: {e}", exc_info=True)
                except Exception as e:
                    st.error(f"❌ Validation failed: {str(e)}")
                    logger.error(f"Validation error: {e}", exc_info=True)

with tab[1]:
    # Dashboard tab
    st.header("📊 Product Dashboard")
    st.markdown("View ranked products with viability scores, recommended prices, and risk assessments.")

    if st.session_state.file_id is None:
        st.warning("⚠️ Please upload a file first from the Home page.")
        st.info("👉 Go to **Home / Upload** to upload your Excel file")
    elif st.session_state.results is None:
        st.warning("⚠️ No results available. Please process the uploaded file.")
        if st.button("🚀 Process Products", type="primary"):
            with st.spinner("Processing products..."):
                try:
                    results = api_client.get_results(st.session_state.file_id)
                    st.session_state.results = results
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)
    else:
        results = st.session_state.results
        st.subheader("📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", results.get("total_products", 0))
        with col2:
            high_viability = sum(1 for r in results.get("results", []) if r.get("viability_class", "").lower() == "high")
            st.metric("High Viability", high_viability)
        with col3:
            high_risk = sum(1 for r in results.get("results", []) if r.get("stockout_risk_level", "").lower() == "high")
            st.metric("High Risk", high_risk)
        with col4:
            results_list = results.get("results", [])
            if results_list:
                avg_viability = sum(r.get("viability_score", 0) for r in results_list) / len(results_list)
                st.metric("Avg Viability", f"{avg_viability:.2%}")
            else:
                st.metric("Avg Viability", "N/A")
        st.markdown("---")
        st.subheader("📋 Ranked Products")
        df_data = []
        for result in results.get("results", []):
            df_data.append({
                "Rank": result.get("rank", 0),
                "SKU": result.get("sku", "N/A"),
                "Product Name": result.get("product_name", "N/A"),
                "Viability Score": result.get("viability_score", 0.0),
                "Viability Class": result.get("viability_class", "low").title(),
                "Recommended Price": f"${result.get('recommended_price', 0.0):.2f}",
                "Current Price": f"${result.get('current_price', 0.0):.2f}",
                "Margin %": f"{result.get('margin_percent', 0.0):.1f}%",
                "Stockout Risk": result.get("stockout_risk_level", "low").title(),
                "Risk Score": f"{result.get('stockout_risk_score', 0.0):.2f}",
                "Cluster ID": result.get("cluster_id", "N/A") if result.get("cluster_id") is not None else "N/A",
            })
        if not df_data:
            st.warning("No product data available")
        else:
            df = pd.DataFrame(df_data)
            col1, col2, col3 = st.columns(3)
            with col1:
                viability_options = ["High", "Medium", "Low"]
                viability_filter = st.multiselect(
                    "Filter by Viability", options=viability_options, default=[]
                )
            with col2:
                risk_options = ["High", "Medium", "Low"]
                risk_filter = st.multiselect(
                    "Filter by Risk", options=risk_options, default=[]
                )
            with col3:
                search_sku = st.text_input("Search SKU", "")
            filtered_df = df.copy()
            if viability_filter:
                filtered_df = filtered_df[filtered_df["Viability Class"].isin(viability_filter)]
            if risk_filter:
                filtered_df = filtered_df[filtered_df["Stockout Risk"].isin(risk_filter)]
            if search_sku:
                filtered_df = filtered_df[filtered_df["SKU"].str.contains(search_sku, case=False, na=False)]
            st.dataframe(
                filtered_df,
                width='stretch',
                height=400,
                hide_index=True
            )
            st.caption(f"Showing {len(filtered_df)} of {len(df)} products")
            st.markdown("---")
            st.subheader("🔍 View Product Details")
            sku_options = [r.get("sku", "N/A") for r in results.get("results", []) if r.get("sku")]
            if sku_options:
                selected_sku = st.selectbox(
                    "Select a product to view details",
                    options=sku_options,
                    key="detail_sku_selector"
                )
                if selected_sku:
                    st.session_state.selected_sku = selected_sku
                    if st.button("View Details", type="primary"):
                        st.info("👉 Navigate to **Product Detail** page to see full analysis")
            else:
                st.warning("No products available for selection")

with tab[2]:
    # Product Detail tab
    st.header("🔍 Product Detail Analysis")
    if st.session_state.results is None:
        st.warning("⚠️ No results available. Please process a file first.")
    elif st.session_state.selected_sku is None:
        st.warning("⚠️ Please select a product from the Dashboard.")
        st.info("👉 Go to **Dashboard** and select a product to view details")
    else:
        results = st.session_state.results
        selected_sku = st.session_state.selected_sku
        product = None
        for r in results.get("results", []):
            if r.get("sku") == selected_sku:
                product = r
                break
        if product is None:
            st.error("Product not found")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(product.get("product_name", "Unknown Product"))
                st.write(f"**SKU:** {product.get('sku', 'N/A')}")
                st.write(f"**Rank:** #{product.get('rank', 'N/A')}")
            with col2:
                viability_class = product.get("viability_class", "low").lower()
                viability_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.metric(
                    "Viability",
                    f"{viability_color.get(viability_class, '⚪')} {viability_class.title()}",
                    f"{product.get('viability_score', 0.0):.2%}"
                )
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Viability Score", f"{product.get('viability_score', 0.0):.2%}")
            with col2:
                st.metric("Recommended Price", f"${product.get('recommended_price', 0.0):.2f}")
            with col3:
                st.metric("Margin %", f"{product.get('margin_percent', 0.0):.1f}%")
            with col4:
                risk_level = product.get("stockout_risk_level", "low").lower()
                risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                st.metric(
                    "Stockout Risk",
                    f"{risk_color.get(risk_level, '⚪')} {risk_level.title()}"
                )
            st.markdown("---")
            st.subheader("💰 Pricing Analysis")
            col1, col2, col3 = st.columns(3)
            current_price = product.get("current_price", 0.0)
            recommended_price = product.get("recommended_price", 0.0)
            with col1:
                st.write("**Current Price**")
                st.write(f"${current_price:.2f}")
            with col2:
                st.write("**Recommended Price**")
                st.write(f"${recommended_price:.2f}")
            with col3:
                price_change = recommended_price - current_price
                price_change_pct = (price_change / current_price * 100) if current_price > 0 else 0
                st.write("**Change**")
                if price_change >= 0:
                    st.write(f"🔼 +${price_change:.2f} ({price_change_pct:+.1f}%)")
                else:
                    st.write(f"🔽 ${price_change:.2f} ({price_change_pct:.1f}%)")
            st.markdown("---")
            st.subheader("⚠️ Risk Analysis")
            st.write(f"**Risk Score:** {product.get('stockout_risk_score', 0.0):.2%}")
            st.write(f"**Risk Level:** {product.get('stockout_risk_level', 'low').title()}")
            cluster_id = product.get("cluster_id")
            if cluster_id is not None:
                st.write(f"**Cluster ID:** {cluster_id}")
            st.markdown("---")
            st.subheader("📊 Feature Importance (SHAP)")
            st.info("💡 SHAP values show how each feature contributes to the viability prediction")
            shap_values = product.get("shap_values")
            base_value = product.get("base_value")
            if shap_values and isinstance(shap_values, dict):
                shap_items = list(shap_values.items())
                shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = shap_items[:15]
                if top_features:
                    feature_names = [item[0] for item in top_features]
                    feature_values = [item[1] for item in top_features]
                    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in feature_values]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=feature_values,
                        y=feature_names,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{v:+.4f}" for v in feature_values],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Top 15 Feature Contributions (SHAP Values)",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        height=500,
                        showlegend=False,
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                        margin=dict(l=150, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if base_value is not None:
                        st.caption(f"Base value (expected output): {base_value:.4f}")
                    with st.expander("📋 View All Feature Contributions"):
                        shap_df = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP Value": feature_values,
                            "Impact": ["Positive" if v >= 0 else "Negative" for v in feature_values]
                        })
                        st.dataframe(shap_df, use_container_width=True)
                else:
                    st.warning("No SHAP values available for this product.")
            else:
                st.info("SHAP values are not available for this product. The model may not support SHAP explanations.")

with tab[3]:
    # Export CSV tab
    st.header("📥 Export Results to CSV")
    if st.session_state.file_id is None:
        st.warning("⚠️ No file uploaded. Please upload a file first.")
        st.info("👉 Go to **Home / Upload** to upload your Excel file")
    else:
        st.markdown("Export your analysis results to CSV for import into Amazon, Shopify, or your ERP system.")
        if st.session_state.results:
            results = st.session_state.results
            csv_data = []
            for result in results.get("results", []):
                csv_data.append({
                    "SKU": result.get("sku", ""),
                    "Product Name": result.get("product_name", ""),
                    "Rank": result.get("rank", 0),
                    "Viability Score": result.get("viability_score", 0.0),
                    "Viability Class": result.get("viability_class", "low"),
                    "Recommended Price": result.get("recommended_price", 0.0),
                    "Current Price": result.get("current_price", 0.0),
                    "Margin %": result.get("margin_percent", 0.0),
                    "Stockout Risk Score": result.get("stockout_risk_score", 0.0),
                    "Stockout Risk Level": result.get("stockout_risk_level", "low"),
                    "Cluster ID": result.get("cluster_id", "") if result.get("cluster_id") is not None else "",
                })
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                st.subheader("📋 Export Preview")
                st.dataframe(df_export.head(10), use_container_width=True)
                st.caption(f"Total rows: {len(df_export)}")
            else:
                st.warning("No data available for preview")
        else:
            st.info("💡 Results will be fetched from the server when you export.")
        st.markdown("---")
        if st.button("📥 Export CSV from Server", type="primary"):
            with st.spinner("Generating CSV file..."):
                try:
                    csv_bytes = api_client.export_csv(st.session_state.file_id)
                    file_id_short = st.session_state.file_id[:8] if st.session_state.file_id else "unknown"
                    filename = f"dropsmart_results_{file_id_short}.csv"
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_bytes,
                        file_name=filename,
                        mime="text/csv",
                        type="primary",
                        key="csv_download_button"
                    )
                    st.success("✅ CSV file generated successfully!")
                    st.info("💡 Click the download button above to save the CSV file.")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
                    logger.error(f"CSV export error: {e}", exc_info=True)
        if st.session_state.results:
            st.markdown("---")
            st.subheader("Alternative: Download from Cached Results")
            st.caption("This uses locally cached results. For the latest data, use the server export above.")
            csv_data = []
            for result in st.session_state.results.get("results", []):
                csv_data.append({
                    "SKU": result.get("sku", ""),
                    "Product Name": result.get("product_name", ""),
                    "Rank": result.get("rank", 0),
                    "Viability Score": result.get("viability_score", 0.0),
                    "Viability Class": result.get("viability_class", "low"),
                    "Recommended Price": result.get("recommended_price", 0.0),
                    "Current Price": result.get("current_price", 0.0),
                    "Margin %": result.get("margin_percent", 0.0),
                    "Stockout Risk Score": result.get("stockout_risk_score", 0.0),
                    "Stockout Risk Level": result.get("stockout_risk_level", "low"),
                    "Cluster ID": result.get("cluster_id", "") if result.get("cluster_id") is not None else "",
                })
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                csv_bytes_local = csv_string.encode('utf-8')
                file_id_short = st.session_state.file_id[:8] if st.session_state.file_id else "unknown"
                filename_local = f"dropsmart_results_{file_id_short}.csv"
                st.download_button(
                    label="📥 Download from Cache",
                    data=csv_bytes_local,
                    file_name=filename_local,
                    mime="text/csv",
                    key="csv_download_cache"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; padding: 2rem;">
    <p>🤖 Powered by DropSmart AI | Machine Learning Product Intelligence</p>
</div>
""", unsafe_allow_html=True)
