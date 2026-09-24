import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.logger_config import setup_logging
from src.pipeline import EarningsAnalysisPipeline

logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

# Page Configuration
st.set_page_config(
    page_title="FinSentiment AI | Financial Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    # div[data-testid="stMetric"] {
    #     background-color: #1e293b; /* Dark navy background */
    #     padding: 15px;
    #     border-radius: 10px;
    #     border: 1px solid #334155;
    # }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline() -> EarningsAnalysisPipeline:
    return EarningsAnalysisPipeline()


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Engine Options")
        st.caption("Financial NLP Pipeline Configuration")
        
        input_mode = st.radio(
            "Select Input Source",
            ["Paste Transcript", "Select Stock Ticker (Demo)"],
            index=0
        )
        
        st.divider()
        st.subheader("Model Metadata")
        st.markdown("""
        * **Sentiment Engine:** `ProsusAI/finbert`
        * **Summarizer Engine:** `facebook/bart-large-cnn`
        * **Tokenizer / Segmentation:** SpaCy (`en_core_web_sm`)
        """)
        
        st.divider()
        st.caption("Capstone Project — Financial Intelligence & NLP")
        
    return input_mode


def render_analytics(results: dict):
    metrics = results["metrics"]
    
    # Key Performance Indicators
    st.subheader("📌 Key Metric Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric(
        label="Remarks Net Sentiment",
        value=f"{metrics['remarks_net_sentiment']:.2f}",
        delta=f"{metrics['remarks_net_sentiment']:.2f}"
    )
    kpi2.metric(
        label="Q&A Net Sentiment",
        value=f"{metrics['qa_net_sentiment']:.2f}",
        delta=f"{metrics['qa_net_sentiment']:.2f}"
    )
    
    shift = metrics['sentiment_shift']
    kpi3.metric(
        label="Sentiment Shift (Q&A vs Remarks)",
        value=f"{shift:.2f}",
        delta=f"{shift:.2f}",
        delta_color="normal" if shift >= 0 else "inverse"
    )
    kpi4.metric(
        label="Total Sentences Parsed",
        value=metrics["total_sentences"]
    )

    st.divider()

    # Tabbed Interface for Detailed Views
    tab_summary, tab_visuals, tab_data = st.tabs([
        "📝 Executive Summary", 
        "📊 Sentiment Dynamics", 
        "🔍 Sentence Inspector"
    ])

    with tab_summary:
        st.subheader("Abstractive Executive Summary")
        st.info(results["summary"])
        st.caption("Generated via BART-Large-CNN transformer model.")

    with tab_visuals:
        all_data = []
        for item in results["remarks_details"]:
            all_data.append({**item, "section": "Remarks"})
        for item in results["qa_details"]:
            all_data.append({**item, "section": "Q&A"})

        if all_data:
            df = pd.DataFrame(all_data)
            df['sentence_idx'] = range(1, len(df) + 1)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### Net Sentiment Score Distribution")
                fig_hist = px.histogram(
                    df,
                    x="net_sentiment",
                    color="section",
                    barmode="overlay",
                    nbins=20,
                    color_discrete_map={"Remarks": "#1f77b4", "Q&A": "#ff7f0e"},
                    labels={"net_sentiment": "Net Score (Positive - Negative)"}
                )
                fig_hist.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_right:
                st.markdown("#### Narrative Sentiment Flow")
                fig_line = px.line(
                    df,
                    x="sentence_idx",
                    y="net_sentiment",
                    color="section",
                    markers=True,
                    labels={"sentence_idx": "Sentence Sequence", "net_sentiment": "Net Sentiment"},
                    color_discrete_map={"Remarks": "#1f77b4", "Q&A": "#ff7f0e"}
                )
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig_line.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No structured sentence data available for visualization.")

    with tab_data:
        st.subheader("Raw Sentiment Output Data")
        if all_data:
            st.dataframe(
                df[["section", "label", "net_sentiment", "sentence"]],
                use_container_width=True,
                height=400
            )


def main():
    input_mode = render_sidebar()
    
    st.title("📈 Financial Sentiment & Earnings Analysis Engine")
    st.markdown("Automated sentiment classification and abstractive transcript summarization for financial analysis.")
    st.divider()

    pipeline = load_pipeline()

    default_text = """Executive Presentation:
We delivered exceptional revenue growth of 24% year-over-year, hitting $1.2B in Q4. Gross margins expanded by 150 basis points due to operational efficiency and strong Enterprise adoption. However, supply chain bottlenecks created mild cost headwinds in logistics.

Question-and-Answer Session:
Analyst: Can you expand on the supply chain pressures?
CEO: The inflation impact on freight cost is real and lingering. We expect margin compression in Q1 before recovery in Q2. Demand remains robust, but short-term uncertainty remains high."""

    if input_mode == "Paste Transcript":
        user_input = st.text_area("Paste Earnings Transcript:", value=default_text, height=220)
    else:
        ticker = st.selectbox("Select Sample Company Ticker:", ["AAPL (Q4 2025)", "NVDA (Q3 2025)", "MSFT (Q4 2025)"])
        st.info(f"Loaded pre-stored transcript sample for **{ticker}**.")
        user_input = default_text

    if st.button("🚀 Run Pipeline Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing transcript using FinBERT and BART engines..."):
            try:
                results = pipeline.run(user_input)
                render_analytics(results)
            except Exception as e:
                logger.error(f"Execution Error: {e}", exc_info=True)
                st.error(f"Pipeline processing failed: {str(e)}")


if __name__ == "__main__":
    main()