import logging
import pandas as pd
import plotly.express as px
import streamlit as st

from src.logger_config import setup_logging
from src.pipeline import EarningsAnalysisPipeline
from src.fetcher import fetch_transcript_by_ticker

logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

# Page Setup
st.set_page_config(
    page_title="FinSentiment AI | Financial Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_pipeline() -> EarningsAnalysisPipeline:
    return EarningsAnalysisPipeline()


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Engine Options")
        st.caption("Financial NLP Pipeline Configuration")

        input_mode = st.radio(
            "Select Input Source",
            ["Fetch Live Ticker", "Paste Transcript"],
            index=0
        )

        st.divider()
        st.subheader("Model Metadata")
        st.markdown("""
        * **Sentiment Engine:** `ProsusAI/finbert`
        * **Summarizer Engine:** `facebook/bart-large-cnn`
        * **Tokenizer:** SpaCy (`en_core_web_sm`)
        * **Data Fetcher:** `earningscall` API
        """)

        st.divider()
        st.caption("Financial Intelligence & NLP Pipeline")

    return input_mode


def render_analytics(results: dict):
    metrics = results["metrics"]
    sentence_details = results["sentence_details"]

    st.subheader("📌 Key Metric Indicators")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    net_sentiment = metrics["overall_net_sentiment"]
    total = metrics["total_sentences"]

    kpi1.metric(
        label="Overall Net Sentiment Score",
        value=f"{net_sentiment:.2f}",
        delta=f"{net_sentiment:.2f}",
        delta_color="normal" if net_sentiment >= 0 else "inverse"
    )
    kpi2.metric(
        label="Positive Sentences",
        value=metrics["positive_count"],
        delta=f"{(metrics['positive_count'] / total * 100) if total else 0:.1f}%"
    )
    kpi3.metric(
        label="Negative Sentences",
        value=metrics["negative_count"],
        delta=f"-{(metrics['negative_count'] / total * 100) if total else 0:.1f}%",
        delta_color="inverse"
    )
    kpi4.metric(
        label="Total Sentences Parsed",
        value=total
    )

    st.divider()

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
        if sentence_details:
            df = pd.DataFrame(sentence_details)
            df['sentence_idx'] = range(1, len(df) + 1)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### Overall Sentiment Score Distribution")
                fig_hist = px.histogram(
                    df,
                    x="net_sentiment",
                    nbins=25,
                    color_discrete_sequence=["#1f77b4"],
                    labels={"net_sentiment": "Net Sentiment (Positive - Negative)"}
                )
                fig_hist.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_right:
                st.markdown("#### Narrative Sentiment Flow Across Transcript")
                fig_line = px.line(
                    df,
                    x="sentence_idx",
                    y="net_sentiment",
                    markers=True,
                    labels={"sentence_idx": "Sentence Sequence", "net_sentiment": "Net Sentiment Score"},
                    color_discrete_sequence=["#1f77b4"]
                )
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                fig_line.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No sentence data available for visualization.")

    with tab_data:
        st.subheader("Raw Sentence Output Data")
        if sentence_details:
            df = pd.DataFrame(sentence_details)
            st.dataframe(
                df[["label", "net_sentiment", "sentence"]],
                use_container_width=True,
                height=400
            )


def main():
    input_mode = render_sidebar()

    st.title("📈 Financial Sentiment & Earnings Analysis Engine")
    st.markdown("Automated sentiment classification and abstractive transcript summarization for financial analysis.")
    st.divider()

    pipeline = load_pipeline()

    default_text = """We delivered exceptional revenue growth of 24% year-over-year, hitting $1.2B in Q4. Gross margins expanded by 150 basis points due to operational efficiency and strong Enterprise adoption. However, supply chain bottlenecks created mild cost headwinds in logistics. Inflation impact on freight cost is real and lingering, and we expect slight margin compression in Q1 before recovery in Q2."""

    if "transcript_text" not in st.session_state:
        st.session_state["transcript_text"] = default_text

    if input_mode == "Fetch Live Ticker":
        st.subheader("🌐 Live Earnings Call Transcript Retrieval")
        col_ticker, col_year, col_quarter = st.columns([2, 1, 1])

        with col_ticker:
            ticker = st.text_input("Enter Company Stock Ticker:", value="AAPL").strip().upper()
        with col_year:
            year = st.number_input("Year:", min_value=2020, max_value=2026, value=2024)
        with col_quarter:
            quarter = st.number_input("Quarter:", min_value=1, max_value=4, value=3)

        if st.button("📥 Fetch Transcript", type="secondary"):
            with st.spinner(f"Fetching Q{quarter} {year} transcript for {ticker} via EarningsCall API..."):
                try:
                    fetched_text = fetch_transcript_by_ticker(ticker, year, quarter)
                    if fetched_text:
                        st.session_state["transcript_text"] = fetched_text
                        st.success(f"Successfully retrieved Q{quarter} {year} transcript for **{ticker}**!")
                    else:
                        st.warning(f"No transcript found for {ticker} (Q{quarter} {year}). Displaying fallback sample.")
                except Exception as e:
                    st.error(f"Error retrieving transcript: {str(e)}")

        user_input = st.text_area(
            "Active Transcript Text for Pipeline Analysis:",
            value=st.session_state["transcript_text"],
            height=250
        )

    else:
        user_input = st.text_area("Paste Earnings Transcript:", value=st.session_state["transcript_text"], height=250)

    st.divider()

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