import logging
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.logger_config import setup_logging
from src.loughran_mcdonald import LoughranMcDonaldScorer, transcript_to_lm_records
from src.pipeline import EarningsAnalysisPipeline
from src.fetcher import fetch_transcript_by_ticker

logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

SECTION_LABELS = {"prepared_remarks": "Prepared remarks", "qa": "Q&A", "transcript": "Transcript"}

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


@st.cache_resource
def load_lm_scorer() -> LoughranMcDonaldScorer:
    return LoughranMcDonaldScorer()


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
        * **Summarizer Engine:** `facebook/bart-large-cnn` (section-aware, LM-guided)
        * **Tokenizer:** SpaCy (`en_core_web_sm`)
        * **Data Fetcher:** `earningscall` API
        """)

        st.divider()
        st.caption("Financial Intelligence & NLP Pipeline")

    return input_mode


def render_lm_analytics(records: Dict[str, Any]):
    aggregated = records["aggregated"]
    sentence_scores = records["sentence_scores"]
    label_schema = records["label_schema"]

    st.subheader("📌 Loughran-McDonald Word-Level Metrics")

    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        label="Negative",
        value=f"{aggregated.get('Negative', 0.0):.3f}",
        delta=f"{aggregated.get('Negative', 0.0) * 100:.1f}%",
        delta_color="inverse"
    )
    kpi_cols[1].metric(
        label="Positive",
        value=f"{aggregated.get('Positive', 0.0):.3f}",
        delta=f"{aggregated.get('Positive', 0.0) * 100:.1f}%"
    )
    kpi_cols[2].metric(
        label="Uncertainty",
        value=f"{aggregated.get('Uncertainty', 0.0):.3f}",
        delta=f"{aggregated.get('Uncertainty', 0.0) * 100:.1f}%"
    )
    kpi_cols[3].metric(
        label="Litigious",
        value=f"{aggregated.get('Litigious', 0.0):.3f}",
        delta=f"{aggregated.get('Litigious', 0.0) * 100:.1f}%",
        delta_color="inverse"
    )

    st.divider()

    lm_tab_summary, lm_tab_flow, lm_tab_data = st.tabs([
        "📝 Category Composition",
        "📊 Category Flow",
        "🔍 LM Sentence Scoring"
    ])

    with lm_tab_summary:
        st.subheader("Word-Level Category Composition")
        if aggregated:
            df = pd.DataFrame({
                "Category": label_schema,
                "Share": [aggregated.get(category, 0.0) for category in label_schema]
            })
            fig_bar = px.bar(
                df,
                x="Category",
                y="Share",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"Share": "Category Hit Share", "Category": "Loughran-McDonald Category"}
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### Category Scores")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No Loughran-McDonald scores available for this transcript.")

    with lm_tab_flow:
        if sentence_scores:
            df = pd.DataFrame(sentence_scores)
            df["sentence_idx"] = range(1, len(df) + 1)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### Negative Category Flow")
                fig_neg = px.line(
                    df,
                    x="sentence_idx",
                    y="Negative",
                    markers=True,
                    labels={"sentence_idx": "Sentence Sequence", "Negative": "Negative Share"},
                    color_discrete_sequence=["#d62728"]
                )
                fig_neg.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_neg, use_container_width=True)

            with col_right:
                st.markdown("#### Positive Category Flow")
                fig_pos = px.line(
                    df,
                    x="sentence_idx",
                    y="Positive",
                    markers=True,
                    labels={"sentence_idx": "Sentence Sequence", "Positive": "Positive Share"},
                    color_discrete_sequence=["#2ca02c"]
                )
                fig_pos.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.warning("No sentence-level LM data available for visualization.")

    with lm_tab_data:
        st.subheader("Raw LM Sentence Output Data")
        if sentence_scores:
            df = pd.DataFrame(sentence_scores)
            df["sentence"] = records["sentences"]
            df["sentence_idx"] = range(1, len(df) + 1)
            st.dataframe(
                df[["sentence_idx", "sentence"] + label_schema],
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No sentence-level LM data available for inspection.")


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
        summary_meta = results.get("summary_meta") or {}
        if summary_meta:
            section_bits = [
                f"{SECTION_LABELS.get(section['name'], section['name'].replace('_', ' ').title())} "
                f"{section['selected_chunks']}/{section['chunks']} chunks"
                for section in summary_meta.get("sections", [])
            ]
            st.caption(
                "Section-aware, Loughran-McDonald-guided BART summary: "
                f"{summary_meta.get('boilerplate_removed_count', 0)} boilerplate sentences removed, "
                f"{summary_meta.get('chunks_used', 0)}/{summary_meta.get('chunks_total', 0)} chunks "
                f"selected ({summary_meta.get('input_tokens', 0)} tokens of "
                f"{summary_meta.get('transcript_tokens', 0)})"
                + (f" — {' · '.join(section_bits)}." if section_bits else ".")
            )
            removed_samples = summary_meta.get("boilerplate_removed_samples") or []
            if removed_samples:
                with st.expander("Boilerplate removed before summarizing"):
                    st.caption(
                        "These sentences are stripped so the model reads results and guidance "
                        "instead of safe-harbor text."
                    )
                    for sample in removed_samples:
                        st.markdown(f"- {sample}")
        else:
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
    lm_scorer = load_lm_scorer()

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

    st.subheader("⚙️ Pipeline Selection")
    select_col_left, select_col_right = st.columns(2)

    with select_col_left:
        run_pipeline = st.checkbox(
            "Run FinBERT + BART Pipeline",
            value=True,
            help="Sentence-level sentiment classification and abstractive summarization."
        )

    with select_col_right:
        run_lm = st.checkbox(
            "Run Loughran-McDonald Word-Level Scoring",
            value=True,
            help="Dictionary-based word-level sentiment category shares."
        )

    st.divider()

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not run_pipeline and not run_lm:
            st.warning("Select at least one pipeline before running the analysis.")
            return

        with st.spinner(
            "Analyzing transcript using selected engines... "
            "(a full earnings call takes a minute or two on CPU)"
        ):
            if run_pipeline:
                try:
                    results = pipeline.run(user_input)
                    render_analytics(results)
                except Exception as e:
                    logger.error(f"FinBERT pipeline error: {e}", exc_info=True)
                    st.error(f"FinBERT pipeline processing failed: {str(e)}")
                st.divider()

            if run_lm:
                try:
                    lm_records = transcript_to_lm_records(lm_scorer, user_input)
                    render_lm_analytics(lm_records)
                except Exception as e:
                    logger.error(f"LM scoring error: {e}", exc_info=True)
                    st.error(f"Loughran-McDonald scoring failed: {str(e)}")


if __name__ == "__main__":
    main()