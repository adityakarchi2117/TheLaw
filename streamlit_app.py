"""LEGAL AUTONOMOUS AI PLATFORM — Streamlit Interface.

THREE modes:
  1. Legal Document Simplifier — Upload PDF / DOCX / paste → detect → simplify → risk → Q&A
  2. Legal Case Research Assistant (Lawyer AI) — Describe case → FAISS law search → LLM explanation
  3. Contract Comparison — Upload two contracts → clause diff → risk delta
"""

import streamlit as st
import sys
import os
import uuid
import tempfile
import time
import logging
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.workflow import run_full_analysis, run_qa, reset_retriever, run_case_research
from src.utils.config import (
    check_api_connection,
    get_active_provider_and_model,
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    clear_llm_cache,
)
from src.utils.s3_storage import (
    upload_pdf_to_s3,
    upload_text_to_s3,
    upload_docx_to_s3,
    list_uploaded_documents,
    is_s3_configured,
    get_s3_console_url,
)

# Feature 8: Analytics
try:
    from src.utils.analytics import log_analysis_event, load_analytics, get_summary_stats
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# Feature 6: PDF Export
try:
    from src.utils.pdf_export import generate_analysis_report, generate_case_report
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTS
# =====================================================================

MAX_ANALYSES_PER_SESSION = 10

LANGUAGE_OPTIONS = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi",
    "தமிழ் (Tamil)": "Tamil",
    "বাংলা (Bengali)": "Bengali",
    "తెలుగు (Telugu)": "Telugu",
    "मराठी (Marathi)": "Marathi",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ಕನ್ನಡ (Kannada)": "Kannada",
}

# Page config

st.set_page_config(
    page_title="Legal AI Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark theme CSS

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #FAFAFA; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border-left: 5px solid #667eea;
    border: 1px solid #333;
}
.hero h1 { color: #FAFAFA; font-size: 2.4rem; margin: 0; font-weight: 700; }
.hero p { color: #A0A0A0; margin-top: 0.5rem; font-size: 1.05rem; }

.card {
    background: #1E1E1E;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #333;
}

.badge-legal {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #000; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 13px; display: inline-block;
}
.badge-not-legal {
    background: linear-gradient(135deg, #eb3349, #f45c43);
    color: #fff; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 13px; display: inline-block;
}

.term-pill {
    background: #667eea; color: white;
    padding: 3px 10px; border-radius: 12px;
    font-size: 12px; margin: 2px; display: inline-block;
}

.result-block {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #667eea;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

/* Clean analysis section styling */
[data-testid="stExpander"] details {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #FAFAFA;
}
[data-testid="stMarkdownContainer"] h4 {
    color: #667eea;
    margin-top: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #333;
}

.simplified-block {
    background: rgba(17, 153, 142, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #11998e;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.risk-block {
    background: rgba(235, 51, 73, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #eb3349;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.qa-block {
    background: rgba(102, 126, 234, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #667eea;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.conf-bar { background: #333; border-radius: 8px; height: 8px; overflow: hidden; margin-top: 4px; }
.conf-fill { height: 100%; border-radius: 8px; transition: width .4s ease; }
.conf-high { background: linear-gradient(90deg, #11998e, #38ef7d); }
.conf-med  { background: linear-gradient(90deg, #f7971e, #ffd200); }
.conf-low  { background: linear-gradient(90deg, #eb3349, #f45c43); }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px;
    padding: 0.7rem 1.5rem; font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,0.6);
}

.stTextArea textarea {
    border-radius: 12px; border: 2px solid #333;
    background-color: #262730; color: #FAFAFA;
}
.stTextArea textarea:focus { border-color: #667eea; }

[data-testid="stFileUploader"] {
    border: 2px dashed #667eea; border-radius: 12px;
    padding: 1.5rem; background: rgba(102,126,234,0.08);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

[data-testid="metric-container"] {
    background: #262730; padding: 0.8rem; border-radius: 12px;
}

.footer { text-align: center; padding: 2rem; color: #888; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# Session state initialization

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "case_result" not in st.session_state:
    st.session_state.case_result = None
if "case_history" not in st.session_state:
    st.session_state.case_history = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📄 Document Simplifier"

# Feature 9: Rate limiting session counters
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "case_count" not in st.session_state:
    st.session_state.case_count = 0

# Feature 7: Multi-language default
if "output_language" not in st.session_state:
    st.session_state.output_language = "English"

# Feature 8: Session ID for analytics
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Feature 5: Compare state
if "compare_result" not in st.session_state:
    st.session_state.compare_result = None


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================


def render_confidence_bar(confidence: float):
    pct = int(confidence * 100)
    cls = "conf-high" if confidence >= 0.7 else "conf-med" if confidence >= 0.4 else "conf-low"
    st.markdown(f"""
    <div class="conf-bar"><div class="conf-fill {cls}" style="width:{pct}%"></div></div>
    """, unsafe_allow_html=True)


def render_terms(terms: list):
    if not terms:
        return
    pills = " ".join(f'<span class="term-pill">{t}</span>' for t in terms[:20])
    st.markdown(f'<div style="margin:0.5rem 0">{pills}</div>', unsafe_allow_html=True)


def check_rate_limit(count: int, max_count: int, label: str) -> bool:
    """Check if session rate limit is reached. Returns True if allowed."""
    if count >= max_count:
        st.error(
            f"⚠️ Session limit reached ({max_count} {label}). "
            "Refresh the page to start a new session."
        )
        st.info("💡 Tip: Streamlit Cloud sessions reset on page refresh.")
        return False
    remaining = max_count - count
    if remaining <= 3:
        st.warning(f"⚠️ {remaining} {label} remaining this session.")
    return True


def execute_analysis(input_data, input_type: str):
    """Execute the LangGraph workflow and cache the result."""
    with st.spinner("🔄 Running full analysis pipeline (detect → embed → simplify → risk)…"):
        start = time.time()
        language = st.session_state.get("output_language", "English")
        result = run_full_analysis(input_data, input_type, output_language=language)
        total_ms = int((time.time() - start) * 1000)
        result["processing_time_ms"] = total_ms

    st.session_state.analysis_result = result
    st.session_state.document_loaded = result.get("is_legal", False)
    st.session_state.chat_history = []
    # Feature 9: Increment counter on success
    st.session_state.analysis_count += 1
    return result


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("## ⚖️ Legal AI Platform")
    st.markdown("---")

    st.markdown("### 🔀 Select Product")
    mode_options = ["📄 Document Simplifier", "🔍 Case Research Assistant", "⚖️ Compare Contracts"]
    current_idx = mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in mode_options else 0
    mode = st.radio(
        "Choose your tool:",
        mode_options,
        index=current_idx,
        key="mode_selector",
        label_visibility="collapsed",
    )
    st.session_state.app_mode = mode

    st.markdown("---")

    # Feature 7: Language selector
    st.markdown("### 🌐 Output Language")
    selected_lang_label = st.selectbox(
        "Choose output language:",
        list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.values()).index(st.session_state.output_language)
        if st.session_state.output_language in LANGUAGE_OPTIONS.values()
        else 0,
        key="lang_selector",
        label_visibility="collapsed",
    )
    st.session_state.output_language = LANGUAGE_OPTIONS[selected_lang_label]

    st.markdown("---")

    # Feature 9: Usage indicator
    st.markdown("### 📊 Session Usage")
    total_used = st.session_state.analysis_count + st.session_state.case_count
    st.caption(f"📄 Analyses: **{st.session_state.analysis_count}** / {MAX_ANALYSES_PER_SESSION}")
    st.caption(f"🔍 Case Searches: **{st.session_state.case_count}** / {MAX_ANALYSES_PER_SESSION}")
    usage_pct = min(total_used / (MAX_ANALYSES_PER_SESSION * 2), 1.0)
    st.progress(usage_pct, text=f"{int(usage_pct * 100)}% used")

    st.markdown("---")

    st.markdown("### 🔌 AI Provider & Model")

    active_prov, active_mod = get_active_provider_and_model()

    provider_display_map = {
        "groq": "⚡ Groq (Fast & Free)",
        "gemini": "✨ Google Gemini (Free Tier)",
    }
    provider_keys = ["groq", "gemini"]

    if "selected_llm_provider" not in st.session_state:
        st.session_state.selected_llm_provider = active_prov

    selected_prov_display = st.selectbox(
        "LLM Provider:",
        options=[provider_display_map[p] for p in provider_keys],
        index=provider_keys.index(st.session_state.selected_llm_provider)
        if st.session_state.selected_llm_provider in provider_keys
        else 0,
        key="prov_selector",
    )
    current_prov = [k for k, v in provider_display_map.items() if v == selected_prov_display][0]
    if current_prov != st.session_state.selected_llm_provider:
        st.session_state.selected_llm_provider = current_prov
        st.session_state.selected_llm_model = DEFAULT_MODELS[current_prov]
        clear_llm_cache()

    prov_models = AVAILABLE_MODELS.get(current_prov, [DEFAULT_MODELS[current_prov]])
    current_model = st.session_state.get("selected_llm_model", DEFAULT_MODELS[current_prov])
    if current_model not in prov_models:
        current_model = DEFAULT_MODELS[current_prov]

    selected_model = st.selectbox(
        "Model:",
        options=prov_models,
        index=prov_models.index(current_model) if current_model in prov_models else 0,
        key="model_selector",
    )
    if selected_model != st.session_state.get("selected_llm_model"):
        st.session_state.selected_llm_model = selected_model
        clear_llm_cache()

    with st.expander("🔑 API Key Settings", expanded=False):
        if current_prov == "gemini":
            st.caption("Free Google Gemini API key: [Google AI Studio](https://aistudio.google.com/app/apikey)")
            gemini_input = st.text_input(
                "Gemini API Key:",
                type="password",
                value=st.session_state.get("gemini_api_key_input", ""),
                help="Enter your Gemini API key or set GEMINI_API_KEY in .env / Streamlit Secrets",
                key="gemini_key_widget",
            )
            if gemini_input:
                if gemini_input != st.session_state.get("gemini_api_key_input"):
                    st.session_state.gemini_api_key_input = gemini_input
                    clear_llm_cache()
        else:
            st.caption("Free Groq API key: [Groq Console](https://console.groq.com/keys)")
            groq_input = st.text_input(
                "Groq API Key:",
                type="password",
                value=st.session_state.get("groq_api_key_input", ""),
                help="Enter your Groq API key or set GROQ_API_KEY in .env / Streamlit Secrets",
                key="groq_key_widget",
            )
            if groq_input:
                if groq_input != st.session_state.get("groq_api_key_input"):
                    st.session_state.groq_api_key_input = groq_input
                    clear_llm_cache()

    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Testing connection…"):
            ok, msg = check_api_connection(provider=current_prov, model=selected_model)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")

    if mode == "📄 Document Simplifier":
        st.markdown("""
        **Pipeline 1: Document Simplifier**
        - Upload PDF / DOCX / paste text
        - Local legal detection (no LLM)
        - FAISS vector search
        - Groq LLM for simplification
        - Local risk scoring
        - Interactive Q&A
        """)
    else:
        st.markdown("""
        **Pipeline 2: Case Research**
        - Describe case in natural language
        - FAISS search over Indian laws
        - IPC, CrPC, Evidence Act corpus
        - Groq LLM for explanation ONLY
        - Applicable sections + punishment
        - Winning probability estimate
        """)

    st.markdown("---")

    st.markdown("### ☁️ S3 Storage")
    if is_s3_configured():
        st.success("✅ S3 Connected")
        if st.button("📋 View Uploads", use_container_width=True):
            docs = list_uploaded_documents()
            if docs:
                for d in docs[:5]:
                    st.caption(f"📄 {d['filename']} ({d['size_kb']} KB) — {d['uploaded_at']}")
            else:
                st.info("No documents uploaded yet.")
    else:
        st.warning("⚠️ S3 not configured\nAdd AWS keys to .env")

    # Feature 8: Analytics Dashboard
    if ANALYTICS_AVAILABLE and is_s3_configured():
        st.markdown("---")
        with st.expander("📊 Analytics Dashboard", expanded=False):
            with st.spinner("Loading analytics…"):
                try:
                    import plotly.express as px
                    df = load_analytics(days=7)
                    if df.empty:
                        st.info("No analytics data yet. Analyze a document to start tracking.")
                    else:
                        stats = get_summary_stats(df)

                        # Metrics row
                        am1, am2, am3 = st.columns(3)
                        am1.metric("Total Analyses", stats["total_analyses"])
                        am2.metric("Avg Time", f"{stats['avg_processing_time']/1000:.1f}s")
                        am3.metric("Legal %", f"{stats['legal_percentage']}%")

                        # Risk distribution pie chart
                        if stats["risk_distribution"]:
                            risk_df_data = {
                                "Risk Level": list(stats["risk_distribution"].keys()),
                                "Count": list(stats["risk_distribution"].values()),
                            }
                            fig_pie = px.pie(
                                risk_df_data,
                                names="Risk Level",
                                values="Count",
                                title="Risk Distribution",
                                color="Risk Level",
                                color_discrete_map={
                                    "Critical Risk": "#eb3349",
                                    "High Risk": "#f7971e",
                                    "Moderate Risk": "#ffd200",
                                    "Low Risk": "#11998e",
                                    "Unknown": "#667eea",
                                },
                                hole=0.4,
                            )
                            fig_pie.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#FAFAFA",
                                showlegend=True,
                                margin=dict(t=40, b=10, l=10, r=10),
                                height=250,
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        # Daily bar chart
                        if stats["daily_counts"]:
                            daily_data = {
                                "Date": list(stats["daily_counts"].keys()),
                                "Analyses": list(stats["daily_counts"].values()),
                            }
                            fig_bar = px.bar(
                                daily_data,
                                x="Date",
                                y="Analyses",
                                title="Analyses (Last 7 Days)",
                                color_discrete_sequence=["#667eea"],
                            )
                            fig_bar.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#FAFAFA",
                                xaxis=dict(gridcolor="#333"),
                                yaxis=dict(gridcolor="#333"),
                                margin=dict(t=40, b=10, l=10, r=10),
                                height=200,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # Last 5 analyses
                        if "pipeline" in df.columns:
                            st.markdown("**Recent Activity:**")
                            recent = df.head(5)
                            for _, row in recent.iterrows():
                                ts = str(row.get("timestamp", ""))[:16]
                                pipeline = str(row.get("pipeline", ""))
                                t_ms = row.get("processing_time_ms", 0)
                                t = f"{t_ms/1000:.1f}s" if t_ms else "—"
                                st.caption(f"🕐 {ts} | {pipeline} | {t}")
                except Exception as e:
                    st.warning(f"Analytics unavailable: {e}")

    st.markdown("---")
    st.markdown("### 📖 Tech Stack")
    st.markdown("""
    - **Embeddings**: `sentence-transformers`
    - **Vector DB**: `FAISS`
    - **LLM**: `Groq (Llama 3.3) / Google Gemini`
    - **Workflow**: `LangGraph`
    - **Chains**: `LangChain`
    - **Storage**: `AWS S3`
    """)

    st.markdown("---")

    st.markdown("""
    ⚠️ *For informational purposes only.
    Always consult a legal professional.*
    """)

    # Clear buttons
    if mode == "📄 Document Simplifier" and st.session_state.document_loaded:
        if st.button("🗑️ Clear Document", use_container_width=True):
            reset_retriever()
            st.session_state.analysis_result = None
            st.session_state.chat_history = []
            st.session_state.document_loaded = False
            st.rerun()

    if mode == "🔍 Case Research Assistant" and st.session_state.case_result:
        if st.button("🗑️ Clear Case Results", use_container_width=True):
            st.session_state.case_result = None
            st.session_state.case_history = []
            st.rerun()

    if mode == "⚖️ Compare Contracts" and st.session_state.compare_result:
        if st.button("🗑️ Clear Comparison", use_container_width=True):
            st.session_state.compare_result = None
            st.rerun()


# Hero header

if st.session_state.app_mode == "📄 Document Simplifier":
    st.markdown("""
    <div class="hero">
        <h1>📄 Legal Document Simplifier</h1>
        <p>RAG-powered legal document analysis — simplify, assess risk, and ask questions</p>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.app_mode == "🔍 Case Research Assistant":
    st.markdown("""
    <div class="hero">
        <h1>🔍 Legal Case Research Assistant</h1>
        <p>Describe your legal case — AI searches Indian laws (IPC, CrPC, Evidence Act) and explains applicable sections</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
        <h1>⚖️ Contract Comparison</h1>
        <p>Upload two contract versions — get clause-level diff with risk delta analysis</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# MODE: DOCUMENT SIMPLIFIER
# =====================================================================

if st.session_state.app_mode == "📄 Document Simplifier":

    st.markdown("### 📄 Upload Your Document")
    tab_text, tab_pdf, tab_docx = st.tabs(["📝 Paste Text", "📁 Upload PDF", "📝 Upload Word Doc"])

    with tab_text:
        user_text = st.text_area(
            "Enter legal document text:",
            height=250,
            placeholder=(
                "Paste your contract, agreement, terms of service, or any legal document here…\n\n"
                "Example: This Agreement is entered into as of the Effective Date between "
                "Company Inc. (\"Company\") and the Client…"
            ),
        )
        analyze_text_btn = st.button("🔍 Analyze Document", type="primary",
                                      use_container_width=True, key="analyze_text")

    with tab_pdf:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"],
                                          help="PDF up to 10 MB")
        if uploaded_file:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        analyze_pdf_btn = st.button("🔍 Analyze PDF", type="primary",
                                     use_container_width=True, key="analyze_pdf",
                                     disabled=not uploaded_file)

    with tab_docx:
        uploaded_docx = st.file_uploader(
            "Upload a Word document",
            type=["docx"],
            help="DOCX up to 10 MB",
            key="docx_uploader",
        )
        if uploaded_docx:
            st.info(f"📝 **{uploaded_docx.name}** ({uploaded_docx.size / 1024:.1f} KB)")
        analyze_docx_btn = st.button(
            "🔍 Analyze Word Doc", type="primary",
            use_container_width=True, key="analyze_docx",
            disabled=not uploaded_docx,
        )

    # Trigger analysis

    if analyze_text_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif not check_rate_limit(st.session_state.analysis_count, MAX_ANALYSES_PER_SESSION, "analyses"):
            pass  # Rate limit hit — message already shown
        else:
            execute_analysis(user_text, "text")
            # Upload text to S3 after successful analysis
            if is_s3_configured():
                s3_key = upload_text_to_s3(user_text, "pasted_contract.txt")
                if s3_key:
                    st.toast("☁️ Saved to S3: pasted_contract.txt", icon="✅")

    if analyze_pdf_btn and uploaded_file:
        if not check_rate_limit(st.session_state.analysis_count, MAX_ANALYSES_PER_SESSION, "analyses"):
            pass
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file_bytes = uploaded_file.read()
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                execute_analysis(tmp_path, "pdf")
                # Upload to S3 after successful analysis
                if is_s3_configured():
                    s3_key = upload_pdf_to_s3(file_bytes, uploaded_file.name)
                    if s3_key:
                        st.toast(f"☁️ Saved to S3: {uploaded_file.name}", icon="✅")
            finally:
                os.unlink(tmp_path)

    if analyze_docx_btn and uploaded_docx:
        if not check_rate_limit(st.session_state.analysis_count, MAX_ANALYSES_PER_SESSION, "analyses"):
            pass
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                docx_bytes = uploaded_docx.read()
                tmp.write(docx_bytes)
                tmp_path = tmp.name
            try:
                execute_analysis(tmp_path, "docx")
                # Upload DOCX to S3 after successful analysis
                if is_s3_configured():
                    s3_key = upload_docx_to_s3(docx_bytes, uploaded_docx.name)
                    if s3_key:
                        st.toast(f"☁️ Saved to S3: {uploaded_docx.name}", icon="✅")
            finally:
                os.unlink(tmp_path)

    # Results display

    result = st.session_state.analysis_result

    if result:
        st.markdown("---")

        if result.get("error"):
            st.error(f"⚠️ {result['error']}")

        st.markdown("### 📊 Analysis Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if result.get("is_legal"):
                st.markdown('<span class="badge-legal">✅ LEGAL DOCUMENT</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-not-legal">❌ NOT LEGAL</span>',
                            unsafe_allow_html=True)

        with col2:
            conf = result.get("legal_confidence", 0)
            st.metric("Confidence", f"{int(conf * 100)}%")
            render_confidence_bar(conf)

        with col3:
            st.metric("Legal Terms", len(result.get("detected_terms", [])))

        with col4:
            st.metric("Chunks Indexed", result.get("chunk_count", 0))

        explanation = result.get("legal_explanation", "")
        if explanation:
            st.info(explanation)

        terms = result.get("detected_terms", [])
        if terms:
            with st.expander("🏷️ Detected Legal Terms", expanded=False):
                render_terms(terms)

        if not result.get("is_legal"):
            st.warning("This does not appear to be a legal document. "
                        "Upload a contract, agreement, or terms of service to get full analysis.")

            with st.expander("📄 Input Text Preview"):
                raw = result.get("raw_text", "")
                st.text(raw[:2000] + ("…" if len(raw) > 2000 else ""))

        else:
            simplified = result.get("simplified_text", "")
            if simplified and not simplified.startswith("["):
                st.markdown("### ✨ Simplified Version")
                st.markdown(simplified)

                # Feature 6: PDF Export
                if PDF_EXPORT_AVAILABLE:
                    try:
                        pdf_bytes = generate_analysis_report(result)
                        st.download_button(
                            "📄 Download Full PDF Report",
                            data=pdf_bytes,
                            file_name="legal_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        # Upload PDF report to S3
                        if is_s3_configured():
                            from src.utils.s3_storage import upload_pdf_to_s3 as _upload
                            _upload(pdf_bytes, "legal_analysis_report.pdf", prefix="reports")
                    except Exception as e:
                        logger.warning(f"PDF generation failed: {e}")
                        st.download_button(
                            "📥 Download Simplified Text",
                            data=simplified,
                            file_name="simplified_legal_document.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                else:
                    st.download_button(
                        "📥 Download Simplified Text",
                        data=simplified,
                        file_name="simplified_legal_document.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            elif simplified:
                st.warning(simplified)

            risk = result.get("risk_analysis", "")
            if risk and not risk.startswith("["):
                st.markdown("### ⚠️ Risk Analysis")
                st.markdown(risk)
            elif risk:
                st.warning(risk)

            with st.expander("📜 Original Legal Text", expanded=False):
                raw = result.get("raw_text", "")
                st.text(raw[:8000] + ("…" if len(raw) > 8000 else ""))

            # Feature 4: Clause Breakdown Tab
            try:
                from src.segmentation import segment_into_clauses
                from src.classification import get_classifier
                from src.chains.clause_chain import simplify_clauses_batch

                raw_text = result.get("raw_text", "")
                clauses = segment_into_clauses(raw_text)

                if clauses:
                    st.markdown("### 📋 Clause Breakdown")

                    # Classify clauses
                    classifier = get_classifier()
                    clause_texts = [c["text"] for c in clauses]
                    classifications = classifier.classify_batch(clause_texts)

                    # Merge classifications into clause dicts
                    for clause, clf in zip(clauses, classifications):
                        clause["type"] = clf["type"]
                        clause["risk_level"] = clf["risk_level"]
                        clause["risk_score"] = clf["risk_score"]
                        clause["confidence"] = clf["confidence"]

                    # Filter controls
                    fcol1, fcol2 = st.columns([1, 1])
                    with fcol1:
                        show_critical = st.checkbox("🔴 Show High Risk Only", key="filter_critical")
                    with fcol2:
                        simplify_clauses = st.checkbox("✨ Show Plain English", key="simplify_clauses")

                    # Optionally simplify visible clauses
                    if simplify_clauses:
                        with st.spinner("✨ Simplifying clauses…"):
                            visible = [c for c in clauses if not show_critical or c["risk_level"] == "High"]
                            simplified_clauses = simplify_clauses_batch(visible, max_clauses=15)
                            simplified_map = {c["index"]: c for c in simplified_clauses}
                    else:
                        simplified_map = {}

                    risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

                    for i, clause in enumerate(clauses):
                        if show_critical and clause.get("risk_level") != "High":
                            continue

                        clf = clause
                        emoji = risk_emoji.get(clf.get("risk_level", "Low"), "⚪")
                        heading = clause.get("heading", f"Clause {i+1}")[:60]

                        with st.expander(
                            f"{emoji} **#{clause['index']}** {heading} — "
                            f"{clf.get('type', 'General')} ({clf.get('risk_level', 'Low')})",
                            expanded=False,
                        ):
                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                st.markdown(f"**Type:** {clf.get('type', 'General')}")
                                st.markdown(f"**Risk:** {clf.get('risk_level', 'Low')} ({int(clf.get('risk_score', 0)*100)}%)")
                                st.markdown(f"**Confidence:** {int(clf.get('confidence', 0)*100)}%")
                            with col_b:
                                st.markdown("**Original Clause:**")
                                st.text(clause["text"][:500])

                            # Show plain English if simplification was requested
                            sc = simplified_map.get(clause["index"])
                            if sc:
                                st.markdown("---")
                                st.markdown(f"**✨ Plain English:** {sc.get('summary', '')}")
                                if sc.get("negotiation_tip"):
                                    st.markdown(f"**💡 Tip:** {sc.get('negotiation_tip', '')}")

            except Exception as e:
                logger.warning(f"Clause breakdown unavailable: {e}")

            proc_ms = result.get("processing_time_ms", 0)
            if proc_ms:
                st.caption(f"⏱️ Total processing time: {proc_ms / 1000:.1f}s")

        # Contract Q&A

        if result.get("is_legal") and st.session_state.document_loaded:
            st.markdown("---")
            st.markdown("### 💬 Ask Questions About This Contract")

            for entry in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    st.markdown(entry["answer"])

            question = st.chat_input("Ask a question about the contract…")

            if question:
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document…"):
                        answer = run_qa(question)
                    st.markdown(answer)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                })

            st.markdown("**💡 Try asking:**")
            suggested = [
                "What are the main obligations?",
                "Can they terminate anytime?",
                "What are my risks?",
                "Any penalty clauses?",
                "What if there's a breach?",
            ]
            cols = st.columns(len(suggested))
            for i, sq in enumerate(suggested):
                with cols[i]:
                    if st.button(sq, key=f"sq_{i}", use_container_width=True):
                        with st.spinner("Searching document…"):
                            answer = run_qa(sq)
                        st.session_state.chat_history.append({
                            "question": sq,
                            "answer": answer,
                        })
                        st.rerun()


    # Feature 8: Log analysis event
    if result and ANALYTICS_AVAILABLE:
        try:
            log_analysis_event({
                "session_id": st.session_state.session_id,
                "pipeline": "simplifier",
                "doc_type": "document",
                "is_legal": result.get("is_legal", False),
                "risk_level": "Unknown",
                "processing_time_ms": result.get("processing_time_ms", 0),
                "chunk_count": result.get("chunk_count", 0),
                "language": st.session_state.output_language,
            })
        except Exception:
            pass


# =====================================================================
# MODE: CASE RESEARCH ASSISTANT
# =====================================================================

elif st.session_state.app_mode == "🔍 Case Research Assistant":

    st.markdown("### 📝 Describe Your Legal Case")
    st.markdown("""
    <div class="card">
        <p style="color:#A0A0A0; margin:0;">
        Describe the incident or legal situation in natural language. The AI will search
        through <strong>IPC</strong>, <strong>CrPC</strong>, and <strong>Indian Evidence Act</strong>
        to find applicable sections, punishments, and provide legal guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    case_description = st.text_area(
        "Case Description:",
        height=200,
        placeholder=(
            "Example: A person was stabbed during a robbery attempt at night. "
            "The victim survived but suffered serious injuries. The attacker also "
            "threatened the victim's family. What sections apply?\n\n"
            "Or: Someone posted defamatory content about a public figure on social media..."
        ),
        key="case_input",
    )

    # Suggested case examples
    st.markdown("**💡 Quick Examples:**")
    example_cols = st.columns(3)
    examples = [
        ("🔪 Robbery & Assault", "A gang of 3 people robbed a jewellery shop at night using weapons. They assaulted the shop owner causing grievous injuries. One of the accused was a minor."),
        ("💻 Cyber Crime", "Someone hacked into another person's bank account and transferred money. They also used fake identity documents for KYC verification."),
        ("🏠 Domestic Violence", "A woman is facing continuous physical and mental abuse from her husband and in-laws over dowry demands. She wants to file a case."),
    ]

    for i, (label, desc) in enumerate(examples):
        with example_cols[i]:
            if st.button(label, key=f"case_ex_{i}", use_container_width=True):
                st.session_state["case_prefill"] = desc
                st.rerun()

    # Apply prefill
    if "case_prefill" in st.session_state:
        case_description = st.session_state.pop("case_prefill")

    analyze_case_btn = st.button(
        "🔍 Research Applicable Laws",
        type="primary",
        use_container_width=True,
        key="analyze_case",
    )

    # Trigger case research

    if analyze_case_btn:
        if not case_description or not case_description.strip():
            st.warning("⚠️ Please describe the legal case or incident.")
        elif not check_rate_limit(st.session_state.case_count, MAX_ANALYSES_PER_SESSION, "case searches"):
            pass  # Rate limit hit
        else:
            with st.spinner("🔄 Searching Indian law corpus (FAISS) → Analyzing with LLM…"):
                start = time.time()
                case_result = run_case_research(case_description.strip())
                total_ms = int((time.time() - start) * 1000)
                case_result["processing_time_ms"] = total_ms

            st.session_state.case_result = case_result
            st.session_state.case_history.append({
                "description": case_description.strip(),
                "result": case_result,
            })
            # Feature 9: Increment counter
            st.session_state.case_count += 1

            # Feature 8: Log case research event
            if ANALYTICS_AVAILABLE:
                try:
                    log_analysis_event({
                        "session_id": st.session_state.session_id,
                        "pipeline": "case_research",
                        "doc_type": "case_description",
                        "is_legal": True,
                        "risk_level": "N/A",
                        "processing_time_ms": case_result.get("processing_time_ms", 0),
                        "chunk_count": case_result.get("case_sections_found", 0),
                        "language": st.session_state.output_language,
                    })
                except Exception:
                    pass

    # Case results display

    case_result = st.session_state.case_result

    if case_result:
        st.markdown("---")

        if case_result.get("error"):
            st.error(f"⚠️ {case_result['error']}")

        # Metrics row
        st.markdown("### 📊 Research Results")

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Sections Found", case_result.get("case_sections_found", 0))
        with mcol2:
            proc_ms = case_result.get("processing_time_ms", 0)
            st.metric("Processing Time", f"{proc_ms / 1000:.1f}s")
        with mcol3:
            n_laws = len(case_result.get("retrieved_laws", []))
            st.metric("Laws Retrieved", n_laws)

        # Retrieved law sections
        retrieved_laws = case_result.get("retrieved_laws", [])
        if retrieved_laws:
            st.markdown("### 📜 Applicable Law Sections")

            for i, law in enumerate(retrieved_laws):
                section = law.get("section", "N/A")
                title = law.get("title", "N/A")
                act = law.get("act_name", "N/A")
                score = law.get("confidence", 0)
                crime = law.get("crime", "")
                punishment = law.get("punishment", "")
                jail = law.get("jail_term", "")
                fine = law.get("fine", "")
                bailable = law.get("bailable", "")
                cognizable = law.get("cognizable", "")

                pct = int(score * 100) if score <= 1 else int(score)

                with st.expander(
                    f"**Section {section}** — {title} ({act}) | Relevance: {pct}%",
                    expanded=(i < 3),
                ):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**🏛️ Act:** {act}")
                        st.markdown(f"**📌 Section:** {section}")
                        st.markdown(f"**⚖️ Crime:** {crime}")
                        st.markdown(f"**🔒 Cognizable:** {cognizable}")
                    with cols[1]:
                        st.markdown(f"**⚠️ Punishment:** {punishment}")
                        st.markdown(f"**🏢 Jail Term:** {jail}")
                        st.markdown(f"**💰 Fine:** {fine}")
                        st.markdown(f"**🔓 Bailable:** {bailable}")

                    render_confidence_bar(score if score <= 1 else score / 100)

        # LLM Analysis
        analysis = case_result.get("case_analysis", "")
        if analysis and not analysis.startswith("["):
            st.markdown("### 🧠 AI Legal Analysis")

            # Parse and render each section cleanly
            sections = analysis.split("\n## ")

            for idx_s, section in enumerate(sections):
                if idx_s == 0 and not section.startswith("##"):
                    # First block (may be intro or "## Case Analysis" without leading \n)
                    section_text = section.lstrip("# ").strip()
                    if section_text:
                        # Check if it starts with a heading
                        if section.strip().startswith("##"):
                            heading = section.strip().split("\n", 1)
                            title = heading[0].lstrip("# ").strip()
                            body = heading[1].strip() if len(heading) > 1 else ""
                            st.markdown(f"#### {title}")
                            if body:
                                st.markdown(body)
                        else:
                            st.markdown(section.strip())
                else:
                    # Subsequent sections: split heading from body
                    parts = section.split("\n", 1)
                    title = parts[0].lstrip("# ").strip()
                    body = parts[1].strip() if len(parts) > 1 else ""

                    # Color-coded containers per section type
                    if "applicable" in title.lower():
                        icon = "📜"
                    elif "confidence" in title.lower():
                        icon = "📊"
                    elif "recommended" in title.lower() or "action" in title.lower():
                        icon = "✅"
                    elif "winning" in title.lower() or "probability" in title.lower():
                        icon = "🎯"
                    elif "disclaimer" in title.lower():
                        icon = "⚠️"
                    elif "case" in title.lower() and "analysis" in title.lower():
                        icon = "🔍"
                    else:
                        icon = "📌"

                    with st.container():
                        st.markdown(f"#### {icon} {title}")

                        if body:
                            # Handle sub-sections (### headings for individual laws)
                            sub_sections = body.split("\n### ")

                            if len(sub_sections) > 1:
                                # First part before any ### 
                                intro = sub_sections[0].strip()
                                if intro:
                                    st.markdown(intro)

                                # Each law section as an expander
                                for sub in sub_sections[1:]:
                                    sub_parts = sub.split("\n", 1)
                                    sub_title = sub_parts[0].strip()
                                    sub_body = sub_parts[1].strip() if len(sub_parts) > 1 else ""

                                    with st.expander(f"⚖️ {sub_title}", expanded=True):
                                        if sub_body:
                                            st.markdown(sub_body)
                            else:
                                st.markdown(body)

                        st.markdown("---")

            # Feature 6: PDF export for case reports
            if PDF_EXPORT_AVAILABLE:
                try:
                    pdf_bytes = generate_case_report(case_result)
                    st.download_button(
                        "📄 Download Case Report PDF",
                        data=pdf_bytes,
                        file_name="case_research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception:
                    st.download_button(
                        "📥 Download Case Analysis",
                        data=analysis,
                        file_name="case_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            else:
                st.download_button(
                    "📥 Download Case Analysis",
                    data=analysis,
                    file_name="case_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        elif analysis:
            st.warning(analysis)

    # Case history

    history = st.session_state.case_history
    if len(history) > 1:
        st.markdown("---")
        with st.expander("📋 Previous Case Searches", expanded=False):
            for i, entry in enumerate(reversed(history[:-1])):
                st.markdown(f"**Case {len(history) - 1 - i}:** {entry['description'][:100]}…")
                n = entry['result'].get('case_sections_found', 0)
                st.caption(f"Found {n} applicable sections")


# =====================================================================
# MODE: CONTRACT COMPARISON
# =====================================================================

else:

    st.markdown("### 📂 Upload Two Contract Versions")
    col_orig, col_rev = st.columns(2)

    with col_orig:
        st.markdown("**📄 Original Contract**")
        compare_text_a = st.text_area(
            "Paste original contract:",
            height=200,
            placeholder="Paste the original contract text here...",
            key="compare_a",
        )

    with col_rev:
        st.markdown("**📄 Revised Contract**")
        compare_text_b = st.text_area(
            "Paste revised contract:",
            height=200,
            placeholder="Paste the revised contract text here...",
            key="compare_b",
        )

    compare_btn = st.button(
        "⚖️ Compare Contracts",
        type="primary",
        use_container_width=True,
        key="compare_btn",
    )

    if compare_btn:
        if not compare_text_a.strip() or not compare_text_b.strip():
            st.warning("⚠️ Please paste both contract versions.")
        elif not check_rate_limit(st.session_state.analysis_count, MAX_ANALYSES_PER_SESSION, "analyses"):
            pass
        else:
            with st.spinner("🔄 Comparing contracts (segment → classify → diff → analyze)..."):
                try:
                    from src.comparison import segment_and_align, calculate_risk_delta
                    from src.classification import get_classifier
                    from src.segmentation import segment_into_clauses

                    # 1. Segment and align
                    changes = segment_and_align(compare_text_a, compare_text_b)

                    # 2. Classify both sets for risk delta
                    classifier = get_classifier()
                    clauses_a = segment_into_clauses(compare_text_a)
                    clauses_b = segment_into_clauses(compare_text_b)

                    risks_a = classifier.classify_batch([c["text"] for c in clauses_a]) if clauses_a else []
                    risks_b = classifier.classify_batch([c["text"] for c in clauses_b]) if clauses_b else []

                    risk_delta = calculate_risk_delta(risks_a, risks_b)

                    # 3. LLM overall summary
                    try:
                        from src.chains.compare_chain import overall_comparison_summary
                        summary = overall_comparison_summary(changes, risk_delta)
                    except Exception:
                        summary = ""

                    st.session_state.compare_result = {
                        "changes": changes,
                        "risk_delta": risk_delta,
                        "summary": summary,
                    }
                    st.session_state.analysis_count += 1

                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    logger.error(f"Comparison error: {e}")

    # Display comparison results
    comp = st.session_state.compare_result
    if comp:
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")

        changes = comp["changes"]
        delta = comp["risk_delta"]

        # Metrics
        n_unchanged = sum(1 for c in changes if c["status"] == "unchanged")
        n_modified = sum(1 for c in changes if c["status"] == "modified")
        n_added = sum(1 for c in changes if c["status"] == "added")
        n_removed = sum(1 for c in changes if c["status"] == "removed")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("✅ Unchanged", n_unchanged)
        mc2.metric("✏️ Modified", n_modified)
        mc3.metric("➕ Added", n_added)
        mc4.metric("➖ Removed", n_removed)

        # Risk delta
        st.markdown("### 📈 Risk Delta")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Original Risk", f"{delta['original_avg_score']:.0%}")
        rc2.metric("Revised Risk", f"{delta['revised_avg_score']:.0%}")
        direction_emoji = "🔺" if delta["direction"] == "increased" else "🔻" if delta["direction"] == "decreased" else "➡️"
        rc3.metric("Direction", f"{direction_emoji} {delta['direction'].title()}")

        if delta.get("new_high_risk", 0) > 0:
            st.warning(f"⚠️ {delta['new_high_risk']} new high-risk clauses detected in revision!")

        # Clause-level diff
        st.markdown("### 📝 Clause Diff")
        status_emoji = {"unchanged": "✅", "modified": "✏️", "added": "➕", "removed": "➖"}

        for i, change in enumerate(changes):
            emoji = status_emoji.get(change["status"], "❓")
            sim = change.get("similarity", 0)
            label = f"{emoji} Clause {i+1} — {change['status'].title()}"
            if change["status"] == "modified":
                label += f" ({int(sim * 100)}% similar)"

            with st.expander(label, expanded=(change["status"] == "modified")):
                if change["status"] in ("modified", "removed"):
                    st.markdown("**Original:**")
                    st.text(change["original"][:500])
                if change["status"] in ("modified", "added"):
                    st.markdown("**Revised:**")
                    st.text(change["revised"][:500])
                if change["status"] == "unchanged":
                    st.text(change["original"][:300])

        # LLM Summary
        summary = comp.get("summary", "")
        if summary:
            st.markdown("### 🧠 AI Comparison Summary")
            st.markdown(summary)


# Footer

st.markdown("""
<div class="footer">
    <p>⚖️ <strong>Legal Autonomous AI Platform</strong> | RAG + LangGraph + LangChain + FAISS + Groq</p>
    <p style="opacity:0.6; font-size:0.85rem">
        Built by Aditya & AI ❤️ — For informational purposes only. Consult a legal professional.
    </p>
</div>
""", unsafe_allow_html=True)
