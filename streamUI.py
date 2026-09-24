"""
🎥 Video Agent — Streamlit UI
--------------------------------
Drop this file in the ROOT of your "Video Agent" project
(the same folder as main.py, next to the `core/` and `utlis/` packages)
and run:

    streamlit run app.py

It reuses your existing functions directly — nothing about your
core/ or utlis/ code is changed.
"""

import os
import re
import shutil
import time
import glob

import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------------------------
# YOUR EXISTING PROJECT FUNCTIONS (unchanged)
# ------------------------------------------------------------------
from utlis.audio_processor import process_audio, DOWNLOAD_DIR
from core.transcriber import transcribe_all
from core.summarize import (
    summarize,
    generate_title,
    translate_summary,
    shorten_summary,
)
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question
from core.vector_store import CHROMA_DIR

load_dotenv()

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Video Agent",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
def init_state():
    defaults = {
        "processed": False,
        "result": None,
        "chat_history": [],
        "translated_summary": None,
        "translated_lang": None,
        "shortened_summary": None,
        "source_label": None,
        "processing_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def delete_downloaded_audio() -> int:
    """Delete every file inside the downloads folder. Returns count removed."""
    if not os.path.isdir(DOWNLOAD_DIR):
        return 0
    removed = 0
    for f in glob.glob(os.path.join(DOWNLOAD_DIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
                removed += 1
        except Exception:
            pass
    return removed


def clear_vector_db() -> bool:
    """Wipe the Chroma persist directory so a new video starts clean."""
    if os.path.isdir(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            return True
        except Exception:
            return False
    return False


def save_uploaded_file(uploaded_file) -> str:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def full_reset():
    delete_downloaded_audio()
    clear_vector_db()
    st.session_state.processed = False
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.translated_summary = None
    st.session_state.translated_lang = None
    st.session_state.shortened_summary = None
    st.session_state.source_label = None


def clean_ai_text(text: str) -> str:
    """Normalize stray raw HTML the model sometimes emits (e.g. <br>) into
    plain markdown line breaks, so it never shows up as literal tag text."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<br\s*/?>", "  \n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p\s*/?>", "\n\n", text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------
# CSS  — main theme is strictly BLACK + GOLD/ORANGE.
# A few small, deliberate accent colours (green/blue/red) are used
# ONLY as tiny status/category markers, never as background colours.
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
    /* ---- MAIN THEME: black + gold ---- */
    --gold: #F2A93B;
    --gold-strong: #D98C1A;
    --gold-soft: rgba(242,169,59,0.14);
    --bg: #000000;
    --surface: #121212;
    --surface-2: #1A1712;
    --border: #2E2A22;
    --text: #F5F0E6;
    --text-dim: #A39B8C;

    /* ---- tiny status/category accents only ---- */
    --ok: #4ADE80;
    --danger: #FF6B6B;
    --info-blue: #5FA8E8;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif; }

.stApp{
    background:
        radial-gradient(circle at 15% -10%, rgba(242,169,59,0.10) 0%, transparent 40%),
        var(--bg);
    color: var(--text);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{
    background: #0A0A0A;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4{
    color: var(--gold);
}

/* ---- Hero header ---- */
.va-hero{
    text-align:center;
    padding: 30px 20px 22px 20px;
    animation: fadeInUp .7s ease;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.va-hero h1{
    font-size: 2.7rem;
    font-weight: 800;
    margin-bottom: 4px;
    background: linear-gradient(90deg, #FFD98A 0%, var(--gold) 45%, #FF9A3C 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    background-size: 200% auto;
    animation: shimmer 5s linear infinite;
}
.va-hero p{ color: var(--text-dim); font-size: 1.02rem; margin-top:0; }

/* ---- Cards ---- */
.va-card{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--card-accent, var(--gold));
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    animation: fadeInUp .5s ease;
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
.va-card:hover{
    border-color: var(--card-accent, var(--gold));
    box-shadow: 0 8px 22px rgba(242,169,59,0.12);
    transform: translateY(-2px);
}

.va-card h4{
    margin-top:0; color: var(--card-accent, var(--gold));
    display:flex; align-items:center; gap:8px;
    font-size: 1.05rem;
}
.va-card .va-body{ color: var(--text); line-height:1.65; white-space: pre-wrap; }

/* ---- Feature chips on welcome screen ---- */
.va-chip-grid{ display:flex; flex-wrap:wrap; gap:12px; justify-content:center; margin-top:22px;}
.va-chip{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: .92rem;
    color: var(--text-dim);
    min-width: 150px;
    transition: transform .2s ease, border-color .2s ease, color .2s ease;
}
.va-chip:hover{
    transform: translateY(-4px) scale(1.03);
    border-color: var(--gold);
    color: var(--gold);
    box-shadow: 0 6px 18px rgba(242,169,59,0.18);
}

/* ---- Status pill ---- */
.va-pill{
    display:inline-flex; align-items:center; gap:6px;
    padding: 5px 12px; border-radius: 999px;
    font-size: .78rem; font-weight:600;
    border: 1px solid var(--border);
}
.va-pill.ok{ color: var(--ok); border-color: rgba(74,222,128,.3); background: rgba(74,222,128,.08);}
.va-pill.bad{ color: var(--danger); border-color: rgba(255,107,107,.3); background: rgba(255,107,107,.08);}
.va-dot{ width:8px; height:8px; border-radius:50%; background:currentColor; animation: pulse 1.6s infinite; }

/* ---- Buttons (default = black surface with gold outline on hover) ---- */
.stButton > button{
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    transition: all .2s ease !important;
}
.stButton > button:hover{
    border-color: var(--gold) !important;
    color: #000 !important;
    background: var(--gold) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(242,169,59,.35);
}

/* ---- Primary CTA buttons (Process Video) — solid gold ---- */
button[kind="primary"], button[kind="primaryFormSubmit"]{
    background: linear-gradient(90deg, var(--gold), var(--gold-strong)) !important;
    border: 1px solid var(--gold) !important;
    color: #000 !important;
    font-weight: 700 !important;
}
button[kind="primary"]:hover{
    box-shadow: 0 8px 20px rgba(242,169,59,.45) !important;
    transform: translateY(-2px);
}

/* ---- Tabs ---- */
button[data-baseweb="tab"]{
    font-weight: 600;
    color: var(--text-dim);
}
button[data-baseweb="tab"][aria-selected="true"]{
    color: var(--gold) !important;
}
div[data-baseweb="tab-highlight"]{ background-color: var(--gold) !important; }

/* ---- Chat bubbles: outer card kept, inner "sub rectangle" removed,
   text made bigger and whiter ---- */
div[data-testid="stChatMessage"]{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    animation: fadeInUp .35s ease;
}
div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] > div{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li{
    color: #FFFFFF !important;
    font-size: 1.08rem !important;
    line-height: 1.7 !important;
}

/* ---- Metrics ---- */
div[data-testid="stMetric"]{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px;
}
div[data-testid="stMetricValue"]{ color: var(--gold) !important; }

/* ---- Small metric-style box for long values (e.g. source URL) ---- */
.va-mini-metric{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    height: 100%;
}
.va-mini-metric .va-mm-label{ color: var(--text-dim); font-size: .82rem; margin-bottom: 6px; }
.va-mini-metric .va-mm-value{
    color: var(--gold);
    font-size: .88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
}

/* ---- Native alert boxes (info / warning / success / error) ----
   restyle to black surface with a coloured LEFT BORDER only,
   so nothing looks like a big flat yellow highlighter box. */
div[data-testid="stAlert"]{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--gold) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
div[data-testid="stAlert"] p{ color: var(--text) !important; }

/* ---- Expanders ---- */
details{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
summary{ color: var(--gold) !important; font-weight:600; }

/* ---- Inputs ---- */
input, textarea, .stTextInput input, .stTextArea textarea{
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
input:focus, textarea:focus{ border-color: var(--gold) !important; box-shadow: 0 0 0 1px var(--gold) !important; }

/* ---- Divider ---- */
hr{ border-color: var(--border) !important; }

@keyframes fadeInUp{
    from{ opacity:0; transform: translateY(10px); }
    to{ opacity:1; transform: translateY(0); }
}
@keyframes pulse{
    0%{ opacity:1; } 50%{ opacity:.35; } 100%{ opacity:1; }
}
@keyframes shimmer{
    0%{ background-position: 0% center; }
    100%{ background-position: 200% center; }
}

/* ---- Section boxes (hero / chips / info-bar / sidebar boxes are each their own rectangle) ---- */
.va-section{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 18px;
    animation: fadeInUp .5s ease;
    transition: border-color .2s ease;
}
.va-section:hover{ border-color: var(--gold); }
.va-section.spaced{ margin-top: 30px; }
.va-section.tight{ padding: 16px 18px; margin-bottom: 14px; }
.va-info-line{
    display:flex; align-items:center; gap:12px;
    color: var(--text); font-size: 1rem;
}
.va-info-line .va-hand{ font-size: 1.4rem; animation: pulse 1.6s infinite; }

code{
    background: var(--surface-2) !important;
    color: var(--gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
}

/* ---- Source toggle buttons — reliably scoped via Streamlit's auto
   `.st-key-<key>` class (not guessed internal radio/segmented DOM).
   YouTube = red, Upload = blue, white text always, identical pill shape. */
.st-key-src_youtube_btn button,
.st-key-src_upload_btn button{
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
.st-key-src_youtube_btn button p,
.st-key-src_upload_btn button p{
    color: #FFFFFF !important;
}

.st-key-src_youtube_btn button{ background: #E53935 !important; }
.st-key-src_youtube_btn button:hover{ background: #FF4B45 !important; }
.st-key-src_youtube_btn button[kind="primary"]{
    background: #E53935 !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(229,57,53,.55) !important;
}

.st-key-src_upload_btn button{ background: #2563EB !important; }
.st-key-src_upload_btn button:hover{ background: #3B82F6 !important; }
.st-key-src_upload_btn button[kind="primary"]{
    background: #2563EB !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.55) !important;
}

/* ---- Real bordered content boxes (st.container(border=True)) ----
   Used for Title / Summary / Action Items / Decisions / Questions /
   Transcript so the AI's markdown (**bold**, lists, etc.) is parsed
   properly by Streamlit instead of being frozen as literal text. */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--gold) !important;
    border-radius: 14px !important;
}
.va-card-title{
    font-family: 'Poppins', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--gold);
    margin: 4px 0 14px 0;
    display: flex; align-items: center; gap: 10px;
}
div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] li{
    color: #F5F0E6 !important;
    font-size: 1.04rem !important;
    line-height: 1.8 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] strong{
    color: #FFFFFF !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3{
    color: var(--gold) !important;
    margin-top: 18px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] ol,
div[data-testid="stVerticalBlockBorderWrapper"] ul{
    padding-left: 1.5em;
    margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


_card_seq = {"n": 0}


def card(title: str, body: str, accent: str = "var(--gold)", key: str = None) -> None:
    """Render one bordered black-surface box with a big title and real
    markdown body (so **bold**, lists, etc. from the AI render properly)."""
    if key is None:
        _card_seq["n"] += 1
        key = f"card_{_card_seq['n']}"
    with st.container(border=True, key=key):
        st.markdown(
            f'<div class="va-card-title" style="color:{accent};">{title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(clean_ai_text(body), unsafe_allow_html=True)


# ------------------------------------------------------------------
# PIPELINE (same steps as main.run_pipeline, shown as an animated
# percentage progress bar instead of a step-by-step text log)
# ------------------------------------------------------------------
def run_full_pipeline(source: str):
    bar = st.progress(0, text="🚀 0% — Starting...")
    try:
        bar.progress(10, text="📥 10% — Downloading / converting / chunking audio...")
        chunks = process_audio(source)

        bar.progress(35, text=f"🗣️ 35% — Transcribing {len(chunks)} chunk(s) with Whisper...")
        transcript = transcribe_all(chunks)

        bar.progress(50, text="🏷️ 50% — Generating title with Groq...")
        title = generate_title(transcript)

        bar.progress(65, text="📝 65% — Generating summary with Groq...")
        summary = summarize(transcript)

        bar.progress(75, text="✅ 75% — Extracting action items...")
        action_items = extract_action_items(transcript)

        bar.progress(82, text="🔑 82% — Extracting key decisions...")
        decisions = extract_key_decisions(transcript)

        bar.progress(88, text="❓ 88% — Extracting open questions...")
        questions = extract_questions(transcript)

        bar.progress(93, text="🧹 93% — Clearing previous knowledge base...")
        clear_vector_db()

        bar.progress(97, text="🔍 97% — Building the RAG knowledge base for chat...")
        rag_chain = build_rag_chain(transcript)

        bar.progress(100, text="✅ 100% — Done!")
        time.sleep(0.5)
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(f"Something went wrong: {e}")
        return None

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    if GROQ_API_KEY:
        status_pill = '<span class="va-pill ok"><span class="va-dot"></span>Groq API connected</span>'
    else:
        status_pill = '<span class="va-pill bad"><span class="va-dot"></span>GROQ_API_KEY missing</span>'

    # ---- Header box: title + tagline + status pill, all in one rectangle ----
    st.markdown(
        f"""
<div class="va-section tight">
    <h3 style="margin:0 0 6px 0;">🎥 Video Agent</h3>
    <p style="color:var(--text-dim); margin:0 0 14px 0; font-size:.92rem;">
        Turn any video into a summary, insights, and a chat assistant.
    </p>
    {status_pill}
</div>
""",
        unsafe_allow_html=True,
    )
    if not GROQ_API_KEY:
        st.caption("Add `GROQ_API_KEY=...` to your .env file.")

    st.markdown("#### 📂 Source")

    if "source_type" not in st.session_state:
        st.session_state.source_type = "YouTube URL"

    col_yt, col_up = st.columns(2)
    with col_yt:
        yt_clicked = st.button(
            "📺 YouTube URL",
            use_container_width=True,
            key="src_youtube_btn",
            type="primary" if st.session_state.source_type == "YouTube URL" else "secondary",
        )
    with col_up:
        up_clicked = st.button(
            "📁 Upload local file",
            use_container_width=True,
            key="src_upload_btn",
            type="primary" if st.session_state.source_type == "Upload local file" else "secondary",
        )

    if yt_clicked:
        st.session_state.source_type = "YouTube URL"
        st.rerun()
    if up_clicked:
        st.session_state.source_type = "Upload local file"
        st.rerun()

    source_type = st.session_state.source_type

    source_value = None
    if source_type == "YouTube URL":
        url_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        source_value = url_value.strip() if url_value else None
        label_for_source = url_value
    else:
        uploaded = st.file_uploader(
            "Upload video or audio",
            type=["mp4", "mov", "mkv", "wav", "mp3", "m4a", "webm"],
        )
        if uploaded is not None:
            source_value = save_uploaded_file(uploaded)
            label_for_source = uploaded.name
        else:
            label_for_source = None

    process_clicked = st.button(
        "🚀 Process Video",
        use_container_width=True,
        disabled=not GROQ_API_KEY,
        type="primary",
    )

    st.divider()
    st.markdown("#### 🧰 Session Tools")

    col_a, col_b = st.columns(2)
    with col_a:
        reset_chat_clicked = st.button("🔄 Reset Chat", use_container_width=True)
    with col_b:
        remove_video_clicked = st.button("🗑️ Remove Video", use_container_width=True)

    new_video_clicked = st.button("🆕 Start New Video (Full Reset)", use_container_width=True)

    st.markdown(
        f"""
<div class="va-section tight">
    <p style="margin:0 0 10px 0;"><b style="color:var(--gold);">Reset Chat</b> clears the conversation only.</p>
    <p style="margin:0 0 10px 0;"><b style="color:var(--gold);">Remove Video</b> deletes the downloaded/processed audio files in <code>{DOWNLOAD_DIR}/</code>.</p>
    <p style="margin:0;"><b style="color:var(--gold);">Full Reset</b> clears everything, including the knowledge base.</p>
</div>
""",
        unsafe_allow_html=True,
    )

# ---- Sidebar button actions ----
if process_clicked:
    if not source_value:
        st.sidebar.warning("Please provide a YouTube URL or upload a file first.")
    else:
        result = run_full_pipeline(source_value)
        if result:
            st.session_state.result = result
            st.session_state.processed = True
            st.session_state.chat_history = []
            st.session_state.translated_summary = None
            st.session_state.shortened_summary = None
            st.session_state.source_label = label_for_source
            st.rerun()

if reset_chat_clicked:
    st.session_state.chat_history = []
    st.toast("Chat history cleared.", icon="🔄")
    st.rerun()

if remove_video_clicked:
    n = delete_downloaded_audio()
    st.toast(f"🗑️ Removed {n} audio file(s) from '{DOWNLOAD_DIR}/'.", icon="🗑️")

if new_video_clicked:
    full_reset()
    st.toast("Session fully reset. Ready for a new video!", icon="🆕")
    st.rerun()

# ------------------------------------------------------------------
# MAIN AREA
# ------------------------------------------------------------------
# ---- Section 1: Header / hero box ----
st.markdown(
    """
<div class="va-hero va-section">
    <h1>🎥 Video Agent</h1>
    <p>Transcribe, summarize, extract insights and chat with any video — powered by Whisper + Groq + RAG.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.processed:
    # ---- Section 2: Feature chips box ----
    st.markdown(
        """
<div class="va-section">
<div class="va-chip-grid">
    <div class="va-chip">🎵 Audio extraction</div>
    <div class="va-chip">🗣️ Whisper transcription</div>
    <div class="va-chip">🏷️ AI title & summary</div>
    <div class="va-chip">✅ Action items</div>
    <div class="va-chip">🔑 Key decisions</div>
    <div class="va-chip">❓ Open questions</div>
    <div class="va-chip">💬 Chat with your video</div>
    <div class="va-chip">🌐 Translate & shorten</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---- Section 3: Info bar box (extra top margin separates it from chips) ----
    st.markdown(
        """
<div class="va-section spaced">
    <div class="va-info-line">
        <span class="va-hand">👈</span>
        <span>Add a YouTube URL or upload a file in the sidebar, then click <b>Process Video</b> to get started.</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

else:
    result = st.session_state.result

    card("📌 Title", result["title"].strip(), accent="var(--gold)", key="card_title")

    word_count = len(result["transcript"].split())
    source_display = re.sub(r"^https?://(www\.)?", "", st.session_state.source_label or "—")
    if len(source_display) > 26:
        source_display = source_display[:23] + "..."

    c1, c2, c3 = st.columns(3)
    c1.metric("📝 Transcript words", f"{word_count:,}")
    c2.metric("🎬 Source", source_display)
    c3.metric("💬 Chat messages", len(st.session_state.chat_history))

    tab_overview, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Questions", "📜 Transcript", "💬 Chat"]
    )

    # ---- SUMMARY TAB ----
    with tab_overview:
        card("📋 Video Summary", result["summary"], accent="var(--gold)", key="card_summary")

        # Full-width, stacked (not split into two half-screen columns)
        with st.expander("✂️ Shorten this summary"):
            if st.button("Generate shorter version", key="shorten_btn"):
                with st.spinner("Shortening..."):
                    st.session_state.shortened_summary = shorten_summary(result["summary"])
            if st.session_state.shortened_summary:
                card(
                    "✂️ Shorter Summary",
                    st.session_state.shortened_summary,
                    accent="var(--ok)",
                    key="card_short",
                )

        with st.expander("🌐 Translate this summary"):
            lang = st.text_input("Target language", placeholder="e.g. French, Urdu, Spanish")
            if st.button("Translate", key="translate_btn") and lang.strip():
                with st.spinner(f"Translating into {lang}..."):
                    st.session_state.translated_summary = translate_summary(result["summary"], lang)
                    st.session_state.translated_lang = lang
            if st.session_state.translated_summary:
                card(
                    f"🌐 Summary ({st.session_state.translated_lang})",
                    st.session_state.translated_summary,
                    accent="var(--info-blue)",
                    key="card_translated",
                )

        st.download_button(
            "⬇️ Download Summary (.txt)",
            data=result["summary"],
            file_name="summary.txt",
            use_container_width=True,
        )

    # ---- ACTION ITEMS TAB (small green accent = "done/to-do" cue) ----
    with tab_actions:
        card("✅ Action Items", result["action_items"], accent="var(--ok)", key="card_actions")

    # ---- DECISIONS TAB (main gold accent) ----
    with tab_decisions:
        card("🔑 Key Decisions", result["key_decisions"], accent="var(--gold)", key="card_decisions")

    # ---- QUESTIONS TAB (small blue accent) ----
    with tab_questions:
        card("❓ Open Questions", result["open_questions"], accent="var(--info-blue)", key="card_questions")

    # ---- TRANSCRIPT TAB ----
    with tab_transcript:
        card("📜 Full Transcript", result["transcript"], accent="var(--gold)", key="card_transcript")
        st.download_button(
            "⬇️ Download Transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            use_container_width=True,
        )

    # ---- CHAT TAB ----
    with tab_chat:
        st.caption("Ask anything about the video, or use a quick action below.")

        quick_prompts = {
            "📝 Quiz (5Q)": "Create a 5-question quiz based on this video, with an answer key at the end.",
            "🎯 MCQs (5)": "Create 5 multiple-choice questions (4 options each) from this video, and give the correct answers at the end.",
            "💡 Explain Simply": "Explain the main topic of this video in simple, beginner-friendly terms.",
            "⭐ Key Points": "List the most important points from this video as clear bullet points.",
            "📖 Study Notes": "Create detailed, well-structured study notes from this video.",
        }

        qcols = st.columns(len(quick_prompts))
        quick_clicked = None
        for col, (label, prompt) in zip(qcols, quick_prompts.items()):
            with col:
                if st.button(label, use_container_width=True, key=f"qa_{label}"):
                    quick_clicked = prompt

        st.divider()

        chat_box = st.container()
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(clean_ai_text(msg["content"]), unsafe_allow_html=True)

        user_input = st.chat_input("Ask a question, request a quiz, ask for an explanation...")

        prompt_to_send = quick_clicked or user_input

        if prompt_to_send:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_send})
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = ask_question(result["rag_chain"], prompt_to_send)
                except Exception as e:
                    answer = f"⚠️ Error while answering: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

st.markdown(
    "<div style='text-align:center; color:#5A5A6E; font-size:.8rem; margin-top:30px;'>"
    "Video Agent · Whisper + Groq + LangChain RAG</div>",
    unsafe_allow_html=True,
)