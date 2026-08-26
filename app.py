"""
Reddit's Feelings — Search any topic, brand, or product to analyze Reddit's consensus and sentiment.
"""

import textwrap
import datetime as dt
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from reddit_client import get_reddit_client
from nlp_pipeline import clean_text, sentiment_label_and_score, top_keyphrases, compute_weighted_metrics

# ────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ────────────────────────────────────────────────────────────────────────
TIME_FILTERS = {
    "Past 24 Hours": "day",
    "Past Week": "week",
    "Past Month": "month",
    "Past Year": "year",
    "All Time": "all",
}
MAX_SEARCHES_PER_SESSION = 15

SUGGESTED_TOPICS = [
    "Steam Deck",
    "iPhone 16 Pro",
    "Notion",
    "Tesla",
    "Cyberpunk 2077",
    "Diamond Gym",
]

# ────────────────────────────────────────────────────────────────────────
# MARKDOWN HELPER
# ────────────────────────────────────────────────────────────────────────
def md(html: str) -> None:
    """Render an HTML block via st.markdown, safely stripped of the source
    indentation that would otherwise be misread by the Markdown parser as
    a fenced code block."""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def dot(color: str) -> str:
    """A small inline colored dot used in place of emoji icons."""
    return f'<span class="status-dot" style="background:{color};"></span>'


# ────────────────────────────────────────────────────────────────────────
# ICON SET — minimal stroke-based line icons (no emoji, no gradients).
# All share the same visual language: 1.8px stroke, rounded joins, and
# inherit color from their container via currentColor.
# ────────────────────────────────────────────────────────────────────────
def icon(name: str, size: int = 20) -> str:
    paths = {
        # magnifying glass — search / scrape step
        "search": '<circle cx="10.5" cy="10.5" r="6.5"/><line x1="15.3" y1="15.3" x2="20" y2="20"/>',
        # waveform — the "pulse" of a community's sentiment; the brand mark
        "pulse": '<polyline points="1.5 12 6.5 12 9 4.5 14.5 19.5 17 12 22.5 12"/>',
        # concentric rings — verdict / consensus step
        "target": '<circle cx="12" cy="12" r="8.2"/><circle cx="12" cy="12" r="4.2"/><circle cx="12" cy="12" r="0.6" fill="currentColor" stroke="none"/>',
        # magnifying glass with a slash — empty state
        "search-off": '<circle cx="10.2" cy="10.2" r="6.2"/><line x1="14.8" y1="14.8" x2="20.5" y2="20.5"/><line x1="5.8" y1="20.5" x2="20.5" y2="5.8"/>',
        # speech bubble — discussion feed
        "chat": '<path d="M4 5.5h16v11H9.5L5 20.5V16.5H4z"/>',
    }
    return f"""<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none"
        stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        {paths[name]}
    </svg>"""


def build_favicon() -> Image.Image:
    """A small indigo pulse mark, drawn at runtime — avoids relying on an
    emoji glyph for the browser tab icon."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, size - 2, size - 2), radius=16, fill=(79, 70, 229, 255))
    pts = [(12, 34), (22, 34), (27, 16), (37, 50), (43, 34), (52, 34)]
    draw.line(pts, fill=(255, 255, 255, 255), width=5, joint="curve")
    return img


# ────────────────────────────────────────────────────────────────────────
# STYLING & DESIGN SYSTEM
# ────────────────────────────────────────────────────────────────────────
def load_custom_theme() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --border-color: #E2E8F0;
        --primary-indigo: #4F46E5;
        --primary-indigo-hover: #4338CA;
        --reddit-orange: #FF4500;
        --positive-green: #10B981;
        --positive-bg: #ECFDF5;
        --positive-border: #A7F3D0;
        --neutral-slate: #64748B;
        --neutral-bg: #F1F5F9;
        --negative-red: #EF4444;
        --negative-bg: #FEF2F2;
        --negative-border: #FECACA;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background-color: var(--bg-main);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1140px;
    }

    /* ---- Smooth Card Containers ---- */
    .rf-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    /* ---- Small inline status dot (replaces emoji icons) ---- */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
        vertical-align: middle;
    }

    /* ---- App Header ---- */
    .app-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .session-meta {
        text-align: right;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        padding-top: 0.4rem;
    }

    /* ---- Executive Verdict Banner ---- */
    .verdict-container {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1.5px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.25rem;
    }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }

    .score-circle {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 3px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        flex-shrink: 0;
    }
    .score-num {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        color: var(--text-primary);
    }
    .score-denom {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }

    .summary-box {
        background: #F1F5F9;
        border-left: 4px solid var(--primary-indigo);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        font-size: 0.98rem;
        line-height: 1.6;
        color: var(--text-primary);
        margin-top: 1rem;
    }

    /* ---- Two-Tone Sentiment Result Cards ---- */
    .tone-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .tone-card {
        border-radius: 18px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        border: 1.5px solid transparent;
        position: relative;
        overflow: hidden;
    }
    .tone-card.positive {
        background: var(--positive-bg);
        border-color: var(--positive-border);
    }
    .tone-card.negative {
        background: var(--negative-bg);
        border-color: var(--negative-border);
    }
    .tone-label-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .tone-label {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .tone-card.positive .tone-label { color: #047857; }
    .tone-card.negative .tone-label { color: #B91C1C; }
    .tone-tag {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        background: rgba(255,255,255,0.7);
    }
    .tone-card.positive .tone-tag { color: #047857; }
    .tone-card.negative .tone-tag { color: #B91C1C; }
    .tone-value {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .tone-card.positive .tone-value { color: var(--positive-green); }
    .tone-card.negative .tone-value { color: var(--negative-red); }
    .tone-copy {
        margin-top: 0.6rem;
        font-size: 0.86rem;
        line-height: 1.45;
        color: var(--text-secondary);
    }
    .neutral-note {
        text-align: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-muted);
        margin: 0.25rem 0 1.5rem 0;
    }

    /* ---- Keyphrase Chips ---- */
    .phrase-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.4rem 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        color: #334155;
        margin: 0 0.4rem 0.5rem 0;
        transition: all 0.15s ease;
    }
    .phrase-chip:hover {
        border-color: var(--primary-indigo);
        color: var(--primary-indigo);
        background: #EEF2FF;
    }
    .phrase-count {
        color: var(--text-muted);
        font-size: 0.75rem;
    }

    /* ---- Evidence Feed Post Cards ---- */
    .post-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .post-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .post-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .post-sub {
        color: var(--reddit-orange);
        background: #FFF5F2;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-weight: 700;
    }
    .post-title-link {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        text-decoration: none;
        line-height: 1.4;
    }
    .post-title-link:hover {
        color: var(--primary-indigo);
        text-decoration: underline;
    }
    .post-body-snippet {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    .pill-pos {
        background: var(--positive-bg);
        color: var(--positive-green);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .pill-neg {
        background: var(--negative-bg);
        color: var(--negative-red);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .pill-neu {
        background: var(--neutral-bg);
        color: var(--neutral-slate);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* ---- Hero Search Input Overrides ---- */
    div[data-baseweb="input"] > div {
        border-radius: 12px !important;
        border: 2px solid var(--border-color) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within > div {
        border-color: var(--primary-indigo) !important;
    }

    /* ---- Button System (harmonized to the indigo accent) ---- */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.25rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-indigo) !important;
        border: 1px solid var(--primary-indigo) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px -2px rgba(79, 70, 229, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--primary-indigo-hover) !important;
        border-color: var(--primary-indigo-hover) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--primary-indigo) !important;
        color: var(--primary-indigo) !important;
    }

    .hero-box {
        text-align: center;
        padding: 3rem 1.5rem 2rem 1.5rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.05);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        margin-bottom: 0.6rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    div[class*="st-key-chip_wrap"] button {
        border-radius: 9999px !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.85rem !important;
    }
    div[class*="st-key-chip_wrap"] button:hover {
        border-color: var(--primary-indigo) !important;
        color: var(--primary-indigo) !important;
        background: #EEF2FF !important;
    }

    /* ---- Brand mark (header + hero) ---- */
    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: #EEF2FF;
        color: var(--primary-indigo);
        margin-right: 0.6rem;
        flex-shrink: 0;
    }

    /* ---- "How it works" tile strip ---- */
    .flow-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 0.5rem 0 0.5rem 0;
    }
    .flow-tile {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.25rem 1.25rem 1.4rem 1.25rem;
        text-align: left;
    }
    .flow-tile-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: #EEF2FF;
        color: var(--primary-indigo);
        margin-bottom: 0.85rem;
    }
    .flow-tile-step {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .flow-tile-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
    }
    .flow-tile-copy {
        font-size: 0.85rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }

    /* ---- Empty state ---- */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 2.5rem 1.5rem;
        gap: 0.65rem;
    }
    .empty-state-icon { color: var(--text-muted); }
    .empty-state-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .empty-state-copy {
        font-size: 0.9rem;
        color: var(--text-secondary);
        max-width: 440px;
    }

    /* ---- Section heading with icon ---- */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--text-primary);
    }
    .section-heading .brand-mark { margin-right: 0; width: 26px; height: 26px; }

    @media (max-width: 700px) {
        .tone-grid { grid-template-columns: 1fr; }
        .flow-strip { grid-template-columns: 1fr; }
    }
    </style>
    """

def styled_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#475569"),
        margin=dict(l=10, r=10, t=15, b=15),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig

# ────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ────────────────────────────────────────────────────────────────────────
def _row_from_post(post) -> dict:
    return {
        "subreddit": str(post.subreddit),
        "title": post.title,
        "body": post.selftext or "",
        "score": post.score,
        "num_comments": post.num_comments,
        "created": dt.datetime.utcfromtimestamp(post.created_utc),
        "url": f"https://reddit.com{post.permalink}",
    }


def _is_relevant(row: dict, query_terms: list) -> bool:
    """Require every significant query word to actually appear in the post's
    title or body.

    Reddit's own search endpoint matches loosely — multi-word queries are
    effectively OR-ed together and results get stemmed/expanded — so a
    query like a person's name (e.g. two words, each individually common
    elsewhere) can pull back posts that only share one incidental word with
    the query and are otherwise completely off-topic. This is a cheap
    safety net that filters those out after the fact.
    """
    haystack = f"{row['title']} {row['body']}".lower()
    return all(term in haystack for term in query_terms)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_reddit_posts(query: str, subreddits: str, time_filter: str, limit: int) -> pd.DataFrame:
    reddit = get_reddit_client()
    target = subreddits.replace(" ", "") if subreddits != "all" else "all"
    sub = reddit.subreddit(target)

    def _run_search(q: str) -> list:
        return [
            _row_from_post(p)
            for p in sub.search(q, sort="relevance", time_filter=time_filter, limit=limit)
        ]

    # Search as an exact phrase first — this is what stops a multi-word
    # query being split into an OR search by Reddit and matching posts that
    # only contain one of the words.
    rows = _run_search(f'"{query}"') if " " in query else _run_search(query)

    # A strict phrase search can come back thin for topics people discuss
    # more loosely (misspellings, abbreviations, etc). If it did, widen the
    # search — but keep the relevance filter below as a safety net so we
    # don't reintroduce the off-topic results the phrase search avoided.
    if len(rows) < min(10, limit):
        rows = _run_search(query)

    query_terms = [t for t in query.lower().split() if len(t) > 1]
    if query_terms:
        rows = [r for r in rows if _is_relevant(r, query_terms)]

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────
# MAIN APP ORCHESTRATION
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit's Feelings — Sentiment Consensus Engine",
    page_icon=build_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(load_custom_theme(), unsafe_allow_html=True)

if "search_count" not in st.session_state:
    st.session_state.search_count = 0
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# Top Header Bar
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    md(f"""
        <div style="display:flex; align-items:center; padding-bottom: 0.5rem;">
            <span class="brand-mark">{icon('pulse', 17)}</span>
            <span class="app-title">Reddit's Feelings</span>
        </div>
    """)
with col_h2:
    searches_remaining = max(0, MAX_SEARCHES_PER_SESSION - st.session_state.search_count)
    md(f"""
        <div class="session-meta">
            {searches_remaining} searches remaining this session
        </div>
    """)

# ────────────────────────────────────────────────────────────────────────
# SEARCH CONTROLS & HERO SECTION
# ────────────────────────────────────────────────────────────────────────
if not st.session_state.active_query:
    # Landing Hero Page
    md("""
        <div class="hero-box">
            <div class="hero-title">What does Reddit think?</div>
            <div class="hero-subtitle">
                Enter any product, brand, media, or topic to instantly analyze community sentiment, pros & cons, and consensus across thousands of discussions.
            </div>
        </div>
    """)

    with st.container():
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            search_input = st.text_input(
                "Search topic",
                placeholder="e.g. Steam Deck, Notion, PlayStation 5, Cyberpunk 2077...",
                label_visibility="collapsed",
                key="hero_query_input",
            )
        with sc2:
            search_btn = st.button("Analyze Consensus", type="primary", use_container_width=True)

        with st.expander("Advanced Search Filters (Subreddits, Time Range, Limit)", expanded=False):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                subreddits_val = st.text_input("Subreddits (all or comma-separated)", value="all", key="hero_subs")
            with fcol2:
                time_range_val = st.selectbox("Time range", list(TIME_FILTERS.keys()), index=2, key="hero_time")
            with fcol3:
                max_posts_val = st.slider("Max discussions to scrape", 25, 200, 75, step=5, key="hero_limit")

        # Trending Chips
        md("<div style='font-size:0.85rem; font-weight:700; color:#64748B; margin: 1.2rem 0 0.5rem 0;'>TRY POPULAR SEARCHES:</div>")
        chip_cols = st.columns(len(SUGGESTED_TOPICS))
        for idx, topic_str in enumerate(SUGGESTED_TOPICS):
            with chip_cols[idx]:
                with st.container(key=f"chip_wrap_{idx}"):
                    if st.button(topic_str, key=f"chip_{idx}", use_container_width=True):
                        st.session_state.search_count += 1
                        st.session_state.active_query = topic_str
                        st.rerun()

        # "How it works" — fills the space below the fold on first load with
        # something functional (what actually happens) rather than filler.
        md(f"""
            <div class="flow-strip">
                <div class="flow-tile">
                    <div class="flow-tile-icon">{icon('search', 18)}</div>
                    <div class="flow-tile-step">Step 1</div>
                    <div class="flow-tile-title">Scrape the discussion</div>
                    <div class="flow-tile-copy">Pulls matching posts straight from Reddit's public search across the subreddits and time range you choose.</div>
                </div>
                <div class="flow-tile">
                    <div class="flow-tile-icon">{icon('pulse', 18)}</div>
                    <div class="flow-tile-step">Step 2</div>
                    <div class="flow-tile-title">Score the sentiment</div>
                    <div class="flow-tile-copy">Each post is scored with VADER sentiment analysis and weighted by its upvotes and comment count.</div>
                </div>
                <div class="flow-tile">
                    <div class="flow-tile-icon">{icon('target', 18)}</div>
                    <div class="flow-tile-step">Step 3</div>
                    <div class="flow-tile-title">Read the consensus</div>
                    <div class="flow-tile-copy">Get a single 0–100 verdict, the themes driving it, and the actual posts behind the number.</div>
                </div>
            </div>
        """)

        if search_btn and search_input.strip():
            st.session_state.search_count += 1
            st.session_state.active_query = search_input.strip()
            st.rerun()

else:
    # Results View Header Search Bar
    with st.container():
        sc1, sc2, sc3 = st.columns([3, 1, 1])
        with sc1:
            search_input = st.text_input(
                "Topic",
                value=st.session_state.active_query,
                label_visibility="collapsed",
                key="results_query_input",
            )
        with sc2:
            new_search_btn = st.button("Update Search", type="primary", use_container_width=True)
        with sc3:
            clear_btn = st.button("New Search", use_container_width=True)

        with st.expander("Adjust Scope & Time Filter", expanded=False):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                subreddits_val = st.text_input("Subreddits", value="all", key="res_subs")
            with fcol2:
                time_range_val = st.selectbox("Time range", list(TIME_FILTERS.keys()), index=2, key="res_time")
            with fcol3:
                max_posts_val = st.slider("Max discussions", 25, 200, 75, step=5, key="res_limit")

    if clear_btn:
        st.session_state.active_query = ""
        st.rerun()

    if new_search_btn and search_input.strip():
        st.session_state.search_count += 1
        st.session_state.active_query = search_input.strip()
        st.rerun()

# ────────────────────────────────────────────────────────────────────────
# RESULTS RENDER PIPELINE
# ────────────────────────────────────────────────────────────────────────
if st.session_state.active_query:
    query = st.session_state.active_query
    subs = st.session_state.get("res_subs", "all")
    t_filter = TIME_FILTERS[st.session_state.get("res_time", "Past Month")]
    limit_cnt = st.session_state.get("res_limit", 75)

    if st.session_state.search_count >= MAX_SEARCHES_PER_SESSION:
        st.error("Session search limit reached — please refresh the page to reset.")
    else:
        with st.status(f"Scanning Reddit discussions for '{query}'...", expanded=True) as status:
            df = fetch_reddit_posts(query, subs, t_filter, limit_cnt)
            status.update(label="Analyzing VADER sentiment vectors & engagement weights...")

            if not df.empty:
                df["combined"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
                df["cleaned"] = df["combined"].apply(clean_text)
                results = df["combined"].apply(sentiment_label_and_score)
                df["sentiment"] = results.apply(lambda x: x[0])
                df["compound"] = results.apply(lambda x: x[1])
                df["ts"] = df["created"].dt.strftime("%b %d, %Y")

            status.update(label="Consensus analysis complete!", state="complete", expanded=False)

        if df.empty:
            md(f"""
                <div class="rf-card empty-state">
                    <div class="empty-state-icon">{icon('search-off', 40)}</div>
                    <div class="empty-state-title">No discussions found for "{query}"</div>
                    <div class="empty-state-copy">
                        Nothing matched in the selected time range and subreddits. Try widening the time range,
                        searching "all" subreddits, or simplifying the query to fewer, more distinctive words.
                    </div>
                </div>
            """)
        else:
            metrics = compute_weighted_metrics(df, query)

            # ─────────────────────────────────────────────────────────────
            # 1. EXECUTIVE VERDICT BANNER
            # ─────────────────────────────────────────────────────────────
            md(f"""
                <div class="verdict-container">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.5rem;">
                        <div style="flex:1; min-width:280px;">
                            <div style="font-size:0.85rem; font-weight:700; color:#64748B; letter-spacing:0.05em; margin-bottom:0.5rem; text-transform:uppercase;">
                                COMMUNITY CONSENSUS VERDICT
                            </div>
                            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.75rem; flex-wrap:wrap;">
                                <span class="verdict-badge" style="background-color:{metrics['badge_color']};">
                                    ● {metrics['verdict']}
                                </span>
                                <span style="font-size:1.1rem; font-weight:700; color:#0F172A;">
                                    {query}
                                </span>
                            </div>
                            <div class="summary-box">
                                {metrics['summary_text']}
                            </div>
                        </div>
                        <div class="score-circle">
                            <span class="score-num">{metrics['score']}</span>
                            <span class="score-denom">OUT OF 100</span>
                        </div>
                    </div>
                </div>
            """)

            # ─────────────────────────────────────────────────────────────
            # 1b. TWO-TONE SENTIMENT RESULT CARDS
            # ─────────────────────────────────────────────────────────────
            md(f"""
                <div class="tone-grid">
                    <div class="tone-card positive">
                        <div class="tone-label-row">
                            <span class="tone-label">Positive</span>
                            <span class="tone-tag">Favorable</span>
                        </div>
                        <div class="tone-value">{metrics['pos_pct']:.0%}</div>
                        <div class="tone-copy">Share of tracked discussions with a clearly positive tone toward "{query}".</div>
                    </div>
                    <div class="tone-card negative">
                        <div class="tone-label-row">
                            <span class="tone-label">Negative</span>
                            <span class="tone-tag">Critical</span>
                        </div>
                        <div class="tone-value">{metrics['neg_pct']:.0%}</div>
                        <div class="tone-copy">Share of tracked discussions raising complaints or concerns about "{query}".</div>
                    </div>
                </div>
                <div class="neutral-note">{metrics['neu_pct']:.0%} of discussions were neutral in tone</div>
            """)

            # Quick Metric Grid
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Discussions Scraped", f"{len(df):,}")
            with mcol2:
                st.metric("Total Upvotes", f"{metrics['total_upvotes']:,}")
            with mcol3:
                st.metric("Total Comments", f"{metrics['total_comments']:,}")
            with mcol4:
                top_sub_name = f"r/{df['subreddit'].mode()[0]}" if not df.empty else "N/A"
                st.metric("Primary Subreddit", top_sub_name)

            st.write("")

            # ─────────────────────────────────────────────────────────────
            # 2. KEY HIGHLIGHTS & EXTRACTED THEMES
            # ─────────────────────────────────────────────────────────────
            md(f"""
                <h3 class="section-heading">
                    <span class="brand-mark">{icon('pulse', 15)}</span> Key Themes &amp; Community Opinions
                </h3>
            """)

            hcol1, hcol2 = st.columns(2)

            with hcol1:
                md(f"""
                    <div class="rf-card">
                        <div style="font-weight:800; font-size:1.1rem; color:#10B981; margin-bottom:1rem; display:flex; align-items:center;">
                            {dot('#10B981')} What Reddit Loves
                        </div>
                """)
                pos_posts = df.sort_values(["compound", "score"], ascending=[False, False]).head(3)
                for _, p in pos_posts.iterrows():
                    md(f"""
                        <div style="margin-bottom:0.85rem; padding-bottom:0.75rem; border-bottom:1px solid #F1F5F9;">
                            <div style="font-size:0.78rem; font-weight:700; color:#64748B;">r/{p['subreddit']} &middot; {p['score']:,} upvotes</div>
                            <a href="{p['url']}" target="_blank" class="post-title-link" style="font-size:0.95rem;">{p['title']}</a>
                        </div>
                    """)
                md("</div>")

            with hcol2:
                md(f"""
                    <div class="rf-card">
                        <div style="font-weight:800; font-size:1.1rem; color:#EF4444; margin-bottom:1rem; display:flex; align-items:center;">
                            {dot('#EF4444')} Main Complaints & Critiques
                        </div>
                """)
                neg_posts = df.sort_values(["compound", "score"], ascending=[True, False]).head(3)
                for _, p in neg_posts.iterrows():
                    md(f"""
                        <div style="margin-bottom:0.85rem; padding-bottom:0.75rem; border-bottom:1px solid #F1F5F9;">
                            <div style="font-size:0.78rem; font-weight:700; color:#64748B;">r/{p['subreddit']} &middot; {p['score']:,} upvotes</div>
                            <a href="{p['url']}" target="_blank" class="post-title-link" style="font-size:0.95rem;">{p['title']}</a>
                        </div>
                    """)
                md("</div>")

            # Keyphrase Chips
            md("<div style='font-size:0.9rem; font-weight:700; color:#475569; margin: 0.5rem 0 0.4rem 0;'>FREQUENT PHRASES & TOPICS:</div>")
            kws = top_keyphrases(df["cleaned"], exclude_terms=query.lower().split(), n=14)
            if kws:
                chips_html = "".join(
                    f'<span class="phrase-chip">{w} <span class="phrase-count">({c})</span></span>' for w, c in kws
                )
                md(f"<div>{chips_html}</div>")

            st.write("")
            st.divider()

            # ─────────────────────────────────────────────────────────────
            # 3. FILTERABLE EVIDENCE DISCUSSION FEED
            # ─────────────────────────────────────────────────────────────
            md(f"""
                <h3 class="section-heading">
                    <span class="brand-mark">{icon('chat', 15)}</span> Top Reddit Discussions
                </h3>
            """)

            feed_tab1, feed_tab2, feed_tab3, feed_tab4 = st.tabs(
                ["All Top Posts", "Positive Vibes", "Critiques & Concerns", "Most Upvoted"]
            )

            def render_post_feed(posts_subset):
                if posts_subset.empty:
                    st.info("No posts matching this filter.")
                    return
                for _, row in posts_subset.iterrows():
                    body_snippet = (row["body"][:220] + "...") if len(row["body"]) > 220 else row["body"]
                    sentiment_pill = (
                        f'<span class="pill-pos">+{row["compound"]:.2f} Pos</span>'
                        if row["compound"] >= 0.05
                        else f'<span class="pill-neg">{row["compound"]:.2f} Neg</span>'
                        if row["compound"] <= -0.05
                        else '<span class="pill-neu">Neutral</span>'
                    )

                    md(f"""
                        <div class="post-card">
                            <div class="post-meta">
                                <span class="post-sub">r/{row['subreddit']}</span>
                                <span>&middot;</span>
                                <span>{row['score']:,} upvotes</span>
                                <span>&middot;</span>
                                <span>{row['num_comments']:,} comments</span>
                                <span>&middot;</span>
                                <span>{row['ts']}</span>
                                <span style="margin-left:auto;">{sentiment_pill}</span>
                            </div>
                            <a href="{row['url']}" target="_blank" class="post-title-link">{row['title']}</a>
                            {f'<div class="post-body-snippet">{body_snippet}</div>' if body_snippet else ''}
                        </div>
                    """)

            with feed_tab1:
                render_post_feed(df.sort_values("score", ascending=False).head(15))
            with feed_tab2:
                render_post_feed(df[df["sentiment"] == "Positive"].sort_values("score", ascending=False).head(15))
            with feed_tab3:
                render_post_feed(df[df["sentiment"] == "Negative"].sort_values("score", ascending=False).head(15))
            with feed_tab4:
                render_post_feed(df.sort_values("score", ascending=False).head(15))

            st.write("")

            # ─────────────────────────────────────────────────────────────
            # 4. COLLAPSIBLE ADVANCED ANALYTICS & RAW DATA
            # ─────────────────────────────────────────────────────────────
            with st.expander("Advanced Analytics, Charts & CSV Export", expanded=False):
                st.markdown("#### Sentiment & Subreddit Distribution")
                cc1, cc2 = st.columns(2)
                with cc1:
                    counts = df["sentiment"].value_counts()
                    fig = go.Figure(go.Bar(
                        x=list(counts.index),
                        y=list(counts.values),
                        marker_color=[
                            {"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"}.get(k, "#64748B")
                            for k in counts.index
                        ],
                    ))
                    fig.update_layout(title="Raw Post Count by Sentiment Category")
                    st.plotly_chart(styled_fig(fig), use_container_width=True)

                with cc2:
                    sub_counts = df["subreddit"].value_counts().head(8)
                    fig2 = go.Figure(go.Bar(
                        x=list(sub_counts.values),
                        y=[f"r/{s}" for s in sub_counts.index],
                        orientation="h",
                        marker_color="#4F46E5",
                    ))
                    fig2.update_layout(title="Top Subreddits by Discussion Volume")
                    fig2.update_yaxes(autorange="reversed")
                    st.plotly_chart(styled_fig(fig2), use_container_width=True)

                st.markdown("#### Sentiment Timeline")
                timeline = df.copy()
                timeline["date"] = timeline["created"].dt.date
                daily = timeline.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
                fig3 = go.Figure()
                for label, color in (("Positive", "#10B981"), ("Neutral", "#64748B"), ("Negative", "#EF4444")):
                    if label in daily.columns:
                        fig3.add_trace(go.Bar(x=daily.index, y=daily[label], name=label, marker_color=color))
                fig3.update_layout(barmode="stack", title="Daily Discussion Sentiment Volume")
                st.plotly_chart(styled_fig(fig3), use_container_width=True)

                st.markdown("#### Raw Scraped Data Table")
                st.dataframe(
                    df[["subreddit", "title", "score", "num_comments", "ts", "sentiment", "compound", "url"]],
                    use_container_width=True,
                    height=300,
                )
                st.download_button(
                    "Download Scraped Data (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"reddit_feelings_{query.replace(' ', '_')}.csv",
                    mime="text/csv",
                )