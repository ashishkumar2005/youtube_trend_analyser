import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="YT Trending Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.database import (
    get_latest_trending,
    get_total_video_count,
    get_all_collected_videos,
)
from src.data_cleaner import clean_and_engineer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

COUNTRY_NAMES = {
    "US": "United States",
    "IN": "India",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
}

def apply_country_names(df):
    if "country" in df.columns:
        df = df.copy()
        df["country"] = df["country"].map(COUNTRY_NAMES).fillna(df["country"])
    return df

PREDICT_FEATURES = [
    "like_view_ratio",
    "comment_view_ratio",
    "engagement_score",
    "title_length",
    "title_word_count",
    "title_has_number",
    "title_exclamation",
    "is_hd",
    "tag_count",
    "publish_hour",
    "title_sentiment",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f7f8fc !important;
    color: #1a1d2e !important;
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
.block-container { padding: 2.5rem 3rem !important; max-width: 100% !important; }

/* ── SIDEBAR ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #1b3a4b 0%, #0d2233 60%, #091929 100%) !important;
    border-right: none !important;
    min-width: 210px !important;
    max-width: 230px !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow: hidden !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}
section[data-testid="stSidebar"] * { color: #8fb8cc !important; }
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 8px !important;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    margin: 1px 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8fb8cc !important;
    line-height: 1 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #4fc3f7 !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSidebarCollapsedControl"] {
    background: #0d2233 !important;
    border: none !important;
}
[data-testid="stSidebarCollapsedControl"] svg { fill: #4fc3f7 !important; }

/* ── KPI Cards ───────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 0 0 36px 0;
}
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #eceef5;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff4444, #ff7070);
}
.kpi-label {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #8b92a5; margin-bottom: 8px;
}
.kpi-value {
    font-size: 30px; font-weight: 700;
    color: #1a1d2e; letter-spacing: -0.03em;
    line-height: 1.1; margin-bottom: 8px;
}
.kpi-sub { font-size: 12px; color: #8b92a5; line-height: 1.5; }

/* ── Page hero ──────────────────────────── */
.page-hero { margin: 0 0 32px 0; }
.page-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 30px; font-weight: 400;
    color: #1a1d2e; letter-spacing: -0.02em;
    line-height: 1.2; margin: 0 0 8px 0;
}
.page-sub { font-size: 14px; color: #8b92a5; margin: 0; line-height: 1.6; }

/* ── Section header ─────────────────────── */
.section-hdr {
    display: flex; align-items: center;
    gap: 12px; margin: 36px 0 18px 0;
}
.section-hdr-line { flex: 1; height: 1px; background: #eceef5; }
.section-hdr-text {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #ff4444; white-space: nowrap;
}

/* ── Chart card ─────────────────────────── */
.chart-card {
    background: #ffffff;
    border: 1px solid #eceef5;
    border-radius: 14px;
    padding: 26px 26px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 22px;
}
.chart-question {
    font-size: 15px; font-weight: 600;
    color: #1a1d2e; margin-bottom: 6px; line-height: 1.4;
}
.chart-context {
    font-size: 12px; color: #8b92a5;
    margin-bottom: 18px; line-height: 1.6;
}
.chart-insight-bar {
    background: #fff8e8;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 11px 16px; margin-top: 14px;
    font-size: 12px; color: #664400; line-height: 1.6;
}

/* ── Insight cards ──────────────────────── */
.insights-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin-bottom: 12px;
}
.insight-card {
    background: #ffffff; border: 1px solid #eceef5;
    border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.insight-emoji { font-size: 20px; margin-bottom: 10px; display: block; }
.insight-headline { font-size: 14px; font-weight: 600; color: #1a1d2e; line-height: 1.4; margin-bottom: 7px; }
.insight-detail   { font-size: 12px; color: #8b92a5; line-height: 1.6; }
.insight-tag      { display: inline-block; margin-top: 10px; padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.tag-green  { background: #e8f8ef; color: #1a7a45; }
.tag-red    { background: #fff0f0; color: #cc2222; }
.tag-blue   { background: #eef4ff; color: #2255cc; }
.tag-amber  { background: #fff8e8; color: #885500; }
.tag-purple { background: #f3eeff; color: #5522bb; }

/* ── Country stat cards ─────────────────── */
.country-stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin-bottom: 28px;
}
.country-stat-card {
    background: #ffffff; border: 1px solid #eceef5;
    border-radius: 14px; padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.cs-label {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #8b92a5; margin-bottom: 12px;
}
.cs-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 7px 0;
    border-bottom: 1px solid #f3f4f8; font-size: 13px;
}
.cs-row:last-child { border-bottom: none; }
.cs-country { color: #1a1d2e; font-weight: 500; }
.cs-value   { color: #ff4444; font-weight: 700; font-size: 13px; }

/* ── Action cards ───────────────────────── */
.action-card {
    background: #ffffff; border: 1px solid #eceef5;
    border-radius: 14px; padding: 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.action-num {
    width: 28px; height: 28px; background: #ff4444; color: white;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 700; margin-bottom: 12px;
}
.action-title { font-size: 14px; font-weight: 600; color: #1a1d2e; margin-bottom: 8px; line-height: 1.4; }
.action-desc  { font-size: 12px; color: #8b92a5; line-height: 1.7; }

/* ── Predict cards ──────────────────────── */
.result-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 16px; margin: 18px 0 24px;
}
.result-card {
    background: #ffffff; border: 1.5px solid #eceef5;
    border-radius: 14px; padding: 20px 22px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.result-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #8b92a5; margin-bottom: 9px;
}
.result-value { font-size: 22px; font-weight: 700; color: #1a1d2e; line-height: 1.2; }
.tip-item {
    background: #fff8e8; border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0; padding: 13px 18px;
    margin: 9px 0; font-size: 13px; color: #664400; line-height: 1.7;
}
.ml-info-box {
    background: #eef4ff; border-left: 4px solid #2255cc;
    border-radius: 0 10px 10px 0; padding: 14px 18px;
    margin: 0 0 20px 0; font-size: 13px; color: #1a1d2e; line-height: 1.7;
}

/* ── Inputs ─────────────────────────────── */
.stSelectbox label, .stMultiSelect label,
.stTextInput label, .stNumberInput label, .stSlider label {
    color: #5a6075 !important; font-size: 12px !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
}
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid #dde1ee !important;
    border-radius: 10px !important;
    color: #1a1d2e !important; font-size: 13px !important;
}
[data-baseweb="popover"] {
    background: #ffffff !important; border: 1px solid #eceef5 !important;
    border-radius: 12px !important; box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
}
[data-baseweb="menu"]    { background: #ffffff !important; }
[data-baseweb="option"]  {
    background: #ffffff !important; color: #1a1d2e !important;
    font-size: 13px !important; padding: 10px 16px !important;
}
[data-baseweb="option"]:hover { background: #f7f8fc !important; }
[data-baseweb="tag"] {
    background: #e8f4fd !important; color: #1b3a4b !important;
    border-radius: 6px !important;
}
.stTextInput input, .stNumberInput input {
    background: #ffffff !important; border: 1.5px solid #dde1ee !important;
    border-radius: 10px !important; color: #1a1d2e !important;
    font-size: 13px !important; padding: 10px 14px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff4444, #cc2222) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-size: 14px !important;
    font-weight: 600 !important; padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(255,68,68,0.3) !important;
}
[data-testid="stDataFrame"] {
    border: 1.5px solid #eceef5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stSuccess > div { background: #e8f8ef !important; border: 1px solid #a8dfc0 !important; border-radius: 10px !important; color: #1a5c38 !important; }
.stWarning > div { background: #fff8e8 !important; border: 1px solid #f5d888 !important; border-radius: 10px !important; color: #664400 !important; }
.stError   > div { background: #fff0f0 !important; border: 1px solid #ffb0b0 !important; border-radius: 10px !important; color: #cc2222 !important; }

/* ── Sidebar brand ──────────────────────── */
.s-brand {
    padding: 22px 16px 18px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
.s-brand-name {
    font-family: 'DM Serif Display', serif !important;
    font-size: 18px !important; color: #ffffff !important;
}
.s-brand-dot { color: #4fc3f7; }
.s-brand-tag {
    font-size: 11px !important; color: #4a7a8a !important;
    margin-top: 5px !important; font-weight: 500 !important;
    display: flex !important; align-items: center !important; gap: 6px !important;
}
.s-nav-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #2d5a6a !important; padding: 6px 20px;
}
.s-footer {
    font-size: 11px; color: #2d5a6a !important;
    padding: 14px 16px; border-top: 1px solid rgba(255,255,255,0.06);
    line-height: 1.9; margin-top: auto;
}
.s-live-dot {
    display: inline-block; width: 7px; height: 7px;
    background: #22c55e; border-radius: 50%;
    animation: pulse 2s infinite; flex-shrink: 0;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
hr { border: none !important; border-top: 1.5px solid #eceef5 !important; margin: 28px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#5a6075", size=12),
    xaxis=dict(gridcolor="#eceef5", linecolor="#eceef5",
               tickcolor="#eceef5", tickfont=dict(color="#8b92a5", size=11)),
    yaxis=dict(gridcolor="#eceef5", linecolor="#eceef5",
               tickcolor="#eceef5", tickfont=dict(color="#8b92a5", size=11)),
    hoverlabel=dict(bgcolor="#1a1d2e", bordercolor="#1a1d2e",
                    font=dict(color="#ffffff", size=12, family="DM Sans")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#eceef5",
                font=dict(color="#5a6075", size=11)),
    margin=dict(l=0, r=0, t=10, b=0),
    # title_font=dict(color="#1a1d2e", size=14, family="DM Sans"),
)
COLORS = ["#ff4444", "#3b82f6", "#10b981", "#f59e0b",
          "#8b5cf6", "#ec4899", "#06b6d4"]


# ── Number formatter ─────────────────────────────────────────
def fmt(n):
    try:
        n = float(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(int(n))
    except: return str(n)

def fmt_df(n):
    try:
        n = float(n)
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(int(n))
    except: return str(n)


# ── Load and cache data ───────────────────────────────────────
@st.cache_data(ttl=600)
def load_latest(limit: int = 500) -> pd.DataFrame:
    """
    Load only the LATEST snapshot per country (250 videos max).
    Used for the Live Feed table display.
    Cached for 10 minutes.
    """
    df_raw = get_latest_trending(limit=limit)
    if df_raw.empty:
        return pd.DataFrame()
    df = clean_and_engineer(df_raw)
    df = apply_country_names(df)
    if "category_name" in df.columns:
        df = df[df["category_name"] != "Unknown"].reset_index(drop=True)
    return df


@st.cache_data(ttl=600)
def load_all_history(limit: int = 50000) -> pd.DataFrame:
    """
    Load ALL videos ever collected across every run.
    Used for Analysis, Country Compare, Historical, and ML training.
    This is the full dataset that grows every 8 hours.
    Cached for 10 minutes.
    """
    df_raw = get_all_collected_videos(limit=limit)
    if df_raw.empty:
        return pd.DataFrame()
    df = clean_and_engineer(df_raw)
    df = apply_country_names(df)
    if "category_name" in df.columns:
        df = df[df["category_name"] != "Unknown"].reset_index(drop=True)
    return df


# ── Train simple Logistic Regression on live data ─────────────
@st.cache_data(ttl=600)
def train_simple_model(_df):
    df = _df.copy()

    required = PREDICT_FEATURES + ["is_trending_high"]
    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        return None, None

    df_model = df[required].dropna()
    if len(df_model) < 30:
        return None, None

    X = df_model[PREDICT_FEATURES].values
    y = df_model["is_trending_high"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

def generate_insights(df):
    insights = []

    if "category_name" in df.columns and "hours_to_trend" in df.columns:
        cat_speed = df.groupby("category_name")["hours_to_trend"].median().sort_values()
        if len(cat_speed) >= 2:
            fastest = cat_speed.index[0]
            slowest = cat_speed.index[-1]
            ratio   = cat_speed.iloc[-1] / max(cat_speed.iloc[0], 1)
            insights.append({
                "emoji": "⚡",
                "headline": f"{fastest} videos go viral {ratio:.0f}x faster than {slowest}",
                "detail": f"{fastest} takes ~{cat_speed.iloc[0]:.0f}h vs {cat_speed.iloc[-1]:.0f}h for {slowest}.",
                "tag": "Speed Insight", "tag_class": "tag-green"
            })

    if "publish_hour" in df.columns:
        hr  = df.groupby("publish_hour")["view_count"].mean()
        bhr = hr.idxmax()
        ratio = hr.max() / max(hr.min(), 1)
        insights.append({
            "emoji": "🕐",
            "headline": f"Upload at {bhr}:00 UTC for {ratio:.1f}x more views",
            "detail": f"Videos at {bhr}:00 UTC average {fmt(hr.max())} views — the peak hour.",
            "tag": "Best Time", "tag_class": "tag-blue"
        })

    if "is_short" in df.columns:
        short_pct = df["is_short"].mean() * 100
        if short_pct > 10:
            insights.append({
                "emoji": "📱",
                "headline": f"{short_pct:.0f}% of trending videos are Shorts",
                "detail": "Short-form content gets a separate algorithm boost. Consider Shorts.",
                "tag": "Format Trend", "tag_class": "tag-purple"
            })

    if "country" in df.columns and "engagement_score" in df.columns:
        eng   = df.groupby("country")["engagement_score"].mean().sort_values(ascending=False)
        top_c = eng.index[0]
        insights.append({
            "emoji": "🌍",
            "headline": f"{top_c} has the highest audience engagement",
            "detail": f"Viewers in {top_c} engage {eng.iloc[0]*100:.1f}% — the most interactive market.",
            "tag": "Top Market", "tag_class": "tag-amber"
        })

    if "sentiment_label" in df.columns:
        sv = df.groupby("sentiment_label")["view_count"].mean()
        if "positive" in sv and "negative" in sv:
            ratio = sv["positive"] / max(sv["negative"], 1)
            if ratio > 1.1:
                insights.append({
                    "emoji": "😊",
                    "headline": f"Positive titles get {ratio:.1f}x more views",
                    "detail": "Uplifting titles consistently outperform negative framing in views.",
                    "tag": "Title Strategy", "tag_class": "tag-green"
                })

    if "category_name" in df.columns:
        top_cat = df.groupby("category_name")["view_count"].mean().idxmax()
        top_avg = df.groupby("category_name")["view_count"].mean().max()
        insights.append({
            "emoji": "🏆",
            "headline": f"{top_cat} earns the highest average views",
            "detail": f"A trending {top_cat} video averages {fmt(top_avg)} views right now.",
            "tag": "Top Category", "tag_class": "tag-red"
        })

    return insights[:6]


def generate_actions(df):
    actions = []
    if "publish_hour" in df.columns:
        bh = df.groupby("publish_hour")["view_count"].mean().idxmax()
        actions.append({"num": "1", "title": f"Upload between {bh}:00 – {(bh+2)%24}:00 UTC",
                         "desc": "This window produces the highest-viewed trending videos."})
    if "tag_count" in df.columns:
        avg_tags = int(df["tag_count"].median())
        actions.append({"num": "2", "title": f"Use at least {avg_tags} tags per video",
                         "desc": f"Trending videos use a median of {avg_tags} tags for better discoverability."})
    if "category_name" in df.columns:
        top3 = df.groupby("category_name")["view_count"].mean()\
                  .sort_values(ascending=False).head(3).index.tolist()
        actions.append({"num": "3", "title": f"Focus on {', '.join(top3[:2])} content",
                         "desc": "These categories dominate trending right now."})
    if "title_length" in df.columns:
        avg_len = int(df["title_length"].median())
        actions.append({"num": "4", "title": f"Keep titles around {avg_len} characters",
                         "desc": f"Median trending title is {avg_len} chars — short enough to read, long enough for context."})
    if "is_short" in df.columns and df["is_short"].mean() > 0.15:
        actions.append({"num": "5", "title": "Experiment with YouTube Shorts",
                         "desc": f"{df['is_short'].mean()*100:.0f}% of trending content is under 60s."})
    if "is_hd" in df.columns and df["is_hd"].mean() > 0.6:
        actions.append({"num": "6", "title": "Always upload in HD (1080p or higher)",
                         "desc": f"{df['is_hd'].mean()*100:.0f}% of trending videos are HD."})
    return actions[:6]


# ── Section header helper ─────────────────────────────────────
def sh(text):
    st.markdown(
        f'<div class="section-hdr">'
        f'<span class="section-hdr-text">{text}</span>'
        f'<div class="section-hdr-line"></div></div>',
        unsafe_allow_html=True
    )


# ── KPI cards ─────────────────────────────────────────────────
def kpi_cards(df: pd.DataFrame, show_total: bool = False):
    """
    Render the 4 KPI cards at the top of each page.

    show_total=True  → 'Videos Tracked' shows cumulative ALL-TIME count
    show_total=False → 'Videos Tracked' shows count from current df
    """
    if show_total:
        # Real cumulative count from the full database
        total_count = get_total_video_count()
        total = f"{total_count:,}"
        sub_label = f"Total videos collected across all runs"
    else:
        # Count of rows in the passed DataFrame
        total = f"{len(df):,}"
        sub_label = f"Trending videos across {df['country'].nunique()} countries"

    avg_views  = fmt(df["view_count"].mean())
    eng_rate   = f"{df['like_view_ratio'].mean()*100:.1f}%"
    viral_time = f"{df['hours_to_trend'].median():.0f}h"

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Videos Tracked</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-sub">{sub_label}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Average Views</div>
        <div class="kpi-value">{avg_views}</div>
        <div class="kpi-sub">Average views per trending video</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Engagement Rate</div>
        <div class="kpi-value">{eng_rate}</div>
        <div class="kpi-sub">Percentage of viewers who liked</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Time to Go Viral</div>
        <div class="kpi-value">{viral_time}</div>
        <div class="kpi-sub">Median hours from upload to trending</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

#  PAGE: LIVE FEED

def page_live(df_latest: pd.DataFrame):
    """
    Live Feed page.
    df_latest = only the most recent collection run (25 videos).
    The KPI 'Videos Tracked' shows the ALL-TIME cumulative total.
    """
    st.markdown(
        '<div class="page-hero">'
        '<p class="page-title">Live Trending Feed</p>'
        '<p class="page-sub">Real-time YouTube trending videos — '
        'auto-refreshed every 24 hours across 5 countries</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # show_total=True → reads cumulative count from database
    kpi_cards(df_latest, show_total=True)

    sh("Filter & Explore")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        country = st.selectbox(
            "Country",
            ["All"] + sorted(df_latest["country"].unique().tolist())
        )
    with c2:
        valid_cats = sorted([
            c for c in df_latest["category_name"].dropna().unique().tolist()
            if c != "Unknown"
        ])
        category = st.selectbox("Category", ["All"] + valid_cats)
    with c3:
        top_n = st.selectbox("Top Results", [5, 10, 25, 50, 100], index=1)

    # Deduplicate: show each video once with latest stats
    df_dedup = (
        df_latest
        .sort_values("fetched_at" if "fetched_at" in df_latest.columns else "run_at",
                     ascending=False)
        .drop_duplicates(subset=["video_id", "country"], keep="first")
        .reset_index(drop=True)
    )

    filtered = df_dedup.copy()
    if country  != "All": filtered = filtered[filtered["country"]       == country]
    if category != "All": filtered = filtered[filtered["category_name"] == category]

    display = filtered[[
        "title", "channel_title", "country", "view_count",
        "like_count", "comment_count", "category_name",
        "views_per_hour",
    ]].copy()
    display = display.sort_values("view_count", ascending=False).head(top_n).reset_index(drop=True)
    display["view_count"]     = display["view_count"].apply(fmt_df)
    display["like_count"]     = display["like_count"].apply(fmt_df)
    display["comment_count"]  = display["comment_count"].apply(fmt_df)
    display["views_per_hour"] = display["views_per_hour"].apply(fmt_df)
    display.columns = [
        "Video Title", "Channel", "Country", "Views",
        "Likes", "Comments", "Category", "Views/Hour"
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    total_in_db = get_total_video_count()
    st.info(
        f"📊 **{total_in_db:,} videos** collected in total across all runs. "
        f"The table above shows the **latest snapshot** ({len(df_dedup)} unique videos). "
        f"'Videos Tracked' in the KPI shows the real cumulative total."
    )
#  PAGE: ANALYSIS

def page_analysis(df):
    st.markdown(
        '<div class="page-hero">'
        '<p class="page-title">Trend Analysis</p>'
        '<p class="page-sub">Understand what is driving YouTube trends right now</p>'
        '</div>',
        unsafe_allow_html=True
    )
    kpi_cards(df)

    sh("Key Insights — What The Data Is Telling You")
    insights = generate_insights(df)
    if insights:
        for row_start in range(0, len(insights), 3):
            row  = insights[row_start:row_start+3]
            html = "".join([f"""
            <div class="insight-card">
              <span class="insight-emoji">{i['emoji']}</span>
              <div class="insight-headline">{i['headline']}</div>
              <div class="insight-detail">{i['detail']}</div>
              <span class="insight-tag {i['tag_class']}">{i['tag']}</span>
            </div>""" for i in row])
            st.markdown(f'<div class="insights-grid">{html}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    sh("What Should You Do? — Data-Backed Recommendations")
    st.markdown(
        "<p style='font-size:13px;color:#8b92a5;margin:-10px 0 20px'>"
        "Actions derived directly from live trending data.</p>",
        unsafe_allow_html=True
    )
    actions = generate_actions(df)
    for i in range(0, len(actions), 3):
        row  = actions[i:i+3]
        cols = st.columns(len(row))
        for col, act in zip(cols, row):
            with col:
                st.markdown(f"""<div class="action-card">
                  <div class="action-num">{act['num']}</div>
                  <div class="action-title">{act['title']}</div>
                  <div class="action-desc">{act['desc']}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

#  PAGE: COUNTRY COMPARE

def page_country(df):
    st.markdown(
        '<div class="page-hero">'
        '<p class="page-title">Country Comparison</p>'
        '<p class="page-sub">See how trending patterns differ across global markets — simple bar charts, easy to read</p>'
        '</div>',
        unsafe_allow_html=True
    )

    countries = st.multiselect(
        "Select countries to compare",
        options=sorted(df["country"].unique().tolist()),
        default=sorted(df["country"].unique().tolist())
    )
    if not countries:
        st.warning("Please select at least one country.")
        return

    dfc = df[df["country"].isin(countries)]

    sh("Country Highlights at a Glance")

    top_cat_per = (
        dfc.groupby(["country", "category_name"]).size()
           .reset_index(name="count")
           .sort_values("count", ascending=False)
           .groupby("country").first().reset_index()
    )
    top_eng   = dfc.groupby("country")["like_view_ratio"].mean().sort_values(ascending=False)
    top_viral = dfc.groupby("country")["hours_to_trend"].median().sort_values()

    cat_rows   = "".join([
        f'<div class="cs-row"><span class="cs-country">{r["country"]}</span>'
        f'<span class="cs-value">{r["category_name"]}</span></div>'
        for _, r in top_cat_per.iterrows()
    ])
    eng_rows   = "".join([
        f'<div class="cs-row"><span class="cs-country">{c}</span>'
        f'<span class="cs-value">{v*100:.1f}%</span></div>'
        for c, v in top_eng.items()
    ])
    viral_rows = "".join([
        f'<div class="cs-row"><span class="cs-country">{c}</span>'
        f'<span class="cs-value">{v:.0f}h</span></div>'
        for c, v in top_viral.items()
    ])

    st.markdown(f"""
    <div class="country-stat-row">
      <div class="country-stat-card">
        <div class="cs-label">Top Trending Category</div>{cat_rows}
      </div>
      <div class="country-stat-card">
        <div class="cs-label">Engagement Rate</div>{eng_rows}
      </div>
      <div class="country-stat-card">
        <div class="cs-label">Hours to Trend (Median)</div>{viral_rows}
      </div>
    </div>
    """, unsafe_allow_html=True)

    sh("Which Country Gets The Most Views?")

    st.markdown(
        '<div class="chart-question">Average views per trending video by country</div>'
        '<div class="chart-context">'
        'Taller bar = more views on average. Simple and direct.'
        '</div>',
        unsafe_allow_html=True
    )
    avg_views = (
        dfc.groupby("country")["view_count"]
           .mean().reset_index()
           .sort_values("view_count", ascending=False)
    )
    fig1 = go.Figure(go.Bar(
        x=avg_views["country"],
        y=avg_views["view_count"],
        marker=dict(color=COLORS[:len(avg_views)]),
        text=[fmt(v) for v in avg_views["view_count"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Views: %{text}<extra></extra>"
    ))
    fig1.update_layout(height=320, xaxis_title="Country",
                       yaxis_title="Average Views", **THEME)
    st.plotly_chart(fig1, use_container_width=True)
    top_v = avg_views.iloc[0]["country"]
  
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Where Do Viewers Engage The Most?")

    st.markdown(
        '<div class="chart-question">Engagement rate by country (likes ÷ views)</div>'
        '<div class="chart-context">'
        'Higher % = more viewers liked the video. Shows audience quality, not just quantity.'
        '</div>',
        unsafe_allow_html=True
    )
    eng_data = (
        dfc.groupby("country")["like_view_ratio"]
           .mean().reset_index()
           .sort_values("like_view_ratio", ascending=False)
    )
    eng_data["pct"] = (eng_data["like_view_ratio"] * 100).round(2)
    fig2 = go.Figure(go.Bar(
        x=eng_data["country"],
        y=eng_data["pct"],
        marker=dict(color=COLORS[:len(eng_data)]),
        text=[f"{v:.1f}%" for v in eng_data["pct"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Engagement: %{text}<extra></extra>"
    ))
    fig2.update_layout(height=320, xaxis_title="Country",
                       yaxis_title="Engagement Rate (%)", **THEME)
    st.plotly_chart(fig2, use_container_width=True)
    top_e = eng_data.iloc[0]["country"]
    
    st.markdown('</div>', unsafe_allow_html=True)

    sh("What Kind of Videos Trend in Each Country?")

    st.markdown(
        '<div class="chart-question">Number of trending videos per category per country</div>'
        '<div class="chart-context">'
        'Each group of bars is one country. The tallest bar in each group is that '
        "country's most popular category. Compare countries side by side."
        '</div>',
        unsafe_allow_html=True
    )
    cc = (
        dfc.groupby(["country", "category_name"]).size()
           .reset_index(name="count")
    )
    fig3 = px.bar(
        cc, x="category_name", y="count",
        color="country", barmode="group",
        color_discrete_sequence=COLORS,
        labels={"category_name": "Category", "count": "Videos", "country": "Country"}
    )
    fig3.update_layout(xaxis_tickangle=-35, height=380, **THEME)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

#  PAGE: HISTORICAL

def page_historical(df):
    st.markdown(
        '<div class="page-hero">'
        '<p class="page-title">Historical Trends</p>'
        '<p class="page-sub">See what categories and countries have dominated trending over time</p>'
        '</div>',
        unsafe_allow_html=True
    )

    df = df.copy()
    df["run_at"] = pd.to_datetime(df["run_at"], errors="coerce")
    df = df.dropna(subset=["run_at"])
    df["date"] = df["run_at"].dt.strftime("%Y-%m-%d")

    sh("Most Trending Categories — All Time")
    cat_summary = (
        df.groupby("category_name")
          .agg(total_videos   =("video_id",      "count"),
               avg_views      =("view_count",    "mean"),
               avg_likes      =("like_count",    "mean"),
               avg_hrs_viral  =("hours_to_trend","median"))
          .sort_values("total_videos", ascending=False)
          .reset_index()
    )

    st.markdown(
        '<div class="chart-question">Which categories have appeared on trending the most?</div>'
        '<div class="chart-context">Longer bar = appeared on trending more often.</div>',
        unsafe_allow_html=True
    )
    fig1 = go.Figure(go.Bar(
        x=cat_summary["total_videos"],
        y=cat_summary["category_name"],
        orientation="h",
        marker=dict(
            color=cat_summary["total_videos"],
            colorscale=[[0,"#ffe5e5"],[0.5,"#ff8888"],[1,"#ff4444"]],
            showscale=False
        ),
        hovertemplate="<b>%{y}</b><br>%{x} appearances<extra></extra>"
    ))
    fig1.update_layout(
        height=380,
        yaxis=dict(categoryorder="total ascending",
                   gridcolor="#eceef5", tickfont=dict(size=11, color="#8b92a5")),
        xaxis=dict(title="Total trending appearances",
                   gridcolor="#eceef5", tickfont=dict(size=11, color="#8b92a5")),
        **{k: v for k, v in THEME.items() if k not in ["xaxis","yaxis"]}
    )
    st.plotly_chart(fig1, use_container_width=True)
    top_overall = cat_summary.iloc[0]["category_name"]
    top_count   = int(cat_summary.iloc[0]["total_videos"])
  
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Category Performance Summary")

    st.markdown(
        '<div class="chart-question">How do categories compare on views, likes, and viral speed?</div>',
        unsafe_allow_html=True
    )
    display_cat = cat_summary.copy()
    display_cat["avg_views"] = display_cat["avg_views"].apply(fmt_df)
    display_cat["avg_likes"] = display_cat["avg_likes"].apply(fmt_df)
    display_cat["avg_hrs_viral"] = (
        display_cat["avg_hrs_viral"].round(0).astype(int).astype(str) + "h"
    )
    display_cat.columns = ["Category","Times Trending","Avg Views","Avg Likes","Avg Hours to Viral"]
    st.dataframe(display_cat, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Country Performance — Total Videos Collected")

    st.markdown(
        '<div class="chart-question">How many trending videos collected per country?</div>'
        '<div class="chart-context">Equal bars = balanced data quality across markets.</div>',
        unsafe_allow_html=True
    )
    country_summary = (
        df.groupby("country")
          .agg(total    =("video_id","count"),
               avg_views=("view_count","mean"))
          .reset_index()
          .sort_values("total", ascending=False)
    )
    fig3 = go.Figure(go.Bar(
        x=country_summary["country"],
        y=country_summary["total"],
        marker=dict(color=COLORS[:len(country_summary)]),
        text=country_summary["total"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} videos<extra></extra>"
    ))
    fig3.update_layout(height=280, xaxis_title="Country",
                       yaxis_title="Videos collected", **THEME)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


#  PAGE: PREDICT

def page_predict(df):
    st.markdown(
        '<div class="page-hero">'
        '<p class="page-title">Trending Potential Predictor</p>'
        '<p class="page-sub">Enter your video details — our ML model will tell you the trending probability</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Train model on live data ────────────────────────────────
    model, scaler = train_simple_model(df)

    if model is None:
        st.warning(
            "Not enough data yet to train the ML model. "
            "Please wait for more collection runs (need at least 30 videos). "
            "Check back after the next GitHub Actions run."
        )
        return

    n_training = df.dropna(subset=PREDICT_FEATURES + ["is_trending_high"]).shape[0]
    st.success(f"Model trained on {n_training} trending videos from the live database.")

    # ── User input form ─────────────────────────────────────────
    sh("Enter Your Video Details")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        title         = st.text_input(
            "Video title",
            placeholder="e.g. 10 Things You Never Knew About Space!"
        )
        view_count    = st.number_input("Current views",    min_value=0, value=10000, step=1000)
        like_count    = st.number_input("Current likes",    min_value=0, value=500,   step=100)
        comment_count = st.number_input("Current comments", min_value=0, value=100,   step=10)
        tag_count     = st.slider("Number of tags used", 0, 30, 12)

    with c2:
        duration_min = st.slider("Video length (minutes)", 0, 60, 10)
        pub_hour     = st.slider("Upload hour (UTC 0–23)", 0, 23, 15)
        quality      = st.selectbox("Video quality", ["HD (1080p+)", "SD (720p or below)"])
        is_hd        = 1 if quality == "HD (1080p+)" else 0

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    go_btn = st.button("Predict Trending Potential", type="primary")

    if go_btn:
        if not title.strip():
            st.warning("Please enter a video title first.")
            return

        # ── Build exactly the same 11 features used in training ──
        view_safe = max(view_count, 1)
        try:
            sentiment = TextBlob(title).sentiment.polarity
        except:
            sentiment = 0.0

        user_vector = {
            "like_view_ratio":    like_count    / view_safe,
            "comment_view_ratio": comment_count / view_safe,
            "engagement_score":   (like_count + comment_count) / view_safe,
            "title_length":       len(title),
            "title_word_count":   len(title.split()),
            "title_has_number":   int(any(c.isdigit() for c in title)),
            "title_exclamation":  int("!" in title),
            "is_hd":              is_hd,
            "tag_count":          tag_count,
            "publish_hour":       pub_hour,
            "title_sentiment":    sentiment,
        }

        # Build feature array in the EXACT same order as PREDICT_FEATURES
        X_pred = [[user_vector[f] for f in PREDICT_FEATURES]]

        # Scale and predict
        X_scaled = scaler.transform(X_pred)
        prob     = model.predict_proba(X_scaled)[0][1] * 100
        verdict  = "Will Trend" if prob >= 50 else "Won't Trend"

        prob_color = (
            "#10b981" if prob >= 65
            else "#f59e0b" if prob >= 40
            else "#ff4444"
        )

        # ── Results ────────────────────────────────────────────
        sh("Your Prediction Results")
        st.markdown(f"""
        <div class="result-grid">
          <div class="result-card">
            <div class="result-label">Verdict</div>
            <div class="result-value" style="font-size:15px">
              {"✅" if prob>=50 else "❌"}&nbsp;{verdict}
            </div>
          </div>
          <div class="result-card">
            <div class="result-label">Trending Probability</div>
            <div class="result-value" style="color:{prob_color}">{prob:.1f}%</div>
          </div>
          <div class="result-card">
            <div class="result-label">Title Sentiment</div>
            <div class="result-value">
              {"Positive 😊" if sentiment > 0.1 else "Neutral 😐" if sentiment > -0.1 else "Negative 😟"}
            </div>
          </div>
          <div class="result-card">
            <div class="result-label">Title Length</div>
            <div class="result-value">
              {"Good ✓" if 30<=len(title)<=70 else "Adjust ⚠️"} ({len(title)}c)
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob, 1),
            domain={"x": [0,1], "y": [0,1]},
            title={"text": "Trending Probability",
                   "font": {"color": "#8b92a5", "size": 13, "family": "DM Sans"}},
            number={"suffix": "%",
                    "font": {"color": "#1a1d2e", "size": 44, "family": "DM Sans"}},
            gauge={
                "axis":  {"range": [0,100], "tickfont": {"color":"#8b92a5","size":11}},
                "bar":   {"color": prob_color},
                "bgcolor": "#f7f8fc", "bordercolor": "#eceef5",
                "steps": [
                    {"range":[0,40],  "color":"#fff0f0"},
                    {"range":[40,65], "color":"#fff8e8"},
                    {"range":[65,100],"color":"#e8f8ef"},
                ],
                "threshold": {
                    "line": {"color":"#1a1d2e","width":2},
                    "thickness": 0.8, "value": 65
                }
            }
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
            height=260,
            margin=dict(l=30,r=30,t=20,b=0)
        )
        st.plotly_chart(gauge, use_container_width=True)

        tips = []
        if sentiment < 0:
            tips.append("Your title has a negative tone — try reframing it positively to boost clicks")
        if tag_count < 10:
            tips.append(f"Only {tag_count} tags — trending videos use 12–20 tags for better reach")
        if pub_hour not in range(13, 21):
            tips.append(f"Uploading at {pub_hour}:00 UTC — peak window is 13:00–20:00 UTC")
        if len(title) < 30:
            tips.append("Title is too short — aim for 40–60 characters for better CTR")
        if len(title) > 80:
            tips.append("Title is too long — YouTube truncates after ~70 characters")
        if not any(c.isdigit() for c in title):
            tips.append("Add a number (e.g. '5 Reasons...') — numbered titles get higher CTR")
        if not is_hd:
            tips.append("Upload in HD (1080p+) — over 90% of trending videos are HD")
        if like_count / view_safe < 0.02:
            tips.append("Low like ratio — add a strong call-to-action asking viewers to like")

        if tips:
            sh("How To Improve Your Chances")
            for tip in tips:
                st.markdown(f'<div class="tip-item">→ &nbsp;{tip}</div>', unsafe_allow_html=True)
        else:
            st.success("Your video looks well-optimised! High chance of trending.")


#  MAIN

def main():
    with st.sidebar:
        st.markdown("""
        <div class="s-brand">
          <div class="s-brand-name">YT Trending<span class="s-brand-dot">.</span></div>
          <div class="s-brand-tag">
            <span class="s-live-dot"></span>Intelligence Dashboard
          </div>
        </div>
        <div class="s-nav-label">Navigation</div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=["Live Feed", "Analysis", "Country Compare", "Historical", "Predict"],
            label_visibility="collapsed"
        )

       

        st.markdown("""
        <div class="s-footer">
          <div>Data refreshes every 24 hours</div>
          <div>YouTube Data API v3</div>
          <div>US · IN · GB · CA · AU</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Load data based on which page is selected ──────────────────
    # Live Feed  → latest snapshot only (fast, 250 videos)
    # All others → full history (richer data for analysis and ML)

    if page == "Live Feed":
        df_latest = load_latest()
        if df_latest.empty:
            st.error(
                "No data found yet. Go to GitHub → Actions → "
                "YouTube Trending Data Collector → Run workflow to collect data now."
            )
            return
        page_live(df_latest)

    elif page == "Predict":
        # Predict needs full history for ML training
        df_all = load_all_history()
        if df_all.empty:
            st.error("No data found. Run a collection first.")
            return
        page_predict(df_all)

    else:
        # Analysis, Country Compare, Historical — all use full history
        df_all = load_all_history()
        if df_all.empty:
            st.error("No data found. Run a collection first.")
            return

        if   page == "Analysis":        page_analysis(df_all)
        elif page == "Country Compare": page_country(df_all)
        elif page == "Historical":      page_historical(df_all)


if __name__ == "__main__":
    main()
