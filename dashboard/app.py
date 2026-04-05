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

from src.database import get_latest_trending
from src.data_cleaner import clean_and_engineer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from config import MODELS_PATH
import mysql.connector

# ── Country code → full name mapping (applied everywhere) ──
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f7f8fc !important;
    color: #1a1d2e !important;
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
.block-container { padding: 2.5rem 3rem !important; max-width: 100% !important; }

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

section[data-testid="stSidebar"] * {
    color: #8fb8cc !important;
}

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
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: flex !important;
    align-items: center !important;
    flex-shrink: 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] .stRadio > div > label [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8fb8cc !important;
    line-height: 1 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
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
    flex-shrink: 0 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    background: #0d2233 !important;
    border: none !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapsedControl"] svg { fill: #4fc3f7 !important; }
[data-testid="collapsedControl"] { visibility: visible !important; background: #0d2233 !important; }
[data-testid="collapsedControl"] svg { fill: #4fc3f7 !important; }

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
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b92a5;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #1a1d2e;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 8px;
}
.kpi-sub {
    font-size: 12px;
    color: #8b92a5;
    line-height: 1.5;
}

.page-hero { margin: 0 0 32px 0; }
.page-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 30px;
    font-weight: 400;
    color: #1a1d2e;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 8px 0;
}
.page-sub {
    font-size: 14px;
    color: #8b92a5;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

.section-hdr {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 18px 0;
}
.section-hdr-line { flex: 1; height: 1px; background: #eceef5; }
.section-hdr-text {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #ff4444;
    white-space: nowrap;
}

.chart-card {
    background: #ffffff;
    border: 1px solid #eceef5;
    border-radius: 14px;
    padding: 26px 26px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 22px;
}
.chart-question {
    font-size: 15px;
    font-weight: 600;
    color: #1a1d2e;
    margin-bottom: 6px;
    line-height: 1.4;
}
.chart-context {
    font-size: 12px;
    color: #8b92a5;
    margin-bottom: 18px;
    line-height: 1.6;
}
.chart-insight-bar {
    background: #fff8e8;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 11px 16px;
    margin-top: 14px;
    font-size: 12px;
    color: #664400;
    line-height: 1.6;
}

.insights-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 12px;
}
.insight-card {
    background: #ffffff;
    border: 1px solid #eceef5;
    border-radius: 14px;
    padding: 20px 22px;
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

.country-stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.country-stat-card {
    background: #ffffff;
    border: 1px solid #eceef5;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.cs-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b92a5;
    margin-bottom: 12px;
}
.cs-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #f3f4f8;
    font-size: 13px;
}
.cs-row:last-child { border-bottom: none; }
.cs-country { color: #1a1d2e; font-weight: 500; }
.cs-value   { color: #ff4444; font-weight: 700; font-size: 13px; }

.action-card { background: #ffffff; border: 1px solid #eceef5; border-radius: 14px; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.action-num  { width: 28px; height: 28px; background: #ff4444; color: white; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; margin-bottom: 12px; }
.action-title { font-size: 14px; font-weight: 600; color: #1a1d2e; margin-bottom: 8px; line-height: 1.4; }
.action-desc  { font-size: 12px; color: #8b92a5; line-height: 1.7; }

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #eceef5 !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b92a5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 9px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] { background: #1a1d2e !important; color: #ffffff !important; }

.stSelectbox label, .stMultiSelect label,
.stTextInput label, .stNumberInput label, .stSlider label {
    color: #5a6075 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 4px !important;
}

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid #dde1ee !important;
    border-radius: 10px !important;
    color: #1a1d2e !important;
    font-size: 13px !important;
    padding: 2px 4px !important;
}
[data-baseweb="popover"] { background: #ffffff !important; border: 1px solid #eceef5 !important; border-radius: 12px !important; box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important; }
[data-baseweb="menu"]    { background: #ffffff !important; }
[data-baseweb="option"]  { background: #ffffff !important; color: #1a1d2e !important; font-size: 13px !important; padding: 10px 16px !important; }
[data-baseweb="option"]:hover { background: #f7f8fc !important; }
[data-baseweb="tag"]     { background: #e8f4fd !important; color: #1b3a4b !important; border-radius: 6px !important; }

.stTextInput input, .stNumberInput input {
    background: #ffffff !important;
    border: 1.5px solid #dde1ee !important;
    border-radius: 10px !important;
    color: #1a1d2e !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}
.stTextInput input::placeholder { color: #c0c5d5 !important; }
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #ff4444 !important;
    box-shadow: 0 0 0 3px rgba(255,68,68,0.1) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff4444, #cc2222) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-size: 14px !important;
    font-weight: 600 !important; padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(255,68,68,0.3) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #ffffff !important; color: #1a1d2e !important;
    border: 1.5px solid #dde1ee !important; border-radius: 10px !important;
    font-size: 13px !important; font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    border: 1.5px solid #eceef5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    margin-top: 4px !important;
}

.stSuccess > div { background: #e8f8ef !important; border: 1px solid #a8dfc0 !important; border-radius: 10px !important; color: #1a5c38 !important; padding: 14px 18px !important; }
.stWarning > div { background: #fff8e8 !important; border: 1px solid #f5d888 !important; border-radius: 10px !important; color: #664400 !important; padding: 14px 18px !important; }
.stError   > div { background: #fff0f0 !important; border: 1px solid #ffb0b0 !important; border-radius: 10px !important; color: #cc2222 !important; padding: 14px 18px !important; }
.stInfo    > div { background: #eef4ff !important; border: 1px solid #b0c8ff !important; border-radius: 10px !important; color: #2244aa !important; padding: 14px 18px !important; }

.s-brand {
    padding: 22px 16px 18px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
.s-brand-name {
    font-family: 'DM Serif Display', serif !important;
    font-size: 18px !important;
    color: #ffffff !important;
    letter-spacing: -0.01em !important;
}
.s-brand-dot { color: #4fc3f7; }
.s-brand-tag {
    font-size: 11px !important;
    color: #4a7a8a !important;
    margin-top: 5px !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.s-nav-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2d5a6a !important;
    padding: 6px 20px 6px 20px;
}
.s-footer {
    font-size: 11px;
    color: #2d5a6a !important;
    padding: 14px 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    line-height: 1.9;
    margin-top: auto;
}
.s-live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.result-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin: 18px 0 24px; }
.result-card { background: #ffffff; border: 1.5px solid #eceef5; border-radius: 14px; padding: 20px 22px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.result-label { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #8b92a5; margin-bottom: 9px; }
.result-value { font-size: 22px; font-weight: 700; color: #1a1d2e; line-height: 1.2; }
.tip-item { background: #fff8e8; border-left: 3px solid #f59e0b; border-radius: 0 10px 10px 0; padding: 13px 18px; margin: 9px 0; font-size: 13px; color: #664400; line-height: 1.7; }
hr { border: none !important; border-top: 1.5px solid #eceef5 !important; margin: 28px 0 !important; }
</style>
""", unsafe_allow_html=True)

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
    title_font=dict(color="#1a1d2e", size=14, family="DM Sans"),
)
COLORS = ["#ff4444","#3b82f6","#10b981","#f59e0b","#8b5cf6","#ec4899","#06b6d4"]


# ── FIX 1: fmt and fmt_df now safely handle NaN / None ──
def fmt(n):
    try:
        n = float(n)
        if np.isnan(n): return "N/A"
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(int(n))
    except:
        return "N/A"


def fmt_df(n):
    try:
        n = float(n)
        if np.isnan(n): return "N/A"
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(int(n))
    except:
        return "N/A"


# ── FIX 2: Empty DataFrame now includes ALL columns used across every page ──
def empty_dataframe():
    return pd.DataFrame({
        "video_id":        pd.Series([], dtype="str"),
        "title":           pd.Series([], dtype="str"),
        "channel_title":   pd.Series([], dtype="str"),
        "country":         pd.Series([], dtype="str"),
        "view_count":      pd.Series([], dtype="float"),
        "like_count":      pd.Series([], dtype="float"),
        "comment_count":   pd.Series([], dtype="float"),
        "category_name":   pd.Series([], dtype="str"),
        "views_per_hour":  pd.Series([], dtype="float"),
        "fetched_at":      pd.Series([], dtype="object"),
        "like_view_ratio": pd.Series([], dtype="float"),
        "hours_to_trend":  pd.Series([], dtype="float"),
        "engagement_score":pd.Series([], dtype="float"),
        "title_sentiment": pd.Series([], dtype="float"),
        "sentiment_label": pd.Series([], dtype="str"),
        "is_short":        pd.Series([], dtype="float"),
        "is_hd":           pd.Series([], dtype="float"),
        "tag_count":       pd.Series([], dtype="float"),
        "title_length":    pd.Series([], dtype="float"),
        "publish_hour":    pd.Series([], dtype="float"),
    })


@st.cache_data(ttl=600)
def load_data(limit=10000):
    df_raw = get_latest_trending(limit=limit)
    if df_raw is None or df_raw.empty:
        return empty_dataframe()
    df = clean_and_engineer(df_raw)
    df = apply_country_names(df)
    return df


def generate_insights(df):
    insights = []
    if df.empty:
        return insights
    if "category_name" in df.columns and "hours_to_trend" in df.columns:
        cat_speed = df.groupby("category_name")["hours_to_trend"].median().sort_values().dropna()
        if len(cat_speed) >= 2:
            fastest = cat_speed.index[0]
            slowest = cat_speed.index[-1]
            ratio   = cat_speed.iloc[-1] / max(cat_speed.iloc[0], 1)
            insights.append({"emoji":"⚡","headline":f"{fastest} videos go viral {ratio:.0f}x faster than {slowest}","detail":f"{fastest} takes ~{cat_speed.iloc[0]:.0f}h to trend vs {cat_speed.iloc[-1]:.0f}h for {slowest}.","tag":"Speed Insight","tag_class":"tag-green"})
    if "publish_hour" in df.columns and not df["publish_hour"].dropna().empty:
        hr    = df.groupby("publish_hour")["view_count"].mean().dropna()
        if not hr.empty:
            bhr   = hr.idxmax()
            ratio = hr.max() / max(hr.min(), 1)
            insights.append({"emoji":"🕐","headline":f"Upload at {int(bhr)}:00 UTC for {ratio:.1f}x more views","detail":f"Videos at {int(bhr)}:00 UTC average {fmt(hr.max())} views — the highest performing hour.","tag":"Best Time","tag_class":"tag-blue"})
    if "is_short" in df.columns:
        short_pct = df["is_short"].mean() * 100
        if short_pct > 10:
            insights.append({"emoji":"📱","headline":f"{short_pct:.0f}% of trending videos are Shorts","detail":"Short-form content is heavily favoured by the YouTube algorithm. Consider Shorts for faster exposure.","tag":"Format Trend","tag_class":"tag-purple"})
    if "country" in df.columns and "engagement_score" in df.columns:
        eng = df.groupby("country")["engagement_score"].mean().dropna().sort_values(ascending=False)
        if not eng.empty:
            top_c = eng.index[0]
            insights.append({"emoji":"🌍","headline":f"{top_c} has the highest audience engagement","detail":f"Viewers in {top_c} engage {eng.iloc[0]*100:.1f}% on trending videos — the most interactive market.","tag":"Top Market","tag_class":"tag-amber"})
    if "sentiment_label" in df.columns:
        sv = df.groupby("sentiment_label")["view_count"].mean().dropna()
        if "positive" in sv and "negative" in sv:
            ratio = sv["positive"] / max(sv["negative"], 1)
            if ratio > 1.1:
                insights.append({"emoji":"😊","headline":f"Positive titles get {ratio:.1f}x more views","detail":"Uplifting and exciting titles consistently outperform negative framing in views and CTR.","tag":"Title Strategy","tag_class":"tag-green"})
    if "category_name" in df.columns and not df["category_name"].dropna().empty:
        avg_by_cat = df.groupby("category_name")["view_count"].mean().dropna()
        if not avg_by_cat.empty:
            top_cat = avg_by_cat.idxmax()
            top_avg = avg_by_cat.max()
            insights.append({"emoji":"🏆","headline":f"{top_cat} earns the highest average views","detail":f"A trending {top_cat} video currently averages {fmt(top_avg)} views — the top performing category.","tag":"Top Category","tag_class":"tag-red"})
    return insights[:6]


def generate_actions(df):
    actions = []
    if df.empty:
        return actions
    if "publish_hour" in df.columns and not df["publish_hour"].dropna().empty:
        hr_avg = df.groupby("publish_hour")["view_count"].mean().dropna()
        if not hr_avg.empty:
            bh = int(hr_avg.idxmax())
            actions.append({"num":"1","title":f"Upload between {bh}:00 – {(bh+2)%24}:00 UTC","desc":"This window produces the highest-viewed trending videos. Set a recurring upload schedule here."})
    if "tag_count" in df.columns and not df["tag_count"].dropna().empty:
        avg_tags = int(df["tag_count"].median())
        actions.append({"num":"2","title":f"Use at least {avg_tags} tags per video","desc":f"Trending videos use a median of {avg_tags} tags. More relevant tags = better algorithm discoverability."})
    if "category_name" in df.columns and not df["category_name"].dropna().empty:
        top3 = df.groupby("category_name")["view_count"].mean().dropna().sort_values(ascending=False).head(3).index.tolist()
        if top3:
            actions.append({"num":"3","title":f"Focus on {', '.join(top3[:2])} content","desc":"These categories dominate trending right now. Aligning your content here gives you the best shot."})
    if "title_length" in df.columns and not df["title_length"].dropna().empty:
        avg_len = int(df["title_length"].median())
        actions.append({"num":"4","title":f"Keep titles around {avg_len} characters","desc":f"Median trending title is {avg_len} characters. Too short = lacks context; too long = truncated."})
    if "is_short" in df.columns and df["is_short"].mean() > 0.15:
        actions.append({"num":"5","title":"Experiment with YouTube Shorts","desc":f"{df['is_short'].mean()*100:.0f}% of trending content is under 60s. Shorts get a separate algorithm boost."})
    if "is_hd" in df.columns and df["is_hd"].mean() > 0.6:
        actions.append({"num":"6","title":"Always upload in HD (1080p or higher)","desc":f"{df['is_hd'].mean()*100:.0f}% of trending videos are HD. Low-resolution content rarely trends."})
    return actions[:6]


def sh(text):
    st.markdown(f'<div class="section-hdr"><span class="section-hdr-text">{text}</span><div class="section-hdr-line"></div></div>', unsafe_allow_html=True)


# ── FIX 3: kpi_cards guards against empty df ──
def kpi_cards(df):
    if df.empty:
        st.info("No data available yet. Run the scheduler to start collecting trending videos.")
        return
    avg_views  = fmt(df["view_count"].mean())
    eng_rate   = f"{df['like_view_ratio'].mean()*100:.1f}%" if "like_view_ratio" in df.columns else "N/A"
    viral_time = f"{df['hours_to_trend'].median():.0f}h"   if "hours_to_trend"  in df.columns else "N/A"
    total      = f"{len(df):,}"
    n_countries = df["country"].nunique() if "country" in df.columns else 0
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Videos Tracked</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-sub">Trending videos across {n_countries} countries</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Average Views</div>
        <div class="kpi-value">{avg_views}</div>
        <div class="kpi-sub">Average views per trending video in this batch</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Engagement Rate</div>
        <div class="kpi-value">{eng_rate}</div>
        <div class="kpi-sub">Percentage of viewers who liked — shows audience quality</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Time to Go Viral</div>
        <div class="kpi-value">{viral_time}</div>
        <div class="kpi-sub">Median hours from upload to appearing on trending</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════
# PAGE: LIVE FEED
# ═══════════════════════════════════════
def page_live(df):
    st.markdown('<div class="page-hero"><p class="page-title">Live Trending Feed</p><p class="page-sub">Real-time YouTube trending videos — auto-refreshed every 3 hours across 5 countries</p></div>', unsafe_allow_html=True)
    kpi_cards(df)

    if df.empty:
        return

    sh("Filter & Explore")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        country = st.selectbox("Country", ["All"] + sorted(df["country"].unique().tolist()))
    with c2:
        cats     = ["All"] + sorted(df["category_name"].dropna().unique().tolist())
        category = st.selectbox("Category", cats)
    with c3:
        top_n = st.selectbox("Top Results", [5, 10, 25, 50, 100], index=1)

    filtered = df.copy()
    if country  != "All": filtered = filtered[filtered["country"]       == country]
    if category != "All": filtered = filtered[filtered["category_name"] == category]

    # Only show columns that exist in the dataframe
    desired_cols = ["title","channel_title","country","view_count",
                    "like_count","comment_count","category_name",
                    "views_per_hour","fetched_at"]
    existing_cols = [c for c in desired_cols if c in filtered.columns]

    display = filtered[existing_cols].copy()
    display = display.sort_values("view_count", ascending=False).head(top_n).reset_index(drop=True)

    for col in ["view_count","like_count","comment_count","views_per_hour"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_df)

    rename_map = {
        "title":         "Video Title",
        "channel_title": "Channel",
        "country":       "Country",
        "view_count":    "Views",
        "like_count":    "Likes",
        "comment_count": "Comments",
        "category_name": "Category",
        "views_per_hour":"Views/Hour",
        "fetched_at":    "Fetched At",
    }
    display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns}, inplace=True)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════
# PAGE: ANALYSIS
# ═══════════════════════════════════════
def page_analysis(df):
    st.markdown('<div class="page-hero"><p class="page-title">Trend Analysis</p><p class="page-sub">Understand what is driving YouTube trends — simplified for everyone</p></div>', unsafe_allow_html=True)

    kpi_cards(df)

    if df.empty:
        return

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
    else:
        st.info("Not enough data to generate insights yet.")

    sh("What Should You Do? — Data-Backed Recommendations")
    st.markdown("<p style='font-size:13px;color:#8b92a5;margin:-10px 0 20px'>Actions derived directly from live trending data. Apply these today.</p>", unsafe_allow_html=True)
    actions = generate_actions(df)
    if actions:
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
    else:
        st.info("Not enough data to generate recommendations yet.")


# ═══════════════════════════════════════
# PAGE: COUNTRY COMPARE
# ═══════════════════════════════════════
def page_country(df):
    st.markdown('<div class="page-hero"><p class="page-title">Country Comparison</p><p class="page-sub">See how trending patterns differ across global markets</p></div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data available yet. Run the scheduler to start collecting trending videos.")
        return

    countries = st.multiselect(
        "Select countries to compare",
        options=sorted(df["country"].unique().tolist()),
        default=sorted(df["country"].unique().tolist())
    )
    if not countries:
        st.warning("Please select at least one country.")
        return

    dfc = df[df["country"].isin(countries)]

    sh("Country Highlights")

    top_cat_per_country = (dfc.groupby(["country","category_name"]).size()
                             .reset_index(name="count")
                             .sort_values("count", ascending=False)
                             .groupby("country").first().reset_index())
    top_eng   = dfc.groupby("country")["like_view_ratio"].mean().sort_values(ascending=False)
    top_viral = dfc.groupby("country")["hours_to_trend"].median().sort_values()

    cat_rows   = "".join([f'<div class="cs-row"><span class="cs-country">{r["country"]}</span><span class="cs-value">{r["category_name"]}</span></div>' for _, r in top_cat_per_country.iterrows()])
    eng_rows   = "".join([f'<div class="cs-row"><span class="cs-country">{c}</span><span class="cs-value">{v*100:.1f}%</span></div>' for c, v in top_eng.items()])
    viral_rows = "".join([f'<div class="cs-row"><span class="cs-country">{c}</span><span class="cs-value">{v:.0f}h</span></div>' for c, v in top_viral.items()])

    st.markdown(f"""
    <div class="country-stat-row">
      <div class="country-stat-card">
        <div class="cs-label">Top Category per Country</div>
        {cat_rows}
      </div>
      <div class="country-stat-card">
        <div class="cs-label">Engagement Rate by Country</div>
        {eng_rows}
      </div>
      <div class="country-stat-card">
        <div class="cs-label">Fastest Trending Country</div>
        {viral_rows}
      </div>
    </div>
    """, unsafe_allow_html=True)

    sh("Distribution Charts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-question">Which country gets the most views per trending video?</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-context">Box plot showing the spread of view counts per country. The middle line is the median; dots above are viral outliers.</div>', unsafe_allow_html=True)
        fig1 = px.box(dfc, x="country", y="view_count", color="country",
                      color_discrete_sequence=COLORS,
                      labels={"view_count":"Views","country":"Country"})
        fig1.update_layout(showlegend=False, height=300, **THEME)
        st.plotly_chart(fig1, use_container_width=True)
        top_v = dfc.groupby("country")["view_count"].median().idxmax()
        st.markdown(f'<div class="chart-insight-bar">💡 <b>{top_v}</b> has the highest median views per trending video.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-question">Where do viewers engage the most?</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-context">Engagement score combines likes and comments relative to views. Higher = more interactive audience.</div>', unsafe_allow_html=True)
        fig2 = px.box(dfc, x="country", y="engagement_score", color="country",
                      color_discrete_sequence=COLORS,
                      labels={"engagement_score":"Engagement","country":"Country"})
        fig2.update_layout(showlegend=False, height=300, **THEME)
        st.plotly_chart(fig2, use_container_width=True)
        top_e = dfc.groupby("country")["engagement_score"].mean().idxmax()
        st.markdown(f'<div class="chart-insight-bar">💡 <b>{top_e}</b> viewers engage the most — expect more comments and likes per view here.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-question">Which content categories are trending in each country?</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-context">Grouped bars show trending video counts per category per country. Compare to spot regional content preferences.</div>', unsafe_allow_html=True)
    cc   = dfc.groupby(["country","category_name"]).size().reset_index(name="count")
    fig3 = px.bar(cc, x="category_name", y="count", color="country", barmode="group",
                  color_discrete_sequence=COLORS,
                  labels={"category_name":"Category","count":"Videos","country":"Country"})
    fig3.update_layout(xaxis_tickangle=-35, height=360, **THEME)
    st.plotly_chart(fig3, use_container_width=True)

    united_states_row = cc[cc["country"]=="United States"]
    india_row         = cc[cc["country"]=="India"]
    top_us = united_states_row.nlargest(1,"count")["category_name"].values[0] if len(united_states_row) else None
    top_in = india_row.nlargest(1,"count")["category_name"].values[0] if len(india_row) else None
    parts  = []
    if top_us: parts.append(f"<b>{top_us}</b> dominates in the United States")
    if top_in: parts.append(f"<b>{top_in}</b> leads in India")
    if parts:
        st.markdown(f'<div class="chart-insight-bar">💡 {", while ".join(parts)}. Understanding regional preferences helps you tailor content for specific markets.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Full Engagement Summary")
    eng = dfc.groupby("country").agg(
        avg_views       =("view_count",      "mean"),
        avg_likes       =("like_count",      "mean"),
        avg_comments    =("comment_count",   "mean"),
        engagement_rate =("like_view_ratio", "mean"),
        avg_sentiment   =("title_sentiment", "mean"),
        hrs_to_viral    =("hours_to_trend",  "median")
    ).round(2).reset_index()
    eng["avg_views"]    = eng["avg_views"].apply(fmt_df)
    eng["avg_likes"]    = eng["avg_likes"].apply(fmt_df)
    eng["avg_comments"] = eng["avg_comments"].apply(fmt_df)
    eng.columns = ["Country","Avg Views","Avg Likes","Avg Comments",
                   "Engagement Rate","Avg Sentiment","Hours to Viral"]
    st.dataframe(eng, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════
# PAGE: HISTORICAL
# ═══════════════════════════════════════
def page_historical(df):
    st.markdown('<div class="page-hero"><p class="page-title">Historical Trends</p><p class="page-sub">See what categories and countries have been dominating trending over time</p></div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data available yet. Run the scheduler to start collecting trending videos.")
        return

    df = df.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df = df.dropna(subset=["fetched_at"])
    df["date"] = df["fetched_at"].dt.strftime("%Y-%m-%d")

    sh("Most Trending Categories — All Time")
    cat_summary = (df.groupby("category_name")
                     .agg(total_videos  =("video_id",       "count"),
                          avg_views     =("view_count",     "mean"),
                          avg_likes     =("like_count",     "mean"),
                          avg_hrs_viral =("hours_to_trend", "median"))
                     .sort_values("total_videos", ascending=False)
                     .reset_index())

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-question">Which categories have appeared on trending the most?</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-context">Total trending appearances per category across all data collected. The more you run the scheduler, the richer this becomes.</div>', unsafe_allow_html=True)
    fig1 = go.Figure(go.Bar(
        x=cat_summary["total_videos"],
        y=cat_summary["category_name"],
        orientation="h",
        marker=dict(color=cat_summary["total_videos"],
                    colorscale=[[0,"#ffe5e5"],[0.5,"#ff8888"],[1,"#ff4444"]],
                    showscale=False),
        hovertemplate="<b>%{y}</b><br>%{x} total trending appearances<extra></extra>"
    ))
    fig1.update_layout(
        height=380,
        yaxis=dict(categoryorder="total ascending", gridcolor="#eceef5",
                   tickfont=dict(size=11, color="#8b92a5")),
        xaxis=dict(title="Total trending appearances", gridcolor="#eceef5",
                   tickfont=dict(size=11, color="#8b92a5")),
        **{k:v for k,v in THEME.items() if k not in ["xaxis","yaxis"]}
    )
    st.plotly_chart(fig1, use_container_width=True)
    top_overall = cat_summary.iloc[0]["category_name"]
    top_count   = int(cat_summary.iloc[0]["total_videos"])
    st.markdown(f'<div class="chart-insight-bar">💡 <b>{top_overall}</b> has appeared on trending {top_count} times — the most consistent category in the dataset.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Category Performance Summary")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-question">How do categories compare on views, likes, and speed?</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-context">Full breakdown of every category — trending frequency, average views earned, and how fast it typically reaches trending.</div>', unsafe_allow_html=True)
    display_cat = cat_summary.copy()
    display_cat["avg_views"]     = display_cat["avg_views"].apply(fmt_df)
    display_cat["avg_likes"]     = display_cat["avg_likes"].apply(fmt_df)
    display_cat["avg_hrs_viral"] = display_cat["avg_hrs_viral"].round(0).fillna(0).astype(int).astype(str) + "h"
    display_cat.columns = ["Category","Times Trending","Avg Views","Avg Likes","Avg Hours to Viral"]
    st.dataframe(display_cat, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sh("Country Performance — Total Videos Collected")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-question">How many trending videos have been collected per country?</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-context">Total trending videos tracked per country across all collection runs. Equal bars = balanced data quality per market.</div>', unsafe_allow_html=True)
    country_summary = (df.groupby("country")
                         .agg(total     =("video_id",         "count"),
                              avg_views =("view_count",       "mean"),
                              avg_eng   =("engagement_score", "mean"))
                         .reset_index()
                         .sort_values("total", ascending=False))
    fig3 = go.Figure(go.Bar(
        x=country_summary["country"],
        y=country_summary["total"],
        marker=dict(color=COLORS[:len(country_summary)]),
        hovertemplate="<b>%{x}</b><br>%{y} videos collected<extra></extra>"
    ))
    fig3.update_layout(height=280, xaxis_title="Country", yaxis_title="Videos collected", **THEME)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════
def page_predict():
    st.markdown('<div class="page-hero"><p class="page-title">Trending Potential Predictor</p><p class="page-sub">Enter your video details — the ML model tells you if it will trend and how to improve</p></div>', unsafe_allow_html=True)
    try:
        classifier = joblib.load(f"{MODELS_PATH}classifier.pkl")
    except Exception:
        st.error("Models not found. Run `python models/trend_classifier.py` first.")
        return

    sh("Your Video Details")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        title         = st.text_input("Video title", placeholder="e.g. 10 Things You Didn't Know About Space!")
        view_count    = st.number_input("Current views",    min_value=0, value=10000, step=1000)
        like_count    = st.number_input("Current likes",    min_value=0, value=500,   step=100)
        comment_count = st.number_input("Current comments", min_value=0, value=100,   step=10)
    with c2:
        duration_min = st.slider("Video length (minutes)", 0, 60, 10)
        tag_count    = st.slider("Number of tags used",    0, 30, 12)
        pub_hour     = st.slider("Upload hour (UTC 0–23)", 0, 23, 15)
        quality      = st.selectbox("Video quality", ["HD (1080p+)", "SD (720p or below)"])
        is_hd        = quality == "HD (1080p+)"

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    go_btn = st.button("Predict Trending Potential", type="primary")

    if go_btn:
        if not title:
            st.warning("Please enter a video title first.")
            return
        sentiment = TextBlob(title).sentiment.polarity
        view_safe = max(view_count, 1)
        X = [[
            view_count, like_count, comment_count,
            float(duration_min), 0, int(is_hd),
            pub_hour, 1, 0,
            like_count / view_safe, comment_count / view_safe,
            (like_count + comment_count) / view_safe,
            tag_count, len(title), len(title.split()),
            int(any(c.isdigit() for c in title)),
            int(title.isupper()),
            int("!" in title), int("?" in title),
            sentiment, sentiment, 0, 0
        ]]
        will_trend = classifier.predict(X)[0]
        prob       = classifier.predict_proba(X)[0][1] * 100
        prob_color = "#10b981" if prob >= 65 else "#f59e0b" if prob >= 40 else "#ff4444"
        verdict    = "Will Trend" if will_trend else "Won't Trend"
        verdict_icon = "✅" if will_trend else "❌"

        sh("Prediction Results")
        st.markdown(f"""
        <div class="result-grid">
          <div class="result-card">
            <div class="result-label">Verdict</div>
            <div class="result-value" style="font-size:15px">{verdict_icon}&nbsp;{verdict}</div>
          </div>
          <div class="result-card">
            <div class="result-label">Trend Probability</div>
            <div class="result-value" style="color:{prob_color}">{prob:.1f}%</div>
          </div>
          <div class="result-card">
            <div class="result-label">Title Sentiment</div>
            <div class="result-value">{"Positive" if sentiment > 0.1 else "Neutral" if sentiment > -0.1 else "Negative"}</div>
          </div>
          <div class="result-card">
            <div class="result-label">Title Length</div>
            <div class="result-value">{"Good" if 30 <= len(title) <= 70 else "Adjust"} ({len(title)}c)</div>
          </div>
        </div>""", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(prob, 1),
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Trending Score","font":{"color":"#8b92a5","size":13,"family":"DM Sans"}},
            number={"suffix":"%","font":{"color":"#1a1d2e","size":44,"family":"DM Sans"}},
            gauge={
                "axis":  {"range":[0,100],"tickcolor":"#eceef5","tickfont":{"color":"#8b92a5","size":11}},
                "bar":   {"color": prob_color},
                "bgcolor":"#f7f8fc","bordercolor":"#eceef5",
                "steps":[{"range":[0,40],"color":"#fff0f0"},{"range":[40,65],"color":"#fff8e8"},{"range":[65,100],"color":"#e8f8ef"}],
                "threshold":{"line":{"color":"#1a1d2e","width":2},"thickness":0.8,"value":65}
            }
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans"), height=260,
                            margin=dict(l=30,r=30,t=20,b=0))
        st.plotly_chart(gauge, use_container_width=True)

        tips = []
        if sentiment < 0:                           tips.append("Your title has a negative tone — reframe it positively to boost click-through rate")
        if tag_count < 10:                          tips.append(f"You only have {tag_count} tags — trending videos use 12–20 tags for better algorithm reach")
        if pub_hour not in range(13, 20):           tips.append(f"You're uploading at {pub_hour}:00 UTC — peak trending window is 13:00–20:00 UTC")
        if len(title) < 30:                         tips.append("Your title is too short — aim for 40–60 characters for better click-through rate")
        if len(title) > 80:                         tips.append("Your title is too long — YouTube truncates it after ~70 characters in search results")
        if not any(c.isdigit() for c in title):     tips.append("Add a number to your title (e.g. '5 Reasons...') — numbered titles get significantly higher CTR")
        if not is_hd:                               tips.append("Upload in HD (1080p+) — over 90% of trending videos are high-definition quality")
        if tips:
            sh("How To Improve Your Chances")
            for tip in tips:
                st.markdown(f'<div class="tip-item">→ &nbsp;{tip}</div>', unsafe_allow_html=True)
        else:
            st.success("Your video looks well-optimised. High chance of trending!")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    # ================= SIDEBAR =================
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
            options=[
                "Live Feed",
                "Analysis",
                "Country Compare",
                "Historical",
                "Predict"
            ],
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="s-footer">
          <div>Data refreshes every 3 hours</div>
          <div>YouTube Data API v3</div>
          <div>US · IN · GB · CA · AU</div>
        </div>
        """, unsafe_allow_html=True)

    # ── FIX 4: Load data for all non-Predict pages; always produce a valid df ──
    if page != "Predict":
        df = load_data()
        if df is None or df.empty:
            st.warning("⚠️ No data available yet. Run the scheduler to populate the database.")
            df = empty_dataframe()
    else:
        df = None

    # ================= PAGE ROUTING =================
    try:
        if page == "Live Feed":
            page_live(df)
        elif page == "Analysis":
            page_analysis(df)
        elif page == "Country Compare":
            page_country(df)
        elif page == "Historical":
            page_historical(df)
        elif page == "Predict":
            page_predict()
    except Exception as e:
        st.error("Something went wrong while rendering the page.")
        st.exception(e)


if __name__ == "__main__":
    main()
