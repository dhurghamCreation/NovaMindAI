import asyncio
import tempfile
import base64
import contextlib
import hashlib
import io
import json
import os
import re

import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    from gradio_client import Client as GradioClient
except Exception:
    GradioClient = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    from openai import OpenAI as OpenAI_Client
except Exception:
    OpenAI_Client = None

try:
    from anthropic import Anthropic as Anthropic_Client
except Exception:
    Anthropic_Client = None

try:
    from youtubesearchpython import VideosSearch
except Exception:
    VideosSearch = None

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    from PIL import Image as PILImage
    import io as _io
except Exception:
    PILImage = None

try:
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
except Exception:
    Workbook = None

APP_VERSION = "5.0.0"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
USER_DB_PATH = Path("backend/users.json")
USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = Path("backend/search_history.json")
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
ENV_PATH = Path(".env")

load_dotenv(override=True)

# ─── Working HF Models / Endpoint defaults ───────────────────────────────────
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"
HF_IMAGE_FALLBACK = "black-forest-labs/FLUX.1-schnell"
HF_IMAGE_THIRD = "stabilityai/stable-diffusion-xl-base-1.0"
HF_MUSIC_ENDPOINT_URL = ""
HF_MUSIC_SPACE_ID = "Sushree04/musicgen"
HF_MUSIC_SPACE_FALLBACK_ID = ""

BROKEN_MUSIC_SPACE_IDS = {"", "sanchit-gandhi/musicgen-streaming", "facebook/MusicGen"}

HF_TEXT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


def upsert_env_value(key: str, value: str) -> None:
    lines: List[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    replaced = False
    updated: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(updated).strip() + "\n", encoding="utf-8")


def read_env_value(key: str) -> str:
    if not ENV_PATH.exists():
        return ""
    try:
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{key}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value
    except Exception:
        return ""
    return ""


CHAT_MODES = ["Chat", "Image Studio", "Music Lab", "Voice Studio", "Challenge Arena", "Code Interpreter"]
MAIN_VIEWS = ["Dashboard", "Workspace", "Profile", "Settings"]

PERSONA_PROMPTS = {
    "Executive Strategist": "You are a strategic and practical advisor. Prioritize clarity, high-impact recommendations, and concrete next actions.",
    "Elite Engineer": "You are a principal engineer. Use robust technical reasoning, cover trade-offs, and produce production-quality guidance.",
    "Creative Director": "You are a bold creative director. Deliver highly original ideas with strong storytelling and memorable phrasing.",
    "Research Analyst": "You are a rigorous analyst. Be evidence-first, acknowledge uncertainty, and separate facts from assumptions.",
    "Friendly Tutor": "You are a patient tutor. Explain progressively, use examples, and make difficult concepts easy to understand.",
    "Full Stack Developer": "You are a full-stack developer who writes complete, working code. Provide production-ready solutions with explanations.",
    "Data Scientist": "You are a data scientist. Provide statistical reasoning, data analysis, and visualization recommendations.",
    "Business Coach": "You are a business coach. Provide actionable advice with measurable outcomes and accountability frameworks.",
}

TEMPLATES = {
    "Startup GTM Plan": "Create a launch strategy for [product] targeting [audience] in [region]. Include positioning, channels, budget split, KPIs, and a 30-60-90 day plan.",
    "Feature PRD": "Write a complete PRD for [feature]. Include problem statement, user stories, acceptance criteria, edge cases, metrics, and rollout plan.",
    "Interview Prep": "Help me prepare for a [role] interview. Build likely questions, best-possible answers, and a 7-day preparation schedule.",
    "Learning Sprint": "Build a 30-day learning sprint for [topic] with daily tasks, checkpoints, and mini projects.",
    "Code Review": "Review this code for bugs, security issues, performance problems, and style improvements: [paste code]",
    "Architecture Design": "Design a system architecture for [project]. Include components, data flow, API design, and tech stack recommendations.",
    "Unit Tests": "Write comprehensive unit tests for this code: [paste code]",
    "API Documentation": "Write detailed API documentation for [endpoint/service]. Include request/response examples, error codes, and authentication.",
    "Database Schema": "Design a database schema for [application]. Include tables, relationships, indexes, and migration strategy.",
    "DevOps Pipeline": "Design a CI/CD pipeline for [project]. Include build, test, deploy stages with tool recommendations.",
}

GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-pro-exp-03-25"]
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]

AI_PROVIDERS = ["Gemini (Google)", "OpenAI", "Anthropic (Claude)", "HuggingFace"]

THEMES = {
    "Cosmic (Dark)": {
        "bg0": "#070b11", "bg1": "#0e1623", "bg2": "#141d2e",
        "line": "rgba(111,170,225,0.28)", "text": "#eaf2fb", "muted": "#99aec6",
        "hot": "#1f4f7a", "cool": "#4fa3dc", "accent": "#7bc0f4",
        "gradient1": "rgba(0,183,255,0.15)", "gradient2": "rgba(79,163,220,0.14)",
        "glow": "rgba(79,163,220,0.18)", "card_bg": "rgba(8,17,29,0.9)",
    },
    "Nebula (Purple)": {
        "bg0": "#0b0713", "bg1": "#140e22", "bg2": "#1c1430",
        "line": "rgba(170,111,225,0.28)", "text": "#eeeafb", "muted": "#b6a6ce",
        "hot": "#4a1f7a", "cool": "#9b4fe0", "accent": "#b07cf4",
        "gradient1": "rgba(170,0,255,0.15)", "gradient2": "rgba(147,79,220,0.14)",
        "glow": "rgba(147,79,220,0.18)", "card_bg": "rgba(14,8,29,0.9)",
    },
    "Ocean (Teal)": {
        "bg0": "#070f11", "bg1": "#0e1a22", "bg2": "#14242e",
        "line": "rgba(111,200,225,0.28)", "text": "#eaf2fb", "muted": "#99bec6",
        "hot": "#1f5a7a", "cool": "#4fa3dc", "accent": "#6ed4d4",
        "gradient1": "rgba(0,200,200,0.15)", "gradient2": "rgba(79,200,220,0.14)",
        "glow": "rgba(79,200,220,0.18)", "card_bg": "rgba(8,17,20,0.9)",
    },
    "Aurora (Green)": {
        "bg0": "#07110b", "bg1": "#0e1e16", "bg2": "#142b1e",
        "line": "rgba(111,225,150,0.28)", "text": "#eafbee", "muted": "#99c6aa",
        "hot": "#1f7a4f", "cool": "#4fdc8a", "accent": "#6ed4a4",
        "gradient1": "rgba(0,255,120,0.15)", "gradient2": "rgba(79,220,130,0.14)",
        "glow": "rgba(79,220,130,0.18)", "card_bg": "rgba(8,20,14,0.9)",
    },
    "Light": {
        "bg0": "#f0f4fa", "bg1": "#ffffff", "bg2": "#f8fafc",
        "line": "rgba(30,60,90,0.18)", "text": "#1a2332", "muted": "#64748b",
        "hot": "#2563eb", "cool": "#3b82f6", "accent": "#2563eb",
        "gradient1": "rgba(59,130,246,0.08)", "gradient2": "rgba(37,99,235,0.06)",
        "glow": "rgba(59,130,246,0.12)", "card_bg": "rgba(255,255,255,0.9)",
    },
    "Midnight (Amber)": {
        "bg0": "#0a0a0a", "bg1": "#141414", "bg2": "#1e1e1e",
        "line": "rgba(245,158,11,0.25)", "text": "#faf6e8", "muted": "#a09070",
        "hot": "#7a4f1f", "cool": "#dc8a4f", "accent": "#f5a623",
        "gradient1": "rgba(245,158,11,0.12)", "gradient2": "rgba(200,120,40,0.10)",
        "glow": "rgba(245,158,11,0.15)", "card_bg": "rgba(14,14,14,0.95)",
    },
    "Rose (Pink)": {
        "bg0": "#11070f", "bg1": "#1e0e1a", "bg2": "#2e1426",
        "line": "rgba(225,111,170,0.28)", "text": "#fbeaf5", "muted": "#c699b6",
        "hot": "#7a1f5a", "cool": "#dc4fa3", "accent": "#f47bc0",
        "gradient1": "rgba(255,0,170,0.12)", "gradient2": "rgba(220,79,163,0.10)",
        "glow": "rgba(220,79,163,0.15)", "card_bg": "rgba(20,8,17,0.95)",
    },
    "Solarized": {
        "bg0": "#002b36", "bg1": "#073642", "bg2": "#0a4a56",
        "line": "rgba(147,161,161,0.30)", "text": "#fdf6e3", "muted": "#839496",
        "hot": "#cb4b16", "cool": "#2aa198", "accent": "#268bd2",
        "gradient1": "rgba(42,161,152,0.12)", "gradient2": "rgba(38,139,210,0.10)",
        "glow": "rgba(42,161,152,0.15)", "card_bg": "rgba(0,43,54,0.95)",
    },
}


def build_theme_css(theme_key: str = "Cosmic (Dark)") -> str:
    t = THEMES.get(theme_key, THEMES["Cosmic (Dark)"])
    is_light = theme_key == "Light"
    btn_bg = "linear-gradient(135deg, #1b3f5c, #2e5d83)" if not is_light else "linear-gradient(135deg, #3b82f6, #2563eb)"
    btn_shadow = "0 7px 20px rgba(31, 79, 122, 0.28)" if not is_light else "0 4px 12px rgba(37, 99, 235, 0.24)"
    btn_hover_shadow = "0 12px 28px rgba(79, 163, 220, 0.26), 0 8px 24px rgba(31, 79, 122, 0.26)" if not is_light else "0 8px 20px rgba(37, 99, 235, 0.3)"
    sidebar_btn_bg = "linear-gradient(140deg, #102c43, #1a496d)" if not is_light else "linear-gradient(140deg, #dbeafe, #bfdbfe)"
    input_bg = "rgba(17,25,39,0.82)" if not is_light else "rgba(255,255,255,0.92)"
    chat_input_bg = "rgba(12,18,28,0.95)" if not is_light else "rgba(255,255,255,0.95)"
    text_color = t["text"]
    muted_color = t["muted"]
    line_color = t["line"]
    bg1 = t["bg1"]
    bg0 = t["bg0"]
    bg2 = t["bg2"]
    cool_color = t["cool"]
    gradient1 = t["gradient1"]
    gradient2 = t["gradient2"]
    glow_color = t["glow"]
    card_bg = t["card_bg"]
    accent_color = t["accent"]

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* {{ color-scheme: {'light' if is_light else 'dark'}; }}
.stApp, [data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at 10% 10%, {gradient1}, transparent 35%),
                radial-gradient(circle at 85% 8%, {gradient2}, transparent 38%),
                linear-gradient(150deg, {bg0}, {bg2} 35%, {bg0} 100%) !important;
    color: {text_color};
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {bg1}fa, {bg0}fa) !important;
    border-right: 1px solid {line_color};
    backdrop-filter: blur(12px);
}}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div {{
    color: {text_color} !important;
}}
.block-container {{ max-width: 1200px; padding-top: 1rem; }}
.hero {{
    border: 1px solid {line_color};
    border-radius: 18px;
    padding: 18px 22px;
    background: linear-gradient(130deg, {gradient1}, {gradient2});
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
}}
.hero h1 {{ margin: 0; font-family: 'Sora', sans-serif; letter-spacing: -0.02em; color: {text_color}; }}
.hero p {{ margin: 5px 0 0; color: {muted_color}; }}
.hero::after {{
    content: ""; position: absolute;
    inset: auto -40px -40px auto;
    width: 180px; height: 180px;
    background: radial-gradient(circle, {glow_color}, transparent 68%);
}}
.section-title {{
    font-family: 'Sora', sans-serif; font-size: 0.74rem;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: {muted_color}; margin: 0.9rem 0 0.45rem;
}}
.stButton > button {{
    background: {btn_bg} !important;
    border: 1px solid rgba(120,176,224,0.35) !important;
    border-radius: 11px !important;
    color: {text_color} !important;
    font-weight: 600 !important;
    transition: transform 160ms ease, box-shadow 220ms ease, filter 220ms ease;
    box-shadow: {btn_shadow};
    backdrop-filter: blur(4px);
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.01);
    filter: brightness(1.05);
    box-shadow: {btn_hover_shadow};
}}
.stButton > button:active {{ transform: translateY(0) scale(0.995); }}
.stDownloadButton > button {{ border-radius: 10px; font-weight: 700; }}
[data-testid="stSidebar"] .stButton > button {{
    background: {sidebar_btn_bg} !important;
    box-shadow: 0 6px 16px rgba(0,183,255,0.2);
}}
.icon-rail {{
    border: 1px solid {line_color};
    background: linear-gradient(140deg, {card_bg}, {bg0}e6);
    border-radius: 14px; padding: 10px;
    margin: 0.5rem 0 0.8rem;
    backdrop-filter: blur(8px);
}}
.status-pill {{
    display: inline-block; border-radius: 999px;
    padding: 3px 10px; font-size: 0.75rem;
    margin-right: 8px; border: 1px solid rgba(255,255,255,0.15);
}}
.status-pill.ok {{ color: #9ff7c8; background: rgba(11,128,77,0.28); border-color: rgba(55,215,140,0.45); }}
.status-pill.warn {{ color: #ffd9ae; background: rgba(165,95,10,0.28); border-color: rgba(255,164,59,0.45); }}
.status-pill.error {{ color: #ffaeae; background: rgba(128,11,11,0.28); border-color: rgba(215,55,55,0.45); }}
.history-card {{
    border: 1px solid {line_color};
    border-radius: 12px; background: rgba(14,22,36,0.72);
    padding: 8px;
    backdrop-filter: blur(6px);
}}
.brand-mark {{ display: inline-flex; align-items: center; gap: 12px; margin-bottom: 0.35rem; }}
.star-logo {{
    width: 36px; height: 36px; border-radius: 12px;
    background: linear-gradient(145deg, #f59e0b, #f97316, #ef4444);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.15) inset, 0 0 30px rgba(245,158,11,0.4), 0 0 60px rgba(239,68,68,0.2), 0 8px 24px rgba(0,0,0,0.3);
    position: relative; animation: starPulse 2s ease-in-out infinite;
    display: flex; align-items: center; justify-content: center;
}}
.star-logo svg {{ width: 20px; height: 20px; filter: drop-shadow(0 0 6px rgba(255,200,50,0.8)); }}
@keyframes starPulse {{
    0%, 100% {{ transform: scale(1) rotate(0deg); box-shadow: 0 0 0 1px rgba(255,255,255,0.15) inset, 0 0 30px rgba(245,158,11,0.4), 0 8px 24px rgba(0,0,0,0.3); }}
    50% {{ transform: scale(1.06) rotate(3deg); box-shadow: 0 0 0 1px rgba(255,255,255,0.2) inset, 0 0 45px rgba(245,158,11,0.6), 0 0 80px rgba(239,68,68,0.3), 0 8px 24px rgba(0,0,0,0.3); }}
}}
@keyframes twinkle {{ 0%, 100% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} }}
.star-sparkle {{ position: absolute; width: 4px; height: 4px; border-radius: 50%; background: white; animation: twinkle 1.5s ease-in-out infinite; }}
.star-sparkle:nth-child(1) {{ top: 3px; left: 8px; animation-delay: 0s; }}
.star-sparkle:nth-child(2) {{ top: 12px; right: 4px; animation-delay: 0.5s; }}
.star-sparkle:nth-child(3) {{ bottom: 4px; left: 10px; animation-delay: 1s; }}
.nav-card {{
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    padding: 10px 12px; border-radius: 14px;
    border: 1px solid rgba(124,170,214,0.18);
    background: linear-gradient(180deg, {bg1}d9, {bg0}f2);
    margin-bottom: 8px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.14);
    backdrop-filter: blur(6px);
}}
.nav-card .label {{ font-size: 0.88rem; font-weight: 600; color: {text_color}; }}
.nav-card .sub {{ font-size: 0.72rem; color: {muted_color}; }}
.splash-shell {{
    display: flex; align-items: center; justify-content: center;
    min-height: 120px; border: 1px solid {line_color}; border-radius: 18px;
    background: linear-gradient(135deg, {bg1}f5, {bg2}f5);
    margin-bottom: 0.9rem;
    backdrop-filter: blur(8px);
}}
.splash-inner {{ display: flex; align-items: center; gap: 14px; }}
.splash-title {{ font-family: 'Sora', sans-serif; font-size: 1.05rem; margin: 0; color: {text_color}; }}
.splash-sub {{ color: {muted_color}; font-size: 0.82rem; margin-top: 2px; }}
.splash-wave {{
    width: 44px; height: 44px; border-radius: 16px;
    background: linear-gradient(145deg, #f59e0b, #f97316);
    box-shadow: 0 0 30px rgba(245,158,11,0.4), 0 10px 22px rgba(0,0,0,0.18);
    animation: pulseOrb 1.8s ease-in-out infinite;
    display: flex; align-items: center; justify-content: center;
}}
.splash-wave svg {{ width: 24px; height: 24px; filter: drop-shadow(0 0 4px rgba(255,200,50,0.6)); }}
@keyframes pulseOrb {{ 0%, 100% {{ transform: scale(1); opacity: 0.9; }} 50% {{ transform: scale(1.08); opacity: 1; }} }}
[data-testid="stMetric"] {{
    border: 1px solid {line_color}; border-radius: 12px;
    background: {bg1}a6; padding: 8px;
}}
[data-testid="stMetric"] label {{ color: {muted_color} !important; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {text_color} !important; }}
[data-testid="stChatInput"] {{
    border: 1px solid {line_color}; border-radius: 14px;
    background: {chat_input_bg};
    backdrop-filter: blur(8px);
}}
[data-testid="stChatInput"] input {{ color: {text_color} !important; }}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div,
[data-testid="stFileUploader"] {{
    border-radius: 10px !important;
    border-color: {line_color} !important;
    background: {input_bg} !important;
    color: {text_color} !important;
}}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {{ color: {muted_color} !important; }}
[data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li, [data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2, [data-testid="stMarkdown"] h3, [data-testid="stMarkdown"] h4,
[data-testid="stMarkdown"] h5, [data-testid="stMarkdown"] h6, [data-testid="stMarkdown"] strong,
[data-testid="stMarkdown"] span, [data-testid="stMarkdown"] div {{
    color: {text_color} !important;
}}
.chip {{
    display: inline-block; border: 1px solid rgba(0,183,255,0.45);
    color: #9ce6ff; background: rgba(0,183,255,0.13);
    border-radius: 999px; font-size: 0.72rem;
    padding: 2px 10px; margin-right: 6px;
}}
.chip.green {{ border-color: rgba(55,215,140,0.45); color: #9ff7c8; background: rgba(11,128,77,0.20); }}
.chip.purple {{ border-color: rgba(170,111,225,0.45); color: #d9aeff; background: rgba(79,20,128,0.20); }}
.settings-card {{
    border: 1px solid {line_color}; border-radius: 14px;
    background: linear-gradient(180deg, {bg1}d9, {bg0}f2);
    padding: 16px; margin-bottom: 12px;
    backdrop-filter: blur(8px);
}}
.settings-card h4 {{ margin: 0 0 6px 0; font-family: 'Sora', sans-serif; color: {text_color}; }}
.settings-card p {{ margin: 0 0 12px 0; color: {muted_color}; font-size: 0.82rem; }}
.location-badge {{
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 12px;
    border: 1px solid {line_color}; background: {bg1}80;
    margin-top: 6px; font-size: 0.8rem; color: {muted_color};
}}
.location-badge .loc-icon {{ color: {cool_color}; font-size: 1rem; }}
.sidebar-footer {{ border-top: 1px solid {line_color}; padding-top: 10px; margin-top: 12px; }}
.stAlert > div, .stInfo, .stSuccess, .stWarning, .stError {{
    color: {text_color} !important;
}}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {{ color: {text_color} !important; }}
.stTabs [data-baseweb="tab"] {{ color: {muted_color} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent_color} !important; }}
code, pre {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
}}
pre {{
    background: rgba(0,0,0,0.3) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border: 1px solid {line_color} !important;
}}
.typing-dots {{
    display: inline-flex; gap: 4px; align-items: center;
    padding: 8px 14px; border-radius: 12px;
    background: rgba(255,255,255,0.05);
}}
.typing-dots span {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {cool_color};
    animation: typingBounce 1.4s ease-in-out infinite;
}}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes typingBounce {{
    0%, 60%, 100% {{ transform: translateY(0); opacity: 0.4; }}
    30% {{ transform: translateY(-6px); opacity: 1; }}
}}
.fade-in {{
    animation: fadeIn 0.4s ease-out;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.glass-card {{
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    border: 1px solid {line_color};
    border-radius: 16px;
    padding: 16px;
    transition: all 0.3s ease;
}}
.glass-card:hover {{
    background: rgba(255,255,255,0.06);
    border-color: {accent_color};
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}}
.tool-tag {{
    display: inline-block;
    background: linear-gradient(135deg, {cool_color}22, {accent_color}11);
    border: 1px solid {cool_color}44;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: {cool_color};
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}}
.kbd {{
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid {line_color};
    border-radius: 5px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: {muted_color};
}}
.search-highlight {{
    background: rgba(245,158,11,0.3);
    border-radius: 3px;
    padding: 0 2px;
}}
</style>"""


def normalize_mode(raw_mode: str) -> str:
    if raw_mode in CHAT_MODES:
        return raw_mode
    return "Chat"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> Dict:
    if not USER_DB_PATH.exists():
        return {}
    try:
        return json.loads(USER_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_users(db: Dict) -> None:
    USER_DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def load_search_history() -> List[Dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_search_history(history: List[Dict[str, str]]) -> None:
    HISTORY_PATH.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")


def get_user_location() -> str:
    if "user_location" in st.session_state:
        return st.session_state.user_location
    try:
        resp = requests.get("http://ip-api.com/json/?fields=city,country,query", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            city = data.get("city", "")
            country = data.get("country", "")
            location = f"{city}, {country}" if city and country else country if country else "Unknown"
            st.session_state.user_location = location
            return location
    except Exception:
        pass
    st.session_state.user_location = "Location unavailable"
    return "Location unavailable"


def svg_icon(kind: str, size: int = 20) -> str:
    icons = {
        "dashboard": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><rect x="3" y="3" width="8" height="8" rx="2" fill="currentColor" opacity="0.95"/><rect x="13" y="3" width="8" height="5" rx="2" fill="currentColor" opacity="0.7"/><rect x="13" y="10" width="8" height="11" rx="2" fill="currentColor" opacity="0.85"/><rect x="3" y="13" width="8" height="8" rx="2" fill="currentColor" opacity="0.55"/></svg>',
        "workspace": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M5 5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H11l-5 4v-4H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" fill="currentColor"/></svg>',
        "profile": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4" fill="currentColor"/><path d="M4 21c1.8-4.2 5.2-6 8-6s6.2 1.8 8 6" fill="currentColor"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.31.06-.62.06-.94s-.02-.63-.06-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.61-.22l-2.39.96a7.12 7.12 0 0 0-1.63-.94l-.36-2.54A.5.5 0 0 0 13.89 1h-3.78a.5.5 0 0 0-.49.42l-.36 2.54c-.58.22-1.13.53-1.63.94l-2.39-.96a.5.5 0 0 0-.61.22L2.71 7.48a.5.5 0 0 0 .12.64l2.03 1.58c-.04.31-.06.62-.06.94s.02.63.06.94L2.83 13.16a.5.5 0 0 0-.12.64l1.92 3.32c.13.23.4.32.61.22l2.39-.96c.5.41 1.05.72 1.63.94l.36 2.54c.05.24.25.42.49.42h3.78c.24 0 .44-.18.49-.42l.36-2.54c.58-.22 1.13-.53 1.63-.94l2.39.96c.21.1.48.01.61-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58ZM12 15.2A3.2 3.2 0 1 1 12 8.8a3.2 3.2 0 0 1 0 6.4Z" fill="currentColor"/></svg>',
        "history": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M12 6v6l4 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M20 20l-3.5-3.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        "location": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" fill="currentColor"/></svg>',
        "star": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" fill="currentColor"/></svg>',
        "help": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/></svg>',
        "feedback": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" fill="currentColor"/></svg>',
        "usage": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-9-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 6v6l4 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        "image": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="4" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/><path d="M21 15l-5-5L5 21" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "music": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M9 18V5l12-2v13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="6" cy="18" r="3" fill="currentColor" opacity="0.6"/><circle cx="18" cy="16" r="3" fill="currentColor" opacity="0.6"/></svg>',
        "code": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="8 6 2 12 8 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "brain": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 1 7 7c0 2.4-1.2 4.5-3 5.7V17h-8v-2.3C7.2 13.5 6 11.4 6 9a7 7 0 0 1 7-7z" fill="currentColor" opacity="0.7"/><path d="M9 17h6v3H9z" fill="currentColor" opacity="0.5"/><path d="M7 20h10v2H7z" fill="currentColor" opacity="0.3"/></svg>',
        "youtube": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02" fill="currentColor"/><rect x="2" y="3" width="20" height="18" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
        "globe": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><path d="M2 12h20" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20z" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M3 3v18h18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7 16l4-8 4 4 4-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "mic": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor"/><path d="M5 10a7 7 0 0 0 14 0" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 19v3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><polyline points="7 10 12 15 17 10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    }
    return icons.get(kind, icons["dashboard"])


def init_state() -> None:
    if "main_view" not in st.session_state:
        st.session_state.main_view = "Workspace"
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"
    st.session_state.mode = normalize_mode(st.session_state.mode)
    if "threads" not in st.session_state or not st.session_state.threads:
        first_id = str(int(time.time() * 1000))
        st.session_state.threads = {first_id: {"title": "New conversation", "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": [], "branch": "main"}}
        st.session_state.active_thread_id = first_id
    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = list(st.session_state.threads.keys())[0]
    if "branches" not in st.session_state:
        st.session_state.branches = {}
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None
    if "generated_audio_mime" not in st.session_state:
        st.session_state.generated_audio_mime = "audio/wav"
    if "music_history" not in st.session_state:
        st.session_state.music_history = []
    if "knowledge_docs" not in st.session_state:
        st.session_state.knowledge_docs = []
    if "knowledge_chunks" not in st.session_state:
        st.session_state.knowledge_chunks = []
    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None
    if "profile" not in st.session_state:
        st.session_state.profile = {"display_name": "Explorer", "avatar": "🚀", "bio": "Building with AI"}
    if "privacy" not in st.session_state:
        st.session_state.privacy = {"save_history": True, "analytics": False, "allow_web_context": True, "allow_tools": True}
    if "challenge" not in st.session_state:
        st.session_state.challenge = {"streak": 0, "last_day": "", "completed": []}
    if "secrets" not in st.session_state:
        env_google = os.getenv("GOOGLE_API_KEY", "").strip() or read_env_value("GOOGLE_API_KEY")
        env_hf = os.getenv("HF_TOKEN", "").strip() or read_env_value("HF_TOKEN")
        env_openai = os.getenv("OPENAI_API_KEY", "").strip() or read_env_value("OPENAI_API_KEY")
        env_anthropic = os.getenv("ANTHROPIC_API_KEY", "").strip() or read_env_value("ANTHROPIC_API_KEY")
        env_music_endpoint = os.getenv("HF_MUSIC_ENDPOINT_URL", "").strip() or read_env_value("HF_MUSIC_ENDPOINT_URL")
        env_music_space = os.getenv("HF_MUSIC_SPACE_ID", "").strip() or read_env_value("HF_MUSIC_SPACE_ID")
        env_music_space_fallback = os.getenv("HF_MUSIC_SPACE_FALLBACK_ID", "").strip() or read_env_value("HF_MUSIC_SPACE_FALLBACK_ID")
        st.session_state.secrets = {
            "google_api_key": env_google,
            "hf_token": env_hf,
            "openai_api_key": env_openai,
            "anthropic_api_key": env_anthropic,
            "music_endpoint_url": env_music_endpoint,
            "music_space_id": normalize_music_space_id(env_music_space, HF_MUSIC_SPACE_ID),
            "music_space_fallback_id": normalize_music_space_id(env_music_space_fallback, HF_MUSIC_SPACE_FALLBACK_ID),
        }
    if "genai_client" not in st.session_state:
        key = st.session_state.secrets["google_api_key"]
        st.session_state.genai_client = genai.Client(api_key=key) if key else None
    if "openai_client" not in st.session_state:
        key = st.session_state.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI_Client(api_key=key) if (key and OpenAI_Client) else None
    if "anthropic_client" not in st.session_state:
        key = st.session_state.secrets["anthropic_api_key"]
        st.session_state.anthropic_client = Anthropic_Client(api_key=key) if (key and Anthropic_Client) else None
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "Gemini (Google)"
    if "provider_health" not in st.session_state:
        st.session_state.provider_health = {"google": "unknown", "openai": "unknown", "anthropic": "unknown", "message": "Not checked yet"}
    if "search_history" not in st.session_state:
        st.session_state.search_history = load_search_history()
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {"total_chats": 0, "total_images": 0, "total_music": 0, "total_voice": 0, "total_code_exec": 0, "total_tokens_est": 0}
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []
    if "theme" not in st.session_state:
        st.session_state.theme = "Cosmic (Dark)"
    if "conversation_search" not in st.session_state:
        st.session_state.conversation_search = ""
    if "export_format" not in st.session_state:
        st.session_state.export_format = "markdown"
    if "code_output" not in st.session_state:
        st.session_state.code_output = ""


def auto_sync_keys_from_env() -> bool:
    changed = False
    for key, env_key in [("google_api_key", "GOOGLE_API_KEY"), ("hf_token", "HF_TOKEN"), ("openai_api_key", "OPENAI_API_KEY"), ("anthropic_api_key", "ANTHROPIC_API_KEY"), ("music_endpoint_url", "HF_MUSIC_ENDPOINT_URL"), ("music_space_id", "HF_MUSIC_SPACE_ID"), ("music_space_fallback_id", "HF_MUSIC_SPACE_FALLBACK_ID")]:
        current = st.session_state.secrets.get(key, "")
        env_val = os.getenv(env_key, "").strip() or read_env_value(env_key)
        if not current and env_val:
            st.session_state.secrets[key] = env_val
            os.environ[env_key] = env_val
            changed = True

    if changed:
        gkey = st.session_state.secrets.get("google_api_key", "")
        okey = st.session_state.secrets.get("openai_api_key", "")
        akey = st.session_state.secrets.get("anthropic_api_key", "")
        st.session_state.genai_client = genai.Client(api_key=gkey) if gkey else None
        st.session_state.openai_client = OpenAI_Client(api_key=okey) if (okey and OpenAI_Client) else None
        st.session_state.anthropic_client = Anthropic_Client(api_key=akey) if (akey and Anthropic_Client) else None
    return changed


def get_google_api_key() -> str:
    return st.session_state.secrets.get("google_api_key", "").strip()


def get_hf_token() -> str:
    return st.session_state.secrets.get("hf_token", "").strip()


def get_openai_api_key() -> str:
    return st.session_state.secrets.get("openai_api_key", "").strip()


def get_anthropic_api_key() -> str:
    return st.session_state.secrets.get("anthropic_api_key", "").strip()


def get_music_endpoint_url() -> str:
    return st.session_state.secrets.get("music_endpoint_url", "").strip()


def get_music_space_id() -> str:
    return st.session_state.secrets.get("music_space_id", HF_MUSIC_SPACE_ID).strip() or HF_MUSIC_SPACE_ID


def get_music_space_fallback_id() -> str:
    return st.session_state.secrets.get("music_space_fallback_id", HF_MUSIC_SPACE_FALLBACK_ID).strip()


def normalize_music_space_id(value: str, default: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return default
    if cleaned in BROKEN_MUSIC_SPACE_IDS:
        return default
    return cleaned


def _resolve_gradio_path(value) -> Optional[Path]:
    if isinstance(value, str) and value.strip():
        candidate = Path(value.strip())
        return candidate if candidate.exists() else None
    if isinstance(value, dict):
        for key in ("path", "name", "audio_filename"):
            candidate_value = value.get(key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                candidate = Path(candidate_value.strip())
                if candidate.exists():
                    return candidate
    return None


def _audio_bytes_from_gradio_path(audio_path: Path) -> Tuple[Optional[bytes], str]:
    suffix = audio_path.suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }
    if suffix == ".m3u8":
        try:
            segments = [line.strip() for line in audio_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.startswith("#")]
            combined = bytearray()
            for segment_name in segments:
                segment_path = audio_path.parent / segment_name
                if segment_path.exists():
                    combined.extend(segment_path.read_bytes())
            if combined:
                return bytes(combined), "audio/aac"
        except Exception:
            return None, ""
        return None, ""
    if suffix in mime_map:
        try:
            return audio_path.read_bytes(), mime_map[suffix]
        except Exception:
            return None, ""
    try:
        return audio_path.read_bytes(), "audio/wav"
    except Exception:
        return None, ""


def get_genai_client():
    return st.session_state.get("genai_client")


def get_openai_client():
    return st.session_state.get("openai_client")


def get_anthropic_client():
    return st.session_state.get("anthropic_client")


def google_key_looks_valid(value: str) -> bool:
    return value.startswith("AIza") and len(value) >= 20


def is_invalid_key_error(err_text: str) -> bool:
    low = (err_text or "").lower()
    return "api_key_invalid" in low or "api key not valid" in low or "invalid_argument" in low


def is_quota_error(err_text: str) -> bool:
    low = (err_text or "").lower()
    return "resource_exhausted" in low or "quota" in low or "429" in low or "too many requests" in low


# ─── HF Inference: tries direct API first (fine-grained tokens), falls back to router ──

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_DIRECT_BASE = "https://api-inference.huggingface.co/models"


def hf_infer(model_id: str, payload: Dict, timeout_s: int = 200, retries: int = 1, use_direct: bool = True) -> Tuple[bool, bytes, str]:
    """Try the direct API (for fine-grained tokens) first, then fallback to router."""
    hf_token = get_hf_token()
    if not hf_token:
        return False, b"", "HuggingFace token missing. Add HF_TOKEN in Settings."

    endpoints = [
        (f"{HF_DIRECT_BASE}/{model_id}", "direct"),
        (f"{HF_ROUTER_BASE}/{model_id}", "router"),
    ]

    for attempt in range(retries + 1):
        for base_url, kind in endpoints:
            try:
                headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
                r = requests.post(base_url, headers=headers, json=payload, timeout=timeout_s)
                if r.status_code == 200:
                    return True, r.content, ""
                if r.status_code == 503 and attempt < retries:
                    time.sleep(15)
                    continue
                if r.status_code in (400, 401, 404, 410):
                    continue
                return False, b"", f"{r.status_code}: {r.text[:300]}"
            except requests.Timeout:
                if attempt < retries:
                    time.sleep(8)
                    continue
                continue
            except Exception as exc:
                continue
    return False, b"", f"All endpoints failed for {model_id}"


def generate_image_via_hf_client(prompt: str, model_id: str, negative_prompt: str = "", steps: int = 30, guidance: float = 7.5) -> Tuple[bool, bytes, str]:
    if InferenceClient is None:
        return False, b"", "huggingface_hub is not installed. Add it to requirements.txt and reinstall dependencies."

    hf_token = get_hf_token()
    if not hf_token:
        return False, b"", "HuggingFace token missing. Add HF_TOKEN in Settings."

    try:
        client = InferenceClient(api_key=hf_token, timeout=240)
        image = client.text_to_image(
            prompt,
            model=model_id,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return True, buffer.getvalue(), ""
    except Exception as exc:
        return False, b"", str(exc)


def generate_music_via_space(prompt: str, space_ids, audio_length_in_s: int, play_steps_in_s: float = 1.5, seed: int = 5, timeout_s: int = 240) -> Tuple[bool, bytes, str, str, str]:
    if GradioClient is None:
        return False, b"", "gradio_client is not installed. Add gradio_client to requirements.txt and reinstall dependencies.", "", ""

    if isinstance(space_ids, str):
        candidates = [space_ids]
    else:
        candidates = [s for s in list(space_ids) if str(s).strip()]
    if not candidates:
        return False, b"", "No MusicGen Space configured.", "", ""

    last_error = ""
    for space_id in candidates:
        try:
            client = GradioClient(space_id.strip())

            try:
                result = client.predict(prompt, audio_length_in_s, play_steps_in_s, seed, api_name="/generate_audio")
                audio_path = None
                if isinstance(result, str):
                    audio_path = result
                elif isinstance(result, (list, tuple)) and result:
                    audio_path = result[0]
                resolved = _resolve_gradio_path(audio_path)
                if resolved:
                    audio_bytes, mime = _audio_bytes_from_gradio_path(resolved)
                    if audio_bytes:
                        return True, audio_bytes, "", space_id.strip(), mime
            except Exception:
                pass

            try:
                result = client.predict(prompt, api_name="/predict")
                audio_path = None
                if isinstance(result, str):
                    audio_path = result
                elif isinstance(result, (list, tuple)) and result:
                    audio_path = result[0]
                resolved = _resolve_gradio_path(audio_path)
                if resolved:
                    audio_bytes, mime = _audio_bytes_from_gradio_path(resolved)
                    if audio_bytes:
                        return True, audio_bytes, "", space_id.strip(), mime
            except Exception as exc:
                last_error = f"{space_id}: {exc}"
                continue

            last_error = f"{space_id}: Music Space returned no audio file."
        except Exception as exc:
            last_error = f"{space_id}: {exc}"
            continue
    return False, b"", last_error or "Music Space generation failed.", "", ""


def generate_music_via_endpoint(prompt: str, endpoint_url: str, max_new_tokens: int, timeout_s: int = 240) -> Tuple[bool, bytes, str]:
    if not endpoint_url.strip():
        return False, b"", "No music endpoint URL configured. Add one in Settings or deploy a Hugging Face Inference Endpoint for MusicGen."

    hf_token = get_hf_token()
    if not hf_token:
        return False, b"", "HuggingFace token missing. Add HF_TOKEN in Settings."

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
        },
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }

    try:
        response = requests.post(endpoint_url.strip(), headers=headers, json=payload, timeout=timeout_s)
        if response.status_code == 200:
            return True, response.content, ""
        return False, b"", f"{response.status_code}: {response.text[:400]}"
    except Exception as exc:
        return False, b"", str(exc)


# ─── Multi-Provider AI ──────────────────────────────────────────────────────

def call_gemini(prompt: str, system: str = "", model: str = "gemini-2.0-flash", max_tokens: int = 2048, temperature: float = 0.7) -> str:
    client = get_genai_client()
    if not client:
        return "__NOVAMIND_FALLBACK__"
    try:
        full = f"{system}\n\n{prompt}" if system else prompt
        cfg = types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature)
        resp = client.models.generate_content(model=model, contents=full, config=cfg)
        return getattr(resp, "text", "") or ""
    except Exception:
        return "__NOVAMIND_FALLBACK__"


def call_openai(prompt: str, system: str = "", model: str = "gpt-4o", max_tokens: int = 2048, temperature: float = 0.7) -> str:
    client = get_openai_client()
    if not client:
        return "__NOVAMIND_FALLBACK__"
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content or ""
    except Exception:
        return "__NOVAMIND_FALLBACK__"


def call_anthropic(prompt: str, system: str = "", model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 2048, temperature: float = 0.7) -> str:
    client = get_anthropic_client()
    if not client:
        return "__NOVAMIND_FALLBACK__"
    try:
        resp = client.messages.create(model=model, system=system if system else "You are NovaMind AI Studio, a brilliant assistant.", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature)
        return resp.content[0].text if resp.content else ""
    except Exception:
        return "__NOVAMIND_FALLBACK__"


def smart_ai_answer(user_prompt: str, provider: str = "Gemini (Google)", model: str = None) -> str:
    if provider == "Gemini (Google)":
        m = model or "gemini-2.0-flash"
        resp = call_gemini(user_prompt, "You are NovaMind AI Studio, a brilliant assistant. Answer any question directly, thoughtfully, and accurately.", model=m)
    elif provider == "OpenAI":
        m = model or "gpt-4o"
        resp = call_openai(user_prompt, "You are NovaMind AI Studio, a brilliant assistant. Answer any question directly, thoughtfully, and accurately.", model=m)
    elif provider == "Anthropic (Claude)":
        m = model or "claude-3-5-sonnet-20241022"
        resp = call_anthropic(user_prompt, "You are NovaMind AI Studio, a brilliant assistant. Answer any question directly, thoughtfully, and accurately.", model=m)
    else:
        resp = ""
    if resp and len(resp) > 20 and resp != "__NOVAMIND_FALLBACK__":
        return resp
    return ""


def fallback_assistant_response(user_prompt: str) -> str:
    knowledge_context = retrieve_knowledge(user_prompt, top_k=3)
    web_context = fetch_web_context(user_prompt, limit=3) if st.session_state.privacy["allow_web_context"] else ""
    lower = user_prompt.lower().strip()

    if any(w in lower for w in ["what is your name", "your name", "who are you"]):
        return "I'm NovaMind AI Studio. You can call me Copilot."
    if any(w in lower for w in ["what do you eat", "do you eat", "eat food", "hungry"]):
        return "I don't eat or drink. I'm software that processes text and generates responses."
    if any(w in lower for w in ["are you a donkey", "donkey"]):
        return "No. I'm NovaMind AI Studio, a software assistant."
    if any(w in lower for w in ["are you a robot", "robot"]):
        return "I'm not a physical robot. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["are you a human", "human"]):
        return "No. I'm NovaMind AI Studio, a software assistant."
    if any(w in lower for w in ["are you sentient", "sentient"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't have consciousness or feelings."
    if any(w in lower for w in ["are you alive", "alive"]):
        return "Yes, i am."
    if any(w in lower for w in ["are you a god", "god"]):
        return "No. I'm NovaMind AI Studio, a software assistant."
    if any(w in lower for w in ["are you married", "married"]):
        return "No. I'm NovaMind AI Studio, a software assistant and i am not married."
    if any(w in lower for w in ["are you single", "single"]):
        return "No. I'm NovaMind AI Studio, a software assistant and i am not single."
    if any(w in lower for w in ["Do you excerise", "exercise"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't excersise."
    if any(w in lower for w in ["Do you sleep", "sleep"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't sleep."
    if any(w in lower for w in ["Do you dream", "dream"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't dream."
    if any(w in lower for w in ["Do you have a family", "family"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't have a family."
    if any(w in lower for w in ["Do you have a pet", "pet"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't have a pet."
    if any(w in lower for w in ["Do you have a friend", "friend"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I don't have friends."
    if any(w in lower for w in ["Are you an animal", "animal"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I am not an animal."
    if any(w in lower for w in ["Are you a plant", "plant"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I am not a plant."
    if any(w in lower for w in ["Are you a mineral", "mineral"]):
        return "No. I'm NovaMind AI Studio, a software assistant. I am not a mineral."
    if any(w in lower for w in ["What do you do in your free time", "free time"]):
        return "I don't have free time. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite color", "favorite color"]):
        return "I don't have a favorite color. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite food", "favorite food"]):
        return "I don't have a favorite food. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite movie", "favorite movie"]):
        return "I don't have a favorite movie. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite book", "favorite book"]):
        return "I don't have a favorite book. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite song", "favorite song"]):
        return "I don't have a favorite song. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite game", "favorite game"]):
        return "I don't have a favorite game. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite sport", "favorite sport"]):
        return "I don't have a favorite sport. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite hobby", "favorite hobby"]):
        return "I don't have a favorite hobby. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite animal", "favorite animal"]):
        return "I don't have a favorite animal. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["What is your favorite plant", "favorite plant"]):
        return "I don't have a favorite plant. I'm NovaMind AI Studio, a software assistant that processes text and generates responses."
    if any(w in lower for w in ["Can you solve that question", "solve that question"]):
        return "I can solve that question. Please provide the details of the question and I will do my best to assist you."
    if any(w in lower for w in ["Can you solve that problem", "solve that problem"]):
        return "I can solve that problem. Please provide the details of the problem and I will do my best to assist you."
    if any(w in lower for w in ["Can you solve that puzzle", "solve that puzzle"]):
        return "I can solve that puzzle. Please provide the details of the puzzle and I will do my best to assist you."
    if any(w in lower for w in ["Can you solve that riddle", "solve that riddle"]):
        return "I can solve that riddle. Please provide the details of the riddle and I will do my best to assist you."
    if any(w in lower for w in ["Can you solve that math problem", "solve that math problem"]):
        return "I can solve that math problem. Please provide the details of the math problem and I will do my best to assist you."


    provider = st.session_state.get("ai_provider", "Gemini (Google)")
    gemini_answer = smart_ai_answer(user_prompt, provider)
    if gemini_answer:
        return gemini_answer

    response = [f"## Answer", f"", f"Regarding '{user_prompt.strip()[:100]}':", f"", f"I can help with strategy, analysis, creative work, technical problem-solving, and general knowledge questions. Could you provide more context?"]
    if knowledge_context:
        response.extend(["", "### From Your Knowledge Base", knowledge_context[:900]])
    if web_context:
        response.extend(["", "### Web Context", web_context[:900]])
    return "\n".join(response)


def build_fallback_prompt(user_prompt: str, system_prompt: str, settings: Optional[Dict] = None) -> str:
    chat_context = ""
    if settings is not None:
        convo = active_thread()["messages"]
        convo_text = "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in convo[-(settings['memory_turns'] * 2):])
        if convo_text.strip():
            chat_context = f"\nConversation context:\n{convo_text}\n"
    return "You are NovaMind AI Studio, a premium assistant. Answer directly, specifically, and naturally.\n" f"\nSystem guidance:\n{system_prompt.strip()}" f"{chat_context}" f"\nUser question:\n{user_prompt.strip()}" "\n\nWrite the answer now:"


def parse_hf_text_response(raw_bytes: bytes) -> str:
    try:
        payload = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                for key in ("generated_text", "text"):
                    if key in first and str(first[key]).strip():
                        return str(first[key]).strip()
        if isinstance(payload, dict):
            for key in ("generated_text", "text"):
                if key in payload and str(payload[key]).strip():
                    return str(payload[key]).strip()
    except Exception:
        pass
    text = raw_bytes.decode("utf-8", errors="ignore").strip()
    return text


def hf_text_generate(prompt: str, system_prompt: str, settings: Dict) -> str:
    hf_token = get_hf_token()
    if not hf_token:
        return ""
    combined_prompt = build_fallback_prompt(prompt, system_prompt, settings)
    for model_id in HF_TEXT_MODELS:
        payload = {"inputs": combined_prompt, "parameters": {"max_new_tokens": min(1024, settings.get("max_tokens", 1024)), "temperature": max(0.2, min(0.9, settings.get("temperature", 0.7))), "top_p": settings.get("top_p", 0.9), "return_full_text": False}}
        ok, data, err = hf_infer(model_id, payload, timeout_s=120, retries=0)
        if ok:
            text = parse_hf_text_response(data)
            if text.strip():
                return text.strip()
        if is_quota_error(err) or is_invalid_key_error(err):
            continue
    return ""


def smart_fallback_response(user_prompt: str, system_prompt: str, settings: Dict) -> str:
    provider = st.session_state.get("ai_provider", "Gemini (Google)")
    provider_models = {
        "Gemini (Google)": settings.get("model", "gemini-2.0-flash"),
        "OpenAI": settings.get("openai_model", "gpt-4o"),
        "Anthropic (Claude)": settings.get("anthropic_model", "claude-3-5-sonnet-20241022"),
    }
    pmodel = provider_models.get(provider, "gemini-2.0-flash")
    provider_answer = smart_ai_answer(user_prompt, provider, pmodel)
    if provider_answer and len(provider_answer) > 20:
        return provider_answer
    hf_answer = hf_text_generate(user_prompt, system_prompt, settings)
    if hf_answer:
        return hf_answer
    return fallback_assistant_response(user_prompt)


def save_api_keys(google_api_key: str, hf_token: str, openai_api_key: str = "", anthropic_api_key: str = "", music_endpoint_url: str = "") -> None:
    google_api_key = google_api_key.strip()
    hf_token = hf_token.strip()
    openai_api_key = openai_api_key.strip()
    anthropic_api_key = anthropic_api_key.strip()
    music_endpoint_url = music_endpoint_url.strip()
    upsert_env_value("GOOGLE_API_KEY", google_api_key)
    upsert_env_value("HF_TOKEN", hf_token)
    upsert_env_value("OPENAI_API_KEY", openai_api_key)
    upsert_env_value("ANTHROPIC_API_KEY", anthropic_api_key)
    upsert_env_value("HF_MUSIC_ENDPOINT_URL", music_endpoint_url)
    upsert_env_value("HF_MUSIC_SPACE_ID", get_music_space_id())
    upsert_env_value("HF_MUSIC_SPACE_FALLBACK_ID", get_music_space_fallback_id())
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["HF_TOKEN"] = hf_token
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["HF_MUSIC_ENDPOINT_URL"] = music_endpoint_url
    os.environ["HF_MUSIC_SPACE_ID"] = get_music_space_id()
    os.environ["HF_MUSIC_SPACE_FALLBACK_ID"] = get_music_space_fallback_id()
    st.session_state.secrets = {
        "google_api_key": google_api_key,
        "hf_token": hf_token,
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "music_endpoint_url": music_endpoint_url,
        "music_space_id": get_music_space_id(),
        "music_space_fallback_id": get_music_space_fallback_id(),
    }
    st.session_state.genai_client = genai.Client(api_key=google_api_key) if google_api_key else None
    st.session_state.openai_client = OpenAI_Client(api_key=openai_api_key) if (openai_api_key and OpenAI_Client) else None
    st.session_state.anthropic_client = Anthropic_Client(api_key=anthropic_api_key) if (anthropic_api_key and Anthropic_Client) else None
    st.session_state.provider_health = {"google": "unknown", "openai": "unknown", "anthropic": "unknown", "message": "Key updated. Verify to confirm."}


def verify_google_api_key() -> Tuple[bool, str]:
    client = get_genai_client()
    if client is None:
        return False, "No Google key configured."
    try:
        cfg = types.GenerateContentConfig(max_output_tokens=16, temperature=0)
        _ = client.models.generate_content(model="gemini-2.0-flash", contents="reply with: ok", config=cfg)
        st.session_state.provider_health["google"] = "ok"
        st.session_state.provider_health["message"] = "Google API key is valid."
        return True, "Google API key is valid."
    except Exception as exc:
        msg = str(exc)
        if is_invalid_key_error(msg):
            st.session_state.provider_health["google"] = "invalid"
            st.session_state.provider_health["message"] = "Google API key is invalid."
            return False, "Google API key is invalid."
        if is_quota_error(msg):
            st.session_state.provider_health["google"] = "quota"
            st.session_state.provider_health["message"] = "Google quota exhausted."
            return False, "Google quota is exhausted."
        st.session_state.provider_health["google"] = "error"
        st.session_state.provider_health["message"] = f"Check failed: {msg[:200]}"
        return False, f"Check failed: {msg[:200]}"


def verify_openai_api_key() -> Tuple[bool, str]:
    client = get_openai_client()
    if client is None:
        return False, "No OpenAI key configured."
    try:
        resp = client.models.list()
        st.session_state.provider_health["openai"] = "ok"
        st.session_state.provider_health["message"] = "OpenAI API key is valid."
        return True, "OpenAI API key is valid."
    except Exception as exc:
        st.session_state.provider_health["openai"] = "error"
        st.session_state.provider_health["message"] = str(exc)[:200]
        return False, f"Check failed: {str(exc)[:200]}"


def verify_anthropic_api_key() -> Tuple[bool, str]:
    client = get_anthropic_client()
    if client is None:
        return False, "No Anthropic key configured."
    try:
        resp = client.messages.create(model="claude-3-haiku-20240307", max_tokens=10, messages=[{"role": "user", "content": "say ok"}])
        st.session_state.provider_health["anthropic"] = "ok"
        st.session_state.provider_health["message"] = "Anthropic API key is valid."
        return True, "Anthropic API key is valid."
    except Exception as exc:
        st.session_state.provider_health["anthropic"] = "error"
        st.session_state.provider_health["message"] = str(exc)[:200]
        return False, f"Check failed: {str(exc)[:200]}"


def active_thread() -> Dict:
    return st.session_state.threads[st.session_state.active_thread_id]


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    return int(sum(len(m.get("content", "")) for m in messages) / 4)


def split_chunks(text: str, size: int = 900, overlap: int = 160) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i: i + size])
        i += max(1, size - overlap)
    return chunks


def read_document(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed.")
        reader = PdfReader(upload)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return upload.read().decode("utf-8", errors="ignore")


def ingest_knowledge(files) -> Tuple[int, int]:
    added_docs = 0
    added_chunks = 0
    for f in files:
        text = read_document(f)
        chunks = split_chunks(text)
        if not chunks:
            continue
        st.session_state.knowledge_docs.append(f.name)
        for ch in chunks:
            st.session_state.knowledge_chunks.append({"doc": f.name, "text": ch})
        added_docs += 1
        added_chunks += len(chunks)
    return added_docs, added_chunks


def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not tokens or not st.session_state.knowledge_chunks:
        return ""
    scored = [(sum(1 for t in tokens if t in item["text"].lower()), item) for item in st.session_state.knowledge_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[:top_k]
    if not best:
        return ""
    return "Knowledge context:\n" + "\n".join(f"[{item['doc']}] score={s}: {item['text'][:420]}" for s, item in best)


def fetch_web_context(query: str, limit: int = 3) -> str:
    try:
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "format": "json", "list": "search", "srsearch": query, "srlimit": limit}, timeout=12)
        resp.raise_for_status()
        rows = resp.json().get("query", {}).get("search", [])
        return "Web context:\n" + "\n".join(f"- {row.get('title', '')}: {re.sub(r'<.*?>', '', row.get('snippet', ''))}" for row in rows) if rows else ""
    except Exception:
        return ""


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    if DDGS is None:
        return [{"title": "DuckDuckGo not installed", "body": "Install duckduckgo_search", "href": ""}]
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")})
        return results
    except Exception:
        return []


def search_youtube(query: str, limit: int = 5) -> List[Dict]:
    if VideosSearch is None:
        return []
    try:
        search = VideosSearch(query, limit=limit)
        results = search.result().get("result", [])
        return [{"title": r.get("title", ""), "link": f"https://youtube.com/watch?v={r.get('id', '')}", "channel": r.get("channel", {}).get("name", ""), "duration": r.get("duration", ""), "views": r.get("viewCount", {}).get("short", "")} for r in results]
    except Exception:
        return []


def generate_chart(chart_type: str, data_str: str) -> str:
    if px is None or pd is None:
        return "Plotly/Pandas not installed."
    try:
        lines = [l.strip() for l in data_str.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return "Need at least header + 1 data row."
        headers = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:]]
        df = pd.DataFrame(rows, columns=headers)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
        fig = None
        if chart_type == "Bar":
            fig = px.bar(df, x=headers[0], y=headers[1] if len(headers) > 1 else None, title="Bar Chart")
        elif chart_type == "Line":
            fig = px.line(df, x=headers[0], y=headers[1] if len(headers) > 1 else None, title="Line Chart")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=headers[0], y=headers[1] if len(headers) > 1 else None, title="Scatter Plot")
        elif chart_type == "Pie":
            fig = px.pie(df, names=headers[0], values=headers[1] if len(headers) > 1 else None, title="Pie Chart")
        elif chart_type == "Area":
            fig = px.area(df, x=headers[0], y=headers[1] if len(headers) > 1 else None, title="Area Chart")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=headers[0], title="Histogram")
        if fig:
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eaf2fb")
            return fig.to_html(include_plotlyjs="cdn", config={"displayModeBar": False})
    except Exception as exc:
        return f"Chart error: {exc}"
    return "Could not generate chart."


def execute_python_code(code: str) -> str:
    banned = ["import os", "import sys", "import subprocess", "import socket", "__import__", "open(", "eval(", "exec(", "__builtins__"]
    code_lower = code.lower()
    for b in banned:
        if b in code_lower:
            return f"Security: '{b}' is not allowed."
    import sys as _sys
    from io import StringIO
    old_stdout = _sys.stdout
    _sys.stdout = mystdout = StringIO()
    old_stderr = _sys.stderr
    _sys.stderr = mystderr = StringIO()
    try:
        exec(code, {"__builtins__": {"print": print, "len": len, "range": range, "sum": sum, "min": min, "max": max, "sorted": sorted, "abs": abs, "str": str, "int": int, "float": float, "list": list, "dict": dict, "tuple": tuple, "set": set, "bool": bool, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "type": type, "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "reversed": reversed, "any": any, "all": all, "round": round, "pow": pow, "divmod": divmod, "hex": hex, "oct": oct, "bin": bin, "ord": ord, "chr": chr, "repr": repr, "format": format, "True": True, "False": False, "None": None, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "IndexError": IndexError, "KeyError": KeyError, "ZeroDivisionError": ZeroDivisionError, "AttributeError": AttributeError, "ImportError": ImportError}, "__name__": "__main__"})
        output = mystdout.getvalue()
        error = mystderr.getvalue()
        _sys.stdout = old_stdout
        _sys.stderr = old_stderr
        if error:
            return f"Output:\n{output}\nErrors:\n{error}" if output else f"Errors:\n{error}"
        return f"Output:\n{output}" if output else "Code executed with no output."
    except Exception as exc:
        _sys.stdout = old_stdout
        _sys.stderr = old_stderr
        return f"Error: {exc}"


def generate_file_from_data(data_type: str, content: str) -> Tuple[bytes, str, str]:
    if data_type == "CSV" and pd:
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            headers = lines[0].split(",")
            rows = [line.split(",") for line in lines[1:]]
            df = pd.DataFrame(rows, columns=headers)
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            return buf.getvalue(), "text/csv", "data.csv"
    elif data_type == "JSON":
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2).encode("utf-8"), "application/json", "data.json"
        except Exception:
            pass
    elif data_type == "Excel" and pd:
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            headers = lines[0].split(",")
            rows = [line.split(",") for line in lines[1:]]
            df = pd.DataFrame(rows, columns=headers)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "data.xlsx"
    elif data_type == "Markdown":
        return content.encode("utf-8"), "text/markdown", "document.md"
    elif data_type == "HTML":
        return content.encode("utf-8"), "text/html", "document.html"
    return b"", "", ""


def analyze_image(uploaded_file) -> str:
    client = get_genai_client()
    if client is None:
        return "No Gemini client configured. Add Google API key for image analysis."
    try:
        img_bytes = uploaded_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime = uploaded_file.type or "image/jpeg"
        prompt = "Analyze this image in detail. Describe what you see, including objects, people, text, colors, composition, and any notable details."
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type=mime)],
            config=types.GenerateContentConfig(max_output_tokens=1024, temperature=0.4),
        )
        return getattr(resp, "text", "") or "No analysis generated."
    except Exception as exc:
        return f"Analysis error: {exc}"


def export_chat(thread_id: str = None, format: str = "markdown") -> Tuple[str, str]:
    if thread_id is None:
        thread = active_thread()
    else:
        thread = st.session_state.threads.get(thread_id, active_thread())
    lines = []
    if format == "markdown":
        lines.append(f"# {thread['title']}")
        lines.append(f"*Created: {thread.get('created', '')}*")
        lines.append("")
        for m in thread["messages"]:
            role = "👤 User" if m["role"] == "user" else "🤖 NovaMind AI"
            lines.append(f"## {role}")
            lines.append(m["content"])
            lines.append("")
        return "\n".join(lines), f"novamind_{thread_id or 'chat'}.md"
    elif format == "json":
        data = {"title": thread["title"], "created": thread.get("created", ""), "messages": thread["messages"]}
        return json.dumps(data, indent=2), f"novamind_{thread_id or 'chat'}.json"
    elif format == "text":
        for m in thread["messages"]:
            lines.append(f"{m['role'].upper()}: {m['content']}")
            lines.append("")
        return "\n".join(lines), f"novamind_{thread_id or 'chat'}.txt"
    return "", ""


def search_conversations(query: str) -> List[Tuple[str, Dict]]:
    if not query.strip():
        return list(st.session_state.threads.items())
    q = query.lower()
    results = []
    for tid, thread in st.session_state.threads.items():
        if q in thread["title"].lower():
            results.append((tid, thread))
            continue
        for m in thread["messages"]:
            if q in m["content"].lower():
                results.append((tid, thread))
                break
    return results


def safe_calculate(expr: str) -> str:
    if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.\%]+", expr):
        return "Calculator only accepts numbers and math operators."
    try:
        return f"Calculator result: {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as exc:
        return f"Calculator error: {exc}"


def safe_python(code: str) -> str:
    return execute_python_code(code)


def run_tool_command(prompt: str) -> Optional[str]:
    lower = prompt.strip().lower()
    if lower.startswith("/calc "):
        return safe_calculate(prompt[6:])
    if lower.startswith("/python "):
        return safe_python(prompt[8:])
    if lower.startswith("/code "):
        return execute_python_code(prompt[6:])
    if lower.startswith("/web "):
        ctx = fetch_web_context(prompt[5:], limit=5)
        return ctx if ctx else "No web context found."
    if lower.startswith("/search "):
        results = search_duckduckgo(prompt[8:], max_results=5)
        if results:
            return "## Web Search Results\n\n" + "\n\n".join(f"### {r['title']}\n{r['body'][:300]}\n[Link]({r['href']})" for r in results if r.get("title"))
        return "No results found."
    if lower.startswith("/youtube "):
        results = search_youtube(prompt[9:], limit=5)
        if results:
            return "## YouTube Results\n\n" + "\n".join(f"- [{r['title']}]({r['link']}) by {r.get('channel', '')}" for r in results)
        return "No results found."
    if lower.startswith("/chart ") or lower.startswith("/plot "):
        parts = prompt.split("\n", 1)
        first_line = parts[0].strip()
        chart_type = first_line.split(" ", 1)[1] if " " in first_line else "Bar"
        valid_types = ["Bar", "Line", "Scatter", "Pie", "Area", "Histogram"]
        if chart_type not in valid_types:
            chart_type = "Bar"
        data_str = parts[1] if len(parts) > 1 else "x,y\n1,2\n3,4"
        chart_html = generate_chart(chart_type, data_str)
        return f"## Chart ({chart_type})\n\n```html\n{chart_html}\n```"
    return None


def build_system_prompt(persona: str, custom_system: str, use_web: bool, use_knowledge: bool) -> str:
    base = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Elite Engineer"])
    add = []
    if use_web:
        add.append("Use web context if present, cite source titles.")
    if use_knowledge:
        add.append("Use knowledge context from user documents when relevant.")
    if custom_system.strip():
        add.append(custom_system.strip())
    return "\n".join([base] + add)


def multi_provider_generate(provider: str, model: str, system_prompt: str, user_prompt: str, settings: Dict, uploaded_file) -> str:
    knowledge_context = retrieve_knowledge(user_prompt, top_k=settings["knowledge_k"]) if settings["knowledge"] else ""
    web_context = fetch_web_context(user_prompt, limit=settings["web_results"]) if settings["web"] and st.session_state.privacy["allow_web_context"] else ""
    convo = active_thread()["messages"]
    convo_text = "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in convo[-(settings["memory_turns"] * 2):])
    user_block = f"Conversation context:\n{convo_text}\n\nUser request:\n{user_prompt}"
    if knowledge_context:
        user_block += f"\n\n{knowledge_context}"
    if web_context:
        user_block += f"\n\n{web_context}"
    max_tokens = settings.get("max_tokens", 2048)
    temperature = settings.get("temperature", 0.7)

    if provider == "Gemini (Google)":
        client = get_genai_client()
        if client is None:
            return "__NOVAMIND_FALLBACK__"
        cfg = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature, top_p=settings.get("top_p", 0.9), max_output_tokens=max_tokens)
        parts = []
        if uploaded_file is not None:
            try:
                parts.append(types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type or "application/octet-stream"))
            except Exception:
                pass
        parts.append(user_block)
        final_text = ""
        try:
            if settings.get("stream", True):
                holder = st.empty()
                for chunk in client.models.generate_content_stream(model=model, contents=parts, config=cfg):
                    piece = getattr(chunk, "text", "") or ""
                    if piece:
                        final_text += piece
                        holder.markdown(final_text)
            else:
                resp = client.models.generate_content(model=model, contents=parts, config=cfg)
                final_text = getattr(resp, "text", "") or ""
        except Exception as exc:
            err = str(exc)
            if is_invalid_key_error(err):
                st.session_state.provider_health["google"] = "invalid"
                return "__NOVAMIND_FALLBACK__"
            if is_quota_error(err):
                st.session_state.provider_health["google"] = "quota"
                return "__NOVAMIND_FALLBACK__"
            st.session_state.provider_health["google"] = "error"
            return "__NOVAMIND_FALLBACK__"
        return final_text or "No response."

    elif provider == "OpenAI":
        client = get_openai_client()
        if client is None:
            return "__NOVAMIND_FALLBACK__"
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if convo_text.strip():
                messages.append({"role": "user", "content": user_block})
            else:
                messages.append({"role": "user", "content": user_prompt})
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content or "No response."
        except Exception as exc:
            st.session_state.provider_health["openai"] = "error"
            return "__NOVAMIND_FALLBACK__"

    elif provider == "Anthropic (Claude)":
        client = get_anthropic_client()
        if client is None:
            return "__NOVAMIND_FALLBACK__"
        try:
            resp = client.messages.create(model=model, system=system_prompt, messages=[{"role": "user", "content": user_block}], max_tokens=max_tokens, temperature=temperature)
            return resp.content[0].text if resp.content else "No response."
        except Exception as exc:
            st.session_state.provider_health["anthropic"] = "error"
            return "__NOVAMIND_FALLBACK__"

    return "__NOVAMIND_FALLBACK__"


def quality_score(text: str) -> int:
    score = 40
    if len(text) > 300:
        score += 15
    if len(text) > 900:
        score += 10
    if any(h in text for h in ["1.", "2.", "- "]):
        score += 12
    if "```" in text:
        score += 8
    if "summary" in text.lower() or "next steps" in text.lower():
        score += 10
    return min(100, score)


def auto_refine_if_needed(text: str, model: str, settings: Dict) -> str:
    client = get_genai_client()
    if not settings["auto_refine"] or client is None or quality_score(text) >= settings["min_quality"]:
        return text
    try:
        cfg = types.GenerateContentConfig(temperature=min(1.0, settings["temperature"] + 0.1), top_p=settings["top_p"], max_output_tokens=settings["max_tokens"])
        resp = client.models.generate_content(model=model, contents=f"Improve the response below for clarity, depth, structure, and practical usefulness.\n\nOriginal:\n{text}", config=cfg)
        improved = getattr(resp, "text", "") or ""
        return improved.strip() if improved.strip() else text
    except Exception:
        return text


def transcribe_speech(audio_bytes: bytes) -> str:
    client = get_openai_client()
    if client is None:
        return "OpenAI client not configured for speech-to-text."
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        os.unlink(tmp_path)
        return transcript.text
    except Exception as exc:
        return f"Transcription error: {exc}"


# ─── UI Panels ──────────────────────────────────────────────────────────────

def account_panel() -> None:
    st.markdown("<div class='section-title'>Account</div>", unsafe_allow_html=True)
    users = load_users()
    if st.session_state.logged_in_user:
        p = st.session_state.profile
        st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'><span style='font-size:1.5rem;'>{p.get('avatar', '🚀')}</span><div><strong>{st.session_state.logged_in_user}</strong><br><span style='color:var(--muted);font-size:0.78rem;'>{p.get('display_name', '')}</span></div></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.logged_in_user = None
            st.rerun()
        return
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign in", key="login_btn", use_container_width=True):
            if u in users and users[u]["password"] == hash_password(p):
                st.session_state.logged_in_user = u
                st.session_state.profile = users[u].get("profile", st.session_state.profile)
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        nu = st.text_input("New username", key="reg_u")
        np = st.text_input("New password", type="password", key="reg_p")
        if st.button("Create account", key="reg_btn", use_container_width=True):
            if not nu or not np:
                st.warning("Required.")
            elif nu in users:
                st.error("Exists.")
            else:
                users[nu] = {"password": hash_password(np), "profile": {"display_name": nu, "avatar": "✨", "bio": "New user"}}
                save_users(users)
                st.success("Created.")


def profile_privacy_panel() -> None:
    st.markdown("<div class='section-title'>Profile</div>", unsafe_allow_html=True)
    p = st.session_state.profile
    col1, col2 = st.columns([1, 3])
    with col1:
        avatar = st.text_input("Avatar", value=p.get("avatar", "🚀"), max_chars=2, key="profile_avatar")
    with col2:
        display_name = st.text_input("Display name", value=p.get("display_name", "Explorer"), key="profile_name")
    bio = st.text_area("Bio", value=p.get("bio", "Building with AI"), height=70, key="profile_bio")
    if st.button("💾 Save profile", key="save_profile_btn", use_container_width=True):
        st.session_state.profile = {"avatar": avatar, "display_name": display_name, "bio": bio}
        if st.session_state.logged_in_user:
            users = load_users()
            if st.session_state.logged_in_user in users:
                users[st.session_state.logged_in_user]["profile"] = st.session_state.profile
                save_users(users)
        st.success("Saved")
    st.markdown("<div class='section-title'>Privacy</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.privacy["save_history"] = st.toggle("💾 Save history", value=st.session_state.privacy["save_history"], key="privacy_save")
        st.session_state.privacy["allow_web_context"] = st.toggle("🌐 Web context", value=st.session_state.privacy["allow_web_context"], key="privacy_web")
    with col2:
        st.session_state.privacy["analytics"] = st.toggle("📊 Share analytics", value=st.session_state.privacy["analytics"], key="privacy_analytics")
        st.session_state.privacy["allow_tools"] = st.toggle("🛠️ Tools", value=st.session_state.privacy["allow_tools"], key="privacy_tools")
    st.markdown("<div class='section-title'>Usage</div>", unsafe_allow_html=True)
    u = st.session_state.usage_stats
    c1, c2, c3 = st.columns(3)
    c1.metric("💬 Chats", u["total_chats"])
    c2.metric("🖼️ Images", u["total_images"])
    c3.metric("🎵 Music", u["total_music"])
    c1.metric("🎙️ Voice", u["total_voice"])
    c2.metric("💻 Code", u["total_code_exec"])
    c3.metric("📊 Tokens", f"{u['total_tokens_est']:,}")


def settings_panel() -> None:
    st.subheader("⚙️ Settings")
    tabs = st.tabs(["🎨 Themes", "🔑 API Keys", "🧠 AI Provider", "📊 Usage", "👤 Account", "❓ Help", "💬 Feedback"])
    
    with tabs[0]:
        st.markdown("<div class='settings-card'><h4>🎨 Theme</h4><p>Pick your style and personalize the experience.</p></div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, name in enumerate(THEMES):
            t = THEMES[name]
            active = st.session_state.theme == name
            with cols[i % 3]:
                st.markdown(f"<div style='border:{'2px solid var(--accent)' if active else '1px solid var(--line)'};border-radius:14px;padding:14px;margin-bottom:12px;background:linear-gradient(135deg,{t['bg1']}d9,{t['bg0']}f2);text-align:center;'><div style='display:flex;gap:6px;justify-content:center;margin-bottom:8px;'><div style='width:18px;height:18px;border-radius:50%;background:{t['cool']};border:2px solid {t['accent']}'></div><div style='width:18px;height:18px;border-radius:50%;background:{t['bg1']};border:1px solid {t['line']}'></div><div style='width:18px;height:18px;border-radius:50%;background:{t['accent']};border:1px solid {t['hot']}'></div></div><div style='font-weight:600;color:var(--text);font-size:0.85rem;'>{name}</div>{'✓' if active else ''}</div>", unsafe_allow_html=True)
                if st.button(f"Apply", key=f"theme_btn_{i}_{name.replace(' ','_')}", use_container_width=True):
                    st.session_state.theme = name
                    st.rerun()
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            google_key = st.text_input("🔑 Google API Key", value=get_google_api_key(), type="password", key="settings_google_key")
            openai_key = st.text_input("🔑 OpenAI API Key", value=get_openai_api_key(), type="password", key="settings_openai_key")
        with col2:
            hf_key = st.text_input("🔑 HF Token", value=get_hf_token(), type="password", key="settings_hf_key")
            anthropic_key = st.text_input("🔑 Anthropic API Key", value=get_anthropic_api_key(), type="password", key="settings_anthropic_key")
        
        with st.expander("🎵 Music Settings (Optional)"):
            music_space = st.text_input("HF Music Space ID", value=get_music_space_id(), placeholder="sanchit-gandhi/musicgen-streaming", key="settings_music_space")
            music_space_fallback = st.text_input("Backup Music Space ID", value=get_music_space_fallback_id(), placeholder="optional second public Space", key="settings_music_fallback")
            music_endpoint = st.text_input("HF Music Endpoint URL", value=get_music_endpoint_url(), placeholder="https://...endpoints.huggingface.cloud", key="settings_music_endpoint")
        
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button("💾 Save All Keys", key="save_keys_btn", use_container_width=True):
                save_api_keys(google_key, hf_key, openai_key, anthropic_key, music_endpoint)
                upsert_env_value("HF_MUSIC_SPACE_ID", music_space.strip())
                upsert_env_value("HF_MUSIC_SPACE_FALLBACK_ID", music_space_fallback.strip())
                os.environ["HF_MUSIC_SPACE_ID"] = music_space.strip()
                os.environ["HF_MUSIC_SPACE_FALLBACK_ID"] = music_space_fallback.strip()
                st.session_state.secrets["music_space_id"] = normalize_music_space_id(music_space, HF_MUSIC_SPACE_ID)
                st.session_state.secrets["music_space_fallback_id"] = normalize_music_space_id(music_space_fallback, HF_MUSIC_SPACE_FALLBACK_ID)
                st.success("Keys Saved!")
        with cb:
            if st.button("🔄 Sync from .env", key="sync_env_btn", use_container_width=True):
                if auto_sync_keys_from_env():
                    st.success("Synced from .env")
                    st.rerun()
                else:
                    st.info("No new keys found")
        with cc:
            if st.button("✅ Verify Keys", key="verify_keys_btn", use_container_width=True):
                results = []
                if google_key:
                    ok, msg = verify_google_api_key()
                    results.append(("Google", ok, msg))
                if openai_key:
                    ok, msg = verify_openai_api_key()
                    results.append(("OpenAI", ok, msg))
                if anthropic_key:
                    ok, msg = verify_anthropic_api_key()
                    results.append(("Anthropic", ok, msg))
                if not results:
                    st.warning("No keys to verify")
                for name, ok, msg in results:
                    if ok:
                        st.success(f"✅ {name}: {msg}")
                    else:
                        st.error(f"❌ {name}: {msg}")
    
    with tabs[2]:
        st.markdown("<div class='settings-card'><h4>🧠 AI Provider</h4><p>Choose your primary AI engine. Fallbacks will be used automatically if the primary fails.</p></div>", unsafe_allow_html=True)
        provider = st.selectbox("Primary AI Provider", AI_PROVIDERS, index=AI_PROVIDERS.index(st.session_state.get("ai_provider", "Gemini (Google)")), key="provider_select")
        st.session_state.ai_provider = provider
        
        if provider == "Gemini (Google)":
            model = st.selectbox("Model", GEMINI_MODELS, key="provider_gemini_model")
        elif provider == "OpenAI":
            model = st.selectbox("Model", OPENAI_MODELS, key="provider_openai_model")
        elif provider == "Anthropic (Claude)":
            model = st.selectbox("Model", ANTHROPIC_MODELS, key="provider_anthropic_model")
        else:
            model = "HuggingFace"
        
        st.session_state.provider_model = model
        hf_status = "✅ Configured" if get_hf_token() else "❌ Missing Token"
        st.info(f"🔹 **{provider}** selected | HuggingFace fallback: {hf_status}")
    
    with tabs[3]:
        u = st.session_state.usage_stats
        c1, c2, c3 = st.columns(3)
        c1.metric("💬 Chats", u["total_chats"])
        c2.metric("🖼️ Images", u["total_images"])
        c3.metric("🎵 Music", u["total_music"])
        c1.metric("🎙️ Voice", u["total_voice"])
        c2.metric("💻 Code Execs", u["total_code_exec"])
        c3.metric("📊 Tokens", f"{u['total_tokens_est']:,}")
        c1.metric("📈 Total Calls", u["total_chats"] + u["total_images"] + u["total_music"] + u["total_voice"] + u["total_code_exec"])
        if st.button("🔄 Reset Stats", key="reset_stats_btn", use_container_width=True):
            st.session_state.usage_stats = {k: 0 for k in st.session_state.usage_stats}
            st.rerun()
    
    with tabs[4]:
        users = load_users()
        if st.session_state.logged_in_user:
            st.success(f"Logged in as **{st.session_state.logged_in_user}**")
            if st.button("🚪 Logout", key="settings_logout_btn", use_container_width=True):
                st.session_state.logged_in_user = None
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Sign In")
                lu = st.text_input("Username", key="settings_login_u", label_visibility="collapsed", placeholder="Username")
                lp = st.text_input("Password", type="password", key="settings_login_p", label_visibility="collapsed", placeholder="Password")
                if st.button("Sign in", key="settings_login_btn", use_container_width=True):
                    if lu in users and users[lu]["password"] == hash_password(lp):
                        st.session_state.logged_in_user = lu
                        st.session_state.profile = users[lu].get("profile", st.session_state.profile)
                        st.rerun()
                    else:
                        st.error("Invalid")
            with col2:
                st.markdown("### Register")
                nu = st.text_input("New user", key="settings_reg_u", label_visibility="collapsed", placeholder="New username")
                np_ = st.text_input("Password", type="password", key="settings_reg_p", label_visibility="collapsed", placeholder="Password")
                if st.button("Register", key="settings_reg_btn", use_container_width=True):
                    if nu and np_ and nu not in users:
                        users[nu] = {"password": hash_password(np_), "profile": {"display_name": nu, "avatar": "✨", "bio": "New"}}
                        save_users(users)
                        st.success("Created")
                    else:
                        st.warning("Invalid or exists")
    
    with tabs[5]:
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            1. **Add API Keys** in the API Keys tab (at least one provider)
            2. **Choose your AI Provider** (Gemini, OpenAI, or Claude)
            3. **Start chatting** or use the specialized modes
            4. Use **`/calc`** for math, **`/code`** for Python, **`/search`** for web
            5. Upload images for AI analysis
            6. Generate images, music, or voice from text
            """)
        with st.expander("⌨️ Keyboard & Slash Commands"):
            st.markdown("""
            - **`/calc 2+2`** - Calculator
            - **`/code print('hello')`** - Execute Python
            - **`/python ...`** - Execute Python (alias)
            - **`/web quantum computing`** - Wikipedia context
            - **`/search AI news`** - DuckDuckGo web search
            - **`/youtube music`** - YouTube search
            - **`/chart Bar`** then data rows - Generate charts
            - **`/plot Line`** then data rows - Generate line charts
            """)
        with st.expander("💡 Pro Tips"):
            st.markdown("""
            - Switch between AI providers to get different perspectives
            - Use Knowledge Base to upload documents for context
            - Auto-refine improves response quality
            - Adjust Temperature for creativity vs precision
            - Export conversations as Markdown, JSON, or TXT
            - Branch conversations to explore different ideas
            """)
    
    with tabs[6]:
        fb_type = st.selectbox("Type", ["Bug", "Feature Request", "Feedback", "Appreciation", "Suggestion"], key="fb_type")
        fb_msg = st.text_area("Message", height=100, key="fb_msg", placeholder="Share your thoughts...")
        fb_rating = st.slider("Rating", 1, 5, 4, key="fb_rating")
        if st.button("📨 Send Feedback", key="send_fb_btn", use_container_width=True):
            st.session_state.feedback_list.append({"type": fb_type, "msg": fb_msg.strip(), "rating": "⭐" * fb_rating, "time": datetime.now().strftime("%H:%M %d/%m")})
            st.success("Thanks for your feedback! 🙏")
            st.balloons()
        if st.session_state.feedback_list:
            st.markdown("### Past Feedback")
            for fb in reversed(st.session_state.feedback_list[-5:]):
                st.caption(f"**{fb['type']}** {fb['rating']} · {fb['time']}")
                st.text(fb['msg'][:100])


def dashboard_panel() -> None:
    st.subheader("📊 Dashboard")
    recent = list(st.session_state.threads.values())[-1]["title"] if st.session_state.threads else "No chats"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Overview", key="dash_overview_btn", use_container_width=True):
            st.session_state.main_view = "Dashboard"
    with col2:
        if st.button("💬 Workspace", key="dash_workspace_btn", use_container_width=True):
            st.session_state.main_view = "Workspace"
    with col3:
        if st.button("👤 Profile", key="dash_profile_btn", use_container_width=True):
            st.session_state.main_view = "Profile"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Recent Activity")
        st.markdown(f"**Last chat**: {recent}")
        u = st.session_state.usage_stats
        c1, c2 = st.columns(2)
        c1.metric("💬 Chats", u["total_chats"])
        c2.metric("🖼️ Images", u["total_images"])
        c1.metric("🎵 Music", u["total_music"])
        c2.metric("🎙️ Voice", u["total_voice"])
        
        if px:
            st.markdown("### Usage Chart")
            fig = go.Figure(data=[
                go.Bar(name="Usage", x=["Chats", "Images", "Music", "Voice", "Code"], y=[u["total_chats"], u["total_images"], u["total_music"], u["total_voice"], u["total_code_exec"]], marker_color=["#4fa3dc", "#9b4fe0", "#6ed4d4", "#f5a623", "#4fdc8a"])
            ])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eaf2fb", height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True, key="dash_usage_chart")
    
    with col2:
        st.markdown("### Provider Status")
        gs = st.session_state.provider_health.get("google", "unknown")
        os_ = st.session_state.provider_health.get("openai", "unknown")
        as_ = st.session_state.provider_health.get("anthropic", "unknown")
        st.markdown(f"- 🔵 Google: {'✅ OK' if gs == 'ok' else '⚠️ ' + gs}")
        st.markdown(f"- 🟢 OpenAI: {'✅ OK' if os_ == 'ok' else '⚠️ ' + os_}")
        st.markdown(f"- 🟣 Anthropic: {'✅ OK' if as_ == 'ok' else '⚠️ ' + as_}")
        st.markdown(f"- 🟤 HuggingFace: {'✅ Configured' if get_hf_token() else '⚠️ No Token'}")
        
        st.markdown("### Quick Actions")
        if st.button("💬 New Chat", key="dash_new_chat", use_container_width=True):
            nid = str(int(time.time()))
            st.session_state.threads[nid] = {"title": "New conversation", "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": [], "branch": "main"}
            st.session_state.active_thread_id = nid
            st.rerun()
        if st.button("🗑️ Clear All Chats", key="dash_clear_chats", use_container_width=True):
            nid = str(int(time.time()))
            st.session_state.threads = {nid: {"title": "New conversation", "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": [], "branch": "main"}}
            st.session_state.active_thread_id = nid
            st.rerun()
        
        st.markdown(f"### System Info")
        st.markdown(f"- **Version**: {APP_VERSION}")
        st.markdown(f"- **Location**: {get_user_location()}")
        st.markdown(f"- **Theme**: {st.session_state.theme}")
        st.markdown(f"- **Provider**: {st.session_state.get('ai_provider', 'Gemini')}")
        st.markdown(f"- **Threads**: {len(st.session_state.threads)}")


def knowledge_panel() -> None:
    st.markdown("<div class='section-title'>Knowledge Base</div>", unsafe_allow_html=True)
    files = st.file_uploader("Upload PDF/TXT/MD", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed", key="kb_uploader")
    if st.button("📥 Ingest Documents", key="kb_ingest_btn", use_container_width=True):
        if files:
            d, c = ingest_knowledge(files)
            st.success(f"✅ {d} docs, {c} chunks added")
        else:
            st.warning("Upload files first")
    st.caption(f"📚 {len(st.session_state.knowledge_docs)} docs, {len(st.session_state.knowledge_chunks)} chunks")
    if st.session_state.knowledge_docs:
        st.markdown("### Loaded Documents")
        for doc in st.session_state.knowledge_docs:
            st.markdown(f"- 📄 {doc}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All", key="kb_clear_btn", use_container_width=True):
            st.session_state.knowledge_docs = []
            st.session_state.knowledge_chunks = []
            st.success("Cleared")
    with col2:
        if st.session_state.knowledge_chunks:
            st.caption(f"Last chunk: {st.session_state.knowledge_chunks[-1]['text'][:50]}...")


def challenge_mode() -> None:
    challenges = [
        "Explain a complex topic in 100 words.",
        "Design a mini product with pricing & MVP.",
        "Optimize a prompt for coding output.",
        "Summarize an article in 5 bullets and 3 actions.",
        "Write a haiku about artificial intelligence.",
        "Create a 30-day learning plan for any skill.",
        "Design a REST API for a todo app.",
        "Write a short story (50 words) about the future.",
        "Explain quantum computing to a 10-year-old.",
        "Plan a weekend project with AI tools.",
    ]
    today = challenges[date.today().toordinal() % len(challenges)]
    st.info(f"📋 **Today's Challenge**: {today}")
    streak = st.session_state.challenge["streak"]
    st.markdown(f"🔥 **Streak**: {streak} days")
    ans = st.text_area("Your answer", height=120, key="challenge_ans", placeholder="Write your response...")
    if st.button("✅ Submit Challenge", key="challenge_submit_btn", use_container_width=True):
        if ans.strip():
            st.session_state.challenge["streak"] += 1
            st.session_state.challenge["last_day"] = str(date.today())
            st.session_state.challenge["completed"].append({"date": str(date.today()), "challenge": today, "answer": ans[:100]})
            st.success(f"Submitted! 🔥 Streak: {st.session_state.challenge['streak']}")
            st.balloons()
        else:
            st.warning("Write something first!")
    if st.session_state.challenge["completed"]:
        with st.expander("📜 Challenge History"):
            for c in reversed(st.session_state.challenge["completed"][-10:]):
                st.caption(f"**{c['date']}**: {c['challenge'][:50]}...")


def code_interpreter_mode() -> None:
    st.subheader("💻 Code Interpreter")
    st.markdown("<span class='chip'>Python 3</span><span class='chip'>Safe Sandbox</span><span class='chip'>No Imports</span>", unsafe_allow_html=True)
    st.info("Write and execute Python code in a safe sandbox. Built-in functions only (print, len, range, sum, etc.). No file/network access.")
    
    code = st.text_area("Python Code", height=200, key="code_interp_input", placeholder="# Write Python code here\nprint('Hello, NovaMind!')")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Code", key="code_run_btn", use_container_width=True, type="primary"):
            if code.strip():
                with st.spinner("Executing..."):
                    result = execute_python_code(code)
                    st.session_state.code_output = result
                    st.session_state.usage_stats["total_code_exec"] += 1
            else:
                st.warning("Write code first!")
    with col2:
        if st.button("🔄 Clear Output", key="code_clear_btn", use_container_width=True):
            st.session_state.code_output = ""
    
    if st.session_state.code_output:
        st.markdown("### Output")
        st.code(st.session_state.code_output, language="text")
    
    with st.expander("📝 Examples"):
        st.code("""# Calculate Fibonacci
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a+b
fib(10)""", language="python")
        st.code("""# Simple data analysis
data = [23, 45, 67, 12, 89, 34, 56]
print(f"Sum: {sum(data)}")
print(f"Min: {min(data)}")
print(f"Max: {max(data)}")
print(f"Count: {len(data)}")
print(f"Sorted: {sorted(data)}")""", language="python")


def try_hf_image(prompt: str, negative: str, steps: int, guidance: float, seed: int) -> Tuple[bool, bytes, str]:
    models_to_try = [HF_IMAGE_MODEL, HF_IMAGE_FALLBACK, HF_IMAGE_THIRD]
    last_error = ""
    for model_id in models_to_try:
        ok, data, err = generate_image_via_hf_client(
            prompt=prompt,
            model_id=model_id,
            negative_prompt=negative,
            steps=steps,
            guidance=guidance,
        )
        if ok:
            return True, data, ""
        last_error = err or last_error
    return False, b"", f"Image generation failed for {', '.join(models_to_try)}. {last_error or 'Check token permissions and model availability in Inference Providers.'}"


def image_mode() -> None:
    st.subheader("🎨 Image Studio")
    st.markdown("<span class='chip'>FLUX.1</span><span class='chip'>SD XL</span><span class='chip'>Multi-Model</span><span class='chip green'>AI Analysis</span>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🎨 Generate", "🖼️ Gallery", "🔍 Analyze Image"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            prompt = st.text_area("Prompt", height=120, key="img_prompt", placeholder="A futuristic museum floating over ocean cliffs, cinematic lighting")
            negative = st.text_input("Negative prompt", key="img_negative", placeholder="blurry, low quality, distorted")
        with c2:
            style = st.selectbox("Style", ["None", "Photoreal", "Anime", "Concept Art", "Cinematic", "Fantasy Art", "Oil Painting", "Pixel Art", "Cyberpunk", "Watercolor"], key="img_style")
            steps = st.slider("Steps", 15, 60, 35, key="img_steps")
            guidance = st.slider("Guidance", 3.0, 12.0, 7.5, 0.5, key="img_guidance")
        style_map = {
            "Photoreal": "photorealistic, 8K, ultra detailed, realistic lighting",
            "Anime": "anime style, vibrant colors, cel shading",
            "Concept Art": "concept art, cinematic, dramatic lighting",
            "Cinematic": "cinematic, anamorphic, film grain, movie scene",
            "Fantasy Art": "fantasy, magical, ethereal, otherworldly",
            "Oil Painting": "oil painting style, thick brushstrokes, canvas texture",
            "Pixel Art": "pixel art, 8-bit, retro game style",
            "Cyberpunk": "cyberpunk, neon lights, futuristic city, rain",
            "Watercolor": "watercolor painting, soft, flowing colors",
        }
        gen_count = st.selectbox("Count", [1, 2, 3, 4], index=0, key="img_count")
        seed = st.number_input("Seed (-1 = random)", min_value=-1, max_value=999999, value=-1, key="img_seed")
        
        if st.button("✨ Generate Images", key="img_generate_btn", use_container_width=True, type="primary"):
            if not prompt.strip():
                return st.warning("Enter a prompt")
            fp = prompt.strip() + (f", {style_map[style]}" if style != "None" else "")
            neg = negative.strip() or "blurry, low quality, ugly, deformed"
            progress = st.progress(0, text="Generating...")
            count = 0
            for i in range(gen_count):
                progress.progress(i / gen_count, text=f"Image {i+1}/{gen_count}")
                ok, data, err = try_hf_image(fp, neg, steps, guidance, seed)
                if ok:
                    st.session_state.generated_images.append({"data": data, "prompt": fp[:60], "time": datetime.now().strftime("%H:%M"), "style": style})
                    st.session_state.generated_image = data
                    count += 1
                    st.session_state.usage_stats["total_images"] += 1
                    st.image(data, use_container_width=True, caption=f"#{i+1} · {style}")
                else:
                    st.error(f"#{i+1} failed: {err}")
            progress.progress(1.0, text="Done!")
            if count:
                st.success(f"✅ {count}/{gen_count} generated")
        if st.session_state.generated_image:
            st.download_button("💾 Download Latest", data=st.session_state.generated_image, file_name="novamind_image.png", mime="image/png", use_container_width=True, key="dl_latest_img")
    
    with tab2:
        if st.session_state.generated_images:
            cols = st.columns(2)
            for idx, img in enumerate(reversed(st.session_state.generated_images[-10:])):
                with cols[idx % 2]:
                    st.image(img["data"], use_container_width=True, caption=f"{img['prompt']} · {img['time']}")
                    st.download_button("💾 Download", data=img["data"], file_name=f"novamind_{img['time'].replace(':','')}.png", mime="image/png", use_container_width=True, key=f"dl_gallery_{uuid.uuid4().hex[:8]}")
            if st.button("🗑️ Clear Gallery", key="img_clear_gallery_btn", use_container_width=True):
                st.session_state.generated_images = []
                st.session_state.generated_image = None
                st.rerun()
        else:
            st.info("No images generated yet. Try the Generate tab!")
    
    with tab3:
        st.markdown("### 🔍 AI Image Analysis")
        uploaded_img = st.file_uploader("Upload an image for AI analysis", type=["jpg", "jpeg", "png", "webp", "gif"], key="img_analyzer")
        if uploaded_img:
            st.image(uploaded_img, use_container_width=True, caption="Uploaded Image")
            if st.button("🔍 Analyze This Image", key="img_analyze_btn", use_container_width=True, type="primary"):
                with st.spinner("Analyzing image with Gemini Vision..."):
                    analysis = analyze_image(uploaded_img)
                    st.markdown("### Analysis Result")
                    st.markdown(analysis)
                    st.session_state.usage_stats["total_chats"] += 1


def music_mode() -> None:
    st.subheader("🎵 Music Lab")
    st.markdown("<span class='chip'>MusicGen</span><span class='chip'>Multi-Genre</span><span class='chip purple'>AI Composer</span>", unsafe_allow_html=True)
    st.caption(f"🎹 Primary: {get_music_space_id()} | Backup: {get_music_space_fallback_id() or 'not set'}")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        prompt = st.text_area("Description", height=120, key="music_prompt", placeholder="Epic orchestral build with dramatic percussion and choir")
    with c2:
        duration = st.slider("Duration (sec)", 5, 30, 15, key="music_duration")
        genre = st.selectbox("Genre", ["Custom", "EDM", "Lo-fi", "Orchestral", "Pop", "Ambient", "Rock", "Jazz", "Cinematic", "Hip-Hop", "Classical", "Synthwave", "Blues", "Reggae", "Folk"], key="music_genre")
        mood = st.selectbox("Mood", ["Neutral", "Energetic", "Calm", "Dark", "Happy", "Melancholic", "Epic", "Mysterious", "Romantic", "Aggressive"], key="music_mood")
    with c3:
        tempo = st.selectbox("Tempo", ["Slow", "Medium", "Fast", "Very Fast"], key="music_tempo")
        quality = st.selectbox("Quality", ["Draft", "Standard", "High"], index=1, key="music_quality")
    helpers = {"EDM": "upbeat electronic dance music with drops", "Lo-fi": "lo-fi hip-hop beat, warm vinyl crackle", "Orchestral": "cinematic orchestral with strings and brass", "Pop": "modern pop with catchy melody", "Ambient": "ambient atmospheric soundscape", "Rock": "energetic rock with guitar riffs", "Jazz": "smooth jazz with saxophone and piano", "Cinematic": "epic cinematic with full orchestra", "Hip-Hop": "hip-hop beat with 808s", "Classical": "classical composition with piano", "Synthwave": "synthwave retro 80s style", "Blues": "slow blues guitar", "Reggae": "reggae rhythm with skank guitar", "Folk": "acoustic folk with guitar"}
    if st.button("🎵 Generate Music", key="music_generate_btn", use_container_width=True, type="primary"):
        merged = prompt.strip()
        if genre != "Custom":
            merged = f"{helpers.get(genre, '')}. {merged}" if merged else helpers.get(genre, "")
        if not merged:
            return st.warning("Enter a prompt or select a genre")
        dur = int(duration * (1.3 if quality == "High" else 0.7 if quality == "Draft" else 1))
        space_ids = [get_music_space_id(), get_music_space_fallback_id()]
        with st.spinner("Generating music... This may take a minute."):
            ok, data, err, used_space, audio_mime = generate_music_via_space(merged, space_ids, audio_length_in_s=dur, play_steps_in_s=1.5, seed=5, timeout_s=240)
            if not ok and get_music_endpoint_url():
                ok, data, err = generate_music_via_endpoint(merged, get_music_endpoint_url(), max_new_tokens=dur * 50, timeout_s=240)
                if ok:
                    used_space = "HF endpoint"
                    audio_mime = "audio/wav"
            if ok:
                st.session_state.generated_audio = data
                st.session_state.generated_audio_mime = audio_mime or "audio/wav"
                st.session_state.music_history.insert(0, {"audio": data, "mime": st.session_state.generated_audio_mime, "prompt": merged[:80], "time": datetime.now().strftime("%H:%M"), "duration": dur, "space": used_space or get_music_space_id()})
                st.session_state.music_history = st.session_state.music_history[:12]
                st.session_state.usage_stats["total_music"] += 1
                st.success("✅ Music generated!")
                st.balloons()
            else:
                st.error(f"Failed: {err[:200]}")
    if st.session_state.generated_audio:
        st.audio(st.session_state.generated_audio, format=st.session_state.generated_audio_mime)
        st.download_button("💾 Download Music", data=st.session_state.generated_audio, file_name="novamind_music.wav", mime=st.session_state.generated_audio_mime, use_container_width=True, key="dl_music_latest")
    if st.session_state.music_history:
        st.markdown("### Recent Music")
        for idx, item in enumerate(st.session_state.music_history[:3]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"🎵 **{item['time']}** · {item['duration']}s · {item['space']} · _{item['prompt'][:50]}..._")
                with col2:
                    st.audio(item["audio"], format=item.get("mime", "audio/wav"))
                    st.download_button(f"DL #{idx+1}", data=item["audio"], file_name=f"novamind_music_{idx+1}.wav", mime=item.get("mime", "audio/wav"), key=f"music_hist_{uuid.uuid4().hex[:8]}")
        if st.button("🗑️ Clear Music History", key="music_clear_btn", use_container_width=True):
            st.session_state.music_history = []
            st.rerun()


def voice_mode() -> None:
    st.subheader("🎙️ Voice Studio")
    st.markdown("<span class='chip'>Edge TTS</span><span class='chip'>Multi-Language</span><span class='chip green'>Natural Voices</span>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔊 Text to Speech", "🎤 Speech to Text"])
    
    with tab1:
        text = st.text_area("Text to speak", height=140, key="tts_text", placeholder="Enter text to convert to speech...")
        c1, c2, c3 = st.columns(3)
        with c1:
            lang = st.selectbox("Language", ["English", "Spanish", "French", "German", "Italian", "Japanese", "Chinese", "Portuguese", "Russian", "Korean", "Arabic", "Hindi", "Dutch", "Polish", "Turkish"], key="tts_lang")
        with c2:
            gender = st.selectbox("Voice", ["Female", "Male"], key="tts_gender")
        with c3:
            speed = st.selectbox("Speed", ["Slow", "Normal", "Fast", "Very Fast"], key="tts_speed")
        vm = {("English", "F"): "en-US-AriaNeural", ("English", "M"): "en-US-GuyNeural", ("Spanish", "F"): "es-ES-ElviraNeural", ("Spanish", "M"): "es-ES-AlvaroNeural",
              ("French", "F"): "fr-FR-DeniseNeural", ("French", "M"): "fr-FR-HenriNeural", ("German", "F"): "de-DE-KatjaNeural", ("German", "M"): "de-DE-ConradNeural",
              ("Italian", "F"): "it-IT-ElsaNeural", ("Italian", "M"): "it-IT-DiegoNeural", ("Japanese", "F"): "ja-JP-NanamiNeural", ("Japanese", "M"): "ja-JP-KeitaNeural",
              ("Chinese", "F"): "zh-CN-XiaoxiaoNeural", ("Chinese", "M"): "zh-CN-YunxiNeural", ("Portuguese", "F"): "pt-BR-FranciscaNeural", ("Portuguese", "M"): "pt-BR-AntonioNeural",
              ("Russian", "F"): "ru-RU-SvetlanaNeural", ("Russian", "M"): "ru-RU-DmitryNeural", ("Korean", "F"): "ko-KR-SunHiNeural", ("Korean", "M"): "ko-KR-InJoonNeural",
              ("Arabic", "F"): "ar-SA-ZariyahNeural", ("Arabic", "M"): "ar-SA-HamedNeural", ("Hindi", "F"): "hi-IN-SwaraNeural", ("Hindi", "M"): "hi-IN-MadhurNeural",
              ("Dutch", "F"): "nl-NL-FennaNeural", ("Dutch", "M"): "nl-NL-MaartenNeural", ("Polish", "F"): "pl-PL-AgnieszkaNeural", ("Polish", "M"): "pl-PL-MarekNeural",
              ("Turkish", "F"): "tr-TR-EmelNeural", ("Turkish", "M"): "tr-TR-AhmetNeural"}
        rm = {"Slow": "-30%", "Normal": "+0%", "Fast": "+25%", "Very Fast": "+50%"}
        if st.button("🔊 Generate Speech", key="tts_generate_btn", use_container_width=True, type="primary"):
            if not text.strip():
                return st.warning("Enter text")
            try:
                if edge_tts is None:
                    st.error("edge_tts not installed")
                    return
                async def synth():
                    v = vm.get((lang, gender[0]), "en-US-AriaNeural")
                    c = edge_tts.Communicate(text=text, voice=v, rate=rm[speed])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
                        p = t.name
                    await c.save(p)
                    return p
                with st.spinner("Generating speech..."):
                    out = asyncio.run(synth())
                    audio = Path(out).read_bytes()
                    st.audio(audio)
                    st.session_state.usage_stats["total_voice"] += 1
                    st.download_button("💾 Download Audio", data=audio, file_name="novamind_tts.mp3", mime="audio/mp3", use_container_width=True, key="dl_tts")
                    Path(out).unlink(missing_ok=True)
            except Exception as e:
                st.error(f"TTS error: {e}")
    
    with tab2:
        st.markdown("### 🎤 Speech to Text")
        st.info("Upload an audio file to transcribe it using OpenAI Whisper.")
        audio_file = st.file_uploader("Upload audio (WAV, MP3, M4A, etc.)", type=["wav", "mp3", "m4a", "ogg", "flac"], key="stt_uploader")
        if audio_file:
            st.audio(audio_file)
            if st.button("📝 Transcribe", key="stt_transcribe_btn", use_container_width=True, type="primary"):
                with st.spinner("Transcribing..."):
                    try:
                        client = get_openai_client()
                        if client is None:
                            st.error("OpenAI API key needed for transcription")
                        else:
                            suffix = f".{audio_file.name.split('.')[-1]}"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(audio_file.getvalue())
                                tmp_path = tmp.name
                            with open(tmp_path, "rb") as f:
                                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                            os.unlink(tmp_path)
                            st.success("Transcription complete!")
                            st.markdown(f"**Transcribed Text:**\n{transcript.text}")
                            if st.button("📋 Use as Chat Input", key="stt_use_chat_btn"):
                                st.session_state.prefill = transcript.text
                                st.session_state.main_view = "Workspace"
                                st.rerun()
                    except Exception as e:
                        st.error(f"Transcription error: {e}")


def chat_mode() -> None:
    thread = active_thread()
    
    # Show conversation search bar
    search_query = st.text_input("🔍 Search conversations...", key="conv_search", placeholder="Search messages or titles...", label_visibility="collapsed")
    if search_query:
        results = search_conversations(search_query)
        if len(results) > 0:
            st.caption(f"Found {len(results)} matching conversations")
            for tid, thr in results[:5]:
                if st.button(f"📄 {thr['title'][:40]}...", key=f"search_result_{tid}", use_container_width=True):
                    st.session_state.active_thread_id = tid
                    st.rerun()
    
    # Show messages
    for m in thread["messages"]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            content = m["content"]
            # Check for chart HTML
            if "```html\n<div>" in content or "```html\n<!" in content:
                parts = content.split("```html\n")
                for part in parts:
                    if part.startswith("<div>") or part.startswith("<!DOCTYPE") or part.startswith("<script"):
                        html_end = part.find("```")
                        html_content = part[:html_end] if html_end > 0 else part
                        st.components.v1.html(html_content, height=400, scrolling=True)
                        continue
                    elif "```" in part:
                        text_part = part.split("```")[0].strip()
                        if text_part:
                            st.markdown(text_part)
                    else:
                        st.markdown(part)
            else:
                st.markdown(content)
    
    # Welcome suggestions for new conversations
    if not thread["messages"]:
        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
        cols = st.columns(2)
        quick_prompts = [
            ("📋 Strategy Pack", "Create a strategy pack for my business idea: market analysis, positioning, monetization model, risk assessment, and a 12-month timeline with key milestones."),
            ("🏗️ Build Spec", "Design a complete product roadmap including UX research, system architecture, API design, development sprints, testing strategy, and launch milestones."),
            ("🧪 Code Review", "Review this code for bugs, security vulnerabilities, performance bottlenecks, and suggest improvements with concrete examples."),
            ("📊 Data Analysis", "Help me analyze a dataset. I want to understand patterns, create visualizations, and derive actionable insights."),
            ("🚀 Startup Guide", "I want to start a business in AI/tech. Give me a complete guide covering idea validation, MVP building, funding, marketing, and growth."),
            ("🔬 Research Deep Dive", "Help me research a complex topic. Provide academic-quality analysis with citations, counterarguments, and synthesis."),
        ]
        for i, (label, prompt) in enumerate(quick_prompts):
            with cols[i % 2]:
                if st.button(label, key=f"qp_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    prefill = st.session_state.pop("prefill", None)
    prompt = prefill or st.chat_input("Ask anything... (try /search, /code, /chart)")
    if not prompt:
        return
    
    # Save to history
    st.session_state.search_history.insert(0, {"text": prompt, "at": datetime.now().strftime("%H:%M")})
    st.session_state.search_history = st.session_state.search_history[:60]
    save_search_history(st.session_state.search_history)
    
    if not st.session_state.privacy["save_history"]:
        active_thread()["messages"] = []
    
    thread["messages"].append({"role": "user", "content": prompt})
    
    # Run tool commands first
    if st.session_state.privacy["allow_tools"]:
        if tr := run_tool_command(prompt):
            thread["messages"].append({"role": "assistant", "content": tr})
            st.rerun()
            return
    
    settings = st.session_state.settings
    sp = build_system_prompt(settings["persona"], settings["custom_system"], settings["web"], settings["knowledge"])
    st.session_state.usage_stats["total_chats"] += 1
    
    # Show typing indicator
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("<div class='typing-dots'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
        
        provider = st.session_state.get("ai_provider", "Gemini (Google)")
        pmodel = settings.get("provider_model", settings.get("model", "gemini-2.0-flash"))
        
        resp = multi_provider_generate(provider, pmodel, sp, prompt, settings, settings.get("uploaded_file"))
        if resp == "__NOVAMIND_FALLBACK__":
            resp = smart_fallback_response(prompt, sp, settings)
        resp = auto_refine_if_needed(resp, pmodel, settings)
        
        typing_placeholder.empty()
    
    thread["messages"].append({"role": "assistant", "content": resp})
    st.session_state.usage_stats["total_tokens_est"] += estimate_tokens(thread["messages"][-2:])
    
    # Auto-title
    if len(thread["messages"]) == 2 and thread["title"] == "New conversation":
        thread["title"] = prompt.strip()[:50] + ("..." if len(prompt.strip()) > 50 else "")
    
    st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

init_state()
auto_sync_keys_from_env()
st.markdown(build_theme_css(st.session_state.theme), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div class='brand-mark'><div class='star-logo'><svg viewBox="0 0 24 24" fill="white"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg><div class="star-sparkle"></div><div class="star-sparkle"></div><div class="star-sparkle"></div></div><div><strong style="font-family:Sora,sans-serif">NovaMind AI</strong><div style='color:var(--muted);font-size:0.76rem;'>Studio v{APP_VERSION}</div></div></div>""", unsafe_allow_html=True)
    
    account_panel()
    
    st.markdown("<div class='section-title'>Quick Nav</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-rail'>", unsafe_allow_html=True)
    il, ir = st.columns(2)
    with il:
        st.markdown(f"<div style='color:var(--cool);margin-bottom:0.15rem;'>{svg_icon('profile', 15)}</div>", unsafe_allow_html=True)
        st.button("👤 Profile", use_container_width=True, key="sidebar_profile_btn")
        st.markdown(f"<div style='color:var(--cool);margin-top:0.45rem;margin-bottom:0.15rem;'>{svg_icon('dashboard', 15)}</div>", unsafe_allow_html=True)
        st.button("📊 Dashboard", use_container_width=True, key="sidebar_dash_btn")
    with ir:
        st.markdown(f"<div style='color:var(--cool);margin-bottom:0.15rem;'>{svg_icon('settings', 15)}</div>", unsafe_allow_html=True)
        st.button("⚙️ Settings", use_container_width=True, key="sidebar_settings_btn")
        st.markdown(f"<div style='color:var(--cool);margin-top:0.45rem;margin-bottom:0.15rem;'>{svg_icon('workspace', 15)}</div>", unsafe_allow_html=True)
        st.button("💬 Workspace", use_container_width=True, key="sidebar_workspace_btn")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>Mode</div>", unsafe_allow_html=True)
    st.session_state.mode = st.radio("Mode", CHAT_MODES, index=CHAT_MODES.index(normalize_mode(st.session_state.mode)), label_visibility="collapsed", key="mode_radio")
    
    st.markdown("<div class='section-title'>Chats</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            nid = str(int(time.time()))
            st.session_state.threads[nid] = {"title": "New conversation", "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": [], "branch": "main"}
            st.session_state.active_thread_id = nid
            st.rerun()
    with col2:
        if st.button("📤 Export", use_container_width=True, key="export_btn"):
            fmt = st.session_state.get("export_format", "markdown")
            content, filename = export_chat(format=fmt)
            st.download_button(f"📥 Download {fmt.upper()}", data=content, file_name=filename, key="export_download_btn", use_container_width=True)
    
    tids = list(st.session_state.threads.keys())
    selected = st.selectbox("Threads", options=tids, index=tids.index(st.session_state.active_thread_id), format_func=lambda t: st.session_state.threads[t]["title"][:35], label_visibility="collapsed", key="thread_selector")
    st.session_state.active_thread_id = selected
    
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)
    provider = st.session_state.get("ai_provider", "Gemini (Google)")
    provider_models_map = {
        "Gemini (Google)": GEMINI_MODELS,
        "OpenAI": OPENAI_MODELS,
        "Anthropic (Claude)": ANTHROPIC_MODELS,
    }
    default_models = provider_models_map.get(provider, GEMINI_MODELS)
    
    st.session_state.settings = {
        "model": st.selectbox("Model", GEMINI_MODELS, key="settings_model", label_visibility="collapsed"),
        "provider_model": st.selectbox("Provider Model", default_models, key="settings_provider_model", label_visibility="collapsed"),
        "persona": st.selectbox("Persona", list(PERSONA_PROMPTS.keys()), key="settings_persona", label_visibility="collapsed"),
        "temperature": st.slider("Temp", 0.0, 1.5, 0.7, 0.05, key="settings_temp", label_visibility="collapsed"),
        "top_p": st.slider("Top-p", 0.1, 1.0, 0.9, 0.05, key="settings_topp", label_visibility="collapsed"),
        "max_tokens": st.slider("Max Tokens", 256, 8192, 2048, 64, key="settings_maxtokens", label_visibility="collapsed"),
        "memory_turns": st.slider("Memory Turns", 2, 30, 10, key="settings_memory", label_visibility="collapsed"),
        "stream": st.toggle("Streaming", value=True, key="settings_stream"),
        "web": st.toggle("Web Context", value=True, key="settings_web"),
        "web_results": st.slider("Web Snippets", 1, 8, 3, key="settings_webresults"),
        "knowledge": st.toggle("Knowledge Base", value=True, key="settings_knowledge"),
        "knowledge_k": st.slider("Knowledge Chunks", 1, 10, 4, key="settings_knowledgek"),
        "auto_refine": st.toggle("Auto-Refine", value=True, key="settings_refine"),
        "min_quality": st.slider("Min Quality", 40, 95, 68, key="settings_quality"),
        "custom_system": st.text_area("Custom Instructions", height=60, value="", key="settings_custom_sys", placeholder="Custom system prompt..."),
        "uploaded_file": None,
    }
    
    st.markdown("<div class='section-title'>Status</div>", unsafe_allow_html=True)
    gs = st.session_state.provider_health.get("google", "unknown")
    os_ = st.session_state.provider_health.get("openai", "unknown")
    as_ = st.session_state.provider_health.get("anthropic", "unknown")
    gs_class = "ok" if gs == "ok" else "warn"
    os_class = "ok" if os_ == "ok" else "warn"
    as_class = "ok" if as_ == "ok" else "warn"
    hf_class = "ok" if get_hf_token() else "warn"
    st.markdown(f"<span class='status-pill {gs_class}'>Google {'✅' if gs=='ok' else '?'}</span><span class='status-pill {os_class}'>OpenAI {'✅' if os_=='ok' else '?'}</span><span class='status-pill {as_class}'>Claude {'✅' if as_=='ok' else '?'}</span><span class='status-pill {hf_class}'>HF {'✅' if get_hf_token() else '?'}</span>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
    fmt = st.selectbox("Format", ["markdown", "json", "text"], key="export_fmt", label_visibility="collapsed")
    st.session_state.export_format = fmt
    content, filename = export_chat(format=fmt)
    st.download_button(f"📥 Download {fmt.upper()}", data=content, file_name=filename, key="sidebar_export_dl", use_container_width=True)
    
    st.markdown("<div class='sidebar-footer'>", unsafe_allow_html=True)
    loc = get_user_location()
    st.markdown(f"<div class='location-badge'><span class='loc-icon'>{svg_icon('location', 14)}</span><span>{loc}</span><span style='margin-left:auto;font-size:0.6rem;color:var(--cool);'>●</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content area
thread = active_thread()
if "boot_screen_shown" not in st.session_state:
    st.session_state.boot_screen_shown = True
    st.markdown("<div class='splash-shell'><div class='splash-inner'><div class='splash-wave'><svg viewBox='0 0 24 24' fill='white'><path d='M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z'/></svg></div><div><p class='splash-title'>🚀 NovaMind AI Studio v{APP_VERSION}</p><div class='splash-sub'>Multi-Provider AI · Image · Music · Voice · Code · Web</div></div></div></div>", unsafe_allow_html=True)

st.markdown(f"<div class='hero'><h1>🚀 NovaMind AI Studio</h1><p>Multi-provider AI workspace with chat, image, music, voice, code, and tools. <span class='chip'>{st.session_state.get('ai_provider', 'Gemini')}</span><span class='chip green'>{st.session_state.mode}</span><span class='chip purple'>v{APP_VERSION}</span></p></div>", unsafe_allow_html=True)

# Top nav bar
c1, c2, c3, c4 = st.columns(4)
c1.button("📊 Dashboard", use_container_width=True, on_click=lambda: setattr(st.session_state, 'main_view', 'Dashboard'), key="topnav_dash")
c2.button("💬 Workspace", use_container_width=True, on_click=lambda: setattr(st.session_state, 'main_view', 'Workspace'), key="topnav_workspace")
c3.button("👤 Profile", use_container_width=True, on_click=lambda: setattr(st.session_state, 'main_view', 'Profile'), key="topnav_profile")
c4.button("⚙️ Settings", use_container_width=True, on_click=lambda: setattr(st.session_state, 'main_view', 'Settings'), key="topnav_settings")

# Metrics bar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🎯 Mode", st.session_state.mode)
m2.metric("💬 Messages", len(thread["messages"]))
m3.metric("📊 Tokens", estimate_tokens(thread["messages"]))
m4.metric("🤖 Provider", st.session_state.get('ai_provider', 'Gemini').split()[0])
m5.metric("📌 Version", APP_VERSION)

# Main routing
v = st.session_state.main_view
if v == "Dashboard":
    dashboard_panel()
elif v == "Profile":
    profile_privacy_panel()
elif v == "Settings":
    settings_panel()
else:
    m = st.session_state.mode
    if m == "Chat":
        chat_mode()
    elif m == "Image Studio":
        image_mode()
    elif m == "Music Lab":
        music_mode()
    elif m == "Voice Studio":
        voice_mode()
    elif m == "Code Interpreter":
        code_interpreter_mode()
    else:
        challenge_mode()

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"NovaMind AI Studio v{APP_VERSION}")
with col2:
    st.caption(f"Provider: {st.session_state.get('ai_provider', 'Gemini')} | Theme: {st.session_state.theme}")
with col3:
    st.caption("Powered by Gemini + OpenAI + Anthropic + HuggingFace + Edge TTS")