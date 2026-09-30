import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import shap
from collections import defaultdict
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from fpdf import FPDF
import tempfile, os

import base64
import io


def get_logo_base64():
    # Baca logo (png atau jpg) dan encode ke base64. Kembalikan '' kalau tak ada.
    for fname in ('logo_kemenkes_trace.png', 'logo_rsmc.png', 'logo_rsmc.jpg'):
        try:
            with open(fname, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            continue
    return ''


LOGO_B64 = get_logo_base64()

# ================================
# CONFIG
# ================================
st.set_page_config(
    page_title="Dashboard ABSA Layanan Kesehatan",
    page_icon="🏥",
    layout="wide"
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* --- GLOBAL APP STYLING --- */
.stApp {
    background: linear-gradient(180deg, #F0F7F7 0%, #E8F3F3 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0F172A;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* --- SIDEBAR CUSTOMIZATION --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8F4F4 0%, #F4FBFB 100%) !important;
    border-right: 1px solid rgba(0, 166, 164, 0.18) !important;
    padding-top: 1rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 6px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] > div {
    width: 100% !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #0A2540 !important;
    background: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid rgba(0, 174, 172, 0.15) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(0, 174, 172, 0.12) !important;
    border-color: rgba(0, 174, 172, 0.35) !important;
    transform: translateX(3px);
}

[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    width: 100% !important;
    box-sizing: border-box !important;
    background: linear-gradient(135deg, rgba(0, 174, 172, 0.2) 0%, rgba(10, 37, 64, 0.12) 100%) !important;
    color: #0A2540 !important;
    font-weight: 800 !important;
    border: 1px solid #00AEAC !important;
    box-shadow: 0 4px 12px rgba(0, 174, 172, 0.15) !important;
}

/* --- HERO BANNER (LIGHT THEME) --- */
.hero-banner {
    background: linear-gradient(120deg, #F4FBFB 0%, #EAF7F7 50%, #DCF3F4 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    border: 1px solid rgba(0, 169, 157, 0.15);
    box-shadow: 0 4px 20px rgba(0, 169, 157, 0.06);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: #000000;
    margin: 0 0 6px 0;
    line-height: 1.25;
    letter-spacing: -0.3px;
}

.hero-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.6rem;
    color: #00A99D;
    margin: 0;
    line-height: 1.4;
    font-weight: 700;
}

/* --- METRIC CARDS --- */
.metric-card {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 174, 172, 0.18);
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px -2px rgba(10, 37, 64, 0.05);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px -6px rgba(0, 174, 172, 0.18);
    border-color: rgba(0, 174, 172, 0.4);
}

.metric-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
}

.metric-card.total::after { background: linear-gradient(90deg, #00AEAC, #0A2540); }
.metric-card.positive::after { background: linear-gradient(90deg, #059669, #34D399); }
.metric-card.negative::after { background: linear-gradient(90deg, #E11D48, #FB7185); }
.metric-card.neutral::after { background: linear-gradient(90deg, #D97706, #FBBF24); }

.metric-label {
    font-size: 0.73rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    color: #0A2540;
    line-height: 1;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.metric-pct {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748B;
}

/* --- SECTION HEADERS --- */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0A2540;
    margin: 4px 0 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-subtitle {
    font-size: 0.83rem;
    color: #64748B;
    margin: 0 0 18px 0;
}

/* --- INSIGHT CARDS --- */
.insight-card {
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}

.insight-card:hover {
    transform: translateY(-2px);
}

.insight-best {
    background: linear-gradient(135deg, rgba(5, 150, 105, 0.08) 0%, rgba(5, 150, 105, 0.02) 100%);
    border: 1px solid rgba(5, 150, 105, 0.3);
    box-shadow: 0 4px 16px rgba(5, 150, 105, 0.06);
}

.insight-warn {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.08) 0%, rgba(225, 29, 72, 0.02) 100%);
    border: 1px solid rgba(225, 29, 72, 0.3);
    box-shadow: 0 4px 16px rgba(225, 29, 72, 0.06);
}

.insight-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.insight-best .insight-title { color: #059669; }
.insight-warn .insight-title { color: #E11D48; }

.insight-aspect {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0A2540;
    margin-bottom: 4px;
}

.insight-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}

/* --- BADGES & CHIPS --- */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}

.badge-positive {
    background: rgba(5, 150, 105, 0.12);
    color: #047857;
    border: 1px solid rgba(5, 150, 105, 0.3);
}

.badge-negative {
    background: rgba(225, 29, 72, 0.12);
    color: #BE123C;
    border: 1px solid rgba(225, 29, 72, 0.3);
}

.badge-neutral {
    background: rgba(217, 119, 6, 0.14);
    color: #B45309;
    border: 1px solid rgba(217, 119, 6, 0.35);
}

.badge-aspect {
    background: rgba(10, 37, 64, 0.06);
    color: #0A2540;
    border: 1px solid rgba(10, 37, 64, 0.18);
}

/* --- BUTTONS & CONTROLS --- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00AEAC 0%, #008785 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 4px 14px rgba(0, 174, 172, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 174, 172, 0.45) !important;
}

.stButton > button:not([kind="primary"]) {
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: #00AEAC !important;
    color: #00AEAC !important;
    background: rgba(0, 174, 172, 0.05) !important;
}

/* --- CARD CONTAINERS & EMPTY STATES --- */
.empty-hero-card {
    background: #FFFFFF;
    border: 2px dashed rgba(0, 174, 172, 0.35);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0, 174, 172, 0.05);
}

.empty-hero-icon {
    font-size: 2.8rem;
    margin-bottom: 12px;
}

.empty-hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0A2540;
    margin-bottom: 8px;
}

.empty-hero-desc {
    font-size: 0.88rem;
    color: #64748B;
    max-width: 540px;
    margin: 0 auto 20px auto;
    line-height: 1.6;
}

.sim-result-card {
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 12px;
    background: #FFFFFF;
    border: 1px solid rgba(0, 174, 172, 0.2);
    box-shadow: 0 4px 14px rgba(10, 37, 64, 0.04);
    transition: all 0.2s ease;
}

.sim-result-card:hover {
    box-shadow: 0 8px 24px rgba(10, 37, 64, 0.08);
    transform: translateY(-2px);
}

.sim-sent-positive { border-left: 5px solid #059669; }
.sim-sent-negative { border-left: 5px solid #E11D48; }
.sim-sent-neutral { border-left: 5px solid #D97706; }
</style>
''', unsafe_allow_html=True)


# ================================
# CONSTANTS
# ================================
ASPECT_LABELS = [
    'persyaratan',
    'sistem_mekanisme_prosedur',
    'waktu_penyelesaian',
    'biaya_tarif',
    'produk_spesifikasi_pelayanan',
    'kompetensi_pelaksana',
    'perilaku_pelaksana',
    'penanganan_pengaduan_saran_masukan',
    'sarana_prasarana',
    'empati_dan_komunikasi',
    'keandalan_dan_daya_tanggap',
]
ASPECT_DISPLAY = {
    'persyaratan': 'Persyaratan',
    'sistem_mekanisme_prosedur': 'Sistem & Prosedur',
    'waktu_penyelesaian': 'Waktu Penyelesaian',
    'biaya_tarif': 'Biaya & Tarif',
    'produk_spesifikasi_pelayanan': 'Produk & Spesifikasi Layanan',
    'kompetensi_pelaksana': 'Kompetensi Pelaksana',
    'perilaku_pelaksana': 'Perilaku Pelaksana',
    'penanganan_pengaduan_saran_masukan': 'Penanganan Pengaduan & Saran',
    'sarana_prasarana': 'Sarana & Prasarana',
    'empati_dan_komunikasi': 'Empati & Komunikasi',
    'keandalan_dan_daya_tanggap': 'Keandalan & Daya Tanggap',
}
SENT_EMOJI = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
SENT_COLOR = {'positive': '#00AEAC', 'negative': '#e05a2b', 'neutral': '#C8D400'}
SENT_LABEL = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}

_sastrawi_stopwords = set(StopWordRemoverFactory().get_stop_words())

STOPWORDS_ID = _sastrawi_stopwords | {
    'yg','dgn','utk','krn','dr','pd','tsb','tdk','ny','ku','mu','deh','sih',
    'nih','dong','kok','ya','loh','lah','the','and','was','is','are','were',
    'be','been','to','of','in','on','for','with','a','an','it',
    'cuma','mulu','walaupun','walau','walopun','meski','meskipun','padahal',
    'namun','sambil','sampai','sampe','terus','trus','jadi','kayak','kaya',
    'gitu','gini','banget','bgt','aja','doang','emang','memang','kan','eh',
    'katanya','ternyata','apalagi','soalnya','pokoknya','gara','gara2','rs',
    'sama','nya','nih','gue','gw','lu','elu','banyak','selalu','ngaruh','milih',
}

_STRIP_CHARS = '.,!?()[]{}"' + chr(39) + '-:;'


def clean_token(tok):
    t = str(tok).strip().lower()
    t = re.sub(r'^[^\w]+|[^\w]+$', '', t)   # buang semua simbol non-huruf/angka di ujung kata
    t = re.sub(r'2$', '', t)
    t = re.sub(r'\s+', ' ', t)              # normalisasi spasi ganda/aneh jadi satu spasi
    return t


def parse_sentiments(x):
    # Normalisasi sentiment_label jadi list (mendukung multi-label, list-literal, & string berkoma dari CSV).
    if isinstance(x, list):
        items = x
    elif isinstance(x, str):
        s = x.strip()
        if s.startswith('['):
            try:
                v = ast.literal_eval(s)
                items = v if isinstance(v, list) else [v]
            except Exception:
                items = s.split(',')
        else:
            items = s.split(',')  # mis. "positive, negative" -> 2 item
    else:
        return ['neutral']
    out = []
    for it in items:
        t = str(it).strip().strip(chr(39)).strip(chr(34)).lower()
        if t:
            out.append(t)
    return out if out else ['neutral']


def primary_sentiment(sents):
    # Sentimen utama untuk pewarnaan kartu: negatif > positif > netral.
    for s in ['negative', 'positive', 'neutral']:
        if s in sents:
            return s
    return 'neutral'


# ================================
# MODEL
# ================================
class MultiTaskIndoBERT(nn.Module):
    def __init__(self, num_sentiment=3, num_aspect=12, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiment)
        self.aspect_head = nn.Linear(self.bert.config.hidden_size, num_aspect)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        return self.sentiment_head(pooled), self.aspect_head(pooled)


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskIndoBERT()
    model.load_state_dict(torch.load('multitask_model_p1_42.pt', map_location=device))
    model.eval()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    return model, tokenizer, device


def predict(texts, model, tokenizer, device, max_len=128, sent_threshold=0.5):
    sent_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    enc = tokenizer(texts, max_length=max_len, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        sl, al = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))

    # --- SENTIMEN MULTI-LABEL (sigmoid + threshold) ---
    sent_probs = torch.sigmoid(sl).cpu().numpy()
    sents = []
    for row in sent_probs:
        labels = [sent_map[i] for i in range(len(sent_map)) if row[i] > sent_threshold]
        if not labels:  # fallback: kalau tak ada yg lolos threshold
            labels = [sent_map[int(row.argmax())]]
        sents.append(labels)  # tiap item = list, mis. ['positive', 'negative']

    # --- ASPEK MULTI-LABEL (tetap) ---
    aspects = []
    for row in (torch.sigmoid(al) > 0.5).cpu().numpy().astype(int):
        a = [ASPECT_LABELS[i] for i in range(len(ASPECT_LABELS)) if row[i] == 1]
        aspects.append(a if a else ['lainnya'])
    return sents, aspects


# ================================
# SHAP HELPERS (interpretasi kata pemicu sentimen per aspek)
# ================================
SENT_CLASS_IDX = {'negative': 0, 'neutral': 1, 'positive': 2}


def make_shap_predict_fn(model, tokenizer, device, max_len=128):
    def f(texts):
        texts = [str(t) if str(t).strip() != '' else '.' for t in texts]
        enc = tokenizer(texts, max_length=max_len, padding='max_length',
                        truncation=True, return_tensors='pt')
        with torch.no_grad():
            sl, _ = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        return torch.sigmoid(sl).cpu().numpy()
    return f


@st.cache_resource
def build_shap_explainer(_model, _tokenizer, _device):
    f = make_shap_predict_fn(_model, _tokenizer, _device)
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(f, masker, output_names=['negative', 'neutral', 'positive'])
    return explainer


def compute_aspect_word_drivers(df_in, model, tokenizer, device,
                                 max_rows=150, top_n=10, min_occurrence=2,
                                 max_evals=200, progress_cb=None):
    explainer = build_shap_explainer(model, tokenizer, device)

    work_df = df_in.copy()
    if len(work_df) > max_rows:
        work_df = work_df.sample(n=max_rows, random_state=42)
    work_df = work_df.reset_index(drop=True)

    texts = work_df['text_translated'].astype(str).tolist() if 'text_translated' in work_df.columns else work_df['text'].astype(str).tolist()
    aspects_list = work_df['aspect_classification'].apply(
        lambda a: ast.literal_eval(a) if isinstance(a, str) else a
    ).tolist()
    sents_list = work_df['sentiment_label'].apply(parse_sentiments).tolist()

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    WORD_RE = re.compile(r"\w+")
    total = len(texts)
    PROGRESS_EVERY = 5

    for idx, txt in enumerate(texts):
        n_words = len(WORD_RE.findall(str(txt)))
        if n_words < 2:
            if progress_cb is not None and (idx + 1) % PROGRESS_EVERY == 0:
                progress_cb(idx + 1, total)
            continue

        asps = aspects_list[idx] or ['lainnya']
        sents = [s for s in sents_list[idx] if s in ('positive', 'negative')]
        if not sents:
            if progress_cb is not None and (idx + 1) % PROGRESS_EVERY == 0:
                progress_cb(idx + 1, total)
            continue

        single_shap = None
        try:
            single_shap = explainer([txt], max_evals=max_evals)
        except Exception:
            try:
                single_shap = explainer([txt])
            except Exception:
                single_shap = None

        if single_shap is not None:
            try:
                tokens = single_shap.data[0]
                values = single_shap.values[0]
                for asp in asps:
                    if asp == 'lainnya' or asp not in ASPECT_LABELS:
                        continue
                    for sent in sents:
                        cidx = SENT_CLASS_IDX[sent]
                        for tok, vrow in zip(tokens, values):
                            tok_c = clean_token(tok)
                            if not tok_c or tok_c in STOPWORDS_ID or tok_c.isdigit():
                                continue
                            contrib = float(vrow[cidx])
                            if contrib > 0:
                                accum[asp][sent][tok_c].append(contrib)
            except Exception:
                pass

        if progress_cb is not None and ((idx + 1) % PROGRESS_EVERY == 0 or idx + 1 == total):
            progress_cb(idx + 1, total)

    result = {}
    for asp, sentdict in accum.items():
        asp_out = {}
        for sent in ('positive', 'negative'):
            worddict = sentdict.get(sent, {})
            scored = [
                (w, float(np.mean(vals)), len(vals))
                for w, vals in worddict.items()
                if len(vals) >= min_occurrence
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            asp_out[sent] = scored[:top_n]
        if asp_out.get('positive') or asp_out.get('negative'):
            result[asp] = asp_out
    return result


# ================================
# RECOMMENDATION ENGINE (KB-based)
# ================================
# Jembatan nama aspek: dashboard pakai 'produk_spesifikasi_pelayanan',
# sedangkan recommendation_rules_kb.json pakai 'produk_spesifikasi_jenis_pelayanan'.
ASPECT_KEY_MAP = {
    'produk_spesifikasi_pelayanan': 'produk_spesifikasi_jenis_pelayanan',
}


@st.cache_resource
def load_recommendation_kb():
    try:
        with open('recommendation_rules_kb_v3.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def find_matched_recommendations(aspect, driver_words, kb):
    """
    driver_words: list (word, score, count) dari shap_result[aspect]['negative'] / ['positive'].
    Mencocokkan token kata driver SHAP dengan daftar keyword di recommendation_rules_kb.json,
    lalu mengembalikan sub-rule dengan jumlah kata cocok paling banyak.
    """
    kb_key = ASPECT_KEY_MAP.get(aspect, aspect)
    aspect_rules = kb.get(kb_key)
    if not aspect_rules:
        return None

    driver_tokens = {clean_token(w) for w, _, _ in driver_words}

    best_rule, best_matched = None, []
    for sub_rule in aspect_rules.values():
        kw_tokens = set()
        for kw in sub_rule.get('keywords', []):
            kw_clean = clean_token(kw)
            kw_tokens.add(kw_clean)
            kw_tokens.update(kw_clean.split())  # biar "kurang ramah" tetap match token "ramah"
        matched = sorted(driver_tokens & kw_tokens)
        if matched and len(matched) > len(best_matched):
            best_rule, best_matched = sub_rule, matched

    if not best_rule:
        return None
    return {'rule': best_rule, 'matched_keywords': best_matched}


def render_recommendation_card(match_info):
    rule = match_info['rule']
    matched_kw = match_info['matched_keywords']
    kw_badges = ''.join(f'<span class="badge badge-negative">🔑 {kw}</span>' for kw in matched_kw)
    root_causes = ''.join(f'<li>{rc}</li>' for rc in rule.get('root_cause', []))
    recs = ''.join(f'<li>{r}</li>' for r in rule.get('recommendation', []))
    kpis = ''.join(f'<li>{k}</li>' for k in rule.get('kpi', []))
    priority = rule.get('priority', '-')
    category = rule.get('category', '-')
    st.markdown(f'''
<div class="insight-card insight-warn" style="padding:18px 20px;margin-top:10px;">
<div class="insight-title">💊 Rekomendasi Perbaikan</div>
<div style="font-size:0.82rem;color:#475569;margin:8px 0 12px 0;">Kata kunci terdeteksi: {kw_badges}</div>
<div class="insight-desc" style="margin-bottom:10px;"><b>Interpretasi:</b> {rule.get('interpretation', '-')}</div>
<div class="insight-desc" style="margin-bottom:4px;"><b>Kemungkinan Akar Masalah:</b></div>
<ul style="margin:0 0 10px 18px;padding:0;font-size:0.83rem;color:#475569;">{root_causes}</ul>
<div class="insight-desc" style="margin-bottom:4px;"><b>Rekomendasi Tindakan:</b></div>
<ul style="margin:0 0 10px 18px;padding:0;font-size:0.83rem;color:#0A2540;">{recs}</ul>
<div style="display:flex;gap:8px;margin:8px 0;">
<span class="badge badge-aspect">📁 {category}</span>
<span class="badge badge-negative">🔥 Prioritas: {priority}</span>
</div>
<div class="insight-desc" style="margin-bottom:4px;"><b>KPI Pemantauan:</b></div>
<ul style="margin:0 0 0 18px;padding:0;font-size:0.83rem;color:#475569;">{kpis}</ul>
</div>''', unsafe_allow_html=True)


def chart_word_drivers(words, color):
    if not words:
        return None
    words_sorted = list(reversed(words))
    labels = [w[0] for w in words_sorted]
    values = [w[1] for w in words_sorted]
    counts = [w[2] for w in words_sorted]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=color,
        text=[f"{c}x" for c in counts],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Skor kontribusi: <b>%{x:.3f}</b><extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color='#0A2540', size=11),
        margin=dict(l=10, r=36, t=10, b=10),
        height=max(130, 36 * len(labels)),
        xaxis=dict(
            gridcolor='rgba(0, 174, 172, 0.15)',
            title=None,
            tickfont=dict(size=10, color='#64748B')
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(size=11, color='#0A2540', family='Plus Jakarta Sans')
        ),
        showlegend=False,
    )
    return fig


# ================================
# TRANSLATE & DATE HELPER
# ================================
def detect_and_translate(text):
    text = str(text).strip()
    if not text:
        return text
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'id'
    if lang == 'id':
        return text
    try:
        return GoogleTranslator(source='auto', target='id').translate(text)
    except Exception:
        return text


def parse_published_date(series):
    dt = pd.to_datetime(series, errors='coerce', utc=True)
    return dt.dt.year, dt.dt.month, dt.dt.day


# ================================
# METRICS HELPER
# ================================
def compute_metrics(df):
    total = len(df)
    pos = neg = neu = 0
    asp_sent = {a: {'positive': 0, 'negative': 0, 'neutral': 0} for a in ASPECT_LABELS}
    for _, row in df.iterrows():
        sents = parse_sentiments(row['sentiment_label'])
        if 'positive' in sents:
            pos += 1
        if 'negative' in sents:
            neg += 1
        if 'neutral' in sents:
            neu += 1

        asps = row['aspect_classification']
        if isinstance(asps, str):
            try:
                asps = ast.literal_eval(asps)
            except Exception:
                asps = ['lainnya']
        if not asps:
            asps = ['lainnya']
        for a in asps:
            if a in asp_sent:
                for s in sents:
                    if s in asp_sent[a]:
                        asp_sent[a][s] += 1

    scores = {}
    for a, c in asp_sent.items():
        denom = c['positive'] + c['negative']
        scores[a] = (c['positive'] - c['negative']) / denom if denom > 0 else 0
    return {
        'total': total, 'pos': pos, 'neg': neg, 'neu': neu,
        'pos_pct': pos / total * 100 if total else 0,
        'neg_pct': neg / total * 100 if total else 0,
        'neu_pct': neu / total * 100 if total else 0,
        'asp_sent': asp_sent, 'scores': scores,
        'best': max(scores, key=scores.get) if scores else None,
        'worst': min(scores, key=scores.get) if scores else None,
    }


def chart_bar(asp_sent):
    labels = [ASPECT_DISPLAY[a] for a in ASPECT_LABELS]
    fig = go.Figure()
    for name, color, key in [
        ('Positif', '#059669', 'positive'),
        ('Netral', '#94A3B8', 'neutral'),
        ('Negatif', '#E11D48', 'negative'),
    ]:
        values = [asp_sent[a][key] for a in ASPECT_LABELS]
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=values,
            marker_color=color,
            marker_line_width=0,
            hovertemplate='<b>%{x}</b><br>' + name + ': <b>%{y} ulasan</b><extra></extra>'
        ))
    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color='#475569', size=11),
        legend=dict(
            orientation='h',
            y=1.1,
            x=0.5,
            xanchor='center',
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#0A2540', family='Plus Jakarta Sans')
        ),
        margin=dict(l=10, r=10, t=40, b=100),
        height=440,
        xaxis=dict(
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(size=10, color='#0A2540', family='Plus Jakarta Sans'),
            tickangle=-32,
        ),
        yaxis=dict(
            gridcolor='rgba(0, 174, 172, 0.12)',
            tickfont=dict(size=10, color='#64748B')
        ),
    )
    return fig


def chart_pie(pos, neg, neu):
    total = pos + neg + neu
    pos_pct = (pos / total * 100) if total > 0 else 0
    fig = go.Figure(go.Pie(
        labels=['Positif', 'Negatif', 'Netral'],
        values=[pos, neg, neu],
        hole=0.68,
        marker=dict(
            colors=['#059669', '#E11D48', '#D97706'],
            line=dict(color='#FFFFFF', width=3)
        ),
        textinfo='percent',
        textfont=dict(family='Outfit', size=12, color='#FFFFFF'),
        hovertemplate='<b>%{label}</b><br>%{value} ulasan (%{percent})<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color='#64748B'),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            x=0.88,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#0A2540')
        ),
        margin=dict(l=10, r=90, t=10, b=10),
        height=290,
        annotations=[
            dict(
                text=f'<span style="font-size:1.6rem;font-weight:800;color:#0A2540;font-family:Outfit">{pos_pct:.0f}%</span><br><span style="font-size:0.75rem;color:#64748B;font-family:Plus Jakarta Sans">Positif</span>',
                x=0.42, y=0.5,
                font=dict(family='Outfit'),
                showarrow=False
            )
        ]
    )
    return fig

def fig_to_png_bytes(fig, width=900, height=500, scale=2):
    return fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")


def generate_pdf_report(m, periode_label, source_label):
    pie_fig = chart_pie(m['pos'], m['neg'], m['neu'])
    bar_fig = chart_bar(m['asp_sent'])
    pie_png = fig_to_png_bytes(pie_fig, width=600, height=500)
    bar_png = fig_to_png_bytes(bar_fig, width=900, height=500)

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(10, 37, 64)
    pdf.cell(0, 10, 'Laporan Analisis Sentimen Berbasis Aspek (ABSA)', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 174, 172)
    pdf.cell(0, 8, 'Rumah Sakit Mata Cicendo', ln=True)
    pdf.set_text_color(100, 100, 100)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, f'Sumber data: {source_label}  |  Periode: {periode_label}', ln=True)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(10, 37, 64)
    pdf.cell(0, 8, 'Ringkasan Sentimen', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, f"Total Ulasan: {m['total']:,}", ln=True)
    pdf.cell(0, 6, f"Positif: {m['pos']:,} ({m['pos_pct']:.1f}%)", ln=True)
    pdf.cell(0, 6, f"Negatif: {m['neg']:,} ({m['neg_pct']:.1f}%)", ln=True)
    pdf.cell(0, 6, f"Netral: {m['neu']:,} ({m['neu_pct']:.1f}%)", ln=True)
    pdf.ln(4)

    if m['best'] and m['worst']:
        cb, cw = m['asp_sent'][m['best']], m['asp_sent'][m['worst']]
        tb, tw = sum(cb.values()) or 1, sum(cw.values()) or 1
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(5, 150, 105)
        pdf.cell(0, 7, f"Aspek Terbaik: {ASPECT_DISPLAY[m['best']]} ({cb['positive']/tb*100:.0f}% positif)", ln=True)
        pdf.set_text_color(225, 29, 72)
        pdf.cell(0, 7, f"Perlu Perhatian: {ASPECT_DISPLAY[m['worst']]} ({cw['negative']/tw*100:.0f}% negatif)", ln=True)
        pdf.ln(4)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_pie:
        f_pie.write(pie_png); pie_path = f_pie.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_bar:
        f_bar.write(bar_png); bar_path = f_bar.name

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(10, 37, 64)
    pdf.cell(0, 8, 'Distribusi Sentimen', ln=True)
    pdf.image(pie_path, w=90)
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Distribusi Sentimen per Aspek', ln=True)
    pdf.image(bar_path, w=180)
    pdf.ln(4)
    os.unlink(pie_path); os.unlink(bar_path)

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(10, 37, 64)
    pdf.cell(0, 8, 'Peringkat Aspek SKM', ln=True)
    pdf.ln(2)

    col_widths = [10, 65, 25, 25, 25, 25]
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    for w, h in zip(col_widths, ['#', 'Aspek', 'Positif', 'Netral', 'Negatif', 'Total']):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    for i, (asp, score) in enumerate(sorted(m['scores'].items(), key=lambda x: x[1], reverse=True)):
        c = m['asp_sent'][asp]
        row = [str(i+1), ASPECT_DISPLAY[asp], str(c['positive']), str(c['neutral']),
               str(c['negative']), str(c['positive']+c['neutral']+c['negative'])]
        for w, val in zip(col_widths, row):
            pdf.cell(w, 6, val, border=1)
        pdf.ln()

    return bytes(pdf.output())
# ================================
# RENDER HELPERS
# ================================
SENT_ICON_HTML = {
    'positive': '<span class="badge badge-positive">👍 Positif</span>',
    'negative': '<span class="badge badge-negative">👎 Negatif</span>',
    'neutral': '<span class="badge badge-neutral">— Netral</span>',
}


def render_aspect_badges(aspects):
    if isinstance(aspects, str):
        try:
            aspects = ast.literal_eval(aspects)
        except Exception:
            aspects = [aspects]
    return ''.join([
        f'<span class="badge badge-aspect">{ASPECT_DISPLAY.get(a, a)}</span>'
        for a in aspects
    ])


def render_stars(rating):
    try:
        r = int(float(rating))
    except Exception:
        return '-'
    return f'<span style="color:#F59E0B;font-size:0.95rem;">{"★" * r}</span><span style="color:#CBD5E1;font-size:0.95rem;">{"☆" * (5 - r)}</span>'


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    logo_img_tag = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:40px;height:40px;border-radius:10px;object-fit:contain;background:#ffffff;padding:3px;box-shadow:0 4px 10px rgba(0,174,172,0.15);flex-shrink:0;">' if LOGO_B64 else '<span style="font-size:2rem;">🏥</span>'
    st.markdown(f'''
<div style="background:#FFFFFF;border:1px solid rgba(0,174,172,0.2);border-radius:16px;padding:16px;margin-bottom:20px;box-shadow:0 4px 16px rgba(0,174,172,0.06);">
    <div style="display:flex;align-items:center;gap:12px;">
        {logo_img_tag}
        <div>
            <div style="font-family:\'Outfit\',sans-serif;font-size:0.98rem;font-weight:800;color:#0A2540;line-height:1.2;">RS Mata Cicendo</div>
            <div style="font-size:0.7rem;color:#00AEAC;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;margin-top:2px;">Kemenkes RI</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

    menu_items = [
        ("📊", "Hasil Analisis"),
        ("🧪", "Prediksi Dataset"),
        ("🔍", "Simulasi Prediksi"),
        ("🧠", "Indikator Penilaian Pelayanan"),
    ]
    menu_labels = [f"{icon}  {label}" for icon, label in menu_items]
    menu_selected = st.radio("", menu_labels, label_visibility="collapsed", key="main_menu")
    menu = menu_selected.split("  ", 1)[1]

    st.markdown('''
<div style="margin-top:40px;padding:14px;background:rgba(0,174,172,0.06);border:1px solid rgba(0,174,172,0.2);border-radius:12px;text-align:center;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:700;color:#00AEAC;text-transform:uppercase;letter-spacing:1px;">System Status</div>
    <div style="font-size:0.8rem;font-weight:700;color:#0A2540;margin-top:4px;">⚡ IndoBERT Multi-Task</div>
    <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">11 Aspek Pelayanan Enabled</div>
</div>
''', unsafe_allow_html=True)

# ================================
# HEADER
# ================================
logo_header_tag = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:130px;width:auto;object-fit:contain;flex-shrink:0;">'

st.markdown(f'''
<div class="hero-banner">
    <div style="display:flex;align-items:center;gap:22px;">
        {logo_header_tag}
        <div style="flex:1;">
            <h1 class="hero-title">Dashboard Analisis Sentimen<br>Berbasis Aspek (ABSA)</h1>
            <p class="hero-subtitle">Rumah Sakit Mata Cicendo</p>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# ================================
# PAGE: HASIL ANALISIS
# ================================
if menu == "Hasil Analisis":
    uploaded = st.file_uploader("Upload hasil prediksi (CSV/Excel)", type=['csv', 'xlsx'])

    if not uploaded:
        st.markdown('''
<div class="empty-hero-card">
    <div class="empty-hero-icon">📊</div>
    <div class="empty-hero-title">Belum ada file data yang diupload</div>
    <div class="empty-hero-desc">
        Silakan upload file CSV atau Excel hasil prediksi ulasan di atas untuk menampilkan dashboard ringkasan sentimen, distribusi 11 aspek Pelayanan, serta detail data ulasan.
    </div>
    <div style="display:inline-flex;gap:12px;justify-content:center;flex-wrap:wrap;">
        <span class="badge badge-aspect">📁 Mendukung .csv & .xlsx</span>
        <span class="badge badge-positive">👍 Sentimen Multi-Label</span>
        <span class="badge badge-aspect">🏷️ 11 Aspek Pelayanan</span>
    </div>
</div>
''', unsafe_allow_html=True)


    if uploaded:
        df = pd.read_csv(uploaded, sep=None, engine='python') if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        if 'sentiment_label_reviewed' in df.columns:
            df['sentiment_label'] = df['sentiment_label_reviewed']
        if 'aspect_classification_reviewed' in df.columns:
            df['aspect_classification'] = df['aspect_classification_reviewed']

        for col in ['sentiment_label', 'aspect_classification']:
            if col not in df.columns:
                st.error(f"❌ Kolom '{col}' (atau versi _reviewed-nya) tidak ditemukan di file.")
                st.stop()

        def safe_parse(x):
            if isinstance(x, list):
                return x
            try:
                return ast.literal_eval(str(x))
            except Exception:
                return ['lainnya']
        df['aspect_classification'] = df['aspect_classification'].apply(safe_parse)

        if all(c in df.columns for c in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(
                dict(year=df['year'], month=df['month'], day=df['day']),
                errors='coerce'
            )
        else:
            df['date'] = pd.NaT

        st.success(f"✅ {len(df)} data berhasil dimuat!")

        st.session_state['hasil_analisis_df'] = df

        st.markdown(
            '<div class="section-title">📅 Periode Data</div>'
            '<div class="section-subtitle">Pilih rentang waktu untuk menampilkan dashboard</div>',
            unsafe_allow_html=True
        )
        fp1, fp2 = st.columns([2, 3])
        with fp1:
            periode = st.selectbox(
                "Tampilkan data",
                ["Semua Data", "Tahun Ini", "3 Bulan Terakhir", "6 Bulan Terakhir", "Periode Kustom"]
            )

        df_filtered = df.copy()
        valid_dates = df['date'].dropna()

        if periode != "Semua Data" and not valid_dates.empty:
            max_date = valid_dates.max()

            if periode == "Tahun Ini":
                df_filtered = df[df['date'].dt.year == max_date.year]
            elif periode == "3 Bulan Terakhir":
                cutoff = max_date - pd.DateOffset(months=3)
                df_filtered = df[df['date'] >= cutoff]
            elif periode == "6 Bulan Terakhir":
                cutoff = max_date - pd.DateOffset(months=6)
                df_filtered = df[df['date'] >= cutoff]
            elif periode == "Periode Kustom":
                with fp2:
                    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
                    date_range = st.date_input(
                        "Pilih rentang tanggal",
                        value=(min_d, max_d),
                        min_value=min_d,
                        max_value=max_d
                    )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_d, end_d = date_range
                    df_filtered = df[
                        (df['date'].dt.date >= start_d) & (df['date'].dt.date <= end_d)
                    ]

        st.markdown(
            f"<div style='font-size:0.78rem;color:#4a7a8a;margin:4px 0 18px 0;'>"
            f"Menampilkan <b>{len(df_filtered):,}</b> dari {len(df):,} total ulasan</div>",
            unsafe_allow_html=True
        )

        m = compute_metrics(df_filtered)

        c1, c2, c3, c4 = st.columns(4)
        CARD_ICON_SVG = {
            'total': '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1B3A6B" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
            'positive': '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00AEAC" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>',
            'negative': '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#e05a2b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>',
            'neutral': '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#8a9200" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>',
        }

        for col, cls, label, val, pct in [
            (c1, 'total', 'Total Ulasan', m['total'], 'Keseluruhan data'),
            (c2, 'positive', 'Sentimen Positif', m['pos'], f"{m['pos_pct']:.1f}% dari total"),
            (c3, 'negative', 'Sentimen Negatif', m['neg'], f"{m['neg_pct']:.1f}% dari total"),
            (c4, 'neutral', 'Sentimen Netral', m['neu'], f"{m['neu_pct']:.1f}% dari total"),
        ]:
            icon_svg = CARD_ICON_SVG[cls]
            with col:
                st.markdown(f'''
<div class="metric-card {cls}">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
<div class="metric-label">{label}</div>
<div style="width:44px;height:44px;border-radius:10px;background:rgba(255,255,255,0.7);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
{icon_svg}
</div>
</div>
<div class="metric-value">{val:,}</div>
<div class="metric-pct">{pct}</div>
</div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        ci, cp = st.columns(2)
        with ci:
            st.markdown(
                '<div class="section-title">💡 Insight Aspek</div><div class="section-subtitle">Aspek terbaik dan yang perlu perhatian</div>',
                unsafe_allow_html=True
            )
            for key, cls, emoji, label in [
                (m['best'], 'insight-best', '⭐', 'Aspek Terbaik'),
                (m['worst'], 'insight-warn', '⚠️', 'Perlu Perhatian')
            ]:
                c = m['asp_sent'][key]
                t = sum(c.values())
                pct_val = (c['positive'] / t * 100 if t > 0 else 0) if label == 'Aspek Terbaik' else (c['negative'] / t * 100 if t > 0 else 0)
                pct_color = '#34d399' if label == 'Aspek Terbaik' else '#f87171'
                pct_label = 'positif' if label == 'Aspek Terbaik' else 'negatif'
                st.markdown(f'''
<div class="insight-card {cls}">
<div class="insight-title">{emoji} {label}</div>
<div class="insight-aspect">{ASPECT_DISPLAY[key]}</div>
<div class="insight-desc">
{c['positive']} positif · {c['neutral']} netral · {c['negative']} negatif
 |  <b style="color:{pct_color}">{pct_val:.0f}% {pct_label}</b>
</div>
</div>''', unsafe_allow_html=True)

        with cp:
            st.markdown(
                '<div class="section-title">🥧 Distribusi Sentimen</div><div class="section-subtitle">Keseluruhan ulasan</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(chart_pie(m['pos'], m['neg'], m['neu']), use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="section-title">📊 Distribusi Sentimen per Aspek</div><div class="section-subtitle">Positif (hijau) · Netral (abu) · Negatif (merah)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_bar(m['asp_sent']), use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="section-title">🏆 Peringkat Aspek SKM</div><div class="section-subtitle">Urutan dari sentimen paling positif ke yang memerlukan perhatian khusus</div>', unsafe_allow_html=True)
        sorted_asp = sorted(m['scores'].items(), key=lambda x: x[1], reverse=True)
        total_all = m['total'] or 1
        html = ''
        for i, (asp, score) in enumerate(sorted_asp):
            c = m['asp_sent'][asp]
            pos_pct = c['positive'] / total_all * 100
            neu_pct = c['neutral'] / total_all * 100
            neg_pct = c['negative'] / total_all * 100
            total_asp = c['positive'] + c['neutral'] + c['negative']
            html += f'''
<div style="background:#FFFFFF;border:1px solid rgba(0,174,172,0.18);border-radius:14px;padding:16px 20px;margin-bottom:12px;box-shadow:0 2px 10px rgba(10,37,64,0.03);">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="display:flex;align-items:center;gap:12px;">
<span style="font-family:'Outfit',sans-serif;font-size:0.8rem;font-weight:800;color:#FFFFFF;background:linear-gradient(135deg,#0A2540,#00AEAC);width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{i+1}</span>
<span style="font-family:'Outfit',sans-serif;font-size:0.98rem;font-weight:700;color:#0A2540;">{ASPECT_DISPLAY[asp]}</span>
</div>
<span style="font-size:0.78rem;color:#64748B;font-family:'JetBrains Mono',monospace;"><b>{total_asp:,}</b> ulasan</span>
</div>
<div style="width:100%;height:12px;border-radius:8px;overflow:hidden;display:flex;background:#F1F5F9;">
<div style="width:{pos_pct:.2f}%;background:linear-gradient(90deg,#059669,#34D399);height:100%;"></div>
<div style="width:{neu_pct:.2f}%;background:#94A3B8;height:100%;"></div>
<div style="width:{neg_pct:.2f}%;background:linear-gradient(90deg,#E11D48,#FB7185);height:100%;"></div>
</div>
<div style="display:flex;gap:18px;margin-top:8px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;">
<span style="font-size:0.74rem;color:#059669;">● Positif <b>{c['positive']:,}</b> <span style="color:#64748B;">({pos_pct:.1f}%)</span></span>
<span style="font-size:0.74rem;color:#64748B;">● Netral <b>{c['neutral']:,}</b> <span style="color:#64748B;">({neu_pct:.1f}%)</span></span>
<span style="font-size:0.74rem;color:#E11D48;">● Negatif <b>{c['negative']:,}</b> <span style="color:#64748B;">({neg_pct:.1f}%)</span></span>
</div>
</div>'''
        st.markdown(html, unsafe_allow_html=True)

        pdf_bytes = generate_pdf_report(m, periode, "Hasil Analisis")
        st.download_button(
            "📄 Download Laporan PDF",
            data=pdf_bytes,
            file_name="laporan_absa.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        with st.expander("📋 Lihat Data Lengkap & Filter Ulasan"):
            fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 3])
            with fc1:
                sort_waktu = st.selectbox("Waktu", ["Semua", "Terbaru", "Terlama"])
            with fc2:
                sort_rating = st.selectbox("Rating", ["Semua", "Tertinggi", "Terendah"])
            with fc3:
                rating_options = ["5 ⭐", "4 ⭐", "3 ⭐", "2 ⭐", "1 ⭐"]
                filter_rating = st.multiselect("Filter Rating", options=rating_options, default=[], placeholder="Semua rating")
            with fc4:
                sent_options = ["Positif", "Netral", "Negatif"]
                filter_sent = st.multiselect("Filter Sentimen", options=sent_options, default=[], placeholder="Semua sentimen")
            with fc5:
                asp_options = [ASPECT_DISPLAY[a] for a in ASPECT_LABELS]
                filter_asp_display = st.multiselect("Filter Aspek", options=asp_options, default=[], placeholder="Semua aspek")
            asp_display_to_key = {v: k for k, v in ASPECT_DISPLAY.items()}
            filter_asp_keys = [asp_display_to_key[d] for d in filter_asp_display if d in asp_display_to_key]

            disp_df = df_filtered.copy()

            if filter_sent:
                sent_map_filter = {"Positif": "positive", "Netral": "neutral", "Negatif": "negative"}
                selected_sent_keys = [sent_map_filter[s] for s in filter_sent]
                disp_df = disp_df[disp_df['sentiment_label'].apply(
                    lambda x: any(s in parse_sentiments(x) for s in selected_sent_keys)
                )]

            if filter_rating and 'stars' in disp_df.columns:
                selected_stars = [int(r[0]) for r in filter_rating]
                disp_df = disp_df[disp_df['stars'].apply(
                    lambda x: int(float(x)) in selected_stars if pd.notna(x) else False
                )]

            if filter_asp_keys:
                disp_df = disp_df[disp_df['aspect_classification'].apply(
                    lambda a: any(
                        k in (ast.literal_eval(a) if isinstance(a, str) else a)
                        for k in filter_asp_keys
                    )
                )]

            has_date = all(c in disp_df.columns for c in ['year', 'month', 'day'])
            has_stars = 'stars' in disp_df.columns

            if sort_waktu == "Terbaru" and has_date:
                disp_df = disp_df.sort_values(['year', 'month', 'day'], ascending=[False, False, False])
            elif sort_waktu == "Terlama" and has_date:
                disp_df = disp_df.sort_values(['year', 'month', 'day'], ascending=[True, True, True])

            if sort_rating == "Tertinggi" and has_stars:
                disp_df = disp_df.sort_values('stars', ascending=False)
            elif sort_rating == "Terendah" and has_stars:
                disp_df = disp_df.sort_values('stars', ascending=True)

            ROWS_PER_PAGE = 10
            total_rows = len(disp_df)
            total_pages = max(1, (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)

            if 'tbl_page' not in st.session_state:
                st.session_state['tbl_page'] = 1
            st.session_state['tbl_page'] = min(st.session_state['tbl_page'], total_pages)

            page = st.session_state['tbl_page']
            start_idx = (page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            page_df = disp_df.iloc[start_idx:end_idx]

            st.markdown(
                f"<div style='font-size:0.8rem;color:#64748B;margin:10px 0;'>"
                f"Menampilkan <b>{total_rows:,}</b> dari {len(df_filtered):,} ulasan</div>",
                unsafe_allow_html=True
            )

            header = '''
<table style="width:100%;border-collapse:collapse;font-family:'Plus Jakarta Sans',sans-serif;background:#FFFFFF;">
<thead>
<tr style="border-bottom:2px solid rgba(0,174,172,0.2);position:sticky;top:0;background:#F8FAFC;z-index:2;">
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;font-family:'JetBrains Mono',monospace;">No</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;font-family:'JetBrains Mono',monospace;">Tahun</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;font-family:'JetBrains Mono',monospace;">Bulan</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;font-family:'JetBrains Mono',monospace;">Tgl</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Rating</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;min-width:180px;font-family:'JetBrains Mono',monospace;">Aspek</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Sentimen</th>
<th style="text-align:left;padding:12px 10px;font-size:0.7rem;font-weight:800;color:#0A2540;letter-spacing:1px;text-transform:uppercase;min-width:260px;font-family:'JetBrains Mono',monospace;">Kutipan Ulasan</th>
</tr>
</thead><tbody>'''

            rows_html = ''
            for idx, (_, row) in enumerate(page_df.iterrows()):
                bg = '#FFFFFF' if idx % 2 == 0 else '#F8FAFC'
                row_num = start_idx + idx + 1
                year_val = str(int(row['year'])) if 'year' in disp_df.columns and pd.notna(row.get('year')) else '-'
                month_val = str(int(row['month'])) if 'month' in disp_df.columns and pd.notna(row.get('month')) else '-'
                day_val = str(int(row['day'])) if 'day' in disp_df.columns and pd.notna(row.get('day')) else '-'
                stars_val = render_stars(row['stars']) if 'stars' in disp_df.columns else '-'
                sents_row = parse_sentiments(row.get('sentiment_label', []))
                sent_html = ' '.join(SENT_ICON_HTML.get(s, s) for s in sents_row)
                asp_html = render_aspect_badges(row.get('aspect_classification', []))
                kutipan = str(row.get('text_translated', ''))
                rows_html += f'''
<tr style="background:{bg};border-bottom:1px solid #F1F5F9;">
<td style="padding:12px 10px;font-size:0.78rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;white-space:nowrap;">{row_num}</td>
<td style="padding:12px 10px;font-size:0.82rem;color:#475569;white-space:nowrap;">{year_val}</td>
<td style="padding:12px 10px;font-size:0.82rem;color:#475569;white-space:nowrap;">{month_val}</td>
<td style="padding:12px 10px;font-size:0.82rem;color:#475569;white-space:nowrap;">{day_val}</td>
<td style="padding:12px 10px;white-space:nowrap;">{stars_val}</td>
<td style="padding:12px 10px;min-width:180px;">{asp_html}</td>
<td style="padding:12px 10px;white-space:nowrap;">{sent_html}</td>
<td style="padding:12px 10px;font-size:0.82rem;color:#0A2540;line-height:1.5;min-width:260px;">{kutipan}</td>
</tr>'''

            st.markdown(
                f'<div style="overflow:auto;max-height:440px;border-radius:16px;border:1px solid rgba(0,174,172,0.2);box-shadow:0 4px 16px rgba(10,37,64,0.04);">'
                f'{header}{rows_html}</tbody></table></div>',
                unsafe_allow_html=True
            )


            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            p1, p2, p3, p4, p5 = st.columns([1, 1, 2, 1, 1])
            with p1:
                if st.button("⏮", use_container_width=True, disabled=(page == 1)):
                    st.session_state['tbl_page'] = 1
                    st.rerun()
            with p2:
                if st.button("◀ Prev", use_container_width=True, disabled=(page == 1)):
                    st.session_state['tbl_page'] -= 1
                    st.rerun()
            with p3:
                st.markdown(
                    f"<div style='text-align:center;padding:6px 0;font-size:0.82rem;color:#4a7a8a;'>"
                    f"Halaman <b style='color:#1B3A6B'>{page}</b> / <b style='color:#1B3A6B'>{total_pages}</b>"
                    f" · {start_idx+1}–{end_idx} dari {total_rows:,}</div>",
                    unsafe_allow_html=True
                )
            with p4:
                if st.button("Next ▶", use_container_width=True, disabled=(page == total_pages)):
                    st.session_state['tbl_page'] += 1
                    st.rerun()
            with p5:
                if st.button("⏭", use_container_width=True, disabled=(page == total_pages)):
                    st.session_state['tbl_page'] = total_pages
                    st.rerun()

# ================================
# PAGE: PREDIKSI DATASET
# ================================
elif menu == "Prediksi Dataset":
    st.markdown(
        '<div class="section-title">🧪 Prediksi Dataset Baru</div>'
        '<div class="section-subtitle">Upload data ulasan mentah (tanggal, rating, teks) untuk diprediksi sentimen dan aspeknya secara otomatis</div>',
        unsafe_allow_html=True
    )

    raw_file = st.file_uploader(
        "Upload data mentah (CSV/Excel) — kolom: publishedAtDate, stars, text",
        type=['csv', 'xlsx'],
        key="raw_uploader"
    )

    if raw_file:
        raw_df = pd.read_csv(raw_file) if raw_file.name.endswith('.csv') else pd.read_excel(raw_file)

        required_cols = {'publishedAtDate', 'stars', 'text'}
        missing = required_cols - set(raw_df.columns)
        if missing:
            st.error(f"❌ Kolom wajib tidak ditemukan: {', '.join(missing)}")
            st.stop()

        st.success(f"✅ {len(raw_df)} baris data berhasil dimuat!")

        raw_df['text'] = raw_df['text'].astype(str).str.strip()
        n_before = len(raw_df)
        raw_df = raw_df[raw_df['text'].notna() & (raw_df['text'] != '') & (raw_df['text'].str.lower() != 'nan')]
        n_skipped = n_before - len(raw_df)
        if n_skipped > 0:
            st.warning(f"⚠️ {n_skipped} baris dilewati karena teks ulasan kosong.")

        st.markdown(f"<div style='font-size:0.85rem;color:#4a7a8a;margin:6px 0 16px 0;'>Siap diproses: <b>{len(raw_df)}</b> ulasan</div>", unsafe_allow_html=True)

        run_btn = st.button("🚀 Jalankan Prediksi", type="primary")

        if run_btn:
            with st.spinner("Memuat model IndoBERT..."):
                try:
                    model, tokenizer, device = load_model()
                except Exception as e:
                    st.error(f"❌ Gagal memuat model: {e}")
                    st.stop()

            raw_df = raw_df.reset_index(drop=True)
            total_n = len(raw_df)

            progress_bar = st.progress(0, text="Memulai proses...")

            text_translated_list = []
            sentiment_list = []
            aspect_list = []

            BATCH_SIZE = 16
            for start in range(0, total_n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total_n)
                batch_texts_raw = raw_df['text'].iloc[start:end].tolist()

                batch_translated = [detect_and_translate(t) for t in batch_texts_raw]
                text_translated_list.extend(batch_translated)

                sents, aspects = predict(batch_translated, model, tokenizer, device)
                sentiment_list.extend(sents)
                aspect_list.extend(aspects)

                pct = end / total_n
                progress_bar.progress(pct, text=f"Memproses {end}/{total_n} ulasan...")

            progress_bar.empty()

            raw_df['text_translated'] = text_translated_list
            raw_df['sentiment_label'] = sentiment_list
            raw_df['aspect_classification'] = aspect_list

            year, month, day = parse_published_date(raw_df['publishedAtDate'])
            raw_df['year'] = year
            raw_df['month'] = month
            raw_df['day'] = day

            st.session_state['predicted_df'] = raw_df
            st.success("✅ Prediksi selesai! Dashboard ditampilkan di bawah.")

    if 'predicted_df' in st.session_state:
        df = st.session_state['predicted_df']

        if all(c in df.columns for c in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(
                dict(year=df['year'], month=df['month'], day=df['day']),
                errors='coerce'
            )
        else:
            df['date'] = pd.NaT

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Dashboard Hasil Prediksi</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">📅 Periode Data</div>'
            '<div class="section-subtitle">Pilih rentang waktu untuk menampilkan dashboard</div>',
            unsafe_allow_html=True
        )
        fp1, fp2 = st.columns([2, 3])
        with fp1:
            periode2 = st.selectbox(
                "Tampilkan data",
                ["Semua Data", "Tahun Ini", "3 Bulan Terakhir", "6 Bulan Terakhir", "Periode Kustom"],
                key="periode_prediksi"
            )

        df_filtered2 = df.copy()
        valid_dates2 = df['date'].dropna()

        if periode2 != "Semua Data" and not valid_dates2.empty:
            max_date2 = valid_dates2.max()

            if periode2 == "Tahun Ini":
                df_filtered2 = df[df['date'].dt.year == max_date2.year]
            elif periode2 == "3 Bulan Terakhir":
                cutoff2 = max_date2 - pd.DateOffset(months=3)
                df_filtered2 = df[df['date'] >= cutoff2]
            elif periode2 == "6 Bulan Terakhir":
                cutoff2 = max_date2 - pd.DateOffset(months=6)
                df_filtered2 = df[df['date'] >= cutoff2]
            elif periode2 == "Periode Kustom":
                with fp2:
                    min_d2, max_d2 = valid_dates2.min().date(), valid_dates2.max().date()
                    date_range2 = st.date_input(
                        "Pilih rentang tanggal",
                        value=(min_d2, max_d2),
                        min_value=min_d2,
                        max_value=max_d2,
                        key="daterange_prediksi"
                    )
                if isinstance(date_range2, tuple) and len(date_range2) == 2:
                    start_d2, end_d2 = date_range2
                    df_filtered2 = df[
                        (df['date'].dt.date >= start_d2) & (df['date'].dt.date <= end_d2)
                    ]

        st.markdown(
            f"<div style='font-size:0.78rem;color:#4a7a8a;margin:4px 0 18px 0;'>"
            f"Menampilkan <b>{len(df_filtered2):,}</b> dari {len(df):,} total ulasan</div>",
            unsafe_allow_html=True
        )

        m = compute_metrics(df_filtered2)

        c1, c2, c3, c4 = st.columns(4)
        for col, cls, label, val, pct in [
            (c1, 'total', 'Total Ulasan', m['total'], 'Keseluruhan data'),
            (c2, 'positive', 'Sentimen Positif', m['pos'], f"{m['pos_pct']:.1f}% dari total"),
            (c3, 'negative', 'Sentimen Negatif', m['neg'], f"{m['neg_pct']:.1f}% dari total"),
            (c4, 'neutral', 'Sentimen Netral', m['neu'], f"{m['neu_pct']:.1f}% dari total"),
        ]:
            with col:
                st.markdown(f'''
<div class="metric-card {cls}">
<div class="metric-label">{label}</div>
<div class="metric-value">{val:,}</div>
<div class="metric-pct">{pct}</div>
</div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        ci, cp = st.columns(2)
        with ci:
            st.markdown(
                '<div class="section-title">💡 Insight Aspek</div><div class="section-subtitle">Aspek terbaik dan yang perlu perhatian</div>',
                unsafe_allow_html=True
            )
            for key, cls, emoji, label in [
                (m['best'], 'insight-best', '⭐', 'Aspek Terbaik'),
                (m['worst'], 'insight-warn', '⚠️', 'Perlu Perhatian')
            ]:
                c = m['asp_sent'][key]
                t = sum(c.values())
                pct_val = (c['positive'] / t * 100 if t > 0 else 0) if label == 'Aspek Terbaik' else (c['negative'] / t * 100 if t > 0 else 0)
                pct_color = '#34d399' if label == 'Aspek Terbaik' else '#f87171'
                pct_label = 'positif' if label == 'Aspek Terbaik' else 'negatif'
                st.markdown(f'''
<div class="insight-card {cls}">
<div class="insight-title">{emoji} {label}</div>
<div class="insight-aspect">{ASPECT_DISPLAY[key]}</div>
<div class="insight-desc">
{c['positive']} positif · {c['neutral']} netral · {c['negative']} negatif
 |  <b style="color:{pct_color}">{pct_val:.0f}% {pct_label}</b>
</div>
</div>''', unsafe_allow_html=True)

        with cp:
            st.markdown(
                '<div class="section-title">🥧 Distribusi Sentimen</div><div class="section-subtitle">Keseluruhan ulasan</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(chart_pie(m['pos'], m['neg'], m['neu']), use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="section-title">📊 Distribusi Sentimen per Aspek</div><div class="section-subtitle">Positif (hijau) · Netral (abu) · Negatif (merah)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_bar(m['asp_sent']), use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="section-title">🏆 Peringkat Aspek SKM</div><div class="section-subtitle">Urutan dari sentimen paling positif ke yang memerlukan perhatian khusus</div>', unsafe_allow_html=True)
        sorted_asp = sorted(m['scores'].items(), key=lambda x: x[1], reverse=True)
        total_all = m['total'] or 1
        html = ''
        for i, (asp, score) in enumerate(sorted_asp):
            c = m['asp_sent'][asp]
            pos_pct = c['positive'] / total_all * 100
            neu_pct = c['neutral'] / total_all * 100
            neg_pct = c['negative'] / total_all * 100
            total_asp = c['positive'] + c['neutral'] + c['negative']
            html += f'''
<div style="background:#FFFFFF;border:1px solid rgba(0,174,172,0.18);border-radius:14px;padding:16px 20px;margin-bottom:12px;box-shadow:0 2px 10px rgba(10,37,64,0.03);">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="display:flex;align-items:center;gap:12px;">
<span style="font-family:'Outfit',sans-serif;font-size:0.8rem;font-weight:800;color:#FFFFFF;background:linear-gradient(135deg,#0A2540,#00AEAC);width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{i+1}</span>
<span style="font-family:'Outfit',sans-serif;font-size:0.98rem;font-weight:700;color:#0A2540;">{ASPECT_DISPLAY[asp]}</span>
</div>
<span style="font-size:0.78rem;color:#64748B;font-family:'JetBrains Mono',monospace;"><b>{total_asp:,}</b> ulasan</span>
</div>
<div style="width:100%;height:12px;border-radius:8px;overflow:hidden;display:flex;background:#F1F5F9;">
<div style="width:{pos_pct:.2f}%;background:linear-gradient(90deg,#059669,#34D399);height:100%;"></div>
<div style="width:{neu_pct:.2f}%;background:#94A3B8;height:100%;"></div>
<div style="width:{neg_pct:.2f}%;background:linear-gradient(90deg,#E11D48,#FB7185);height:100%;"></div>
</div>
<div style="display:flex;gap:18px;margin-top:8px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;">
<span style="font-size:0.74rem;color:#059669;">● Positif <b>{c['positive']:,}</b> <span style="color:#64748B;">({pos_pct:.1f}%)</span></span>
<span style="font-size:0.74rem;color:#64748B;">● Netral <b>{c['neutral']:,}</b> <span style="color:#64748B;">({neu_pct:.1f}%)</span></span>
<span style="font-size:0.74rem;color:#E11D48;">● Negatif <b>{c['negative']:,}</b> <span style="color:#64748B;">({neg_pct:.1f}%)</span></span>
</div>
</div>'''
        st.markdown(html, unsafe_allow_html=True)

        pdf_bytes2 = generate_pdf_report(m, periode2, "Prediksi Dataset")
        st.download_button(
            "📄 Download Laporan PDF",
            data=pdf_bytes2,
            file_name="laporan_prediksi_absa.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf_pred"
        )

        with st.expander("📋 Lihat Data Lengkap"):
            fc1b, fc2b, fc3b, fc4b, fc5b = st.columns([2, 2, 2, 2, 3])
            with fc1b:
                sort_waktu2 = st.selectbox("Waktu", ["Semua", "Terbaru", "Terlama"], key="sort_waktu_pred")
            with fc2b:
                sort_rating2 = st.selectbox("Rating", ["Semua", "Tertinggi", "Terendah"], key="sort_rating_pred")
            with fc3b:
                rating_options2 = ["5 ⭐", "4 ⭐", "3 ⭐", "2 ⭐", "1 ⭐"]
                filter_rating2 = st.multiselect("Filter Rating", options=rating_options2, default=[], placeholder="Semua rating", key="filter_rating_pred")
            with fc4b:
                sent_options2 = ["Positif", "Netral", "Negatif"]
                filter_sent2 = st.multiselect("Filter Sentimen", options=sent_options2, default=[], placeholder="Semua sentimen", key="filter_sent_pred")
            with fc5b:
                asp_options2 = [ASPECT_DISPLAY[a] for a in ASPECT_LABELS]
                filter_asp_display2 = st.multiselect("Filter Aspek", options=asp_options2, default=[], placeholder="Semua aspek", key="filter_asp_pred")
            asp_display_to_key2 = {v: k for k, v in ASPECT_DISPLAY.items()}
            filter_asp_keys2 = [asp_display_to_key2[d] for d in filter_asp_display2 if d in asp_display_to_key2]

            disp_df2 = df_filtered2.copy()

            if filter_sent2:
                sent_map_filter2 = {"Positif": "positive", "Netral": "neutral", "Negatif": "negative"}
                selected_sent_keys2 = [sent_map_filter2[s] for s in filter_sent2]
                disp_df2 = disp_df2[disp_df2['sentiment_label'].apply(
                    lambda x: any(s in parse_sentiments(x) for s in selected_sent_keys2)
                )]

            if filter_rating2 and 'stars' in disp_df2.columns:
                selected_stars2 = [int(r[0]) for r in filter_rating2]
                disp_df2 = disp_df2[disp_df2['stars'].apply(
                    lambda x: int(float(x)) in selected_stars2 if pd.notna(x) else False
                )]

            if filter_asp_keys2:
                disp_df2 = disp_df2[disp_df2['aspect_classification'].apply(
                    lambda a: any(
                        k in (ast.literal_eval(a) if isinstance(a, str) else a)
                        for k in filter_asp_keys2
                    )
                )]

            has_date2 = all(c in disp_df2.columns for c in ['year', 'month', 'day'])
            has_stars2 = 'stars' in disp_df2.columns

            if sort_waktu2 == "Terbaru" and has_date2:
                disp_df2 = disp_df2.sort_values(['year', 'month', 'day'], ascending=[False, False, False])
            elif sort_waktu2 == "Terlama" and has_date2:
                disp_df2 = disp_df2.sort_values(['year', 'month', 'day'], ascending=[True, True, True])

            if sort_rating2 == "Tertinggi" and has_stars2:
                disp_df2 = disp_df2.sort_values('stars', ascending=False)
            elif sort_rating2 == "Terendah" and has_stars2:
                disp_df2 = disp_df2.sort_values('stars', ascending=True)

            ROWS_PER_PAGE2 = 10
            total_rows2 = len(disp_df2)
            total_pages2 = max(1, (total_rows2 + ROWS_PER_PAGE2 - 1) // ROWS_PER_PAGE2)

            if 'tbl_page_pred' not in st.session_state:
                st.session_state['tbl_page_pred'] = 1
            st.session_state['tbl_page_pred'] = min(st.session_state['tbl_page_pred'], total_pages2)

            page2 = st.session_state['tbl_page_pred']
            start_idx2 = (page2 - 1) * ROWS_PER_PAGE2
            end_idx2 = min(start_idx2 + ROWS_PER_PAGE2, total_rows2)
            page_df2 = disp_df2.iloc[start_idx2:end_idx2]

            st.markdown(
                f"<div style='font-size:0.8rem;color:#4a7a8a;margin:10px 0;'>"
                f"Menampilkan <b>{total_rows2:,}</b> dari {len(df_filtered2):,} ulasan</div>",
                unsafe_allow_html=True
            )

            header2 = '''
<table style="width:100%;border-collapse:collapse;font-family:Plus Jakarta Sans,sans-serif;">
<thead>
<tr style="border-bottom:2px solid #b2dede;position:sticky;top:0;background:#f8fcfc;z-index:2;">
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;">No</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;">Tahun</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;">Bulan</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;">Tgl</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;">Rating</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;min-width:180px;">Aspek</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;">Sentimen</th>
<th style="text-align:left;padding:10px 8px;font-size:0.7rem;font-weight:700;color:#1B3A6B;letter-spacing:1px;text-transform:uppercase;min-width:260px;">Kutipan</th>
</tr>
</thead><tbody>'''

            rows_html2 = ''
            for idx, (_, row) in enumerate(page_df2.iterrows()):
                bg2 = '#ffffff' if idx % 2 == 0 else '#f8fcfc'
                row_num2 = start_idx2 + idx + 1
                year_val2 = str(int(row['year'])) if 'year' in disp_df2.columns and pd.notna(row.get('year')) else '-'
                month_val2 = str(int(row['month'])) if 'month' in disp_df2.columns and pd.notna(row.get('month')) else '-'
                day_val2 = str(int(row['day'])) if 'day' in disp_df2.columns and pd.notna(row.get('day')) else '-'
                stars_val2 = render_stars(row['stars']) if 'stars' in disp_df2.columns else '-'
                sents_row2 = parse_sentiments(row.get('sentiment_label', []))
                sent_html2 = ' '.join(SENT_ICON_HTML.get(s, s) for s in sents_row2)
                asp_html2 = render_aspect_badges(row.get('aspect_classification', []))
                kutipan2 = str(row.get('text_translated', ''))
                rows_html2 += f'''
<tr style="background:{bg2};border-bottom:1px solid #e6f0f0;">
<td style="padding:10px 8px;font-size:0.78rem;color:#94a3b8;font-family:Space Mono,monospace;white-space:nowrap;">{row_num2}</td>
<td style="padding:10px 8px;font-size:0.8rem;color:#4a7a8a;white-space:nowrap;">{year_val2}</td>
<td style="padding:10px 8px;font-size:0.8rem;color:#4a7a8a;white-space:nowrap;">{month_val2}</td>
<td style="padding:10px 8px;font-size:0.8rem;color:#4a7a8a;white-space:nowrap;">{day_val2}</td>
<td style="padding:10px 8px;white-space:nowrap;">{stars_val2}</td>
<td style="padding:10px 8px;min-width:180px;">{asp_html2}</td>
<td style="padding:10px 8px;white-space:nowrap;">{sent_html2}</td>
<td style="padding:10px 8px;font-size:0.8rem;color:#1B3A6B;line-height:1.5;min-width:260px;">{kutipan2}</td>
</tr>'''

            st.markdown(
                f'<div style="overflow:auto;max-height:420px;border-radius:12px;border:1px solid #b2dede;">'
                f'{header2}{rows_html2}</tbody></table></div>',
                unsafe_allow_html=True
            )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            pp1, pp2, pp3, pp4, pp5 = st.columns([1, 1, 2, 1, 1])
            with pp1:
                if st.button("⏮", use_container_width=True, disabled=(page2 == 1), key="first_pred"):
                    st.session_state['tbl_page_pred'] = 1
                    st.rerun()
            with pp2:
                if st.button("◀ Prev", use_container_width=True, disabled=(page2 == 1), key="prev_pred"):
                    st.session_state['tbl_page_pred'] -= 1
                    st.rerun()
            with pp3:
                st.markdown(
                    f"<div style='text-align:center;padding:6px 0;font-size:0.82rem;color:#4a7a8a;'>"
                    f"Halaman <b style='color:#1B3A6B'>{page2}</b> / <b style='color:#1B3A6B'>{total_pages2}</b>"
                    f" · {start_idx2+1}–{end_idx2} dari {total_rows2:,}</div>",
                    unsafe_allow_html=True
                )
            with pp4:
                if st.button("Next ▶", use_container_width=True, disabled=(page2 == total_pages2), key="next_pred"):
                    st.session_state['tbl_page_pred'] += 1
                    st.rerun()
            with pp5:
                if st.button("⏭", use_container_width=True, disabled=(page2 == total_pages2), key="last_pred"):
                    st.session_state['tbl_page_pred'] = total_pages2
                    st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        dlf1, dlf2 = st.columns([1, 2])
        with dlf1:
            format_pilihan = st.selectbox(
                "Format file",
                ["Excel (.xlsx)", "CSV (.csv)"],
                key="format_download_pred",
                label_visibility="collapsed"
            )
        with dlf2:
            df_export = df.drop(columns=['date']) if 'date' in df.columns else df
            if format_pilihan == "Excel (.xlsx)":
                from io import BytesIO
                excel_buffer = BytesIO()
                df_export.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    data=excel_buffer.getvalue(),
                    file_name="hasil_prediksi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_pred_xlsx",
                    use_container_width=True
                )
            else:
                csv_export = df_export.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    data=csv_export,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="download_pred_csv",
                    use_container_width=True
                )

# ================================
# PAGE: SIMULASI
# ================================
elif menu == "Simulasi Prediksi":
    st.markdown('<div class="section-title">🔍 Simulasi Prediksi ABSA</div><div class="section-subtitle">Masukkan ulasan layanan kesehatan untuk diprediksi sentimen dan aspeknya</div>', unsafe_allow_html=True)

    with st.spinner("Memuat model IndoBERT..."):
        try:
            model, tokenizer, device = load_model()
            st.success("✅ Model siap!")
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            st.stop()

    input_text = st.text_area(
        "Masukkan ulasan (satu per baris)",
        placeholder="Dokternya sangat ramah dan sabar menjelaskan kondisi saya...\nAntriannya sangat panjang, saya menunggu 3 jam tanpa kejelasan...\nFasilitas ruang tunggu cukup bersih dan nyaman.",
        height=160
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🔍 Prediksi", type="primary", use_container_width=True)

    if predict_btn and input_text.strip():
        texts = [t.strip() for t in input_text.strip().split('\n') if t.strip()]
        with st.spinner(f"Memproses {len(texts)} ulasan..."):
            sents, aspects = predict(texts, model, tokenizer, device)

        st.markdown(f"<div style='margin:20px 0 12px 0;font-size:0.85rem;color:#64748b;'>Hasil prediksi untuk <b style='color:#1B3A6B'>{len(texts)} ulasan</b></div>", unsafe_allow_html=True)

        for i, (text, sents_i, asps) in enumerate(zip(texts, sents, aspects)):
            prim = primary_sentiment(sents_i)
            badges_sent = ''.join(
                f'<span class="badge badge-{s}" style="font-size:0.8rem;">{SENT_EMOJI[s]} {SENT_LABEL[s]}</span>'
                for s in sents_i
            )
            badges_asp = ''.join(
                f'<span class="badge badge-aspect">{ASPECT_DISPLAY.get(a, a)}</span>' for a in asps
            )
            st.markdown(f'''
<div class="sim-result-card sim-sent-{prim}">
<div style="font-size:0.72rem;color:#64748b;font-family:Space Mono,monospace;margin-bottom:8px;">ULASAN #{i+1}</div>
<div style="font-size:0.95rem;color:#1B3A6B;margin-bottom:12px;line-height:1.6;">"{text}"</div>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
{badges_sent}{badges_asp}
</div>
</div>''', unsafe_allow_html=True)

        if len(texts) > 1:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Ringkasan</div>', unsafe_allow_html=True)
            from collections import Counter
            sent_count = Counter([s for sub in sents for s in sub])
            asp_count = Counter([a for asp in aspects for a in asp])
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Distribusi Sentimen**")
                for s, cnt in sent_count.most_common():
                    color = SENT_COLOR[s]
                    pct = cnt / len(texts) * 100
                    st.markdown(f'''
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
<span style="width:80px;font-size:0.82rem;color:#4a7a8a;">{SENT_EMOJI[s]} {SENT_LABEL[s]}</span>
<div style="flex:1;height:8px;background:#e5e7eb;border-radius:4px;">
<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
</div>
<span style="font-size:0.82rem;color:#64748b;width:60px;text-align:right;">{cnt} ({pct:.0f}%)</span>
</div>''', unsafe_allow_html=True)
            with rc2:
                st.markdown("**Aspek yang Muncul**")
                for asp, cnt in asp_count.most_common():
                    st.markdown(f'<span class="badge badge-aspect">{ASPECT_DISPLAY.get(asp, asp)}</span> <span style="font-size:0.8rem;color:#64748b;">{cnt}x</span><br>', unsafe_allow_html=True)

# ================================
# PAGE: INDIKATOR PENILAIAN PELAYANAN
# ================================
elif menu == "Indikator Penilaian Pelayanan":
    st.markdown(
        '<div class="section-title">🧠 Indikator Penilaian Pelayanan per Aspek</div>'
        '<div class="section-subtitle">Menampilkan kata-kata yang paling memengaruhi penilaian pasien terhadap setiap aspek pelayanan. Informasi ini membantu mengidentifikasi faktor yang menjadi kekuatan maupun area yang perlu ditingkatkan.</div>',
        unsafe_allow_html=True
    )

    source_options = []
    if 'predicted_df' in st.session_state:
        source_options.append("Hasil Prediksi Dataset (halaman Prediksi Dataset)")
    if 'hasil_analisis_df' in st.session_state:
        source_options.append("Hasil Analisis yang diupload (halaman Hasil Analisis)")
    source_options.append("Upload file baru")

    src_choice = st.radio("Sumber data", source_options, horizontal=False)

    shap_df = None
    if src_choice == "Hasil Prediksi Dataset (halaman Prediksi Dataset)":
        shap_df = st.session_state['predicted_df']
    elif src_choice == "Hasil Analisis yang diupload (halaman Hasil Analisis)":
        shap_df = st.session_state['hasil_analisis_df']
    else:
        shap_file = st.file_uploader(
            "Upload file (CSV/Excel) — kolom: text_translated (atau text), sentiment_label, aspect_classification",
            type=['csv', 'xlsx'],
            key="shap_uploader"
        )
        if shap_file:
            shap_df = pd.read_csv(shap_file, sep=None, engine='python') if shap_file.name.endswith('.csv') else pd.read_excel(shap_file)
            if 'sentiment_label_reviewed' in shap_df.columns:
                shap_df['sentiment_label'] = shap_df['sentiment_label_reviewed']
            if 'aspect_classification_reviewed' in shap_df.columns:
                shap_df['aspect_classification'] = shap_df['aspect_classification_reviewed']

    if shap_df is not None:
        required = {'sentiment_label', 'aspect_classification'}
        has_text = 'text_translated' in shap_df.columns or 'text' in shap_df.columns
        if not required.issubset(shap_df.columns) or not has_text:
            st.error("❌ File butuh kolom `sentiment_label`, `aspect_classification`, dan `text_translated` (atau `text`).")
            st.stop()

        st.success(f"✅ {len(shap_df)} ulasan siap dianalisis.")

        if all(c in shap_df.columns for c in ['year', 'month', 'day']):
            shap_df = shap_df.copy()
            shap_df['date'] = pd.to_datetime(
                dict(year=shap_df['year'], month=shap_df['month'], day=shap_df['day']),
                errors='coerce'
            )
        else:
            shap_df = shap_df.copy()
            shap_df['date'] = pd.NaT

        has_date_col = shap_df['date'].notna().any()

        if has_date_col:
            st.markdown(
                '<div class="section-title" style="margin-top:8px;">📅 Periode Data</div>'
                '<div class="section-subtitle">Pilih rentang waktu ulasan yang mau dianalisis SHAP-nya</div>',
                unsafe_allow_html=True
            )
            fps1, fps2 = st.columns([2, 3])
            with fps1:
                periode_shap = st.selectbox(
                    "Tampilkan data",
                    ["Semua Data", "Tahun Ini", "3 Bulan Terakhir", "6 Bulan Terakhir", "Periode Kustom"],
                    key="periode_shap"
                )

            shap_df_period = shap_df.copy()
            valid_dates_shap = shap_df['date'].dropna()

            if periode_shap != "Semua Data" and not valid_dates_shap.empty:
                max_date_shap = valid_dates_shap.max()
                if periode_shap == "Tahun Ini":
                    shap_df_period = shap_df[shap_df['date'].dt.year == max_date_shap.year]
                elif periode_shap == "3 Bulan Terakhir":
                    cutoff_shap = max_date_shap - pd.DateOffset(months=3)
                    shap_df_period = shap_df[shap_df['date'] >= cutoff_shap]
                elif periode_shap == "6 Bulan Terakhir":
                    cutoff_shap = max_date_shap - pd.DateOffset(months=6)
                    shap_df_period = shap_df[shap_df['date'] >= cutoff_shap]
                elif periode_shap == "Periode Kustom":
                    with fps2:
                        min_d_s, max_d_s = valid_dates_shap.min().date(), valid_dates_shap.max().date()
                        date_range_shap = st.date_input(
                            "Pilih rentang tanggal",
                            value=(min_d_s, max_d_s),
                            min_value=min_d_s,
                            max_value=max_d_s,
                            key="daterange_shap"
                        )
                    if isinstance(date_range_shap, tuple) and len(date_range_shap) == 2:
                        start_d_s, end_d_s = date_range_shap
                        shap_df_period = shap_df[
                            (shap_df['date'].dt.date >= start_d_s) & (shap_df['date'].dt.date <= end_d_s)
                        ]

            st.markdown(
                f"<div style='font-size:0.78rem;color:#4a7a8a;margin:4px 0 18px 0;'>"
                f"Menampilkan <b>{len(shap_df_period):,}</b> dari {len(shap_df):,} total ulasan</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ File ini tidak punya kolom tanggal (year/month/day), jadi filter periode tidak tersedia — seluruh data dipakai.")
            shap_df_period = shap_df

        st.markdown(
            '<div class="section-subtitle" style="margin-top:8px;">⏳ Mohon tunggu sebentar. Sistem sedang menyiapkan insight dan rekomendasi berdasarkan hasil analisis sehingga membutuhkan waktu beberapa saat.</div>',
            unsafe_allow_html=True
        )

        MAX_SHAP_ROWS = 300
        n_period = len(shap_df_period)

        process_all = False
        if n_period > MAX_SHAP_ROWS:
            process_all = st.checkbox(
                f"Proses semua {n_period:,} ulasan pada periode ini (lewati batas {MAX_SHAP_ROWS}, akan jauh lebih lama)"
            )
            if process_all:
                st.warning(f"⚠️ Memproses semua {n_period:,} ulasan bisa memakan waktu lama.")
            else:
                st.warning(f"⚠️ Periode ini punya {n_period:,} ulasan. Demi performa, sistem hanya akan menghitung sampel acak sebanyak {MAX_SHAP_ROWS} ulasan dari periode ini.")

        max_rows = n_period if process_all else (min(MAX_SHAP_ROWS, n_period) if n_period > 0 else 0)

        sc2, sc3 = st.columns(2)
        with sc2:
            top_n = st.slider("Jumlah kata teratas per aspek", min_value=3, max_value=15, value=8)
        with sc3:
            min_occurrence = st.slider("Minimal kemunculan kata", min_value=1, max_value=5, value=2)

        run_shap = st.button("🧠 Jalankan Analisis SHAP", type="primary", disabled=(n_period == 0))
        if n_period == 0:
            st.error("❌ Tidak ada ulasan pada periode yang dipilih.")

        if run_shap:
            with st.spinner("Memuat model IndoBERT..."):
                try:
                    model, tokenizer, device = load_model()
                except Exception as e:
                    st.error(f"❌ Gagal memuat model: {e}")
                    st.stop()

            progress_bar = st.progress(0, text="Memulai analisis SHAP...")

            def _cb(done, total):
                progress_bar.progress(done / total, text=f"Menghitung SHAP {done}/{total} ulasan...")

            with st.spinner("Menghitung kontribusi kata (bisa memakan waktu beberapa menit)..."):
                shap_result = compute_aspect_word_drivers(
                    shap_df_period, model, tokenizer, device,
                    max_rows=max_rows, top_n=top_n, min_occurrence=min_occurrence,
                    progress_cb=_cb,
                )
            progress_bar.empty()
            st.session_state['shap_result'] = shap_result
            st.session_state['shap_sample_size'] = max_rows
            st.success("✅ Analisis SHAP selesai!")

    if 'shap_result' in st.session_state:
        shap_result = st.session_state['shap_result']
        sample_size = st.session_state.get('shap_sample_size', '?')
        recommendation_kb = load_recommendation_kb()

        if not shap_result:
            st.warning("⚠️ Tidak ada kata yang cukup dominan ditemukan pada sampel ini. Coba perbesar jumlah sampel atau turunkan minimal kemunculan kata.")
        else:
            st.markdown(
                f"<div style='font-size:0.78rem;color:#4a7a8a;margin:16px 0;'>"
                f"Analisis dihitung dari sampel <b>{sample_size}</b> ulasan</div>",
                unsafe_allow_html=True
            )

            view_mode = st.radio(
                "Tampilan",
                ["📌 Per Aspek (aspek → positif/negatif)", "🎭 Per Sentimen (positif/negatif → aspek)"],
                horizontal=True,
            )

            ordered_aspects = [a for a in ASPECT_LABELS if a in shap_result]

            if view_mode.startswith("📌"):
                for asp in ordered_aspects:
                    data = shap_result[asp]
                    st.markdown(f'<div class="section-title" style="margin-top:20px;">📌 {ASPECT_DISPLAY[asp]}</div>', unsafe_allow_html=True)

                    colp, coln = st.columns(2)
                    with colp:
                        st.markdown(
                            '<div class="insight-card insight-best" style="padding:14px 18px;">'
                            '<div class="insight-title">⭐ Pemicu Sentimen Positif</div></div>',
                            unsafe_allow_html=True
                        )
                        pos_words = data.get('positive', [])
                        if pos_words:
                            fig = chart_word_drivers(pos_words, '#059669')
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.markdown("<div style='font-size:0.8rem;color:#94a3b8;padding:8px 0;'>Belum ada kata dominan pada sampel ini.</div>", unsafe_allow_html=True)

                    with coln:
                        st.markdown(
                            '<div class="insight-card insight-warn" style="padding:14px 18px;">'
                            '<div class="insight-title">⚠️ Pemicu Sentimen Negatif</div></div>',
                            unsafe_allow_html=True
                        )
                        neg_words = data.get('negative', [])
                        if neg_words:
                            fig = chart_word_drivers(neg_words, '#E11D48')
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.markdown("<div style='font-size:0.8rem;color:#94a3b8;padding:8px 0;'>Belum ada kata dominan pada sampel ini.</div>", unsafe_allow_html=True)

                    # --- Rekomendasi otomatis dari kata-kata negatif yang terdeteksi SHAP ---
                    neg_words = data.get('negative', [])
                    if neg_words:
                        match_info = find_matched_recommendations(asp, neg_words, recommendation_kb)
                        if match_info:
                            render_recommendation_card(match_info)
            else:
                for sent_key, sent_label, emoji, card_cls, color in [
                    ('positive', 'Sentimen Positif', '⭐', 'insight-best', '#059669'),
                    ('negative', 'Sentimen Negatif', '⚠️', 'insight-warn', '#E11D48'),
                ]:
                    aspects_with_data = [a for a in ordered_aspects if shap_result[a].get(sent_key)]

                    st.markdown(
                        f'<div class="section-title" style="margin-top:28px;">{emoji} {sent_label}</div>'
                        f'<div class="section-subtitle">Kata-kata pemicu {"positif" if sent_key == "positive" else "negatif"} di tiap aspek</div>',
                        unsafe_allow_html=True
                    )

                    if not aspects_with_data:
                        st.markdown("<div style='font-size:0.8rem;color:#94a3b8;padding:8px 0 16px 0;'>Belum ada kata dominan pada sampel ini.</div>", unsafe_allow_html=True)
                        continue

                    cols = st.columns(2)
                    for i, asp in enumerate(aspects_with_data):
                        words = shap_result[asp].get(sent_key, [])
                        with cols[i % 2]:
                            st.markdown(
                                f'<div class="insight-card {card_cls}" style="padding:14px 18px;">'
                                f'<div class="insight-title">{ASPECT_DISPLAY[asp]}</div></div>',
                                unsafe_allow_html=True
                            )
                            fig = chart_word_drivers(words, color)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                            # --- Rekomendasi otomatis (khusus sisi negatif) ---
                            if sent_key == 'negative':
                                match_info = find_matched_recommendations(asp, words, recommendation_kb)
                                if match_info:
                                    with st.expander("💊 Lihat Rekomendasi Perbaikan"):
                                        render_recommendation_card(match_info)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.75rem;color:#94a3b8;'>Catatan: model memprediksi sentimen ulasan secara keseluruhan (bukan per-aspek), "
                "sehingga kata pemicu di atas dihitung dari sentimen keseluruhan ulasan yang mengandung aspek tersebut.</div>",
                unsafe_allow_html=True
            )