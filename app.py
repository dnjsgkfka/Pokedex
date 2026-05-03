import json
from pathlib import Path
import io, base64

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from PIL import Image


st.set_page_config(
    page_title="Pokédex AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    background: #F5F5FF;
    color: #1A1A2E;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui,
        'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}

/* ── hide streamlit chrome (흰색 박스 포함) ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

.block-container { padding: 2rem 4% !important; max-width: 1280px !important; margin: 0 auto; }

/* ── NAV ── */
.nav-bar { display:flex; align-items:center; margin-bottom:2rem; }
.nav-logo-text { font-weight:900; font-size:1.45rem; color:#4F46E5; letter-spacing:-0.03em; }

/* ── LEFT ── */
.eyebrow { font-size:0.75rem; font-weight:700; color:#4F46E5;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px; }
.headline { font-size:clamp(2rem,3vw,2.9rem); font-weight:900; line-height:1.22;
    color:#1A1A2E; margin-bottom:16px; letter-spacing:-0.03em; word-break:keep-all; }
.headline span { color:#4F46E5; }
.desc { font-size:0.93rem; line-height:1.7; color:#6B7280; margin-bottom:32px; word-break:keep-all; }
.field-label { font-size:0.8rem; font-weight:700; color:#374151; margin-bottom:10px; }
.divider { width:100%; height:1px; background:#F0F0FF; margin:24px 0; }
.meta-tag { display:inline-flex; align-items:center; gap:6px; margin-top:32px;
    font-size:0.76rem; color:#9CA3AF; font-weight:500; }
.meta-dot { width:6px; height:6px; border-radius:50%; background:#A5B4FC; display:inline-block; }

/* ── RADIO ── */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display:none !important; }

[data-testid="stRadio"] [role="radiogroup"] {
    display:grid !important;
    grid-template-columns:1fr 1fr !important;
    gap:8px !important;
}
[data-testid="stRadio"] [role="radiogroup"] label {
    background:#F9F9FF !important; border:1.5px solid #E0E0FF !important;
    border-radius:12px !important; padding:10px 14px !important;
    cursor:pointer !important; transition:all .15s !important;
    display:flex !important; align-items:center !important;
    justify-content:center !important;
}
[data-testid="stRadio"] [role="radiogroup"] label:hover {
    border-color:#4F46E5 !important; background:#EEF2FF !important; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none !important; }
[data-testid="stRadio"] [role="radiogroup"] label p {
    font-size:0.79rem !important; font-weight:600 !important;
    color:#6B7280 !important; margin:0 !important; text-align:center !important; }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background:#EEF2FF !important; border-color:#4F46E5 !important; }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color:#4F46E5 !important; }
[data-testid="stRadio"] [role="radiogroup"] label input { display:none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { background:transparent !important; }
[data-testid="stFileUploader"] > div {
    background:#F5F5FF !important; border:2px dashed #C7D2FE !important;
    border-radius:14px !important; padding:26px 20px !important; transition:all .2s;
}
[data-testid="stFileUploader"] > div:hover { border-color:#4F46E5 !important; background:#EEF2FF !important; }
[data-testid="stFileUploader"] label { display:none !important; }
[data-testid="stFileDropzoneInstructions"] { color:#6366F1 !important; font-size:0.87rem !important; font-weight:600 !important; }

/* ── RIGHT / RESULT ── */
.panel-title { font-size:0.97rem; font-weight:800; color:#1A1A2E; margin-bottom:24px; }
.empty-box { display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:12px; background:#F9F9FF; border:2px solid #C7D2FE; border-radius:20px;
    padding:72px 24px; min-height:500px; }
.empty-text { font-size:0.9rem; color:#9CA3AF; font-weight:500; }

/* ── RESULT CARD ── */
.result-card { background:#FAFAFE; border:1px solid #E8E8FF; border-radius:20px; overflow:hidden; }

/* 이미지: 카드 상단 전체 폭 */
.rc-img-wrap { width 100%; height:350px; }
.rc-img-wrap img { width:100%; height:100%; object-fit:cover; display:block; }

/* 이미지 아래 본문 */
.rc-body { padding:18px 22px 22px; }

/* 한 줄 정보 */
.rc-info-row { display:flex; align-items:baseline; gap:8px; margin-bottom:18px; }
.rc-subtitle { font-size:0.75rem; color:#9CA3AF; font-weight:500; white-space:nowrap; }
.rc-name { font-size:1.2rem; font-weight:900; color:#4F46E5; letter-spacing:-0.02em; }
.rc-spacer { flex:1; min-width:12px; }
.rc-match-label { font-size:0.75rem; color:#9CA3AF; font-weight:500; white-space:nowrap; }
.rc-match-pct { font-size:1.2rem; font-weight:900; color:#1A1A2E; letter-spacing:-0.03em; white-space:nowrap; }
.rc-match-pct span { font-size:0.82rem; font-weight:700; color:#6B7280; }

.bar-list { display:flex; flex-direction:column; gap:9px; }
.bar-row { display:grid; grid-template-columns:100px 1fr 34px; align-items:center; gap:9px; }
.bar-label { font-size:0.8rem; font-weight:600; color:#6B7280;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-label.hi { color:#1A1A2E; font-weight:800; }
.bar-track { height:9px; background:#EBEBFF; border-radius:99px; overflow:hidden; position:relative; }
.bar-fill { position:absolute; top:0; left:0; height:100%; border-radius:99px; background:#4F46E5; }
.bar-fill.lo { background:#A5B4FC; }
.bar-pct { font-size:0.78rem; font-weight:700; color:#9CA3AF; text-align:right; }
.bar-pct.hi { color:#4F46E5; }

div[data-testid="stSpinner"] > div { border-top-color:#4F46E5 !important; }
</style>
""", unsafe_allow_html=True)


IMG_SIZE        = 224
EXPERIMENTS_DIR = Path("./experiments")

EXPERIMENT_META = {
    "exp1_resnet50_pretrained_headonly":     "ResNet-50\nHead Only",
    "exp2_resnet50_pretrained_fulltune":     "ResNet-50\nFull Tune",
    "exp3_efficientnet_pretrained_headonly": "EfficientNet\nHead Only",
    "exp4_resnet50_scratch_fulltrain":       "ResNet-50\nScratch",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_data(show_spinner=False)
def load_classes():
    path = EXPERIMENTS_DIR / "classes.json"
    if path.exists():
        return json.loads(path.read_text())
    return [f"Pokemon_{i}" for i in range(150)]


@st.cache_resource(show_spinner=False)
def load_model(exp_name, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "efficientnet" in exp_name:
        weights = EfficientNet_B0_Weights.DEFAULT if "scratch" not in exp_name else None
        model   = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, num_classes))
    else:
        weights = ResNet50_Weights.DEFAULT if "scratch" not in exp_name else None
        model   = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    weight_path = EXPERIMENTS_DIR / exp_name / "best_model.pth"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval(), device


@torch.no_grad()
def predict(model, image, device, classes, top_k=5):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    return [(classes[i], float(p)) for i, p in zip(top_idx, top_probs)]


classes     = load_classes()
num_classes = len(classes)
exp_keys    = list(EXPERIMENT_META.keys())
exp_labels  = list(EXPERIMENT_META.values())

# ── NAV ────────────────────────────────
st.markdown(
    '<div class="nav-bar">'
    '<div class="nav-logo-text">AI POKÉDEX</div>'
    '</div>',
    unsafe_allow_html=True,
)

col_left, col_gap, col_right = st.columns([1, 0.04, 1.15])

# ══════════════ LEFT ══════════════
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="eyebrow">포켓몬 이미지 분석</div>'
        '<div class="headline">이 포켓몬은<br><span>누구일까요?</span></div>'
        '<div class="desc">AI 모델이 사진을 분석해 어떤 포켓몬인지 알려드립니다.<br>'
        '이미지를 업로드하고 결과를 확인해 보세요.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">분석 모델</div>', unsafe_allow_html=True)
    selected_label = st.radio(
        "model",
        options=exp_labels,
        label_visibility="collapsed",
    )
    selected_exp = exp_keys[exp_labels.index(selected_label)]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="field-label">이미지 업로드</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )


# ══════════════ RIGHT ══════════════
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">분석 결과</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown(
            '<div class="empty-box">'
            '<div class="empty-text">이미지를 업로드하면 결과가 여기에 표시됩니다</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        img = Image.open(uploaded)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with st.spinner("분석 중..."):
            model, device = load_model(selected_exp, num_classes)
            preds = predict(model, img, device, classes, top_k=5)

        top_name, top_conf = preds[0]
        top_display = top_name.replace("_", " ").title()

        bar_html = "".join(
            '<div class="bar-row">'
            f'<div class="{"bar-label hi" if i == 0 else "bar-label"}">{name.replace("_", " ").title()}</div>'
            f'<div class="bar-track"><div class="{"bar-fill" if i == 0 else "bar-fill lo"}" style="width:{conf*100:.1f}%"></div></div>'
            f'<div class="{"bar-pct hi" if i == 0 else "bar-pct"}">{int(conf*100)}%</div>'
            '</div>'
            for i, (name, conf) in enumerate(preds)
        )

        card_html = (
            '<div class="result-card">'
            # 이미지: 카드 상단 전체 폭
            f'<div class="rc-img-wrap"><img src="data:image/png;base64,{b64}"/></div>'
            '<div class="rc-body">'
            # 한 줄: 분석된 포켓몬 | 이름 · · · 일치도 | 72%
            '<div class="rc-info-row">'
            '<span class="rc-subtitle">분석된 포켓몬</span>'
            f'<span class="rc-name">{top_display}</span>'
            '<span class="rc-spacer"></span>'
            '<span class="rc-match-label">일치도</span>'
            f'<span class="rc-match-pct">{int(top_conf*100)}<span>%</span></span>'
            '</div>'
            # 바 차트
            f'<div class="bar-list">{bar_html}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
