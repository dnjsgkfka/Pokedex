import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="PokéClassifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0f14; color: #e8eaf0; }
.stApp { background: #0d0f14; }

[data-testid="stSidebar"] { background: #13151c !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #ffcb05 !important; }

.card {
    background: #161921; border: 1px solid #1e2130;
    border-radius: 14px; padding: 24px 28px; margin-bottom: 18px;
}
.card-highlight {
    background: linear-gradient(135deg, #1a1d2e 0%, #161921 100%);
    border: 1px solid #2e3251; border-radius: 14px; padding: 24px 28px; margin-bottom: 18px;
}

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffcb05 30%, #ff6b35 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin: 0;
}
.hero-sub {
    color: #6b7394; font-size: 0.95rem; font-family: 'Space Mono', monospace;
    margin-top: 6px; letter-spacing: 0.05em;
}
.pred-rank { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #6b7394; text-transform: uppercase; letter-spacing: 0.1em; }
.pred-name { font-size: 1.5rem; font-weight: 800; color: #ffcb05; margin: 2px 0 6px; }
.pred-conf { font-family: 'Space Mono', monospace; font-size: 1rem; color: #e8eaf0; }

.stProgress > div > div > div { background: linear-gradient(90deg, #ffcb05, #ff6b35) !important; border-radius: 99px; }

[data-testid="metric-container"] { background: #161921; border: 1px solid #1e2130; border-radius: 10px; padding: 14px 20px; }
[data-testid="stMetricValue"] { color: #ffcb05 !important; font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #6b7394 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #9aa0bc !important; font-size: 0.82rem !important;
    text-transform: uppercase; letter-spacing: 0.07em; font-family: 'Space Mono', monospace;
}
.stButton > button {
    background: linear-gradient(135deg, #ffcb05, #ff8c35); color: #0d0f14;
    font-weight: 700; font-family: 'Syne', sans-serif; border: none;
    border-radius: 8px; padding: 10px 28px; font-size: 0.95rem; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; color: #6b7394; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: #ffcb05 !important; border-bottom: 2px solid #ffcb05 !important; }

hr { border-color: #1e2130 !important; }
details summary { color: #9aa0bc !important; font-family: 'Space Mono', monospace; font-size: 0.82rem; }
.stImage img { border-radius: 12px; }

.exp-badge {
    display: inline-block; background: #1e2130; color: #ffcb05;
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    padding: 3px 10px; border-radius: 99px; letter-spacing: 0.08em;
    border: 1px solid #2e3251; margin-right: 6px; margin-bottom: 4px;
}
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ffcb05, #ff8c35);
    color: #0d0f14; font-family: 'Space Mono', monospace; font-size: 0.7rem;
    padding: 3px 10px; border-radius: 99px; font-weight: 700; letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


IMG_SIZE        = 224
EXPERIMENTS_DIR = Path("./experiments")
TOP_K_DEFAULT   = 5

EXPERIMENT_META = {
    "exp1_resnet50_pretrained_headonly": {
        "label": "EXP 1", "name": "ResNet-50 · Pretrained · Head Only",
        "backbone": "resnet50", "pretrained": True, "color": "#3b82f6",
    },
    "exp2_resnet50_pretrained_fulltune": {
        "label": "EXP 2", "name": "ResNet-50 · Pretrained · Full Fine-Tune",
        "backbone": "resnet50", "pretrained": True, "color": "#22c55e",
    },
    "exp3_efficientnet_pretrained_headonly": {
        "label": "EXP 3", "name": "EfficientNet-B0 · Pretrained · Head Only",
        "backbone": "efficientnet_b0", "pretrained": True, "color": "#a855f7",
    },
    "exp4_resnet50_scratch_fulltrain": {
        "label": "EXP 4", "name": "ResNet-50 · Scratch · Full Train",
        "backbone": "resnet50", "pretrained": False, "color": "#ef4444",
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner=False)
def load_model(exp_name, num_classes):
    meta    = EXPERIMENT_META[exp_name]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if meta["backbone"] == "resnet50":
        weights = ResNet50_Weights.DEFAULT if meta["pretrained"] else None
        model   = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    else:
        weights = EfficientNet_B0_Weights.DEFAULT if meta["pretrained"] else None
        model   = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, num_classes))

    weight_path = EXPERIMENTS_DIR / exp_name / "best_model.pth"
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        st.warning(f"⚠️ Weights not found: {weight_path}")

    return model.to(device).eval(), device


@st.cache_data(show_spinner=False)
def load_classes():
    path = EXPERIMENTS_DIR / "classes.json"
    if path.exists():
        return json.loads(path.read_text())
    return [f"Pokemon_{i}" for i in range(150)]


@st.cache_data(show_spinner=False)
def load_all_results():
    path = EXPERIMENTS_DIR / "all_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@torch.no_grad()
def predict(model, image, device, classes, top_k):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    return [(classes[i], float(p)) for i, p in zip(top_idx, top_probs)]


with st.sidebar:
    st.markdown("## ⚡ PokéClassifier")
    st.markdown("**HW #6 · Computer Vision**  \nSEOULTECH · Transfer Learning")
    st.divider()

    classes     = load_classes()
    num_classes = len(classes)
    st.markdown(f"<span class='exp-badge'>🎯 {num_classes} Classes</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Select Model")
    exp_options = list(EXPERIMENT_META.keys())
    exp_labels  = [f"{EXPERIMENT_META[e]['label']} — {EXPERIMENT_META[e]['name']}" for e in exp_options]
    selected_idx = st.selectbox("Experiment", range(len(exp_options)),
                                 format_func=lambda i: exp_labels[i], label_visibility="collapsed")
    selected_exp = exp_options[selected_idx]

    st.divider()
    st.markdown("### Top-K Predictions")
    top_k = st.slider("K", 1, 10, TOP_K_DEFAULT, label_visibility="collapsed")

    st.divider()
    meta = EXPERIMENT_META[selected_exp]
    st.markdown(f"""
    <div class="card">
      <div class="pred-rank">Current Model</div>
      <div style="font-size:1rem;font-weight:700;color:#e8eaf0;margin:4px 0 10px;">{meta['name']}</div>
      <span class="exp-badge">{'PRETRAINED' if meta['pretrained'] else 'FROM SCRATCH'}</span>
      <span class="exp-badge">{meta['backbone'].upper()}</span>
    </div>
    """, unsafe_allow_html=True)


tab_classify, tab_compare, tab_curves = st.tabs([
    "⚡  CLASSIFY",
    "📊  EXPERIMENT COMPARISON",
    "📈  LEARNING CURVES",
])


with tab_classify:
    st.markdown('<p class="hero-title">PokéClassifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">TRANSFER LEARNING · SEOULTECH · HW #6</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Pokémon Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True, caption="Uploaded Image")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if uploaded:
            with st.spinner("Analyzing..."):
                model, device = load_model(selected_exp, num_classes)
                preds = predict(model, img, device, classes, top_k=top_k)

            top_name, top_conf = preds[0]
            st.markdown(f"""
            <div class="card-highlight">
              <div class="pred-rank">Top Prediction</div>
              <div class="pred-name">#{top_name.replace('_', ' ').title()}</div>
              <div class="pred-conf">{top_conf*100:.2f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Top-{top_k} Predictions**")
            for i, (name, conf) in enumerate(preds):
                rank_color = "#ffcb05" if i == 0 else "#9aa0bc"
                label_col, bar_col, pct_col = st.columns([3, 5, 1])
                with label_col:
                    st.markdown(
                        f"<span style='color:{rank_color};font-family:Space Mono,monospace;font-size:0.82rem;'>"
                        f"{'⭐' if i==0 else f'{i+1}.'} {name.replace('_',' ').title()}</span>",
                        unsafe_allow_html=True
                    )
                with bar_col:
                    st.progress(float(conf))
                with pct_col:
                    st.markdown(
                        f"<span style='font-family:Space Mono,monospace;font-size:0.78rem;color:#6b7394;'>"
                        f"{conf*100:.1f}%</span>",
                        unsafe_allow_html=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 20px;opacity:0.5;">
              <div style="font-size:3rem;margin-bottom:12px;">🎯</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.82rem;color:#6b7394;">
                UPLOAD AN IMAGE TO CLASSIFY
              </div>
            </div>
            """, unsafe_allow_html=True)


with tab_compare:
    st.markdown("## Experiment Comparison")
    all_results = load_all_results()

    if not all_results:
        st.info("🔬 No results found. Run `train.py` first.")
    else:
        metrics       = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
        best_exp      = max(all_results.items(), key=lambda x: x[1].get("test_accuracy", 0))[0]
        best          = all_results[best_exp]

        st.markdown("### 🏆 Best Model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{best['test_accuracy']:.4f}")
        c2.metric("Precision", f"{best['test_precision']:.4f}")
        c3.metric("Recall",    f"{best['test_recall']:.4f}")
        c4.metric("F1 Score",  f"{best['test_f1']:.4f}")
        st.markdown(f"""
        <div style="margin:6px 0 24px;">
          <span class="best-badge">BEST</span>
          <span style="color:#9aa0bc;font-size:0.85rem;margin-left:8px;">
            {EXPERIMENT_META.get(best_exp, {}).get('name', best_exp)}
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Performance Comparison")
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.patch.set_facecolor('#161921')
        ax.set_facecolor('#161921')

        exp_names     = list(all_results.keys())
        x             = np.arange(len(exp_names))
        width         = 0.2
        metric_colors = ["#ffcb05", "#22c55e", "#3b82f6", "#a855f7"]

        for j, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            vals = [all_results[e].get(metric, 0) for e in exp_names]
            bars = ax.bar(x + j * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.85, zorder=3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha='center', va='bottom', color='#e8eaf0', fontsize=6.5, fontfamily='monospace')

        ax.set_xticks(x)
        ax.set_xticklabels([EXPERIMENT_META.get(e, {}).get('label', e) for e in exp_names], color='#9aa0bc', fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", color='#6b7394', fontsize=9)
        ax.tick_params(colors='#6b7394')
        ax.spines[['right', 'top', 'bottom', 'left']].set_color('#1e2130')
        ax.grid(axis='y', color='#1e2130', linewidth=0.8, zorder=0)
        ax.legend(loc='upper right', framealpha=0, labelcolor='#9aa0bc', fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Detailed Results")
        for exp_name, result in all_results.items():
            meta_info = EXPERIMENT_META.get(exp_name, {})
            is_best   = (exp_name == best_exp)
            with st.expander(
                f"{'🏆 ' if is_best else ''}{meta_info.get('label','?')} — {meta_info.get('name', exp_name)}",
                expanded=is_best
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{result['test_accuracy']:.4f}")
                c2.metric("Precision", f"{result['test_precision']:.4f}")
                c3.metric("Recall",    f"{result['test_recall']:.4f}")
                c4.metric("F1",        f"{result['test_f1']:.4f}")
                st.markdown(f"""
                <div style="margin-top:10px;">
                  <span class="exp-badge">{result.get('backbone','').upper()}</span>
                  <span class="exp-badge">{'PRETRAINED' if result.get('pretrained') else 'SCRATCH'}</span>
                  <span class="exp-badge">{'HEAD ONLY' if result.get('freeze_backbone') else 'FULL TUNE'}</span>
                  <span class="exp-badge">⏱ {result.get('training_time_sec', 0)/60:.1f} min</span>
                </div>
                """, unsafe_allow_html=True)


with tab_curves:
    st.markdown("## Learning Curves")
    all_results = load_all_results()

    if not all_results:
        st.info("🔬 No results found. Run `train.py` first.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor('#161921')
        for ax in axes:
            ax.set_facecolor('#161921')
            ax.spines[['right', 'top', 'bottom', 'left']].set_color('#1e2130')
            ax.tick_params(colors='#6b7394')
            ax.grid(color='#1e2130', linewidth=0.8)

        exp_colors = ["#3b82f6", "#22c55e", "#a855f7", "#ef4444"]
        for idx, (exp_name, result) in enumerate(all_results.items()):
            hist = result.get('history', {})
            if not hist:
                continue
            color  = exp_colors[idx % len(exp_colors)]
            label  = EXPERIMENT_META.get(exp_name, {}).get('label', exp_name)
            epochs = range(1, len(hist['train_loss']) + 1)
            axes[0].plot(epochs, hist['val_loss'],  label=label, color=color, lw=2)
            axes[0].plot(epochs, hist['train_loss'], color=color, lw=1, linestyle='--', alpha=0.4)
            axes[1].plot(epochs, hist['val_acc'],   label=label, color=color, lw=2)
            axes[1].plot(epochs, hist['train_acc'],  color=color, lw=1, linestyle='--', alpha=0.4)

        for ax, title, ylabel in zip(axes, ['Validation Loss', 'Validation Accuracy'], ['Loss', 'Accuracy']):
            ax.set_title(title, color='#e8eaf0', fontsize=11)
            ax.set_xlabel('Epoch', color='#6b7394', fontsize=9)
            ax.set_ylabel(ylabel, color='#6b7394', fontsize=9)
            ax.legend(framealpha=0, labelcolor='#9aa0bc', fontsize=9)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        st.markdown("*Solid = Validation · Dashed = Train*")
        st.divider()

        st.markdown("### Individual Experiment Curves")
        cols = st.columns(2)
        for i, exp_name in enumerate(all_results.keys()):
            curve_path = EXPERIMENTS_DIR / exp_name / f"{exp_name}_curve.png"
            meta_info  = EXPERIMENT_META.get(exp_name, {})
            with cols[i % 2]:
                st.markdown(f"**{meta_info.get('label','?')} — {meta_info.get('name', exp_name)}**")
                if curve_path.exists():
                    st.image(str(curve_path), use_column_width=True)
                else:
                    st.markdown(
                        "<div class='card' style='text-align:center;color:#6b7394;font-size:0.8rem;"
                        "font-family:monospace;padding:30px;'>Curve image not found</div>",
                        unsafe_allow_html=True
                    )
