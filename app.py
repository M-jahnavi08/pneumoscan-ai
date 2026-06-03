# app.py — Run with: streamlit run app.py
import streamlit as st
import os
import requests
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg-primary:#080c14; --bg-card:#0d1320; --border:rgba(0,212,255,0.15);
    --accent-cyan:#00d4ff; --accent-green:#00ff88; --accent-red:#ff4466;
    --accent-yellow:#ffd93d; --text-primary:#e8f4fd; --text-muted:#5a7a9a;
    --glow-cyan:0 0 30px rgba(0,212,255,0.3); --glow-green:0 0 30px rgba(0,255,136,0.3);
    --glow-red:0 0 30px rgba(255,68,102,0.3);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
.main .block-container{padding:2rem 3rem!important;max-width:1400px;background:var(--bg-primary);}
#MainMenu,footer,header{visibility:hidden;} .stDeployButton{display:none;}
.hero-header{text-align:center;padding:3rem 2rem 2rem;position:relative;margin-bottom:2rem;}
.hero-header::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:600px;height:200px;background:radial-gradient(ellipse,rgba(0,212,255,0.08) 0%,transparent 70%);pointer-events:none;}
.hero-tag{display:inline-block;font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:0.25em;color:var(--accent-cyan);background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.25);padding:0.3rem 1rem;border-radius:2rem;margin-bottom:1.2rem;text-transform:uppercase;}
.hero-title{font-family:'Space Mono',monospace;font-size:clamp(2rem,5vw,3.5rem);font-weight:700;color:#fff;line-height:1.1;margin:0 0 0.8rem;letter-spacing:-0.02em;}
.hero-title span{color:var(--accent-cyan);text-shadow:var(--glow-cyan);}
.hero-subtitle{font-size:1rem;color:var(--text-muted);font-weight:300;max-width:500px;margin:0 auto;}
.stats-bar{display:flex;justify-content:center;gap:3rem;padding:1.5rem 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);margin:2rem 0;}
.stat-item{text-align:center;}
.stat-value{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:var(--accent-cyan);display:block;text-shadow:var(--glow-cyan);}
.stat-label{font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.12em;margin-top:0.2rem;}
.info-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;}
.info-card-title{font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent-cyan);margin-bottom:0.8rem;}
.tech-badge{display:inline-block;background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);color:var(--accent-cyan);padding:0.2rem 0.7rem;border-radius:4px;font-family:'Space Mono',monospace;font-size:0.65rem;margin:0.2rem;}
.result-normal{background:linear-gradient(135deg,rgba(0,255,136,0.08),rgba(0,255,136,0.03));border:1px solid rgba(0,255,136,0.3);border-left:4px solid var(--accent-green);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;box-shadow:var(--glow-green);}
.result-pneumonia{background:linear-gradient(135deg,rgba(255,68,102,0.08),rgba(255,68,102,0.03));border:1px solid rgba(255,68,102,0.3);border-left:4px solid var(--accent-red);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;box-shadow:var(--glow-red);}
.result-label{font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem;}
.result-value{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;line-height:1;}
.result-normal .result-label{color:rgba(0,255,136,0.6);} .result-normal .result-value{color:var(--accent-green);text-shadow:var(--glow-green);}
.result-pneumonia .result-label{color:rgba(255,68,102,0.6);} .result-pneumonia .result-value{color:var(--accent-red);text-shadow:var(--glow-red);}
.conf-row{display:flex;align-items:center;gap:1rem;margin:0.6rem 0;}
.conf-name{font-family:'Space Mono',monospace;font-size:0.7rem;width:90px;color:var(--text-muted);text-transform:uppercase;}
.conf-bar-bg{flex:1;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;}
.conf-bar-fill-green{height:100%;background:linear-gradient(90deg,var(--accent-green),rgba(0,255,136,0.5));border-radius:3px;}
.conf-bar-fill-red{height:100%;background:linear-gradient(90deg,var(--accent-red),rgba(255,68,102,0.5));border-radius:3px;}
.conf-pct{font-family:'Space Mono',monospace;font-size:0.75rem;width:45px;text-align:right;}
.conf-pct-green{color:var(--accent-green);} .conf-pct-red{color:var(--accent-red);}
.img-panel{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1rem;text-align:center;}
.img-panel-label{font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.5rem;}
.section-heading{font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.22em;text-transform:uppercase;color:var(--accent-cyan);margin:1.8rem 0 0.8rem;display:flex;align-items:center;gap:0.6rem;}
.section-heading::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(0,212,255,0.3),transparent);}
.warning-badge{background:rgba(255,217,61,0.07);border:1px solid rgba(255,217,61,0.2);border-radius:8px;padding:0.7rem 1rem;font-size:0.76rem;color:var(--accent-yellow);margin-top:1rem;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.5rem 0!important;}
::-webkit-scrollbar{width:6px;} ::-webkit-scrollbar-track{background:var(--bg-primary);} ::-webkit-scrollbar-thumb{background:rgba(0,212,255,0.2);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ── Model download using requests ─────────────────────────────────────────────
def download_model():
    file_id = "10DWSq-00Q1zV6vCPWqHy-_Ni4--GEw8C"
    session = requests.Session()

    # Step 1: Get confirmation token
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True
    )

    # Step 2: Find confirmation token in cookies or response
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # Step 3: If token found, re-request with confirmation
    if token:
        response = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": token},
            stream=True
        )
    else:
        # Try new Google Drive download URL format
        response = session.get(
            f"https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True
        )

    # Step 4: Save file
    with open("best_model.pth", "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


# ── Load model ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pth"):
        with st.spinner("Downloading model weights (~50MB)..."):
            download_model()
    m = models.efficientnet_b3(weights=None)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))
    m.load_state_dict(torch.load("best_model.pth", map_location=device))
    m.to(device).eval()
    return m

model = load_model()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        pred   = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, pred].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs   = output.softmax(1)[0].detach().cpu().numpy()
        return cam, pred, probs


# ── Transform & predict ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_and_explain(pil_image):
    img_rgb = pil_image.convert("RGB")
    input_t = transform(img_rgb).unsqueeze(0).to(device)
    input_t.requires_grad_()
    gc = GradCAM(model, model.features[-1])
    cam, pred_idx, probs = gc.generate(input_t)
    img_np  = np.array(img_rgb.resize((224, 224)))
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return pred_idx, probs, img_np, heatmap[:, :, ::-1], overlay[:, :, ::-1]


# ── LAYOUT ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-tag">Deep Learning · Medical Imaging · Explainable AI</div>
    <h1 class="hero-title">PNEUMO<span>SCAN</span></h1>
    <p class="hero-subtitle">AI-powered chest X-ray analysis with real-time explainability using Grad-CAM</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><span class="stat-value">94%</span><div class="stat-label">Test Accuracy</div></div>
    <div class="stat-item"><span class="stat-value">5,863</span><div class="stat-label">Training Images</div></div>
    <div class="stat-item"><span class="stat-value">EfficientB3</span><div class="stat-label">Architecture</div></div>
    <div class="stat-item"><span class="stat-value">Grad-CAM</span><div class="stat-label">Explainability</div></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<div class="section-heading">Upload X-Ray</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop chest X-ray here", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
        <span style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;color:#5a7a9a;text-transform:uppercase;">
            Supports JPG · PNG · JPEG
        </span>
    </div>
    <div class="section-heading">Model Info</div>
    <div class="info-card">
        <div class="info-card-title">Architecture</div>
        <div style="font-size:0.85rem;color:#c0d8f0;margin-bottom:1rem;">
            EfficientNetB3 fine-tuned with transfer learning on 5,863 labeled chest X-rays from the NIH Kaggle dataset.
        </div>
        <div class="info-card-title">Tech Stack</div>
        <div>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">EfficientNet</span>
            <span class="tech-badge">Grad-CAM</span>
            <span class="tech-badge">OpenCV</span>
            <span class="tech-badge">Streamlit</span>
        </div>
    </div>
    <div class="warning-badge">
        ⚠️ For educational & research purposes only. Not a certified medical device. Always consult a qualified radiologist.
    </div>
    """, unsafe_allow_html=True)

with right:
    if uploaded is None:
        st.markdown("""
        <div style="height:480px;background:#0d1320;border:1px solid rgba(0,212,255,0.1);border-radius:16px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;opacity:0.4;">🫁</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;letter-spacing:0.1em;color:#5a7a9a;text-transform:uppercase;margin-bottom:0.5rem;">
                Awaiting X-Ray Input
            </div>
            <div style="font-size:0.82rem;color:#3a5a7a;max-width:280px;line-height:1.6;">
                Upload a chest X-ray on the left to receive an AI diagnosis with explainability heatmap
            </div>
            <div style="margin-top:2rem;display:flex;gap:1.5rem;">
                <div style="text-align:center;"><div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:rgba(0,212,255,0.3);">01</div><div style="font-size:0.7rem;color:#3a5a7a;margin-top:0.2rem;">Upload</div></div>
                <div style="color:#3a5a7a;padding-top:0.3rem;">→</div>
                <div style="text-align:center;"><div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:rgba(0,212,255,0.2);">02</div><div style="font-size:0.7rem;color:#3a5a7a;margin-top:0.2rem;">Analyse</div></div>
                <div style="color:#3a5a7a;padding-top:0.3rem;">→</div>
                <div style="text-align:center;"><div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:rgba(0,212,255,0.2);">03</div><div style="font-size:0.7rem;color:#3a5a7a;margin-top:0.2rem;">Explain</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded)
        with st.spinner("Running inference + generating Grad-CAM..."):
            pred_idx, probs, orig, heatmap, overlay = predict_and_explain(image)

        if pred_idx == 0:
            st.markdown("""
            <div class="result-normal">
                <div class="result-label">Diagnosis Result</div>
                <div class="result-value">✓ NORMAL</div>
                <div style="font-size:0.8rem;color:rgba(0,255,136,0.5);margin-top:0.4rem;">No signs of pneumonia detected</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-pneumonia">
                <div class="result-label">Diagnosis Result</div>
                <div class="result-value">⚠ PNEUMONIA</div>
                <div style="font-size:0.8rem;color:rgba(255,68,102,0.5);margin-top:0.4rem;">Pneumonia indicators detected — consult a radiologist</div>
            </div>""", unsafe_allow_html=True)

        n_pct = int(probs[0] * 100)
        p_pct = int(probs[1] * 100)

        st.markdown('<div class="section-heading">Confidence Scores</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1320;border:1px solid rgba(0,212,255,0.1);border-radius:10px;padding:1rem 1.2rem;">
            <div class="conf-row">
                <div class="conf-name">Normal</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-green" style="width:{n_pct}%"></div></div>
                <div class="conf-pct conf-pct-green">{n_pct}%</div>
            </div>
            <div class="conf-row">
                <div class="conf-name">Pneumonia</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-red" style="width:{p_pct}%"></div></div>
                <div class="conf-pct conf-pct-red">{p_pct}%</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-heading">Grad-CAM Explainability</div>
        <div style="font-size:0.78rem;color:#5a7a9a;margin-bottom:0.8rem;">
            🔴 Red/yellow regions indicate where the AI focused to make its decision
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="img-panel"><div class="img-panel-label">Original X-Ray</div>', unsafe_allow_html=True)
            st.image(orig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="img-panel"><div class="img-panel-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="img-panel"><div class="img-panel-label">Overlay</div>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)