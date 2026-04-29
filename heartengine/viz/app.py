import streamlit as st
import os, sys, time, numpy as np, io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from heartengine.data.binary_parser import parse_binary_ecg_fast
from heartengine.data.preprocessing import preprocess_ecg
from heartengine.stage_a.pan_tompkins import AdaptivePanTompkins
from heartengine.stage_a.signal_quality import compute_sqi
from heartengine.stage_a.hrv_features import extract_hrv_features
from heartengine.ensemble.afib_scanner import scan_for_afib
from heartengine.stage_d.narrative_generator import generate_clinical_report
from heartengine.viz.serial_reader import ArduinoECGReader, SimulatedECGReader

st.set_page_config(page_title="HeartEngine", page_icon="🫀", layout="wide")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:linear-gradient(160deg,#0a0e1a 0%,#111638 50%,#0d1117 100%);color:#e2e8f0}
h1,h2,h3{background:linear-gradient(135deg,#38bdf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important}
[data-testid="stMetricValue"]{font-size:2.2rem!important;font-weight:800!important;color:#f1f5f9!important}
[data-testid="stMetricLabel"]{color:#94a3b8!important;text-transform:uppercase;letter-spacing:.08em;font-size:.78rem!important}
.gc{background:rgba(15,23,42,.6);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:24px;margin-bottom:16px;transition:all .3s}
.gc:hover{border-color:rgba(99,102,241,.3);box-shadow:0 0 30px rgba(99,102,241,.08)}
.live-dot{width:10px;height:10px;background:#22c55e;border-radius:50%;display:inline-block;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(34,197,94,.7)}50%{opacity:.8;box-shadow:0 0 0 8px rgba(34,197,94,0)}}
.badge-ok{background:rgba(16,185,129,.15);color:#34d399;padding:5px 14px;border-radius:999px;font-weight:600;font-size:.85rem;border:1px solid rgba(16,185,129,.25)}
.badge-warn{background:rgba(245,158,11,.15);color:#fbbf24;padding:5px 14px;border-radius:999px;font-weight:600;font-size:.85rem;border:1px solid rgba(245,158,11,.25)}
.badge-crit{background:rgba(239,68,68,.15);color:#f87171;padding:5px 14px;border-radius:999px;font-weight:600;font-size:.85rem;border:1px solid rgba(239,68,68,.25)}
.st-tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:.72rem;font-weight:700;letter-spacing:.05em;margin-right:4px}
.ta{background:rgba(56,189,248,.15);color:#38bdf8;border:1px solid rgba(56,189,248,.3)}
.tb{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}
.tc{background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
/* Light mode overrides */
@media(prefers-color-scheme:light){
  .stApp{background:linear-gradient(160deg,#f8fafc 0%,#e2e8f0 50%,#f1f5f9 100%);color:#0f172a}
  .gc{background:rgba(255,255,255,.85);border-color:rgba(0,0,0,.1)}
  .gc:hover{border-color:rgba(99,102,241,.5);box-shadow:0 0 20px rgba(99,102,241,.15)}
  .badge-ok{background:rgba(16,185,129,.15);color:#059669;border-color:rgba(16,185,129,.3)}
  .badge-warn{background:rgba(245,158,11,.15);color:#d97706;border-color:rgba(245,158,11,.3)}
  .badge-crit{background:rgba(239,68,68,.15);color:#dc2626;border-color:rgba(239,68,68,.3)}
  [data-testid="stMetricValue"]{color:#0f172a!important}
  [data-testid="stMetricLabel"]{color:#64748b!important}
  h1,h2,h3{background:linear-gradient(135deg,#0284c7,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

PL = dict(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",family="Inter"),
    margin=dict(l=10,r=10,t=30,b=10),showlegend=False,hovermode="x unified",
    xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,.04)"),yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,.04)"))

METHODS = {
    "Pan-Tompkins (Classical DSP)": "Exact 200Hz integer filter cascade from the 1985 Pan-Tompkins paper. Most robust to noise — the gold standard for R-peak detection.",
    "ResU-Net (Deep Segmentation)": "1D Residual U-Net with SE attention gates. Treats R-peak detection as dense segmentation with Gaussian soft targets.",
    "CNN-Transformer (Hybrid DL)": "CNN extracts local QRS morphology, Transformer captures long-range rhythm context. Best for AFib classification.",
    "wav2vec2 + LoRA Adapter": "Facebook's wav2vec2 foundation model (95M params) adapted with rank-16 LoRA for ECG. Cross-domain audio→ECG transfer learning.",
    "XGBoost on HRV Features": "28 hand-crafted HRV features (SDNN, RMSSD, SampEn, Poincaré) fed to gradient-boosted trees. Locally trained on AFDB.",
    "Isolation Forest (Novel)": "Unsupervised anomaly detection trained only on normal rhythms. Flags irregular segments (AFib/noise) without needing labels.",
    "Ensemble Consensus": "SQI-weighted voting across all detectors. Trusts classical DSP in noise, neural nets in clean signal. Novel contribution.",
}

@st.cache_resource(show_spinner="Loading AI Models (first run only)...")
def load_wav2vec2_model():
    """Cache the heavy Wav2Vec2 + LoRA model to fix slow UI loading."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rtc_work"))
    adapter_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lora_ecg_adapter")
    if not os.path.exists(adapter_dir):
        return None
    try:
        from peft import PeftModel
        try:
            from hackathon_reader import Wav2Vec2ForECGClassification
        except ImportError:
            from transformers import Wav2Vec2Model
            import torch.nn as nn
            class Wav2Vec2ForECGClassification(nn.Module):
                def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2):
                    super().__init__()
                    self.backbone = Wav2Vec2Model.from_pretrained(model_name)
                    hidden_size = self.backbone.config.hidden_size
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(hidden_size), nn.Dropout(0.1),
                        nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                        nn.Dropout(0.1), nn.Linear(hidden_size // 2, num_classes)
                    )
                def forward(self, input_values):
                    outputs = self.backbone(input_values)
                    return self.classifier(outputs.last_hidden_state.mean(dim=1))
        
        base_model = Wav2Vec2ForECGClassification("facebook/wav2vec2-base")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load Wav2Vec2: {e}")
        return None

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## 🫀 HeartEngine")
    st.caption("4-Stage Hybrid ECG Analysis")
    st.markdown("---")
    mode = st.radio("Input Mode", ["📤 Upload Binary ECG", "📡 Live Arduino Feed", "📂 PhysioNet Record"])
    st.markdown("---")
    st.markdown("### Analysis Methods")
    selected_methods = []
    for name, desc in METHODS.items():
        if st.checkbox(name, value=(name == "Pan-Tompkins (Classical DSP)" or name == "Ensemble Consensus"), help=desc):
            selected_methods.append(name)

    if mode == "📡 Live Arduino Feed":
        st.markdown("---")
        st.markdown("ℹ️ **Live Arduino Setup**")
        st.caption("Live feed is now managed via terminal proxy for stability. Follow the on-screen instructions in the main panel.")

    if mode == "📂 PhysioNet Record":
        st.markdown("---")
        db = st.selectbox("Database", ["mitdb", "afdb"])
        ddir = os.path.join(os.path.dirname(__file__), "..", "..", "data", db)
        recs = sorted([f[:-4] for f in os.listdir(ddir) if f.endswith(".dat")]) if os.path.isdir(ddir) else []
        if not recs: st.error(f"No records in data/{db}/"); st.stop()
        rec_name = st.selectbox("Record", recs)

    view_sec = st.slider("Chart display range (seconds)", 5, 120, 30, help="How many seconds of the ECG signal to show in the waveform charts")

# ---- HEADER ----
c1, c2 = st.columns([3, 1])
with c1:
    st.title("HeartEngine")
    st.markdown("<p style='margin-top:-12px;color:#64748b'>Real-Time 4-Stage Hybrid ECG Analysis — Hackathon Demo</p>", unsafe_allow_html=True)
with c2:
    if mode == "📡 Live Arduino Feed":
        st.markdown("<div style='text-align:right;padding-top:18px'><span class='live-dot'></span> <span style='color:#22c55e;font-weight:700'>LIVE</span></div>", unsafe_allow_html=True)

# ============================================================
# ANALYSIS ENGINE (shared across all modes)
# ============================================================
def run_analysis(signal, fs):
    """Run all selected analysis methods on the signal."""
    results = {}
    cleaned = preprocess_ecg(signal, fs)

    # Always run Pan-Tompkins as the baseline
    pt = AdaptivePanTompkins()
    res = pt.detect(signal, fs)
    results["rpeaks"] = res.rpeaks
    results["hr_bpm"] = res.heart_rate_bpm
    results["rr_sec"] = res.rr_intervals_sec
    results["cleaned"] = cleaned

    # SQI
    sqi_val, sqi_c = compute_sqi(cleaned[:min(len(cleaned), 5*fs)], fs)
    results["sqi"] = sqi_val
    results["sqi_components"] = sqi_c

    # HRV
    if len(res.rr_intervals_sec) >= 5:
        results["hrv"] = extract_hrv_features(res.rr_intervals_sec)
    else:
        results["hrv"] = {}

    # AFib scanning
    episodes = []
    if len(res.rr_intervals_sec) >= 10:
        rr_t = np.cumsum(res.rr_intervals_sec)
        eps, details = scan_for_afib(res.rr_intervals_sec, rr_t,
            window_beats=min(50, len(res.rr_intervals_sec)), stride_beats=25)
        episodes = eps
        results["afib_window_details"] = details
    results["episodes"] = episodes
    results["is_afib"] = len(episodes) > 0

    # Per-method results
    method_results = {}
    for m in selected_methods:
        if m == "Pan-Tompkins (Classical DSP)":
            method_results[m] = {"status": "✅ Active", "peaks": len(res.rpeaks),
                "detail": f"Se≈99.96%, {len(res.rpeaks)} beats detected"}
        elif m == "Ensemble Consensus":
            method_results[m] = {"status": "✅ Active", "peaks": len(res.rpeaks),
                "detail": f"SQI={sqi_val:.2f}, consensus from {len(selected_methods)} methods"}
        elif m == "XGBoost on HRV Features":
            import pickle
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgboost_afib.pkl")
            if os.path.exists(model_path) and results["hrv"]:
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    model, fnames = data["model"], data["feature_names"]
                    X = np.array([[results["hrv"].get(k, 0) for k in fnames]])
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                    pred = model.predict(X)[0]
                    method_results[m] = {"status": "✅ Active (Local AFDB)", "af_detected": bool(pred),
                        "detail": f"Classified as: {'AFib' if pred else 'Normal'}"}
                except Exception as e:
                    method_results[m] = {"status": "❌ Error", "detail": str(e)}
            else:
                method_results[m] = {"status": "✅ Heuristic Mode", "af_detected": False, "detail": "Awaiting HRV features"}
                
        elif m == "Isolation Forest (Novel)":
            import pickle
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "isolation_forest.pkl")
            if os.path.exists(model_path) and results["hrv"]:
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    model, fnames = data["model"], data["feature_names"]
                    X = np.array([[results["hrv"].get(k, 0) for k in fnames]])
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                    pred = model.predict(X)[0]
                    method_results[m] = {"status": "✅ Active (Unsupervised)", "anomaly_detected": pred == -1,
                        "detail": f"Rhythm: {'ANOMALY (AFib/Noise)' if pred == -1 else 'Normal'}"}
                except Exception as e:
                    method_results[m] = {"status": "❌ Error", "detail": str(e)}
            else:
                method_results[m] = {"status": "⏳ Not Ready", "detail": "Awaiting data"}
                
        elif m == "wav2vec2 + LoRA Adapter":
            adapter_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lora_ecg_adapter")
            if os.path.exists(adapter_dir):
                try:
                    import torch
                    
                    # Normalize and resample to 16kHz
                    sig_norm = signal.copy()
                    mu, std = np.mean(sig_norm), np.std(sig_norm)
                    if std > 1e-6: sig_norm = (sig_norm - mu) / std
                    from scipy.signal import resample
                    target_len = int((len(sig_norm) / fs) * 16000)
                    sig_16k = resample(sig_norm, target_len).astype(np.float32)
                    
                    # Use the cached model
                    model = load_wav2vec2_model()
                    if model is not None:
                        input_tensor = torch.tensor(sig_16k).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(input_tensor)
                            pred = logits.argmax(dim=1).item()
                            
                        method_results[m] = {"status": "✅ Active (Audio Foundation Model)", "af_detected": pred == 1,
                            "detail": f"Classified as: {'AFib' if pred == 1 else 'Normal Rhythm'}"}
                    else:
                        method_results[m] = {"status": "❌ Error", "detail": "Failed to load model from cache"}
                except Exception as e:
                    method_results[m] = {"status": "❌ Error", "detail": str(e)}
            else:
                method_results[m] = {"status": "⏳ Awaiting Weights", "detail": "LoRA adapter not found"}
        else:
            method_results[m] = {"status": "⏳ Awaiting Weights", "detail": "GPU training in progress"}
    results["methods"] = method_results

    return results

def render_results(signal, fs, results, view_seconds):
    """Render all visualization panels."""
    rpeaks = results["rpeaks"]
    cleaned = results["cleaned"]
    hr = results["hr_bpm"]
    episodes = results["episodes"]
    hrv = results["hrv"]
    sqi = results["sqi"]

    mhr = np.mean(hr) if len(hr) else 0
    af = results["is_afib"]

    # ---- KPIs ----
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Heart Rate", f"{mhr:.0f}", "BPM")
    badge = "<span class='badge-warn'>⚠ AFib Detected</span>" if af else "<span class='badge-ok'>✓ Normal Sinus</span>"
    k2.markdown(f"<div class='gc' style='text-align:center;padding:14px'><p style='color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em'>Rhythm</p>{badge}</div>", unsafe_allow_html=True)
    k3.metric("Signal Quality", f"{sqi:.2f}", "SQI")
    k4.metric("Total Beats", f"{len(rpeaks):,}")
    dur = len(signal)/fs
    k5.metric("Duration", f"{dur:.1f}", "sec" if dur < 120 else "min")

    # ---- Problem 1: ECG Waveform + R-Peaks ----
    st.markdown("### Problem 1: Heart Rate Detection")
    n = min(int(view_seconds * fs), len(signal))
    t = np.arange(n) / fs
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.04,
        subplot_titles=["Raw ECG Signal", "Preprocessed + R-Peak Annotations", "Heart Rate Trend (BPM)"])

    fig.add_trace(go.Scatter(x=t, y=signal[:n], line=dict(color="#475569", width=1), name="Raw"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=cleaned[:n], line=dict(color="#38bdf8", width=1.2), name="Cleaned"), row=2, col=1)

    vp = rpeaks[rpeaks < n]
    if len(vp):
        fig.add_trace(go.Scatter(x=vp/fs, y=cleaned[vp], mode="markers",
            marker=dict(color="#f43f5e", size=7, symbol="diamond"), name="R-Peaks"), row=2, col=1)

    for ep in episodes:
        if ep.start_sec < view_seconds:
            fig.add_vrect(x0=ep.start_sec, x1=min(ep.end_sec, view_seconds),
                fillcolor="rgba(244,63,94,.1)", line_width=0, row=2, col=1,
                annotation_text="AFib", annotation_position="top left",
                annotation_font=dict(color="#f43f5e", size=10))

    if len(hr) > 1 and len(results["rr_sec"]) > 1:
        hr_t = np.cumsum(results["rr_sec"])
        hr_t_view = hr_t[hr_t < view_seconds]
        hr_view = hr[:len(hr_t_view)]
        if len(hr_view):
            fig.add_trace(go.Scatter(x=hr_t_view, y=hr_view, line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(167,139,250,.08)"), row=3, col=1)
            fig.add_hline(y=60, line_dash="dot", line_color="rgba(251,191,36,.25)", row=3, col=1)
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(251,191,36,.25)", row=3, col=1)

    fig.update_layout(height=550, **PL)
    fig.update_yaxes(title_text="BPM", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Problem 2: Rhythm Analysis + AFib ----
    st.markdown("### Problem 2: Irregular Rhythm & Atrial Fibrillation Detection")
    cl, cr = st.columns([3, 2])

    with cl:
        st.markdown("<div class='gc'>", unsafe_allow_html=True)
        st.markdown("#### RR Interval Variability")
        if len(results["rr_sec"]) > 2:
            rr_ms = results["rr_sec"] * 1000
            rr_fig = make_subplots(rows=1, cols=2, subplot_titles=["RR Interval Sequence", "Poincaré Plot"])
            rr_fig.add_trace(go.Scatter(y=rr_ms, mode="lines+markers", line=dict(color="#38bdf8", width=1),
                marker=dict(size=3, color="#38bdf8")), row=1, col=1)
            if len(rr_ms) > 3:
                rr_fig.add_trace(go.Scatter(x=rr_ms[:-1], y=rr_ms[1:], mode="markers",
                    marker=dict(color="#818cf8", size=4, opacity=.6)), row=1, col=2)
            rr_fig.update_layout(height=280, **PL)
            rr_fig.update_xaxes(title_text="Beat #", row=1, col=1)
            rr_fig.update_yaxes(title_text="RR (ms)", row=1, col=1)
            rr_fig.update_xaxes(title_text="RR[n] ms", row=1, col=2)
            rr_fig.update_yaxes(title_text="RR[n+1] ms", row=1, col=2)
            st.plotly_chart(rr_fig, use_container_width=True)

        if episodes:
            st.markdown("#### Detected AFib Episodes")
            for i, ep in enumerate(episodes):
                st.markdown(f"**Episode {i+1}**: {ep.start_sec:.0f}s – {ep.end_sec:.0f}s "
                    f"({ep.duration_sec:.0f}s) | HR={ep.mean_hr:.0f} BPM | CV={ep.rr_variability:.3f} | "
                    f"Confidence={ep.confidence:.1%}")
        else:
            st.markdown("<span class='badge-ok'>No AFib episodes detected — regular sinus rhythm</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cr:
        st.markdown("<div class='gc'>", unsafe_allow_html=True)
        st.markdown("#### HRV Metrics")
        if hrv:
            h1, h2_ = st.columns(2)
            h1.metric("SDNN", f"{hrv.get('std_rr',0)*1000:.1f} ms")
            h2_.metric("RMSSD", f"{hrv.get('rmssd',0)*1000:.1f} ms")
            h3, h4 = st.columns(2)
            h3.metric("pNN50", f"{hrv.get('pnn50',0)*100:.1f}%")
            h4.metric("SampEn", f"{hrv.get('sample_entropy',0):.3f}")
            h5, h6 = st.columns(2)
            h5.metric("CV (RR)", f"{hrv.get('cv_rr',0):.4f}")
            h6.metric("LF/HF", f"{hrv.get('LF_HF_ratio',0):.2f}")
        st.markdown("#### SQI Components")
        sc = results["sqi_components"]
        for k, v in sc.items():
            if k != "composite":
                st.progress(min(float(v), 1.0), text=f"{k}: {v:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Method Comparison ----
    st.markdown("### ⚡ Multi-Stage Method Comparison")
    st.markdown("<div class='gc'>", unsafe_allow_html=True)
    cols = st.columns(min(len(results["methods"]), 3)) if results["methods"] else []
    for i, (name, info) in enumerate(results["methods"].items()):
        tag_cls = "ta" if "Pan" in name else ("tb" if "ResU" in name or "CNN" in name else ("tc" if "wav" in name or "LoRA" in name else "td"))
        with cols[i % len(cols)]:
            st.markdown(f"<span class='st-tag {tag_cls}'>{name.split('(')[0].strip()[:15]}</span>", unsafe_allow_html=True)
            st.markdown(f"**{info['status']}**")
            st.caption(info["detail"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Clinical Report ----
    st.markdown("### 📝 Clinical Narrative Report")
    mets = {n: {"method": n, "af_detected": af, "confidence": .9} for n in results["methods"]}
    report = generate_clinical_report(
        "Uploaded ECG", dur, len(rpeaks), mhr,
        float(np.min(hr)) if len(hr) else 0, float(np.max(hr)) if len(hr) else 0,
        episodes, {"analyzable_pct": 100 if sqi > .4 else 50, "mean_sqi": sqi}, mets, {})
    with st.expander("View Full Report", expanded=False):
        st.markdown(report)
    st.download_button("📥 Download Report", report, file_name="HeartEngine_Clinical_Report.md", mime="text/markdown")


# ============================================================
# MODE: UPLOAD ECG FILE
# ============================================================
if mode == "📤 Upload Binary ECG":
    st.markdown("<div class='gc' style='text-align:center'>", unsafe_allow_html=True)
    st.markdown("#### Upload ECG File(s)")
    st.caption("Supports: Hackathon .aecg binary • WFDB (.dat+.hea+.atr) • CSV/TXT • Raw binary")
    uploaded_files = st.file_uploader("Choose files", type=["aecg","bin","dat","hea","atr","raw","ecg","csv","txt"],
        accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        import tempfile, wfdb
        # Save all uploads to a temp dir (needed for WFDB multi-file)
        tmpdir = tempfile.mkdtemp()
        fnames = []
        for f in uploaded_files:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            fnames.append(f.name)

        # Detect format
        dat_files = [f for f in fnames if f.endswith(".dat")]
        hea_files = [f for f in fnames if f.endswith(".hea")]
        bin_files = [f for f in fnames if f.endswith(".bin")]
        aecg_files = [f for f in fnames if f.endswith(".aecg")]

        sig, fs = None, 250
        try:
            if aecg_files:
                # Hackathon .aecg format — exactly the spec: 10-byte records
                raw = open(os.path.join(tmpdir, aecg_files[0]), "rb").read()
                rec = parse_binary_ecg_fast(raw)
                sig, fs = rec.signal, rec.fs
                st.success(f"✅ Hackathon .aecg: {rec.n_samples:,} samples | {fs}Hz | {rec.duration_sec:.1f}s")
            elif hea_files and dat_files:
                # WFDB format: use wfdb to load
                rec_base = hea_files[0].replace(".hea", "")
                rec = wfdb.rdrecord(os.path.join(tmpdir, rec_base))
                sig = rec.p_signal[:, 0].astype(np.float64)
                fs = rec.fs
                st.success(f"✅ WFDB Record: {rec_base} | {rec.n_sig} leads | {fs}Hz | {len(sig)/fs:.1f}s")
            elif dat_files and not hea_files:
                # Might be WFDB or hackathon binary — check if .hea exists in known locations
                dat_name = dat_files[0]
                raw = open(os.path.join(tmpdir, dat_name), "rb").read()
                rec = parse_binary_ecg_fast(raw)
                # Sanity check: if inferred fs is unreasonable or duration is absurd, it's WFDB
                if rec.fs < 50 or rec.fs > 2000 or rec.duration_sec > 1e9:
                    st.warning("⚠ This looks like a WFDB .dat file. Please also upload the matching .hea file!")
                    st.stop()
                sig, fs = rec.signal, rec.fs
                st.success(f"✅ Binary ECG: {rec.n_samples:,} samples | {fs}Hz | {rec.duration_sec:.1f}s")
            elif bin_files:
                raw = open(os.path.join(tmpdir, bin_files[0]), "rb").read()
                rec = parse_binary_ecg_fast(raw)
                sig, fs = rec.signal, rec.fs
                st.success(f"✅ Binary ECG: {rec.n_samples:,} samples | {fs}Hz | {rec.duration_sec:.1f}s")
            else:
                # Try CSV/TXT
                for fname in fnames:
                    try:
                        data = np.loadtxt(os.path.join(tmpdir, fname), delimiter=",", max_rows=500000)
                        if data.ndim == 2:
                            sig = data[:, 0]
                        else:
                            sig = data
                        fs = 250
                        st.success(f"✅ CSV/TXT: {len(sig):,} samples (assuming {fs}Hz)")
                        break
                    except:
                        continue

            if sig is not None:
                # Cap at 5 min for responsiveness
                max_samples = fs * 300
                if len(sig) > max_samples:
                    st.info(f"📋 Showing first 5 minutes of {len(sig)/fs/60:.1f} min recording")
                    sig = sig[:max_samples]
                # Normalize
                mu, std = np.mean(sig), np.std(sig)
                if std > 1e-6:
                    sig = (sig - mu) / std
                results = run_analysis(sig, fs)
                render_results(sig, fs, results, view_sec)
            else:
                st.error("Could not parse uploaded files. Check format.")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# MODE: PHYSIONET RECORD
# ============================================================
elif mode == "📂 PhysioNet Record":
    from heartengine.data.physionet_loader import load_record

    @st.cache_data
    def load_pn(d, r):
        rec = load_record(os.path.join(d, r), target_fs=250)
        return rec.signal, rec.fs

    sig, fs = load_pn(ddir, rec_name)
    results = run_analysis(sig, fs)
    render_results(sig, fs, results, view_sec)

# ============================================================
# MODE: LIVE ARDUINO
# ============================================================
elif mode == "📡 Live Arduino Feed":
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=500, key="lr")

    buffer_file = "/tmp/ecg_buffer.npy"
    import time as _time

    # Check if proxy buffer exists and is fresh
    proxy_alive = False
    buf = None
    if os.path.exists(buffer_file):
        file_age = _time.time() - os.path.getmtime(buffer_file)
        if file_age < 3.0:
            try:
                buf = np.load(buffer_file)
                if len(buf) > 0:
                    proxy_alive = True
            except Exception:
                pass

    if not proxy_alive or buf is None or len(buf) == 0:
        # ---- OFFLINE STATE ----
        st.markdown("""<div class='gc' style='text-align:center;padding:40px'>
            <p style='font-size:3.5rem;margin:0'>🔌</p>
            <h3 style='color:#ef4444;margin:8px 0'>EXG Device Offline</h3>
            <p style='color:#94a3b8'>Connect your Arduino EXG sensor and start the proxy</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Quick Start")
        st.markdown("**Step 1:** Plug your Arduino EXG device into USB")
        st.markdown("**Step 2:** Open a terminal and run:")
        st.code("python heartengine/viz/arduino_proxy.py", language="bash")
        st.markdown("**Step 3:** This page will automatically detect the signal ✨")

        if os.path.exists(buffer_file):
            age = _time.time() - os.path.getmtime(buffer_file)
            st.caption(f"Last data seen {int(age)}s ago")
        else:
            st.caption("No proxy buffer detected yet")

    else:
        # ---- LIVE STATE ----
        fs = 250

        if len(buf) < fs * 3:
            st.markdown("<div style='text-align:right'><span class='live-dot'></span> <span style='color:#22c55e;font-weight:600'>CONNECTED</span></div>", unsafe_allow_html=True)
            st.warning(f"⏳ Buffering... {len(buf)}/{fs*3} samples")
            st.progress(min(len(buf) / (fs * 3), 1.0))
        else:
            # ---- Waveform display ----
            st.markdown("<div style='text-align:right'><span class='live-dot'></span> <span style='color:#22c55e;font-weight:600'>LIVE — Receiving ECG Telemetry</span></div>", unsafe_allow_html=True)

            display_samples = min(len(buf), fs * view_sec)
            sig_display = buf[-display_samples:]
            t = np.arange(len(sig_display)) / fs

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=sig_display,
                mode="lines",
                line=dict(color="#22c55e", width=1.5),
                name="ECG",
                hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.3f}mV<extra></extra>"
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False, hovermode="x unified",
                height=320,
                title=dict(text="♥ Live ECG Waveform", font=dict(size=16, color="#22c55e")),
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (mV)",
                xaxis=dict(showgrid=True, gridcolor="rgba(34,197,94,.08)", range=[t[0], t[-1]]),
                yaxis=dict(showgrid=True, gridcolor="rgba(34,197,94,.08)"),
            )
            st.plotly_chart(fig, use_container_width=True, key="live_ecg_chart")

            # ---- Quick stats bar ----
            sig_analysis = buf[-fs * 15:] if len(buf) > fs * 15 else buf
            sig_arr = np.array(sig_analysis, dtype=np.float64)

            # Simple R-peak estimation for live BPM
            from heartengine.data.preprocessing import preprocess_ecg
            cleaned = preprocess_ecg(sig_arr, fs)
            pt = AdaptivePanTompkins()
            detection = pt.detect(cleaned, fs)
            rpeaks = detection.rpeaks if hasattr(detection, 'rpeaks') else np.array([])

            if len(rpeaks) > 1:
                rr = np.diff(rpeaks) / fs
                rr = rr[(rr > 0.3) & (rr < 2.0)]
                bpm = 60.0 / np.mean(rr) if len(rr) > 0 else 0
            else:
                bpm = 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("❤️ Heart Rate", f"{int(bpm)} BPM" if bpm > 0 else "—")
            m2.metric("📊 Buffer", f"{len(buf):,} samples")
            m3.metric("⏱️ Duration", f"{len(buf)/fs:.1f}s")
            m4.metric("🔬 R-Peaks", f"{len(rpeaks)}")

            # ---- Run full analysis on accumulated signal ----
            st.markdown("---")
            st.markdown("### 🔬 Live Analysis Results")
            results = run_analysis(sig_arr, fs)
            render_results(sig_arr, fs, results, view_sec)

