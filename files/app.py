import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── Load artifacts ─────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
pipeline = joblib.load(BASE / 'pipeline_rf.pkl')
le       = joblib.load(BASE / 'label_encoder.pkl')
feat_imp = joblib.load(BASE / 'feature_importance.pkl')

import json
with open(BASE / 'model_summary.json') as f:
    MODEL_SUMMARY = json.load(f)

# ── Constants ───────────────────────────────────────────────────────────────
FEATURES = [
    'math_interest', 'coding_freq', 'stats_interest', 'debugging_pref',
    'visual_creativity', 'teamwork_pref', 'communication',
    'analytical_thinking', 'backend_interest', 'research_interest'
]

LABEL_MAP = {
    'math_interest':       'Math & Logic',
    'coding_freq':         'Coding Frequency',
    'stats_interest':      'Statistics',
    'debugging_pref':      'Debugging',
    'visual_creativity':   'Visual Creativity',
    'teamwork_pref':       'Teamwork',
    'communication':       'Communication',
    'analytical_thinking': 'Analytical Thinking',
    'backend_interest':    'Backend/Systems',
    'research_interest':   'Research',
}

CAREER_INFO = {
    'AI/ML Engineer': {
        'icon': '🤖', 'color': '#00F5A0', 'glow': '0 0 20px #00F5A044',
        'desc': 'Membangun & melatih model AI untuk produk nyata',
        'skills': ['Python', 'TensorFlow/PyTorch', 'Math & Statistics', 'Data Preprocessing'],
        'roadmap': [
            ('0–6 bln',  'Kuasai Python, NumPy, Pandas, Matplotlib'),
            ('6–12 bln', 'Pelajari ML fundamentals: Scikit-learn, regresi, klasifikasi'),
            ('1–2 thn',  'Deep Learning: TensorFlow / PyTorch, CNN, RNN'),
            ('2–3 thn',  'Spesialisasi: NLP, Computer Vision, atau Reinforcement Learning'),
            ('3+ thn',   'Research Engineer / Senior ML Engineer di perusahaan tech'),
        ],
        'avg_salary': 'Rp 12–25 jt/bln',
        'demand': '🔥 Sangat Tinggi',
    },
    'Data Scientist': {
        'icon': '📊', 'color': '#4FC3F7', 'glow': '0 0 20px #4FC3F744',
        'desc': 'Analisis pola data untuk business insight',
        'skills': ['Statistics', 'SQL', 'Python/R', 'Data Visualization'],
        'roadmap': [
            ('0–6 bln',  'Python/R, SQL, statistik dasar, Excel lanjutan'),
            ('6–12 bln', 'EDA, visualisasi data: Tableau / Power BI / Matplotlib'),
            ('1–2 thn',  'ML applied: prediksi, segmentasi, A/B testing'),
            ('2–3 thn',  'Business analytics, storytelling data, cloud (GCP/AWS)'),
            ('3+ thn',   'Lead Data Scientist / Analytics Manager'),
        ],
        'avg_salary': 'Rp 10–20 jt/bln',
        'demand': '🔥 Sangat Tinggi',
    },
    'Backend Engineer': {
        'icon': '⚙️', 'color': '#B39DDB', 'glow': '0 0 20px #B39DDB44',
        'desc': 'Merancang sistem, API, dan arsitektur server-side',
        'skills': ['Node.js/Go/Java', 'Database Design', 'REST/GraphQL', 'System Design'],
        'roadmap': [
            ('0–6 bln',  'Kuasai satu bahasa backend: Node.js / Go / Python / Java'),
            ('6–12 bln', 'Database: SQL + NoSQL, REST API design, authentication'),
            ('1–2 thn',  'System design, microservices, Docker, CI/CD'),
            ('2–3 thn',  'Cloud (AWS/GCP/Azure), Kubernetes, performance optimization'),
            ('3+ thn',   'Senior Backend Engineer / Software Architect'),
        ],
        'avg_salary': 'Rp 10–22 jt/bln',
        'demand': '📈 Tinggi',
    },
    'Frontend/UI-UX': {
        'icon': '🎨', 'color': '#FFB74D', 'glow': '0 0 20px #FFB74D44',
        'desc': 'Mendesain tampilan & pengalaman pengguna yang intuitif',
        'skills': ['React/Vue', 'Figma', 'CSS/Tailwind', 'User Research'],
        'roadmap': [
            ('0–6 bln',  'HTML, CSS, JavaScript dasar, responsive design'),
            ('6–12 bln', 'React / Vue, Figma, design system, accessibility'),
            ('1–2 thn',  'UI/UX research, prototyping, user testing, animasi'),
            ('2–3 thn',  'Full design pipeline, design-to-dev handoff, product thinking'),
            ('3+ thn',   'Senior UI/UX Designer / Frontend Lead / Product Designer'),
        ],
        'avg_salary': 'Rp 8–18 jt/bln',
        'demand': '📈 Tinggi',
    },
    'Cybersecurity': {
        'icon': '🔐', 'color': '#EF5350', 'glow': '0 0 20px #EF535044',
        'desc': 'Melindungi sistem & data dari ancaman siber',
        'skills': ['Networking', 'Linux', 'Ethical Hacking', 'Cryptography'],
        'roadmap': [
            ('0–6 bln',  'Networking fundamentals, Linux, basic scripting'),
            ('6–12 bln', 'CompTIA Security+, ethical hacking basics, CTF challenges'),
            ('1–2 thn',  'Penetration testing, OWASP, vulnerability assessment'),
            ('2–3 thn',  'Red team / blue team, SIEM, incident response, CEH/OSCP'),
            ('3+ thn',   'Security Engineer / Penetration Tester / CISO track'),
        ],
        'avg_salary': 'Rp 12–28 jt/bln',
        'demand': '🚀 Sangat Tinggi & Langka',
    },
}

QUESTIONS = [
    ('math_interest',       '🔢 Seberapa besar minatmu terhadap matematika & logika?'),
    ('stats_interest',      '📉 Seberapa besar minatmu terhadap statistik & analisis data?'),
    ('research_interest',   '🔬 Seberapa besar minatmu terhadap riset & eksperimen?'),
    ('analytical_thinking', '🧠 Seberapa kuat kemampuan analytical thinking-mu?'),
    ('backend_interest',    '🖥️ Seberapa besar minatmu terhadap sistem & backend?'),
    ('coding_freq',         '💻 Seberapa sering kamu menulis kode per minggu?'),
    ('debugging_pref',      '🐛 Seberapa suka kamu debugging & problem solving?'),
    ('visual_creativity',   '🎨 Seberapa tinggi kreativitas visual & minat desain-mu?'),
    ('teamwork_pref',       '🤝 Seberapa besar preferensimu untuk kerja dalam tim?'),
    ('communication',       '🎤 Seberapa percaya diri kemampuan komunikasi & presentasimu?'),
]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Career Path Finder',
    page_icon='🎯',
    layout='centered',
    initial_sidebar_state='collapsed',
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0A0E1A; color: #E8EAF0; }
  .block-container { max-width: 740px; padding-top: 2rem; }
  #MainMenu, footer, header { visibility: hidden; }
  .stSlider > div > div > div { background: #1E2435 !important; }
  .stSlider > div > div > div > div { background: linear-gradient(90deg,#00F5A0,#00B4D8) !important; }
  .stButton > button {
    background: linear-gradient(135deg, #00F5A0, #00B4D8) !important;
    color: #0A0E1A !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: .75rem 2rem !important; font-size: 1rem !important;
    box-shadow: 0 0 24px #00F5A055 !important;
  }
  .stButton > button:hover { box-shadow: 0 0 36px #00F5A088 !important; }
  .glass-card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: .75rem;
  }
  .glass-card.top-card { border: 1.5px solid #00F5A0; box-shadow: 0 0 28px #00F5A022; }
  .sec-label {
    font-size: .65rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #4A5580; margin: 1.8rem 0 .8rem;
  }
  .prog-wrap { background: #1E2435; border-radius: 8px; height: 6px; overflow: hidden; margin-bottom: 1.5rem; }
  .prog-fill  { height: 100%; border-radius: 8px; background: linear-gradient(90deg,#00F5A0,#00B4D8); }
  .hero { text-align: center; padding: 1.5rem 0 .5rem; }
  .hero h1 { font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg,#00F5A0,#00B4D8,#7B7FFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .4rem; }
  .hero p { color: #6B7280; font-size: .95rem; line-height: 1.6; }
  .badge { display:inline-block; font-size:.65rem; padding:4px 12px; border-radius:20px;
    background: rgba(0,245,160,.12); color:#00F5A0; font-weight:600;
    border: 1px solid rgba(0,245,160,.3); margin-bottom:.75rem; }
  .skill-pill { display:inline-block; font-size:.72rem; padding:3px 10px; border-radius:20px;
    background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
    color:#CBD5E1; margin:3px 3px 0 0; }
  .stat-box { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:.7rem 1rem; text-align:center; }
  .stat-val  { font-size:1rem; font-weight:600; }
  .stat-key  { font-size:.7rem; color:#6B7280; margin-top:.15rem; }
  .cbar-bg { background:#1E2435; border-radius:4px; height:6px; overflow:hidden; margin-top:.6rem; }
  .roadmap-item { display:flex; gap:1rem; margin-bottom:.9rem; align-items:flex-start; }
  .slider-q { font-size:.92rem; color:#CBD5E1; margin-bottom:.3rem; font-weight:500; }
  .scale-hint { display:flex; justify-content:space-between; font-size:.7rem; color:#4A5580;
    margin-top:-.4rem; margin-bottom:.6rem; }
  div[data-testid="stSlider"] label { color: #9CA3AF !important; font-size: .85rem !important; }
  .streamlit-expanderHeader { background: rgba(255,255,255,.03) !important;
    border-radius:12px !important; color:#9CA3AF !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {f: 3 for f in FEATURES}

def progress_bar(step, total=4):
    pct = int((step / total) * 100)
    st.markdown(f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

def radar_chart(answers):
    cats = [LABEL_MAP[f] for f in FEATURES]
    vals = [answers[f] for f in FEATURES]
    cats += [cats[0]]; vals += [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(0,245,160,0.12)',
        line=dict(color='#00F5A0', width=2),
        marker=dict(color='#00F5A0', size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,5], tickfont=dict(size=9, color='#4A5580'),
                gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.06)'),
            angularaxis=dict(tickfont=dict(size=10, color='#9CA3AF'),
                gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.06)'),
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=40, r=40), height=320, showlegend=False,
    )
    return fig

def confidence_chart(top3):
    names  = [c for c, _ in top3]
    probs  = [p * 100 for _, p in top3]
    colors = [CAREER_INFO[c]['color'] for c, _ in top3]
    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside', textfont=dict(color='#E8EAF0', size=13),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 115], visible=False),
        yaxis=dict(tickfont=dict(size=13, color='#CBD5E1')),
        margin=dict(t=10, b=10, l=10, r=60), height=160,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Landing (with tab nav)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 0:

    tab1, tab2 = st.tabs(['🏠  Home', '📊  Model Performance'])

    # ── TAB 1: Landing ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="hero">
          <span class="badge">✦ Powered by Machine Learning</span>
          <h1>Career Path Finder</h1>
          <p>Jawab 10 pertanyaan singkat.<br>
             Model Random Forest kami analisis profilmu<br>
             dan rekomendasikan jalur karier yang paling cocok.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-box"><div class="stat-val" style="color:#00F5A0">10</div><div class="stat-key">Pertanyaan</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-box"><div class="stat-val" style="color:#4FC3F7">5</div><div class="stat-key">Career Paths</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-box"><div class="stat-val" style="color:#00F5A0">85%</div><div class="stat-key">Model Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('🚀  Mulai Sekarang', use_container_width=True):
            st.session_state.page = 1
            st.rerun()
        st.markdown('<p style="text-align:center;color:#2D3555;font-size:.75rem;margin-top:1.5rem">Random Forest · SMOTE Augmentation · Explainable AI · sklearn Pipeline</p>', unsafe_allow_html=True)

    # ── TAB 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-label">⚙️ ML Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="font-size:.88rem;color:#CBD5E1;line-height:1.9">
          <b style="color:#00F5A0">Input Features</b>
          &nbsp;→&nbsp; <b style="color:#4FC3F7">SMOTE</b> (oversampling)
          &nbsp;→&nbsp; <b style="color:#B39DDB">StandardScaler</b>
          &nbsp;→&nbsp; <b style="color:#FFB74D">Random Forest</b> (300 trees)
          &nbsp;→&nbsp; <b style="color:#00F5A0">Prediction + Confidence</b>
          <br><br>
          <span style="color:#6B7280;font-size:.78rem">
            Pipeline dibangun menggunakan <code>imblearn.pipeline.Pipeline</code> —
            SMOTE hanya diaplikasikan pada training data, mencegah data leakage.
            Setiap step tersimpan dalam satu objek <code>.pkl</code>.
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">📈 Model Comparison — 4 Algoritma</div>', unsafe_allow_html=True)

        # Build comparison table
        model_names = list(MODEL_SUMMARY.keys())
        metrics     = ['test_acc', 'cv_acc', 'precision', 'recall', 'f1']
        metric_labels = ['Test Accuracy', '5-Fold CV Acc', 'Precision', 'Recall', 'F1 Score']

        # Grouped bar chart
        colors = ['#00F5A0', '#4FC3F7', '#B39DDB', '#FFB74D']
        fig_cmp = go.Figure()
        for i, (mname, col) in enumerate(zip(model_names, colors)):
            vals = [MODEL_SUMMARY[mname][m] * 100 for m in metrics]
            fig_cmp.add_trace(go.Bar(
                name=mname, x=metric_labels, y=vals,
                marker_color=col, opacity=0.85,
                text=[f'{v:.1f}%' for v in vals],
                textposition='outside', textfont=dict(size=9, color='#9CA3AF'),
            ))
        fig_cmp.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#CBD5E1', size=11),
                        bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,.1)', borderwidth=1),
            xaxis=dict(tickfont=dict(color='#9CA3AF', size=10), gridcolor='rgba(255,255,255,.04)'),
            yaxis=dict(range=[0, 110], tickfont=dict(color='#9CA3AF', size=10),
                       gridcolor='rgba(255,255,255,.04)', ticksuffix='%'),
            margin=dict(t=20, b=10, l=10, r=10), height=340,
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})

        # Metrics table
        st.markdown('<div class="sec-label">📋 Tabel Metrik Lengkap</div>', unsafe_allow_html=True)
        rows = []
        best_acc = max(MODEL_SUMMARY[m]['test_acc'] for m in model_names)
        for mname in model_names:
            s = MODEL_SUMMARY[mname]
            is_best = s['test_acc'] == best_acc
            rows.append({
                'Model': ('⭐ ' if is_best else '') + mname,
                'Test Acc': f"{s['test_acc']*100:.2f}%",
                'CV Acc':   f"{s['cv_acc']*100:.2f}%",
                'Precision':f"{s['precision']*100:.2f}%",
                'Recall':   f"{s['recall']*100:.2f}%",
                'F1 Score': f"{s['f1']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion Matrix — RF only
        st.markdown('<div class="sec-label">🔲 Confusion Matrix — Random Forest</div>', unsafe_allow_html=True)
        cm      = np.array(MODEL_SUMMARY['Random Forest']['cm'])
        classes = le.classes_
        fig_cm  = go.Figure(go.Heatmap(
            z=cm, x=classes, y=classes,
            colorscale=[[0,'#0A0E1A'],[0.5,'#003d2e'],[1,'#00F5A0']],
            text=cm, texttemplate='%{text}',
            textfont=dict(size=12, color='white'),
            showscale=True,
            colorbar=dict(tickfont=dict(color='#9CA3AF')),
        ))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='#9CA3AF', size=10),
                       title=dict(text='Predicted', font=dict(color='#6B7280'))),
            yaxis=dict(tickfont=dict(color='#9CA3AF', size=10),
                       title=dict(text='Actual', font=dict(color='#6B7280')), autorange='reversed'),
            margin=dict(t=20, b=10, l=10, r=10), height=360,
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

        # Per-class metrics
        st.markdown('<div class="sec-label">🎯 Per-Class Performance — Random Forest</div>', unsafe_allow_html=True)
        report  = MODEL_SUMMARY['Random Forest']['report']
        pc_rows = []
        career_colors = {c: CAREER_INFO[c]['color'] for c in le.classes_}
        for cls in le.classes_:
            r = report[cls]
            pc_rows.append({
                'Career Path': cls,
                'Precision': f"{r['precision']*100:.1f}%",
                'Recall':    f"{r['recall']*100:.1f}%",
                'F1 Score':  f"{r['f1-score']*100:.1f}%",
                'Support':   int(r['support']),
            })
        st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

        # Feature importance
        st.markdown('<div class="sec-label">🔑 Feature Importance</div>', unsafe_allow_html=True)
        fi_labels = [LABEL_MAP.get(f, f) for f, _ in feat_imp]
        fi_vals   = [v for _, v in feat_imp]
        fig_fi2 = go.Figure(go.Bar(
            x=fi_vals[::-1], y=fi_labels[::-1], orientation='h',
            marker=dict(color=fi_vals[::-1],
                        colorscale=[[0,'#1E2435'],[0.5,'#00B4D8'],[1,'#00F5A0']],
                        showscale=False),
            text=[f'{v:.3f}' for v in fi_vals[::-1]],
            textposition='outside', textfont=dict(color='#9CA3AF', size=10),
        ))
        fig_fi2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, range=[0, max(fi_vals)*1.3]),
            yaxis=dict(tickfont=dict(size=10, color='#CBD5E1')),
            margin=dict(t=10, b=10, l=10, r=60), height=300,
        )
        st.plotly_chart(fig_fi2, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGES 1–3 — Questions
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page in [1, 2, 3]:
    p = st.session_state.page
    progress_bar(p)

    section_titles = {
        1: '🧪 Bagian 1 — Minat Akademik & Riset',
        2: '💻 Bagian 2 — Kemampuan Teknikal',
        3: '🌐 Bagian 3 — Soft Skills & Preferensi',
    }
    page_questions = {
        1: QUESTIONS[:4],
        2: QUESTIONS[4:7],
        3: QUESTIONS[7:],
    }

    st.markdown(f'<div class="sec-label">{section_titles[p]}</div>', unsafe_allow_html=True)

    for feat, label in page_questions[p]:
        st.markdown(f'<div class="slider-q">{label}</div>', unsafe_allow_html=True)
        val = st.slider(label, 1, 5, value=st.session_state.answers[feat],
                        key=f'slider_{feat}', label_visibility='collapsed')
        st.session_state.answers[feat] = val
        st.markdown('<div class="scale-hint"><span>1 — Rendah</span><span>5 — Tinggi</span></div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

    col_back, col_next = st.columns([1, 2])
    with col_back:
        if st.button('← Kembali', use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with col_next:
        label_next = '🔍  Analisis Sekarang →' if p == 3 else 'Lanjut →'
        if st.button(label_next, use_container_width=True):
            st.session_state.page += 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Results
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 4:
    progress_bar(4)

    ans   = st.session_state.answers
    raw   = pd.DataFrame([[ans[f] for f in FEATURES]], columns=FEATURES)
    proba = pipeline.predict_proba(raw)[0]
    top3  = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
    best_career, best_conf = top3[0]
    info  = CAREER_INFO[best_career]

    # Hero
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <div style="font-size:3.5rem">{info['icon']}</div>
      <div style="font-size:.65rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
                  color:{info['color']};margin:.5rem 0 .3rem">Rekomendasi Utama</div>
      <h2 style="font-size:1.9rem;font-weight:700;color:{info['color']};
                 text-shadow:{info['glow']};margin:0">{best_career}</h2>
      <p style="color:#6B7280;margin:.4rem 0 .8rem">{info['desc']}</p>
      <span style="font-size:1.3rem;font-weight:700;color:{info['color']}">{int(best_conf*100)}%</span>
      <span style="font-size:.75rem;color:#4A5580;margin-left:.4rem">confidence</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence chart
    st.markdown('<div class="sec-label">🏆 Top 3 Rekomendasi</div>', unsafe_allow_html=True)
    st.plotly_chart(confidence_chart(top3), use_container_width=True, config={'displayModeBar': False})

    # Cards
    for rank, (career, conf) in enumerate(top3):
        ci = CAREER_INFO[career]
        rank_labels = ['🥇 Rekomendasi Utama', '🥈 Alternatif #2', '🥉 Alternatif #3']
        card_class  = 'glass-card top-card' if rank == 0 else 'glass-card'
        pct = int(conf * 100)
        skills_html = ''.join(f'<span class="skill-pill">{s}</span>' for s in ci['skills'])
        st.markdown(f"""
        <div class="{card_class}">
          <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.6rem">
            <span style="font-size:1.8rem">{ci['icon']}</span>
            <div style="flex:1">
              <div style="font-size:.6rem;color:#4A5580;font-weight:600;letter-spacing:.08em;
                          text-transform:uppercase;margin-bottom:.15rem">{rank_labels[rank]}</div>
              <div style="font-size:1.05rem;font-weight:600;color:{ci['color']}">{career}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:1.3rem;font-weight:700;color:{ci['color']}">{pct}%</div>
            </div>
          </div>
          <div class="cbar-bg">
            <div style="width:{pct}%;height:100%;background:{ci['color']};border-radius:4px"></div>
          </div>
          <div style="margin-top:.75rem">{skills_html}</div>
          <div style="margin-top:.5rem;display:flex;gap:1rem">
            <div class="stat-box" style="flex:1">
              <div class="stat-val" style="font-size:.85rem;color:{ci['color']}">{ci['avg_salary']}</div>
              <div class="stat-key">Avg. Salary</div>
            </div>
            <div class="stat-box" style="flex:1">
              <div class="stat-val" style="font-size:.85rem">{ci['demand']}</div>
              <div class="stat-key">Job Demand</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart
    st.markdown('<div class="sec-label">🕸️ Profil Kepribadian Kamu</div>', unsafe_allow_html=True)
    st.plotly_chart(radar_chart(ans), use_container_width=True, config={'displayModeBar': False})

    # Feature importance
    st.markdown('<div class="sec-label">🔑 Faktor Paling Berpengaruh (Model Insight)</div>', unsafe_allow_html=True)
    top6        = feat_imp[:6]
    feat_labels = [LABEL_MAP.get(f, f) for f, _ in top6]
    feat_vals   = [v for _, v in top6]
    fig_fi = go.Figure(go.Bar(
        x=feat_vals[::-1], y=feat_labels[::-1], orientation='h',
        marker=dict(color=feat_vals[::-1],
                    colorscale=[[0,'#1E2435'],[0.5,'#00B4D8'],[1,'#00F5A0']],
                    showscale=False),
        text=[f'{v:.3f}' for v in feat_vals[::-1]],
        textposition='outside', textfont=dict(color='#9CA3AF', size=11),
    ))
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[0, max(feat_vals) * 1.25]),
        yaxis=dict(tickfont=dict(size=11, color='#CBD5E1')),
        margin=dict(t=10, b=10, l=10, r=60), height=220,
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

    # Roadmap
    st.markdown(f'<div class="sec-label">🗺️ Roadmap Karier — {best_career}</div>', unsafe_allow_html=True)
    roadmap_html = ''
    for i, (period, desc) in enumerate(info['roadmap']):
        opacity = 1 - i * 0.12
        roadmap_html += f"""
        <div class="roadmap-item">
          <div style="padding-top:4px">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{info['color']};opacity:{opacity}"></div>
          </div>
          <div style="flex:1;background:rgba(255,255,255,.03);border-radius:10px;
                      padding:.6rem .9rem;border-left:2px solid {info['color']}44">
            <div style="font-size:.7rem;font-weight:600;color:{info['color']};
                        letter-spacing:.06em;margin-bottom:.2rem">{period}</div>
            <div style="font-size:.88rem;color:#CBD5E1">{desc}</div>
          </div>
        </div>"""
    st.markdown(roadmap_html, unsafe_allow_html=True)

    # Restart
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button('🔄  Mulai Ulang', use_container_width=True):
        st.session_state.page = 0
        st.session_state.answers = {f: 3 for f in FEATURES}
        st.rerun()

    st.markdown('<p style="text-align:center;color:#2D3555;font-size:.72rem;margin-top:1.2rem">Hasil ini bersifat rekomendasi berdasarkan model ML · Bukan keputusan final karier</p>', unsafe_allow_html=True)
