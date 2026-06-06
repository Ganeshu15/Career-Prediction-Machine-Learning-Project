import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

# ── Load artifacts ─────────────────────────────────────────────────────────
BASE = Path(__file__).parent

PIPELINES = {
    'Random Forest':       joblib.load(BASE / 'pipeline_random_forest.pkl'),
    'Logistic Regression': joblib.load(BASE / 'pipeline_logistic_regression.pkl'),
    'Decision Tree':       joblib.load(BASE / 'pipeline_decision_tree.pkl'),
    'SVM':                 joblib.load(BASE / 'pipeline_svm.pkl'),
}
le       = joblib.load(BASE / 'label_encoder.pkl')
feat_imp = joblib.load(BASE / 'feature_importance.pkl')
with open(BASE / 'model_summary.json') as f: MODEL_SUMMARY = json.load(f)
with open(BASE / 'feature_meta.json')  as f: META = json.load(f)

df_raw = pd.read_csv(BASE / 'cs_career_dataset.csv')

SKILL_MAP     = META['skill_map']
SKILL_MAP_INV = {v: k for k, v in SKILL_MAP.items()}

LABEL_MAP = {
    'Age': 'Age', 'GPA': 'GPA',
    'Interested Domain': 'Interested Domain', 'Projects': 'Projects',
    'Python': 'Python Skill', 'SQL': 'SQL Skill', 'Java': 'Java Skill',
    'Gender_Male': 'Gender (Male)',
}

def _hex_with_alpha(hex_color: str, alpha: float = 0.2) -> str:
    """Return an `rgba(r,g,b,a)` string for a `#rrggbb` hex color.
    Falls back to the original string on error.
    """
    try:
        h = hex_color.lstrip('#')
        if len(h) == 3:
            h = ''.join(2 * ch for ch in h)
        if len(h) != 6:
            return hex_color
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    except Exception:
        return hex_color
CAREER_INFO = {
    'AI/ML Engineer':   {'icon':'🤖','color':'#00F5A0','glow':'0 0 20px #00F5A044',
        'desc':'Membangun & melatih model AI untuk produk nyata',
        'skills':['Python','TensorFlow/PyTorch','Mathematics','Data Preprocessing'],
        'roadmap':[('0–6 bln','Python, NumPy, Pandas, Matplotlib'),
                   ('6–12 bln','ML fundamentals: Scikit-learn, klasifikasi, regresi'),
                   ('1–2 thn','Deep Learning: TensorFlow/PyTorch, CNN, RNN'),
                   ('2–3 thn','Spesialisasi: NLP, Computer Vision, RL'),
                   ('3+ thn','Research Engineer / Senior ML Engineer')],
        'avg_salary':'Rp 12–25 jt/bln','demand':'🔥 Sangat Tinggi'},
    'Data Scientist':   {'icon':'📊','color':'#4FC3F7','glow':'0 0 20px #4FC3F744',
        'desc':'Analisis pola data untuk business insight',
        'skills':['Statistics','SQL','Python/R','Data Visualization'],
        'roadmap':[('0–6 bln','Python/R, SQL, statistik dasar'),
                   ('6–12 bln','EDA, visualisasi: Tableau/Power BI'),
                   ('1–2 thn','ML applied: prediksi, segmentasi, A/B testing'),
                   ('2–3 thn','Business analytics, storytelling, cloud'),
                   ('3+ thn','Lead Data Scientist / Analytics Manager')],
        'avg_salary':'Rp 10–20 jt/bln','demand':'🔥 Sangat Tinggi'},
    'Backend Engineer': {'icon':'⚙️','color':'#B39DDB','glow':'0 0 20px #B39DDB44',
        'desc':'Merancang sistem, API, dan arsitektur server-side',
        'skills':['Node.js/Go/Java','Database','REST/GraphQL','System Design'],
        'roadmap':[('0–6 bln','Kuasai satu bahasa backend: Node.js/Go/Java'),
                   ('6–12 bln','Database SQL+NoSQL, REST API, auth'),
                   ('1–2 thn','System design, microservices, Docker, CI/CD'),
                   ('2–3 thn','Cloud AWS/GCP/Azure, Kubernetes'),
                   ('3+ thn','Senior Backend Engineer / Software Architect')],
        'avg_salary':'Rp 10–22 jt/bln','demand':'📈 Tinggi'},
    'Frontend/UI-UX':   {'icon':'🎨','color':'#FFB74D','glow':'0 0 20px #FFB74D44',
        'desc':'Mendesain tampilan & pengalaman pengguna yang intuitif',
        'skills':['React/Vue','Figma','CSS/Tailwind','User Research'],
        'roadmap':[('0–6 bln','HTML, CSS, JavaScript, responsive design'),
                   ('6–12 bln','React/Vue, Figma, design system'),
                   ('1–2 thn','UI/UX research, prototyping, user testing'),
                   ('2–3 thn','Full design pipeline, product thinking'),
                   ('3+ thn','Senior UI/UX Designer / Frontend Lead')],
        'avg_salary':'Rp 8–18 jt/bln','demand':'📈 Tinggi'},
    'Cybersecurity':    {'icon':'🔐','color':'#EF5350','glow':'0 0 20px #EF535044',
        'desc':'Melindungi sistem & data dari ancaman siber',
        'skills':['Networking','Linux','Ethical Hacking','Cryptography'],
        'roadmap':[('0–6 bln','Networking fundamentals, Linux, scripting'),
                   ('6–12 bln','CompTIA Security+, ethical hacking, CTF'),
                   ('1–2 thn','Penetration testing, OWASP, vulnerability assessment'),
                   ('2–3 thn','Red/blue team, SIEM, incident response, CEH/OSCP'),
                   ('3+ thn','Security Engineer / Penetration Tester')],
        'avg_salary':'Rp 12–28 jt/bln','demand':'🚀 Sangat Tinggi & Langka'},
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='Career Path Finder', page_icon='🎯',
                   layout='centered', initial_sidebar_state='collapsed')

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html,body,[class*="css"]{font-family:'Inter',sans-serif;}
  .stApp{background:#0A0E1A;color:#E8EAF0;}
    .block-container{max-width:780px;padding-top:1.5rem;}
  #MainMenu,footer,header{visibility:hidden;}
  .stSlider>div>div>div{background:#1E2435!important;}
  .stSlider>div>div>div>div{background:linear-gradient(90deg,#00F5A0,#00B4D8)!important;}
  .stButton>button{background:linear-gradient(135deg,#00F5A0,#00B4D8)!important;
    color:#0A0E1A!important;font-weight:700!important;border:none!important;
    border-radius:12px!important;padding:.7rem 1.5rem!important;
    box-shadow:0 0 24px #00F5A055!important;}
  .stButton>button:hover{box-shadow:0 0 36px #00F5A088!important;}
    div[data-testid="stTabs"] button{color:#6B7280!important;font-size:.88rem!important;}
  div[data-testid="stTabs"] button[aria-selected="true"]{color:#00F5A0!important;border-bottom:2px solid #00F5A0!important;}
  .glass-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:.75rem;}
  .glass-card.top-card{border:1.5px solid #00F5A0;box-shadow:0 0 28px #00F5A022;}
    .sec-label{font-size:1rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#4A5580;margin:1.6rem 0 .6rem;}
  .prog-wrap{background:#1E2435;border-radius:8px;height:6px;overflow:hidden;margin-bottom:1.5rem;}
  .prog-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#00F5A0,#00B4D8);}
  .hero{text-align:center;padding:1.5rem 0 .5rem;}
  .hero h1{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#00F5A0,#00B4D8,#7B7FFF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.4rem;}
  .hero p{color:#6B7280;font-size:.95rem;line-height:1.6;}
  .badge{display:inline-block;font-size:.65rem;padding:4px 12px;border-radius:20px;background:rgba(0,245,160,.12);color:#00F5A0;font-weight:600;border:1px solid rgba(0,245,160,.3);margin-bottom:.75rem;}
  .skill-pill{display:inline-block;font-size:.72rem;padding:3px 10px;border-radius:20px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#CBD5E1;margin:3px 3px 0 0;}
  .stat-box{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:.7rem 1rem;text-align:center;}
  .stat-val{font-size:1rem;font-weight:600;}
  .stat-key{font-size:.7rem;color:#6B7280;margin-top:.15rem;}
  .cbar-bg{background:#1E2435;border-radius:4px;height:6px;overflow:hidden;margin-top:.6rem;}
  .slider-q{font-size:.92rem;color:#CBD5E1;margin-bottom:.3rem;font-weight:500;}
  .scale-hint{display:flex;justify-content:space-between;font-size:.7rem;color:#4A5580;margin-top:-.4rem;margin-bottom:.6rem;}
  div[data-testid="stSlider"] label{color:#9CA3AF!important;font-size:.85rem!important;}
  div[data-testid="stSelectbox"] label{color:#9CA3AF!important;}
  div[data-testid="stNumberInput"] label{color:#9CA3AF!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'qpage' not in st.session_state:
    st.session_state.qpage = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {
        'Gender': 'Male', 'Age': 22, 'GPA': 3.0,
        'Interested Domain': META['domain_classes'][0],
        'Projects': META['projects_classes'][0],
        'Python': 2, 'SQL': 2, 'Java': 2,
    }

def prog_bar(step, total=3):
    pct = int((step/total)*100)
    st.markdown(f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

def radar_chart(answers):
    feats  = ['Python','SQL','Java','GPA']
    labels = ['Python','SQL','Java','GPA (×1.25)']
    vals   = [answers['Python'], answers['SQL'], answers['Java'], min(answers['GPA']*1.25, 5)]
    vals  += [vals[0]]; labels += [labels[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=labels, fill='toself',
        fillcolor='rgba(0,245,160,0.12)',
        line=dict(color='#00F5A0', width=2), marker=dict(color='#00F5A0', size=5)))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,4], tickfont=dict(size=9,color='#4A5580'),
                gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.06)'),
            angularaxis=dict(tickfont=dict(size=11,color='#9CA3AF'),
                gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.06)')),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=20,l=40,r=40),
        height=280, showlegend=False)
    return fig

def conf_bar(top3):
    names=[c for c,_ in top3]; probs=[p*100 for _,p in top3]
    colors=[CAREER_INFO[c]['color'] for c,_ in top3]
    fig=go.Figure(go.Bar(x=probs,y=names,orientation='h',
        marker=dict(color=colors,opacity=0.85),
        text=[f'{p:.1f}%' for p in probs],textposition='outside',
        textfont=dict(color='#E8EAF0',size=13)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0,115],visible=False),
        yaxis=dict(tickfont=dict(size=13,color='#CBD5E1')),
        margin=dict(t=10,b=10,l=10,r=60),height=160)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_home, tab_finder, tab_eda, tab_model, tab_about = st.tabs([
    '🏠  Home','🎯  Career Finder','📈  EDA','📊  Model Performance','ℹ️  About'
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
      <span class="badge">✦ Powered by Machine Learning</span>
      <h1>Career Path Finder</h1>
      <p>Bingung mau ambil jalur karier apa setelah lulus?<br>
         Isi profil singkat — model ML kami analisis dan rekomendasikan<br>
         career path yang paling cocok untukmu.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#00F5A0">180</div><div class="stat-key">Dataset Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#4FC3F7">5</div><div class="stat-key">Career Paths</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#B39DDB">4</div><div class="stat-key">ML Models</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#FFB74D">84%</div><div class="stat-key">Best CV Acc</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🗺️ Career Paths yang Tersedia</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col,(career,info) in zip(cols, CAREER_INFO.items()):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:1rem .8rem">
              <div style="font-size:1.8rem">{info['icon']}</div>
              <div style="font-size:.75rem;font-weight:600;color:{info['color']};margin-top:.4rem;line-height:1.3">{career}</div>
              <div style="font-size:.65rem;color:#6B7280;margin-top:.3rem">{info['avg_salary']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">⚙️ ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="font-size:.88rem;color:#CBD5E1;line-height:2">
      <b style="color:#00F5A0">Input</b> (Gender, Age, GPA, Domain, Projects, Python, SQL, Java)
      &nbsp;→&nbsp;<b style="color:#4FC3F7">Preprocessing</b> (OneHot + StandardScaler)
      &nbsp;→&nbsp;<b style="color:#B39DDB">SMOTE</b>
      &nbsp;→&nbsp;<b style="color:#FFB74D">Model</b> (RF / LR / DT / SVM)
      &nbsp;→&nbsp;<b style="color:#00F5A0">Top 3 Rekomendasi</b><br>
      <span style="color:#4A5580;font-size:.75rem">
        Pipeline menggunakan <code>imblearn.pipeline.Pipeline</code> —
        SMOTE hanya pada training data untuk mencegah data leakage.
      </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAREER FINDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_finder:
    qp = st.session_state.qpage

    if qp == 0:
        prog_bar(1)
        st.markdown('<div class="sec-label">👤 Profil Diri</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            gender = st.selectbox('Gender', ['Male','Female'],
                index=['Male','Female'].index(st.session_state.answers['Gender']))
        with c2:
            age = st.number_input('Umur', min_value=18, max_value=40,
                value=int(st.session_state.answers['Age']))

        gpa = st.slider('GPA (IPK)', 2.0, 4.0,
            value=float(st.session_state.answers['GPA']), step=0.01, format='%.2f')

        st.markdown('<div class="sec-label">🎯 Minat & Pengalaman</div>', unsafe_allow_html=True)
        domain = st.selectbox('Interested Domain (bidang yang paling diminati)',
            META['domain_classes'],
            index=META['domain_classes'].index(st.session_state.answers['Interested Domain']))
        projects = st.selectbox('Projects (jenis proyek yang pernah dikerjakan)',
            META['projects_classes'],
            index=META['projects_classes'].index(st.session_state.answers['Projects']))

        st.session_state.answers.update({
            'Gender': gender, 'Age': age, 'GPA': gpa,
            'Interested Domain': domain, 'Projects': projects,
        })

        if st.button('Lanjut →', use_container_width=True):
            st.session_state.qpage = 1; st.rerun()

    elif qp == 1:
        prog_bar(2)
        st.markdown('<div class="sec-label">💻 Kemampuan Teknikal</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#6B7280;font-size:.85rem;margin-bottom:1rem">Nilai kemampuan coding kamu (1 = Weak, 2 = Average, 3 = Strong)</p>', unsafe_allow_html=True)

        for skill in ['Python','SQL','Java']:
            st.markdown(f'<div class="slider-q">{"🐍" if skill=="Python" else "🗄️" if skill=="SQL" else "☕"} {skill} Skill</div>', unsafe_allow_html=True)
            val = st.select_slider(skill, options=[1,2,3],
                value=st.session_state.answers[skill],
                format_func=lambda x: SKILL_MAP_INV[x],
                key=f'skill_{skill}', label_visibility='collapsed')
            st.session_state.answers[skill] = val
            st.markdown('<br>', unsafe_allow_html=True)

        c1,c2 = st.columns([1,2])
        with c1:
            if st.button('← Kembali', use_container_width=True, key='b1'):
                st.session_state.qpage=0; st.rerun()
        with c2:
            if st.button('Lanjut →', use_container_width=True, key='n1'):
                st.session_state.qpage=2; st.rerun()

    elif qp == 2:
        prog_bar(3)
        st.markdown('<div class="sec-label">🤖 Pilih Model ML</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#6B7280;font-size:.85rem;margin-bottom:1rem">Pilih algoritma yang ingin digunakan untuk prediksi career path-mu</p>', unsafe_allow_html=True)

        model_descs = {
            'Random Forest':       '🌲 Ensemble of decision trees — robust & accurate',
            'Logistic Regression': '📉 Linear classifier — fast & interpretable',
            'Decision Tree':       '🌿 Single tree — easy to visualize & explain',
            'SVM':                 '🔷 Support Vector Machine — great for small datasets',
        }
        chosen_model = st.radio('', list(PIPELINES.keys()),
            format_func=lambda x: f"{x}  —  {model_descs[x].split('—')[1].strip()}",
            label_visibility='collapsed')

        acc_info = MODEL_SUMMARY[chosen_model]
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00F5A0">{acc_info["test_acc"]*100:.1f}%</div><div class="stat-key">Test Accuracy</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#4FC3F7">{acc_info["cv_acc"]*100:.1f}%</div><div class="stat-key">CV Accuracy</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#FFB74D">{acc_info["f1"]*100:.1f}%</div><div class="stat-key">F1 Score</div></div>', unsafe_allow_html=True)

        c1,c2 = st.columns([1,2])
        with c1:
            if st.button('← Kembali', use_container_width=True, key='b2'):
                st.session_state.qpage=1; st.rerun()
        with c2:
            if st.button('🔍  Analisis Sekarang →', use_container_width=True, key='n2'):
                st.session_state.chosen_model = chosen_model
                st.session_state.qpage=3; st.rerun()

    elif qp == 3:
        ans    = st.session_state.answers
        chosen = st.session_state.get('chosen_model','Random Forest')
        pipe   = PIPELINES[chosen]

        le_domain   = joblib.load(BASE / 'le_domain.pkl')
        le_projects = joblib.load(BASE / 'le_projects.pkl')

        input_df = pd.DataFrame([{
            'Gender':           ans['Gender'],
            'Age':              int(ans['Age']),
            'GPA':              float(ans['GPA']),
            'Interested Domain':le_domain.transform([ans['Interested Domain']])[0],
            'Projects':         le_projects.transform([ans['Projects']])[0],
            'Python':           int(ans['Python']),
            'SQL':              int(ans['SQL']),
            'Java':             int(ans['Java']),
        }])

        proba = pipe.predict_proba(input_df)[0]
        top3  = sorted(zip(le.classes_, proba), key=lambda x:-x[1])[:3]
        best_career, best_conf = top3[0]
        info  = CAREER_INFO[best_career]

        st.markdown(f"""
        <div style="text-align:center;padding:1.2rem 0 .8rem">
          <div style="font-size:3rem">{info['icon']}</div>
          <div style="font-size:.6rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
                      color:{info['color']};margin:.4rem 0 .2rem">Rekomendasi Utama • {chosen}</div>
          <h2 style="font-size:1.8rem;font-weight:700;color:{info['color']};text-shadow:{info['glow']};margin:0">{best_career}</h2>
          <p style="color:#6B7280;margin:.3rem 0 .6rem">{info['desc']}</p>
          <span style="font-size:1.2rem;font-weight:700;color:{info['color']}">{int(best_conf*100)}%</span>
          <span style="font-size:.72rem;color:#4A5580;margin-left:.3rem">confidence</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">🏆 Top 3 Rekomendasi</div>', unsafe_allow_html=True)
        st.plotly_chart(conf_bar(top3), use_container_width=True, config={'displayModeBar':False})

        for rank,(career,conf) in enumerate(top3):
            ci = CAREER_INFO[career]
            rank_labels=['🥇 Rekomendasi Utama','🥈 Alternatif #2','🥉 Alternatif #3']
            pct=int(conf*100)
            skills_html=''.join(f'<span class="skill-pill">{s}</span>' for s in ci['skills'])
            st.markdown(f"""
            <div class="{'glass-card top-card' if rank==0 else 'glass-card'}">
              <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.6rem">
                <span style="font-size:1.8rem">{ci['icon']}</span>
                <div style="flex:1">
                  <div style="font-size:.6rem;color:#4A5580;font-weight:600;text-transform:uppercase;margin-bottom:.1rem">{rank_labels[rank]}</div>
                  <div style="font-size:1rem;font-weight:600;color:{ci['color']}">{career}</div>
                </div>
                <div style="font-size:1.3rem;font-weight:700;color:{ci['color']}">{pct}%</div>
              </div>
              <div class="cbar-bg"><div style="width:{pct}%;height:100%;background:{ci['color']};border-radius:4px"></div></div>
              <div style="margin-top:.7rem">{skills_html}</div>
              <div style="margin-top:.5rem;display:flex;gap:1rem">
                <div class="stat-box" style="flex:1"><div class="stat-val" style="font-size:.82rem;color:{ci['color']}">{ci['avg_salary']}</div><div class="stat-key">Avg. Salary</div></div>
                <div class="stat-box" style="flex:1"><div class="stat-val" style="font-size:.82rem">{ci['demand']}</div><div class="stat-key">Job Demand</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Skill radar
        st.markdown('<div class="sec-label">🕸️ Profil Skill Kamu</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(ans), use_container_width=True, config={'displayModeBar':False})

        # Roadmap
        st.markdown(f'<div class="sec-label">🗺️ Roadmap — {best_career}</div>', unsafe_allow_html=True)
        rm_html=''
        for i,(period,desc) in enumerate(info['roadmap']):
            op=1-i*.12
            rm_html+=f"""<div style="display:flex;gap:1rem;margin-bottom:.9rem;align-items:flex-start">
              <div style="padding-top:4px"><div style="width:10px;height:10px;border-radius:50%;background:{info['color']};opacity:{op}"></div></div>
              <div style="flex:1;background:rgba(255,255,255,.03);border-radius:10px;padding:.6rem .9rem;border-left:2px solid {info['color']}44">
                <div style="font-size:.7rem;font-weight:600;color:{info['color']};margin-bottom:.2rem">{period}</div>
                <div style="font-size:.88rem;color:#CBD5E1">{desc}</div>
              </div></div>"""
        st.markdown(rm_html, unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        if st.button('🔄  Mulai Ulang', use_container_width=True):
            st.session_state.qpage=0
            st.session_state.answers={
                'Gender':'Male','Age':22,'GPA':3.0,
                'Interested Domain':META['domain_classes'][0],
                'Projects':META['projects_classes'][0],
                'Python':2,'SQL':2,'Java':2,
            }
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-label">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Class distribution
    st.markdown('<div class="sec-label">📦 Distribusi Career Path</div>', unsafe_allow_html=True)
    vc = df_raw['career_path'].value_counts().reset_index()
    vc.columns=['career_path','count']
    fig_cls=go.Figure(go.Bar(x=vc['career_path'],y=vc['count'],
        marker_color=[CAREER_INFO[c]['color'] for c in vc['career_path']],
        text=vc['count'],textposition='outside',textfont=dict(color='#9CA3AF',size=11)))
    fig_cls.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#CBD5E1',size=10),gridcolor='rgba(255,255,255,.04)'),
        yaxis=dict(tickfont=dict(color='#9CA3AF',size=10),gridcolor='rgba(255,255,255,.05)'),
        margin=dict(t=20,b=10,l=10,r=10),height=280)
    st.plotly_chart(fig_cls,use_container_width=True,config={'displayModeBar':False})

    # Histogram
    st.markdown('<div class="sec-label">📊 Histogram</div>', unsafe_allow_html=True)
    hist_feat = st.selectbox('Pilih fitur:',['Age','GPA','Python','SQL','Java'],key='hist_feat')
    fig_hist=go.Figure()

    is_discrete = hist_feat in ['Python','SQL','Java']

    if is_discrete:
        # Grouped bar chart untuk fitur diskrit (1=Weak, 2=Average, 3=Strong)
        skill_labels = {1:'Weak', 2:'Average', 3:'Strong'}
        for career in le.classes_:
            sub = df_raw[df_raw['career_path']==career][hist_feat].value_counts().sort_index()
            fig_hist.add_trace(go.Bar(
                x=[skill_labels[i] for i in sub.index],
                y=sub.values, name=career,
                marker_color=CAREER_INFO[career]['color'], opacity=0.85))
        fig_hist.update_layout(barmode='group')
    else:
        # Histogram overlap untuk fitur kontinu
        for career in le.classes_:
            sub=df_raw[df_raw['career_path']==career][hist_feat]
            fig_hist.add_trace(go.Histogram(x=sub,name=career,
                marker_color=CAREER_INFO[career]['color'],opacity=0.7,nbinsx=12))
        fig_hist.update_layout(barmode='overlay')

    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#CBD5E1',size=10),bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.04)',
                   title=dict(text=hist_feat,font=dict(color='#6B7280'))),
        yaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.05)',
                   title=dict(text='Count',font=dict(color='#6B7280'))),
        margin=dict(t=20,b=10,l=10,r=10),height=300)
    st.plotly_chart(fig_hist,use_container_width=True,config={'displayModeBar':False})

    # Boxplot
    st.markdown('<div class="sec-label">📦 Boxplot per Career</div>', unsafe_allow_html=True)
    box_feat=st.selectbox('Pilih fitur:',['Age','GPA','Python','SQL','Java'],key='box_feat')
    fig_box=go.Figure()
    for career in le.classes_:
        sub=df_raw[df_raw['career_path']==career][box_feat]
        fig_box.add_trace(go.Box(y=sub,name=career,
            marker_color=CAREER_INFO[career]['color'],
            line_color=CAREER_INFO[career]['color'],
            fillcolor=_hex_with_alpha(CAREER_INFO[career]['color'], 0.2)))
    fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#CBD5E1',size=10),bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#CBD5E1',size=10),gridcolor='rgba(255,255,255,.04)'),
        yaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.05)',
                   title=dict(text=box_feat,font=dict(color='#6B7280'))),
        margin=dict(t=20,b=10,l=10,r=10),height=300)
    st.plotly_chart(fig_box,use_container_width=True,config={'displayModeBar':False})

    # Scatterplot
    st.markdown('<div class="sec-label">🔵 Scatterplot</div>', unsafe_allow_html=True)
    num_cols=['Age','GPA','Python','SQL','Java']
    c1,c2=st.columns(2)
    with c1: x_feat=st.selectbox('Sumbu X',num_cols,index=0,key='sc_x')
    with c2: y_feat=st.selectbox('Sumbu Y',num_cols,index=1,key='sc_y')
    fig_sc=go.Figure()
    for career in le.classes_:
        sub=df_raw[df_raw['career_path']==career]
        fig_sc.add_trace(go.Scatter(x=sub[x_feat],y=sub[y_feat],mode='markers',name=career,
            marker=dict(color=CAREER_INFO[career]['color'],size=7,opacity=0.75,
                        line=dict(width=0.5,color='rgba(255,255,255,.2)'))))
    fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#CBD5E1',size=10),bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.04)',
                   title=dict(text=x_feat,font=dict(color='#6B7280'))),
        yaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.05)',
                   title=dict(text=y_feat,font=dict(color='#6B7280'))),
        margin=dict(t=20,b=10,l=10,r=10),height=320)
    st.plotly_chart(fig_sc,use_container_width=True,config={'displayModeBar':False})

    # Correlation matrix
    st.markdown('<div class="sec-label">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
    corr=df_raw[num_cols].corr().round(2)
    fig_corr=go.Figure(go.Heatmap(z=corr.values,x=num_cols,y=num_cols,
        colorscale=[[0,'#EF5350'],[0.5,'#1E2435'],[1,'#00F5A0']],
        zmid=0,text=corr.values,texttemplate='%{text}',
        textfont=dict(size=11,color='white'),showscale=True,
        colorbar=dict(tickfont=dict(color='#9CA3AF'))))
    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#9CA3AF',size=11)),
        yaxis=dict(tickfont=dict(color='#9CA3AF',size=11),autorange='reversed'),
        margin=dict(t=20,b=10,l=10,r=10),height=360)
    st.plotly_chart(fig_corr,use_container_width=True,config={'displayModeBar':False})

    # Gender distribution
    st.markdown('<div class="sec-label">👤 Gender Distribution per Career</div>', unsafe_allow_html=True)
    gd=df_raw.groupby(['career_path','Gender']).size().reset_index(name='count')
    fig_gd=go.Figure()
    for g,col in [('Male','#4FC3F7'),('Female','#FFB74D')]:
        sub=gd[gd['Gender']==g]
        fig_gd.add_trace(go.Bar(name=g,x=sub['career_path'],y=sub['count'],
            marker_color=col,opacity=0.85))
    fig_gd.update_layout(barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#CBD5E1',size=10),bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#CBD5E1',size=10),gridcolor='rgba(255,255,255,.04)'),
        yaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.05)'),
        margin=dict(t=20,b=10,l=10,r=10),height=280)
    st.plotly_chart(fig_gd,use_container_width=True,config={'displayModeBar':False})

    # Age distribution per career
    st.markdown('<div class="sec-label">📅 Age Distribution per Career</div>', unsafe_allow_html=True)
    fig_age=go.Figure()
    for career in le.classes_:
        sub=df_raw[df_raw['career_path']==career]['Age']
        fig_age.add_trace(go.Box(x=sub,name=career,
            marker_color=CAREER_INFO[career]['color'],
            line_color=CAREER_INFO[career]['color'],
            fillcolor=_hex_with_alpha(CAREER_INFO[career]['color'], 0.2),
            orientation='h'))
    fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#9CA3AF'),gridcolor='rgba(255,255,255,.04)',
                   title=dict(text='Age',font=dict(color='#6B7280'))),
        yaxis=dict(tickfont=dict(color='#CBD5E1',size=10),gridcolor='rgba(255,255,255,.04)'),
        margin=dict(t=20,b=10,l=10,r=10),height=280)
    st.plotly_chart(fig_age,use_container_width=True,config={'displayModeBar':False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    model_names=['Logistic Regression','Decision Tree','Random Forest','SVM']
    metrics=['test_acc','cv_acc','precision','recall','f1']
    metric_labels=['Test Accuracy','5-Fold CV','Precision','Recall','F1 Score']
    colors_m=['#00F5A0','#4FC3F7','#B39DDB','#FFB74D']

    st.markdown('<div class="sec-label">📈 Model Comparison</div>', unsafe_allow_html=True)
    fig_cmp=go.Figure()
    for mname,col in zip(model_names,colors_m):
        vals=[MODEL_SUMMARY[mname][m]*100 for m in metrics]
        fig_cmp.add_trace(go.Bar(name=mname,x=metric_labels,y=vals,
            marker_color=col,opacity=0.85,
            text=[f'{v:.1f}%' for v in vals],textposition='outside',
            textfont=dict(size=9,color='#9CA3AF')))
    fig_cmp.update_layout(barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#CBD5E1',size=11),bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,.1)',borderwidth=1),
        xaxis=dict(tickfont=dict(color='#9CA3AF',size=10),gridcolor='rgba(255,255,255,.04)'),
        yaxis=dict(range=[0,110],tickfont=dict(color='#9CA3AF',size=10),
                   gridcolor='rgba(255,255,255,.04)',ticksuffix='%'),
        margin=dict(t=20,b=10,l=10,r=10),height=340)
    st.plotly_chart(fig_cmp,use_container_width=True,config={'displayModeBar':False})

    st.markdown('<div class="sec-label">📋 Tabel Metrik</div>', unsafe_allow_html=True)
    best_acc=max(MODEL_SUMMARY[m]['test_acc'] for m in model_names)
    rows=[{'Model':('⭐ ' if MODEL_SUMMARY[m]['test_acc']==best_acc else '')+m,
           'Test Acc':f"{MODEL_SUMMARY[m]['test_acc']*100:.2f}%",
           'CV Acc':f"{MODEL_SUMMARY[m]['cv_acc']*100:.2f}%",
           'Precision':f"{MODEL_SUMMARY[m]['precision']*100:.2f}%",
           'Recall':f"{MODEL_SUMMARY[m]['recall']*100:.2f}%",
           'F1 Score':f"{MODEL_SUMMARY[m]['f1']*100:.2f}%"} for m in model_names]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown('<div class="sec-label">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    cm_model=st.selectbox('Pilih model:',model_names,key='cm_model')
    cm=np.array(MODEL_SUMMARY[cm_model]['cm'])
    fig_cm=go.Figure(go.Heatmap(z=cm,x=le.classes_,y=le.classes_,
        colorscale=[[0,'#0A0E1A'],[0.5,'#003d2e'],[1,'#00F5A0']],
        text=cm,texttemplate='%{text}',textfont=dict(size=12,color='white'),
        showscale=True,colorbar=dict(tickfont=dict(color='#9CA3AF'))))
    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#9CA3AF',size=10),
                   title=dict(text='Predicted',font=dict(color='#6B7280'))),
        yaxis=dict(tickfont=dict(color='#9CA3AF',size=10),
                   title=dict(text='Actual',font=dict(color='#6B7280')),autorange='reversed'),
        margin=dict(t=20,b=10,l=10,r=10),height=360)
    st.plotly_chart(fig_cm,use_container_width=True,config={'displayModeBar':False})

    st.markdown('<div class="sec-label">🎯 Per-Class Performance</div>', unsafe_allow_html=True)
    pc_model=st.selectbox('Pilih model:',model_names,key='pc_model')
    report=MODEL_SUMMARY[pc_model]['report']
    pc_rows=[{'Career Path':cls,
              'Precision':f"{report[cls]['precision']*100:.1f}%",
              'Recall':f"{report[cls]['recall']*100:.1f}%",
              'F1 Score':f"{report[cls]['f1-score']*100:.1f}%",
              'Support':int(report[cls]['support'])} for cls in le.classes_]
    st.dataframe(pd.DataFrame(pc_rows),use_container_width=True,hide_index=True)

    st.markdown('<div class="sec-label">🔑 Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    fi_labels=[LABEL_MAP.get(f,f) for f,_ in feat_imp]
    fi_vals=[v for _,v in feat_imp]
    fig_fi=go.Figure(go.Bar(x=fi_vals[::-1],y=fi_labels[::-1],orientation='h',
        marker=dict(color=fi_vals[::-1],
                    colorscale=[[0,'#1E2435'],[0.5,'#00B4D8'],[1,'#00F5A0']],
                    showscale=False),
        text=[f'{v:.3f}' for v in fi_vals[::-1]],textposition='outside',
        textfont=dict(color='#9CA3AF',size=10)))
    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False,range=[0,max(fi_vals)*1.3]),
        yaxis=dict(tickfont=dict(size=10,color='#CBD5E1')),
        margin=dict(t=10,b=10,l=10,r=60),height=300)
    st.plotly_chart(fig_fi,use_container_width=True,config={'displayModeBar':False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="hero" style="padding:1rem 0 .5rem">
      <span class="badge">ℹ️ Tentang Project</span>
      <h1>Career Path Finder</h1>
      <p>Final Project — Matkul Machine Learning</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🎯 Tentang Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="font-size:.9rem;color:#CBD5E1;line-height:1.8">
      Career Path Finder adalah sistem rekomendasi jalur karier berbasis Machine Learning
      yang membantu para mahasiswa Computer Science menentukan arah karier mereka sesuai profil,
      minat, dan kemampuan mereka.<br><br>
      Sistem ini memakai <b style="color:#00F5A0">Supervised Classification</b> dengan
      menggunakan 4 algoritma ML yang dibandingkan secara eksplisit, dilengkapi dengan
      <b style="color:#4FC3F7">Explainable AI</b> berupa feature importance, skill radar chart,
      dan roadmap karier per jalur.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🗃️ Dataset</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#00F5A0">180</div><div class="stat-key">Total Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#4FC3F7">8</div><div class="stat-key">Features</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#B39DDB">5</div><div class="stat-key">Career Labels</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-val" style="color:#FFB74D">0</div><div class="stat-key">Missing Values</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">⚙️ Tech Stack</div>', unsafe_allow_html=True)
    techs=[('Python 3.11','🐍'),('Scikit-learn','🤖'),('Imbalanced-learn / SMOTE','⚖️'),
           ('Streamlit','🚀'),('Plotly','📊'),('Pandas & NumPy','🔢')]
    cols=st.columns(3)
    for i,(tech,icon) in enumerate(techs):
        with cols[i%3]:
            st.markdown(f'<div class="glass-card" style="text-align:center;padding:.8rem"><div style="font-size:1.4rem">{icon}</div><div style="font-size:.8rem;color:#CBD5E1;margin-top:.3rem">{tech}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">📚 Referensi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="font-size:.85rem;color:#9CA3AF;line-height:2">
      • Breiman, L. (2001). <i>Random Forests.</i> Machine Learning, 45(1), 5–32.<br>
      • Chawla et al. (2002). <i>SMOTE: Synthetic Minority Over-sampling Technique.</i> JAIR.<br>
      • Pedregosa et al. (2011). <i>Scikit-learn: Machine Learning in Python.</i> JMLR.<br>
      • Kaggle Dataset: CS Students Career Dataset (2024).
    </div>""", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#2D3555;font-size:.72rem;margin-top:2rem">Final Project Machine Learning · 2024/2025</p>', unsafe_allow_html=True)