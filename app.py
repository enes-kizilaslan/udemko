import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from pathlib import Path

QUESTION_IDS = ['Q2','Q4','Q8','Q9','Q13','Q14','Q16','Q18','Q19','Q20','Q21','Q25','Q26','Q28','Q29',
    'Q33','Q34','Q35','Q40','Q44','Q45','Q47','Q51','Q52','Q53','Q54','Q60','Q62','Q67',
    'Q71','Q77','Q81','Q82','Q86','Q89','Q93','Q95','Q96','Q105','Q108','Q115','Q116',
    'Q117','Q119','Q125','Q126','Q127','Q128','Q129','Q130','Q133','Q138','Q139','Q140',
    'Q144','Q151','Q158','Q159','Q163','Q166','Q174','Q179','Q184','Q185','Q187','Q192',
    'Q197','Q202','Q203','Q204','Q205','Q210','Q212','Q215','Q219','Q221','Q222','Q224',
    'Q226','Q227','Q229','Q230','Q231','Q232','Q233','Q234','Q235','Q236','Q239','Q241',
    'Q242','Q243','Q249','Q252','Q253']

# Sayfa yapılandırması
st.set_page_config(
    page_title="Çocuk Gelişimi Değerlendirme",
    page_icon="👶",
    layout="wide"
)

# Session state başlatma
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Soruları yükle ve kolon adlarını temizle
@st.cache_data
def load_questions():
    try:
        df = pd.read_csv('SorularFull.csv', sep=';', encoding='windows-1254')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Soru dosyası okuma hatası: {e}")
        return None

# Modelleri yükle
@st.cache_resource
def load_models():
    models_dir = Path("models")
    models = {}
    for model_file in models_dir.glob("*.pkl"):
        models[model_file.stem] = joblib.load(model_file)
    return models

# Girdi hazırlama
def prepare_input_data(answers):
    return np.array([1 if answers.get(q, "Hayır") == "Evet" else 0 for q in QUESTION_IDS]).reshape(1, -1)

# Analiz fonksiyonu
def analyze_answers(answers, models, questions_df):
    input_data = prepare_input_data(answers)
    total_positive = 0
    for _, model in models.items():
        try:
            total_positive += model.predict(input_data)[0]
        except Exception:
            continue
    return {'total_positive': total_positive, 'total_models': len(models)}

# Ana sayfa
def show_home_page():
    st.title("Çocuk Gelişimi Değerlendirme Sistemi")

    questions_df = load_questions()
    if questions_df is None:
        st.stop()

    # Id ve metin kolonlarını otomatik tespit et
    qid_col, qtext_col = questions_df.columns[0], questions_df.columns[1]

    # 1) Hızlı Cevap: sadece rastgele doldur
    if st.button("🎲 Hızlı Cevap"):
        for q in QUESTION_IDS:
            st.session_state.answers[q] = random.choice(["Evet", "Hayır"])
        st.experimental_rerun()

    # Soruları göster ve manuel doldurma
    for q in QUESTION_IDS:
        matched = questions_df[questions_df[qid_col] == q]
        question = matched[qtext_col].values[0] if not matched.empty else q
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{q}: {question}")
        with col2:
            current = st.session_state.answers.get(q)
            idx = 0 if current == "Evet" else 1 if current == "Hayır" else 0
            st.session_state.answers[q] = st.radio(
                "",
                options=["Evet", "Hayır"],
                index=idx,
                key=f"radio_{q}",
                horizontal=True,
                label_visibility="collapsed"
            )

    # 2) Analiz Et: mevcut cevaplarla analiz sayfasına geç
    if st.button("🔍 Analiz Et"):
        st.session_state.page = 'analyzing'
        st.experimental_rerun()

# Analiz sayfası
def show_analyzing_page():
    st.title("Analiz Yapılıyor...")
    bar = st.progress(0)
    try:
        qdf = load_questions()
        models = load_models()
        res = analyze_answers(st.session_state.answers, models, qdf)
        st.session_state.analysis_results = res
        for i in range(100):
            bar.progress(i + 1)
        st.session_state.page = 'results'
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Analiz hatası: {e}")
        if st.button("Ana Sayfaya Dön"):
            st.session_state.page = 'home'
            st.experimental_rerun()

# Sonuç sayfası
def show_results_page():
    st.title("Analiz Sonuçları")
    res = st.session_state.analysis_results
    if not res:
        st.error("Sonuç bulunamadı")
        return
    st.write(f"Toplam Model Sayısı: {res['total_models']}")
    st.write(f"Pozitif Tahmin Sayısı: {res['total_positive']}")
    st.write(f"Negatif Tahmin Sayısı: {res['total_models'] - res['total_positive']}")
    if st.button("🔄 Yeni Test"):
        st.session_state.answers = {}
        st.session_state.analysis_results = None
        st.session_state.page = 'home'
        st.experimental_rerun()

# Sayfa yönlendirmesi
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'analyzing':
    show_analyzing_page()
else:
    show_results_page()
