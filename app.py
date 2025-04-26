import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
from pathlib import Path

# Sayfa yapılandırması
st.set_page_config(
    page_title="Çocuk Gelişimi Değerlendirme",
    page_icon="👶",
    layout="wide"
)

# Başlık
st.title("Çocuk Gelişimi Değerlendirme Sistemi")

# Soruları yükle
@st.cache_data
def load_questions():
    df = pd.read_csv('SorularFull.csv', sep=';')
    return df

# Modelleri yükle
@st.cache_resource
def load_models():
    models_dir = Path("models")
    models = {}
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)
    return models

try:
    questions_df = load_questions()
    models = load_models()
except Exception as e:
    st.error(f"Dosya yükleme hatası: {str(e)}")
    st.stop()

# Soru listesi
QUESTION_IDS = ['Q2','Q4','Q8','Q9','Q13','Q14','Q16','Q18','Q19','Q20','Q21','Q25','Q26','Q28','Q29',
    'Q33','Q34','Q35','Q40','Q44','Q45','Q47','Q51','Q52','Q53','Q54','Q60','Q62','Q67',
    'Q71','Q77','Q81','Q82','Q86','Q89','Q93','Q95','Q96','Q105','Q108','Q115','Q116',
    'Q117','Q119','Q125','Q126','Q127','Q128','Q129','Q130','Q133','Q138','Q139','Q140',
    'Q144','Q151','Q158','Q159','Q163','Q166','Q174','Q179','Q184','Q185','Q187','Q192',
    'Q197','Q202','Q203','Q204','Q205','Q210','Q212','Q215','Q219','Q221','Q222','Q224',
    'Q226','Q227','Q229','Q230','Q231','Q232','Q233','Q234','Q235','Q236','Q239','Q241',
    'Q242','Q243','Q249','Q252','Q253']

# Kullanıcı cevapları için sözlük
if 'answers' not in st.session_state:
    st.session_state.answers = {}

# Hızlı cevap butonu
if st.button("🎲 Hızlı Cevap"):
    for q in QUESTION_IDS:
        st.session_state.answers[q] = random.choice(['Evet', 'Hayır'])
    st.rerun()

# Soruları göster
for q in QUESTION_IDS:
    question_text = questions_df[questions_df['Soru no'] == q]['Soru'].values[0] if len(questions_df[questions_df['Soru no'] == q]) > 0 else q
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"{q}: {question_text}")
    with col2:
        st.session_state.answers[q] = st.radio(
            "Cevap",
            options=["Evet", "Hayır"],
            key=f"radio_{q}",
            horizontal=True,
            label_visibility="collapsed"
        )

# Tahmin butonu
if st.button("🔍 Tahmin Et"):
    with st.spinner('Sonuçlarınız analiz ediliyor...'):
        # Burada model tahminleri yapılacak
        st.success("Analiz tamamlandı!")
        
        # Beceri ve hastalık sonuçları için bölümler
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Beceri Değerlendirmesi")
            # Beceri sonuçları gösterilecek
            
        with col2:
            st.subheader("Hastalık Değerlendirmesi")
            # Hastalık sonuçları gösterilecek 