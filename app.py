import streamlit as st
import pandas as pd
import numpy as np
import glob
from joblib import load
import random
import time
import os
import zipfile

# Sayfa yapılandırması
st.set_page_config(page_title="Gelişimsel Tarama Testi", layout="wide")

# Session state başlatma
if 'page' not in st.session_state:
    st.session_state.page = 'questions'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'answers' not in st.session_state:
    st.session_state.answers = {}

# 1) Modelleri zip dosyasından çıkart ve yükle
st.write("Debug - Model Yükleme:")

# Zip dosyasından modelleri çıkart
if not os.path.exists('models'):
    os.makedirs('models')
    with zipfile.ZipFile('models.zip', 'r') as zip_ref:
        zip_ref.extractall('models')

# Model dosyalarını bul
model_paths = glob.glob('models/*.pkl')
st.write(f"Bulunan model dosyaları: {model_paths}")

if not model_paths:
    st.error("Model dosyaları bulunamadı! Lütfen models.zip dosyasının doğru konumda olduğundan emin olun.")
    st.stop()

models = {}
for p in model_paths:
    try:
        model_name = os.path.basename(p)[:-4]  # .pkl uzantısını kaldır
        models[model_name] = load(p)
        st.write(f"Model yüklendi: {model_name}")
    except Exception as e:
        st.write(f"Hata: {model_name} yüklenirken hata oluştu: {str(e)}")

if not models:
    st.error("Hiçbir model yüklenemedi! Lütfen model dosyalarının doğru formatta olduğundan emin olun.")
    st.stop()

# 2) Soru havuzunu tanımla (95 soru)
QUESTION_POOL = ['Q2','Q4','Q8','Q9','Q13','Q14','Q16','Q18','Q19','Q20','Q21','Q25','Q26','Q28','Q29',
    'Q33','Q34','Q35','Q40','Q44','Q45','Q47','Q51','Q52','Q53','Q54','Q60','Q62','Q67',
    'Q71','Q77','Q81','Q82','Q86','Q89','Q93','Q95','Q96','Q105','Q108','Q115','Q116',
    'Q117','Q119','Q125','Q126','Q127','Q128','Q129','Q130','Q133','Q138','Q139','Q140',
    'Q144','Q151','Q158','Q159','Q163','Q166','Q174','Q179','Q184','Q185','Q187','Q192',
    'Q197','Q202','Q203','Q204','Q205','Q210','Q212','Q215','Q219','Q221','Q222','Q224',
    'Q226','Q227','Q229','Q230','Q231','Q232','Q233','Q234','Q235','Q236','Q239','Q241',
    'Q242','Q243','Q249','Q252','Q253']

# 3) Meta veriler
st.write("\nDebug - Meta Veriler:")
try:
    perf = pd.read_excel("model_performance.xlsx")
    st.write("model_performance.xlsx yüklendi")
    sel_feat = pd.read_excel("selected_features.xlsx")
    st.write("selected_features.xlsx yüklendi")
    cfg = sel_feat.merge(perf, on="Model").set_index("Model")
    st.write("Meta veriler birleştirildi")
except Exception as e:
    st.write(f"Hata: Meta veriler yüklenirken hata oluştu: {str(e)}")

# Rastgele cevaplama fonksiyonu
def randomize_answers():
    return {q: random.choice([True, False]) for q in QUESTION_POOL}

# Yeni test başlatma fonksiyonu
def start_new_test():
    st.session_state.answers = {}
    st.session_state.results = None
    st.session_state.page = 'questions'
    st.rerun()

# Ana sayfa
if st.session_state.page == 'questions':
    st.title("Gelişimsel Tarama Testi")
    st.write("Her soruya Evet/Hayır ile yanıt verin:")

    # Rastgele cevaplama butonu
    if st.button("Rastgele Cevapla"):
        st.session_state.answers = randomize_answers()
        st.rerun()

    # Soruları göster ve cevapları kaydet
    for q in QUESTION_POOL:
        if q in st.session_state.answers:
            # Eğer cevap varsa, radio butonunu o değerle göster
            st.session_state.answers[q] = (st.radio(q, ["Evet","Hayır"], index=1 if st.session_state.answers[q] else 0)=="Evet")
        else:
            # Eğer cevap yoksa, yeni bir radio butonu oluştur
            st.session_state.answers[q] = (st.radio(q, ["Evet","Hayır"])=="Evet")

    if st.button("Tahmin Et"):
        st.session_state.page = 'analyzing'
        st.rerun()

# Analiz sayfası
elif st.session_state.page == 'analyzing':
    st.title("Analiz Yapılıyor")
    st.write("Lütfen bekleyin, sonuçlarınız analiz ediliyor...")
    
    # Analiz simülasyonu
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
    
    # Tahminleri hesapla
    preds, wts = {}, {}
    for name, clf in models.items():
        feats = [f.strip() for f in cfg.loc[name,"Selected_Questions"].split(',')]
        X = np.array([[st.session_state.answers[f] for f in feats]], dtype=int)
        preds[name] = clf.predict(X)[0]
        wts[name]   = (cfg.loc[name,"Train_F1"] + cfg.loc[name,"Test_F1"])/2

    results = {}
    for label in set(name.split('_',2)[2] for name in models):
        ms = [m for m in preds if m.endswith(label)]
        votes  = np.array([preds[m] for m in ms])
        weights= np.array([wts[m]    for m in ms])
        score  = (votes*weights).sum()/weights.sum()
        results[label] = score>=0.5

    st.session_state.results = results
    st.session_state.page = 'results'
    st.rerun()

# Sonuçlar sayfası
elif st.session_state.page == 'results':
    st.title("Sonuçlar")
    
    # Debug bilgisi
    st.write("Debug - Ham Sonuçlar:")
    for lbl, val in st.session_state.results.items():
        st.write(f"{lbl}: {val}")
    
    # Sonuçları göster
    has_issues = False
    for lbl, val in st.session_state.results.items():
        if not val:  # val False ise sorun var demektir
            has_issues = True
            st.write(f"{lbl}: ❌")
        else:
            st.write(f"{lbl}: ✅")

    if not has_issues:
        st.success("Herşey yolunda görünüyor. Harika! 😊")
    else:
        # Yanlış yapılan sorular
        st.subheader("Yanlış Yapılan Sorular")
        for lbl, val in st.session_state.results.items():
            if not val:  # val False ise sorun var demektir
                pool = set()
                for m in models:
                    if m.endswith(lbl):
                        pool |= set(cfg.loc[m,"Selected_Questions"].split(','))
                wrongs = [q for q in pool if not st.session_state.answers[q.strip()]]
                st.write(f"**{lbl}**: {', '.join(wrongs) or 'Yok'}")

    # Ana sayfaya dön butonu
    if st.button("Yeni Test Başlat"):
        start_new_test() 