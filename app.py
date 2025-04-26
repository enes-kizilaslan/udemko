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

# Session state başlatma
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Kategori tanımlamaları
CATEGORIES = {
    'beceriler': {
        'Sosyal': [],
        'Duyusal': [],
        'Motor': [],
        'Dil': [],
        'Ortak_Dikkat': [],
        'Iletisim': []
    },
    'hastaliklar': {
        'Otizm': [],
        'DEHB': [],
        'Dil ve Konuşma Bozuklukları': [],
        'Gelişimsel Koordinasyon Bozukluğu': [],
        'Zihinsel Yetersizlik': []
    }
}

# Soruları yükle
@st.cache_data
def load_questions():
    try:
        df = pd.read_csv('SorularFull.csv', sep=';', encoding='windows-1254')
        return df
    except Exception as e:
        st.error(f"Soru dosyası okuma hatası: {str(e)}")
        return None

# Modelleri yükle ve kategorilere ayır
@st.cache_resource
def load_models():
    models_dir = Path("models")
    models = {}
    
    # Her model dosyası için
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)
        
        # Model ismini parçalara ayır
        parts = model_name.split('_')
        if len(parts) >= 3:
            model_type = parts[0]
            feature_count = parts[1]
            category = '_'.join(parts[2:])
            
            # Kategoriyi belirle ve modeli ilgili listeye ekle
            if category in CATEGORIES['beceriler']:
                CATEGORIES['beceriler'][category].append(model_name)
            elif category in CATEGORIES['hastaliklar']:
                CATEGORIES['hastaliklar'][category].append(model_name)
    
    return models

def prepare_input_data(answers):
    # Cevapları 1 ve 0'a dönüştür
    return np.array([1 if answers.get(q, "Hayır") == "Evet" else 0 for q in QUESTION_IDS]).reshape(1, -1)

def analyze_answers(answers, models, questions_df):
    results = {
        'beceriler': {},
        'hastaliklar': {}
    }
    
    input_data = prepare_input_data(answers)
    
    # Her kategori için analiz yap
    for category_type in ['beceriler', 'hastaliklar']:
        for category, model_names in CATEGORIES[category_type].items():
            if not model_names:  # Eğer kategoride model yoksa atla
                continue
                
            positive_count = 0
            total_count = len(model_names)
            wrong_questions = set()
            
            # Kategorideki her model için tahmin yap
            for model_name in model_names:
                try:
                    prediction = models[model_name].predict(input_data)[0]
                    if prediction == 1:
                        positive_count += 1
                        
                        # Yanlış cevaplanan soruları bul
                        category_questions = questions_df[questions_df[category.replace('_', ' ')] == 'Evet']['Soru no'].tolist()
                        for q in category_questions:
                            if q in answers and answers[q] != questions_df[questions_df['Soru no'] == q]['Sağlıklı Çocukta Beklenen Cevap'].values[0]:
                                wrong_questions.add(q)
                                
                except Exception as e:
                    st.error(f"Model {model_name} için tahmin hatası: {str(e)}")
                    continue
            
            # Sonuçları kaydet
            results[category_type][category] = {
                'pozitif': positive_count,
                'negatif': total_count - positive_count,
                'yanlis_sorular': sorted(list(wrong_questions))
            }
    
    return results

def show_home_page():
    st.title("Çocuk Gelişimi Değerlendirme Sistemi")
    
    questions_df = load_questions()
    if questions_df is None:
        st.stop()
    
    # Hızlı cevap butonu
    if st.button("🎲 Hızlı Cevap"):
        for q in QUESTION_IDS:
            st.session_state.answers[q] = random.choice(['Evet', 'Hayır'])
        st.rerun()
    
    # Soruları göster
    for q in QUESTION_IDS:
        try:
            question_text = questions_df[questions_df['Soru no'] == q]['Soru'].values[0]
        except:
            question_text = q
        
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
        st.session_state.page = 'analyzing'
        st.rerun()

def show_analyzing_page():
    st.title("Analiz Yapılıyor")
    
    # Progress bar göster
    progress_text = "Sonuçlarınız analiz ediliyor..."
    my_bar = st.progress(0, text=progress_text)
    
    # Model yükleme ve analiz
    try:
        questions_df = load_questions()
        models = load_models()
        
        # Analiz işlemi
        results = analyze_answers(st.session_state.answers, models, questions_df)
        st.session_state.analysis_results = results
        
        # Progress bar'ı güncelle
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # Analiz tamamlandığında sonuç sayfasına git
        st.session_state.page = 'results'
        st.rerun()
        
    except Exception as e:
        st.error(f"Analiz sırasında bir hata oluştu: {str(e)}")
        if st.button("Ana Sayfaya Dön"):
            st.session_state.page = 'home'
            st.rerun()

def show_results_page():
    st.title("Analiz Sonuçları")
    
    results = st.session_state.analysis_results
    if not results:
        st.error("Sonuç bulunamadı!")
        return
    
    # Herhangi bir eksiklik veya hastalık var mı kontrol et
    has_issues = False
    for category_type in ['beceriler', 'hastaliklar']:
        for category, data in results[category_type].items():
            if data['pozitif'] > data['negatif']:
                has_issues = True
                break
        if has_issues:
            break
    
    if not has_issues:
        st.success("🎉 Her şey yolunda, takibe devam!")
    else:
        # Beceri sonuçlarını göster
        st.header("Beceri Değerlendirmesi")
        for beceri, data in results['beceriler'].items():
            if data['pozitif'] > data['negatif']:  # Sadece sorunlu becerileri göster
                with st.expander(f"⚠️ {beceri.replace('_', ' ')} Becerisi"):
                    st.write(f"Çalıştırılan Model Sayısı: {data['pozitif'] + data['negatif']}")
                    st.write(f"Pozitif Sonuç: {data['pozitif']}")
                    st.write(f"Negatif Sonuç: {data['negatif']}")
                    if data['yanlis_sorular']:
                        st.write("İlişkili Yanlış Sorular:")
                        for soru in data['yanlis_sorular']:
                            st.write(f"- {soru}")
        
        # Hastalık sonuçlarını göster
        st.header("Hastalık Değerlendirmesi")
        for hastalik, data in results['hastaliklar'].items():
            if data['pozitif'] > data['negatif']:  # Sadece riskli hastalıkları göster
                with st.expander(f"⚠️ {hastalik}"):
                    st.write(f"Çalıştırılan Model Sayısı: {data['pozitif'] + data['negatif']}")
                    st.write(f"Pozitif Sonuç: {data['pozitif']}")
                    st.write(f"Negatif Sonuç: {data['negatif']}")
                    if data['yanlis_sorular']:
                        st.write("İlişkili Yanlış Sorular:")
                        for soru in data['yanlis_sorular']:
                            st.write(f"- {soru}")
    
    # Yeni test başlat butonu
    if st.button("🔄 Yeni Test Başlat"):
        # Session state'i temizle
        st.session_state.answers = {}
        st.session_state.analysis_results = None
        st.session_state.page = 'home'
        st.rerun()

# Sayfa yönlendirmesi
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'analyzing':
    show_analyzing_page()
elif st.session_state.page == 'results':
    show_results_page() 