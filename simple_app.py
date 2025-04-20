import streamlit as st
import pandas as pd
import numpy as np
import json

# Sayfa yapılandırması
st.set_page_config(
    page_title="UDEMKO - Gelişim ve Bozukluk Tahmin Paneli",
    page_icon="🧠",
    layout="wide"
)

# Veri yükleme
@st.cache_data
def load_data():
    # Soruları yükle
    questions_df = pd.read_csv('15-18 ay sorular.csv', sep=';')
    return questions_df

def main():
    st.title("🧠 UDEMKO - Gelişim ve Bozukluk Tahmin Paneli")
    
    # Veriyi yükle
    questions_df = load_data()
    
    # Soru-cevap formu
    st.header("Soruları Yanıtlayın")
    answers = {}
    
    # İlk 10 soruyu göster (test için)
    for i in range(1, 11):
        question_text = questions_df[questions_df['Soru no'] == i]['Soru'].iloc[0]
        
        answer = st.radio(
            f"Soru {i}: {question_text}",
            ["Evet", "Hayır"],
            key=f"q_{i}"
        )
        answers[i] = 1 if answer == "Evet" else 0
    
    # Tahmin yapma
    if st.button("Tahmin Yap"):
        st.header("Sonuçlar")
        
        # Basit bir tahmin (test için)
        total_yes = sum(answers.values())
        if total_yes >= 7:
            st.success("✅ Gelişimsel risk saptanmadı")
        else:
            st.error("⚠️ Riskli durum tespit edildi!")
        
        # Beceri analizi
        st.header("Beceri Analizi")
        for i, answer in answers.items():
            if answer == 0:  # Hayır cevabı
                skills = questions_df[questions_df['Soru no'] == i][
                    ['Sosyal Beceri', 'Duyusal Beceri', 'Motor beceri', 
                     'Dil Becerisi', 'Ortak Dikkat Becerileri', 'İletişim Becerileri']
                ].iloc[0]
                
                missing_skills = [col for col in skills.index if skills[col] == 'Evet']
                if missing_skills:
                    st.warning(f"Soru {i} için eksik beceriler:")
                    for skill in missing_skills:
                        st.write(f"- {skill}")

if __name__ == "__main__":
    main() 