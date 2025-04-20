import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
    
    # Beceri eşleştirmesini oluştur
    skill_mapping = {}
    for _, row in questions_df.iterrows():
        question_num = int(row['Soru no'])
        skills = []
        
        if row['Sosyal Beceri'] == 'Evet':
            skills.append('Sosyal Beceri')
        if row['Duyusal Beceri'] == 'Evet':
            skills.append('Duyusal Beceri')
        if row['Motor beceri'] == 'Evet':
            skills.append('Motor beceri')
        if row['Dil Becerisi'] == 'Evet':
            skills.append('Dil Becerisi')
        if row['Ortak Dikkat Becerileri'] == 'Evet':
            skills.append('Ortak Dikkat Becerileri')
        if row['İletişim Becerileri'] == 'Evet':
            skills.append('İletişim Becerileri')
            
        skill_mapping[question_num] = skills
    
    return questions_df, skill_mapping

# Model yükleme
@st.cache_resource
def load_models():
    models = {}
    model_names = [
        'RandomForest', 'LogisticRegression', 'SVC', 'KNN',
        'GradientBoosting', 'DecisionTree'
    ]
    
    for name in model_names:
        try:
            models[name] = joblib.load(f'models/{name}.joblib')
        except:
            st.error(f"{name} modeli yüklenemedi!")
    
    return models

# Ana uygulama
def main():
    st.title("🧠 UDEMKO - Gelişim ve Bozukluk Tahmin Paneli")
    
    # Veri ve modelleri yükle
    questions_df, skill_mapping = load_data()
    models = load_models()
    
    # Model seçimi
    st.header("1. Model Seçimi")
    selected_model = st.selectbox(
        "Lütfen bir model seçin:",
        list(models.keys())
    )
    
    # Soruları yükle
    try:
        with open('models/model_sorular.json', 'r') as f:
            selected_questions = json.load(f)
    except:
        st.error("Soru listesi yüklenemedi!")
        return
    
    # Soru-cevap formu
    st.header("2. Soruları Yanıtlayın")
    answers = {}
    
    for q_num in selected_questions:
        q_num = int(q_num.replace('Q', ''))
        question_text = questions_df[questions_df['Soru no'] == q_num]['Soru'].iloc[0]
        
        answer = st.radio(
            f"Soru {q_num}: {question_text}",
            ["Evet", "Hayır"],
            key=f"q_{q_num}"
        )
        answers[q_num] = 1 if answer == "Evet" else 0
    
    # Tahmin yapma
    if st.button("Tahmin Yap"):
        # Cevap vektörünü oluştur
        X_pred = pd.DataFrame([answers])
        X_pred.columns = [f"Q{q}" for q in answers.keys()]
        
        # Tahmin yap
        model = models[selected_model]
        prediction = model.predict(X_pred)[0]
        proba = model.predict_proba(X_pred)[0]
        
        # Sonuçları göster
        st.header("3. Tahmin Sonuçları")
        
        if prediction == 1:
            st.error("⚠️ Riskli durum tespit edildi!")
        else:
            st.success("✅ Gelişimsel risk saptanmadı")
        
        # Olasılık dağılımı
        fig = go.Figure(data=[
            go.Bar(
                x=['Risksiz', 'Riskli'],
                y=proba,
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(
            title="Risk Olasılık Dağılımı",
            xaxis_title="Durum",
            yaxis_title="Olasılık"
        )
        st.plotly_chart(fig)
        
        # Beceri analizi
        st.header("4. Beceri Analizi")
        missing_skills = set()
        
        for q_num, answer in answers.items():
            if answer == 0:  # Hayır cevabı
                skills = skill_mapping.get(q_num, [])
                missing_skills.update(skills)
        
        if missing_skills:
            st.warning("⚠️ Aşağıdaki beceri alanlarında eksiklikler tespit edildi:")
            for skill in missing_skills:
                st.write(f"- {skill}")
            
            # Radar chart
            skill_counts = {skill: sum(1 for q in answers if answer == 0 and skill in skill_mapping.get(q, []))
                          for skill in missing_skills}
            
            fig = go.Figure(data=go.Scatterpolar(
                r=list(skill_counts.values()),
                theta=list(skill_counts.keys()),
                fill='toself'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(skill_counts.values())]
                    )
                ),
                title="Eksik Beceri Analizi"
            )
            
            st.plotly_chart(fig)
        else:
            st.success("✅ Tüm beceri alanlarında yeterli gelişim gözlemlendi.")

if __name__ == "__main__":
    main()
