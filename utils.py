import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import zipfile
import tempfile
from app import MODEL_LIST

def load_models(model_zip_path):
    """
    Zip dosyasından modelleri yükler
    """
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model zip dosyası bulunamadı: {model_zip_path}")

    models = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            for model_name in MODEL_LIST:
                model_path = os.path.join(temp_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                else:
                    print(f"Uyarı: {model_name}.joblib dosyası zip içinde bulunamadı")
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    if not models:
        raise Exception("Hiçbir model yüklenemedi!")
    
    return models

def load_feature_lists(feature_file):
    """
    Excel dosyasından özellik listelerini yükler
    """
    feature_lists = {}
    df = pd.read_excel(feature_file, sheet_name=None)
    
    for sheet_name in df.keys():
        feature_lists[sheet_name] = df[sheet_name]['Features'].tolist()
    
    return feature_lists

def load_model_performances(performance_file):
    """
    Excel dosyasından model performanslarını yükler
    """
    performances = {}
    df = pd.read_excel(performance_file)
    
    for _, row in df.iterrows():
        model_name = row['Model']
        performances[model_name] = row['F1_Score']
    
    return performances

def prepare_input_data(answers, feature_lists):
    """
    Kullanıcı cevaplarını model girişi için hazırlar
    """
    input_data = {}
    
    for category, features in feature_lists.items():
        category_data = []
        for feature in features:
            if feature in answers:
                category_data.append(answers[feature])
            else:
                category_data.append(0)  # Varsayılan değer
        input_data[category] = np.array(category_data).reshape(1, -1)
    
    return input_data

def make_predictions(models, input_data, performances):
    """
    Tüm modeller için tahmin yapar ve ağırlıklı sonuçları döndürür
    """
    results = {}
    
    for category in input_data.keys():
        category_predictions = []
        category_weights = []
        
        for model_name, model in models.items():
            if category in model_name:
                prediction = model.predict_proba(input_data[category])[0]
                weight = performances.get(model_name, 1.0)
                
                category_predictions.append(prediction)
                category_weights.append(weight)
        
        if category_predictions:
            # Ağırlıklı ortalama hesaplama
            category_predictions = np.array(category_predictions)
            category_weights = np.array(category_weights)
            weighted_pred = np.average(category_predictions, weights=category_weights, axis=0)
            
            # Sonuçları kaydet
            results[category] = {
                'probability': weighted_pred[1],  # Pozitif sınıf olasılığı
                'prediction': 1 if weighted_pred[1] >= 0.5 else 0
            }
    
    return results 