import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import zipfile
import tempfile
from config import MODEL_LIST, MODEL_ZIP_PATH, FEATURE_FILE, PERFORMANCE_FILE

def load_models() -> Dict[str, Any]:
    """
    Zip dosyasından modelleri yükler.
    """
    if not os.path.exists(MODEL_ZIP_PATH):
        raise FileNotFoundError(f"Model zip dosyası bulunamadı: {MODEL_ZIP_PATH}")
    
    models = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            loaded_models = 0
            for model_name in MODEL_LIST:
                model_path = os.path.join(temp_dir, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    loaded_models += 1
                else:
                    print(f"Uyarı: {model_name}.pkl dosyası bulunamadı")
            
            if loaded_models == 0:
                raise Exception("Hiçbir model yüklenemedi!")
                
        except Exception as e:
            raise Exception(f"Modeller yüklenirken hata oluştu: {str(e)}")
    
    return models

def load_feature_lists(feature_file: str = FEATURE_FILE) -> Dict[str, List[str]]:
    """
    Excel dosyasından özellik listelerini yükler
    """
    feature_lists = {}
    df = pd.read_excel(feature_file, sheet_name=None)
    
    for sheet_name in df.keys():
        feature_lists[sheet_name] = df[sheet_name]['Features'].tolist()
    
    return feature_lists

def load_model_performances(performance_file: str = PERFORMANCE_FILE) -> Dict[str, float]:
    """
    Excel dosyasından model performanslarını yükler
    """
    performances = {}
    df = pd.read_excel(performance_file)
    
    for _, row in df.iterrows():
        model_name = row['Model']
        performances[model_name] = row['F1_Score']
    
    return performances

def prepare_input_data(answers: Dict[str, int], feature_lists: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Kullanıcı cevaplarını model girişi için hazırlar
    """
    input_data = {}
    
    for category, features in feature_lists.items():
        if category != 'all_features':  # all_features kategorisini atla
            category_data = []
            for feature in features:
                if feature in answers:
                    category_data.append(answers[feature])
                else:
                    category_data.append(0)  # Varsayılan değer
            input_data[category] = np.array(category_data).reshape(1, -1)
    
    return input_data

def make_predictions(models: Dict, input_data: Dict[str, np.ndarray], performances: Dict[str, float]) -> Dict[str, Dict]:
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