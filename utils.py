import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def load_models(model_dir: str = "models") -> Dict[str, object]:
    """
    Belirtilen dizindeki tüm model dosyalarını yükler.
    
    Args:
        model_dir: Model dosyalarının bulunduğu dizin
    
    Returns:
        Yüklenen modellerin sözlüğü {model_adı: model_nesnesi}
    """
    models = {}
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"{model_dir} dizini bulunamadı!")
        
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_dir, model_file)
            try:
                model = joblib.load(model_path)
                models[model_file] = model
            except Exception as e:
                print(f"Hata: {model_file} yüklenirken hata oluştu: {str(e)}")
                continue
    
    if not models:
        raise ValueError("Hiçbir model yüklenemedi!")
    
    return models

def load_feature_lists(excel_file: str = "selected_features.xlsx") -> Dict[str, List[str]]:
    """
    Excel dosyasından özellik listelerini yükler.
    
    Args:
        excel_file: Özellik listelerini içeren Excel dosyası
    
    Returns:
        Özellik listelerinin sözlüğü {sayfa_adı: [özellikler]}
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"{excel_file} dosyası bulunamadı!")
        
    feature_lists = {}
    excel = pd.ExcelFile(excel_file)
    
    for sheet_name in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=sheet_name)
        features = df['Feature'].tolist() if 'Feature' in df.columns else df.columns.tolist()
        feature_lists[sheet_name] = features
    
    return feature_lists

def load_model_performances(excel_file: str = "model_performance.xlsx") -> pd.DataFrame:
    """
    Model performans metriklerini yükler.
    
    Args:
        excel_file: Model performanslarını içeren Excel dosyası
    
    Returns:
        Model performanslarını içeren DataFrame
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"{excel_file} dosyası bulunamadı!")
        
    return pd.read_excel(excel_file)

def prepare_input_data(user_answers: Dict[str, int], feature_lists: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Kullanıcı cevaplarını model girişi için hazırlar.
    
    Args:
        user_answers: Kullanıcı cevapları {soru_id: cevap}
        feature_lists: Her model için özellik listeleri
    
    Returns:
        Her model için hazırlanmış giriş verileri
    """
    prepared_data = {}
    
    for sheet_name, features in feature_lists.items():
        data = {}
        for feature in features:
            if feature in user_answers:
                data[feature] = [user_answers[feature]]
            else:
                print(f"Uyarı: {feature} özelliği kullanıcı cevaplarında bulunamadı!")
                data[feature] = [0]  # Varsayılan değer
                
        prepared_data[sheet_name] = pd.DataFrame(data)
    
    return prepared_data

def make_predictions(models: Dict[str, object], 
                    prepared_data: Dict[str, pd.DataFrame],
                    performances: pd.DataFrame) -> Dict[str, float]:
    """
    Tüm modeller için tahmin yapar ve sonuçları birleştirir.
    
    Args:
        models: Yüklenmiş modeller
        prepared_data: Hazırlanmış giriş verileri
        performances: Model performans metrikleri
    
    Returns:
        Her hastalık için tahmin olasılıkları
    """
    predictions = {}
    
    for model_file, model in models.items():
        try:
            # Model adından hastalık ve özellik sayısını çıkar
            parts = model_file.replace('.pkl', '').split('_')
            disease = '_'.join(parts[2:]) if len(parts) > 2 else parts[-1]
            feature_count = parts[1]
            
            # İlgili veri setini al
            data = prepared_data.get(f"{feature_count}_{disease}")
            if data is None:
                continue
                
            # Tahmin yap
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(data)[0][1]
            else:
                prob = float(model.predict(data)[0])
            
            # Hastalık bazında tahminleri topla
            if disease not in predictions:
                predictions[disease] = []
            predictions[disease].append(prob)
            
        except Exception as e:
            print(f"Hata: {model_file} ile tahmin yapılırken hata oluştu: {str(e)}")
            continue
    
    # Her hastalık için ortalama tahmin hesapla
    final_predictions = {}
    for disease, probs in predictions.items():
        if probs:
            final_predictions[disease] = np.mean(probs)
    
    return final_predictions 