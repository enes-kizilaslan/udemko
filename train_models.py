import pandas as pd
from src.data_processing import load_data, preprocess_data
from src.model_training import select_features, train_models, save_models_and_features

def main():
    # Veriyi yükle ve işle
    print("Veri yükleniyor...")
    questions_df, children_df = load_data()
    questions_df, children_df = preprocess_data(questions_df, children_df)
    
    # Özellik seçimi
    print("Özellik seçimi yapılıyor...")
    X = children_df.drop(['ChildID', 'target'], axis=1)
    y = children_df['target']
    selected_features = select_features(X, y)
    
    # Modelleri eğit
    print("Modeller eğitiliyor...")
    models = train_models(X, y, selected_features)
    
    # Modelleri ve özellikleri kaydet
    print("Modeller ve özellikler kaydediliyor...")
    save_models_and_features(models, selected_features)
    
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    main() 