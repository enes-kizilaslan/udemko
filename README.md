# UDEMKO - Gelişim ve Bozukluk Tahmin Paneli

15-18 aylık çocukların gelişimsel durumlarını ve olası zihinsel bozukluk risklerini değerlendiren bir tahmin paneli.

## Proje Yapısı

```
udemko/
├── data/                  # Veri dosyaları
├── models/               # Eğitilmiş modeller
├── notebooks/            # Jupyter notebook'lar
├── src/                  # Kaynak kodlar
│   ├── data_processing.py
│   └── model_training.py
├── app/                  # Streamlit uygulaması
│   └── app.py
├── requirements.txt      # Bağımlılıklar
└── README.md
```

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Modelleri eğitin:
```bash
python train_models.py
```

3. Uygulamayı başlatın:
```bash
streamlit run app/app.py
```

## Kullanım

1. Streamlit arayüzünde bir model seçin
2. Soruları yanıtlayın
3. "Tahmin Yap" butonuna tıklayın
4. Sonuçları ve beceri analizini inceleyin

## Özellikler

- 8 farklı makine öğrenmesi modeli
- 30 önemli soru seçimi
- Gelişimsel beceri analizi
- Görselleştirmeler (radar chart, olasılık dağılımı)
- Kullanıcı dostu arayüz

## Veri Seti

- 253 gelişimsel soru
- 300 çocuk verisi
- 6 farklı beceri alanı
- Risk durumu etiketleri

## Modeller

- Random Forest
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Gradient Boosting
- Decision Tree
- CatBoost
- LightGBM 