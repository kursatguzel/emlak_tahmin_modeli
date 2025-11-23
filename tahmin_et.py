import pandas as pd
import joblib
import os

def fiyat_tahmin_et(mahalle, m2, oda, yas, isitma):
    """
    UiPath'ten gelen tekil ilan bilgilerini alır ve tahmini fiyat döner.
    Girdiler UiPath'ten string veya int olarak gelebilir.
    """
    try:
        model_path = 'emlak_fiyat_tahmin_modeli.pkl' 
        model = joblib.load(model_path)
 
        input_data = pd.DataFrame({
            'mahalle': [str(mahalle)],
            'brut_m2': [float(str(m2).replace(' m2',''))], # UiPath ham yollarsa diye temizlik
            'oda_sayisi': [str(oda)],
            'bina_yasi': [int(str(yas).replace('0 (Yeni)', '0'))],
            'isitma': [str(isitma)]
        })

        tahmin = model.predict(input_data)[0]

        return round(tahmin, 2)
        
    except Exception as e:
        return f"Hata: {str(e)}"
