import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def veri_temizle(df):
    
    df['fiyat'] = df['fiyat'].astype(str).str.replace(' TL', '').str.replace('.', '').astype(float)
     
    df['brut_m2'] = df['brut_m2'].astype(str).str.replace(' m2', '').astype(float)
    
    df['bina_yasi'] = df['bina_yasi'].astype(str).str.replace('0 (Yeni)', '0').astype(int)
    
    return df

print("Veri yükleniyor...")
df = pd.read_csv('sahibinden_ham_veri.csv')

df = veri_temizle(df)

X = df[['mahalle', 'brut_m2', 'oda_sayisi', 'bina_yasi', 'isitma']]
y = df['fiyat']

kategorik_ozellikler = ['mahalle', 'oda_sayisi', 'isitma']
sayisal_ozellikler = ['brut_m2', 'bina_yasi']

numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, sayisal_ozellikler),
        ('cat', categorical_transformer, kategorik_ozellikler)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

print("Model eğitiliyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model Başarısı (R2 Score): {score:.2f}")

joblib.dump(model, 'emlak_fiyat_tahmin_modeli.pkl')
print("Model 'emlak_fiyat_tahmin_modeli.pkl' olarak kaydedildi.")
