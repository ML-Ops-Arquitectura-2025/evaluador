#!/usr/bin/env python3
"""
Script para generar datos de prueba y modelo de ejemplo
Utiliza para testing del evaluador de modelos climáticos
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_climate_data(n_samples=2000, random_state=42):
    """Genera datos climáticos sintéticos para centro de ski"""
    
    np.random.seed(random_state)
    
    # Features climáticas
    data = pd.DataFrame({
        'humidity': np.random.uniform(30, 95, n_samples),           # Humedad (%)
        'wind_speed': np.random.uniform(0, 30, n_samples),         # Velocidad viento (km/h)
        'pressure': np.random.uniform(970, 1030, n_samples),       # Presión (hPa)
        'altitude': np.random.uniform(1500, 3500, n_samples),      # Altitud (m)
        'hour': np.random.randint(0, 24, n_samples),               # Hora del día
        'month': np.random.randint(1, 13, n_samples),              # Mes del año
        'cloud_cover': np.random.uniform(0, 100, n_samples)       # Cobertura nubosa (%)
    })
    
    # Crear variable objetivo: temperatura
    # Modelo físico simplificado con relaciones realistas
    temperature = (
        # Efecto altitud (temperatura baja con altitud)
        -0.008 * data['altitude'] +
        
        # Efecto humedad (mayor humedad, menor temperatura en montaña)
        -0.05 * data['humidity'] +
        
        # Efecto viento (viento fuerte enfría)
        -0.1 * data['wind_speed'] +
        
        # Efecto presión (alta presión = mejor tiempo)
        0.02 * data['pressure'] +
        
        # Ciclo diario (más frío en madrugada)
        -2 * np.cos(2 * np.pi * data['hour'] / 24) +
        
        # Ciclo estacional (más frío en invierno)
        -8 * np.cos(2 * np.pi * (data['month'] - 1) / 12) +
        
        # Efecto nubosidad
        -0.02 * data['cloud_cover'] +
        
        # Constante base
        25 +
        
        # Ruido aleatorio
        np.random.normal(0, 1.5, n_samples)
    )
    
    data['temperature'] = temperature
    
    return data

def train_model(data, target_col='temperature', test_size=0.2, random_state=42):
    """Entrena un modelo Random Forest con los datos"""
    
    feature_cols = [col for col in data.columns if col != target_col]
    
    X = data[feature_cols]
    y = data[target_col]
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Métricas básicas
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"R² Train: {train_score:.4f}")
    print(f"R² Test: {test_score:.4f}")
    
    return model, X_test, y_test, feature_cols

def main():
    """Función principal para generar datos y modelo de prueba"""
    
    print("🌨️  Generando datos climáticos sintéticos...")
    
    # Generar datos
    data = generate_climate_data(n_samples=2000)
    
    print(f"✅ Generados {len(data)} registros")
    print(f"📊 Columnas: {list(data.columns)}")
    print(f"🌡️  Temperatura promedio: {data['temperature'].mean():.2f}°C")
    print(f"📈 Rango temperatura: {data['temperature'].min():.2f}°C a {data['temperature'].max():.2f}°C")
    
    # Dividir en train y test
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    
    # Entrenar modelo
    print("\n🤖 Entrenando modelo Random Forest...")
    model, X_test, y_test, feature_cols = train_model(train_data)
    
    # Guardar archivos
    print("\n💾 Guardando archivos...")
    
    # Crear directorio examples si no existe
    os.makedirs('examples', exist_ok=True)
    
    # Guardar datasets
    train_data.to_csv('examples/train_data.csv', index=False)
    test_data.to_csv('examples/test_data.csv', index=False)
    
    # Guardar modelo
    joblib.dump(model, 'examples/model.pkl')
    
    # Guardar información del modelo
    model_info = {
        'feature_cols': feature_cols,
        'target_col': 'temperature',
        'n_samples_train': len(train_data),
        'n_samples_test': len(test_data),
        'model_type': 'RandomForestRegressor'
    }
    
    import json
    with open('examples/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Archivos guardados:")
    print("   - examples/train_data.csv (datos entrenamiento)")
    print("   - examples/test_data.csv (datos evaluación)")
    print("   - examples/model.pkl (modelo entrenado)")
    print("   - examples/model_info.json (información del modelo)")
    
    # Ejemplo de evaluación
    print("\n🧪 Ejemplo de evaluación:")
    print("python evaluate.py \\")
    print("  --model_path examples/model.pkl \\")
    print("  --data_path examples/test_data.csv \\")
    print("  --target_col temperature \\")
    print("  --threshold 2.0 \\")
    print("  --verbose")
    
    print("\n🎿 ¡Datos de prueba listos para el centro de ski!")

if __name__ == '__main__':
    main()
