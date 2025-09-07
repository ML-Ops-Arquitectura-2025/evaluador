@echo off
REM Script de prueba simple para el evaluador

echo 🌨️ Evaluador de Modelos ML - Centro de Ski
echo ==========================================

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado
    exit /b 1
)

echo ✅ Python encontrado

REM Instalar dependencias
echo � Instalando dependencias...
pip install -r requirements.txt

REM Generar datos de prueba
echo 🎲 Generando datos de prueba...
python -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

np.random.seed(42)
data = pd.DataFrame({
    'humedad': np.random.uniform(40, 90, 100),
    'viento': np.random.uniform(5, 25, 100),
    'presion': np.random.uniform(990, 1020, 100),
})

data['temperatura'] = -0.1 * data['humedad'] - 0.2 * data['viento'] + 0.05 * data['presion'] + np.random.normal(0, 1.5, 100)

X = data[['humedad', 'viento', 'presion']]
y = data['temperatura']
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

data.to_csv('datos_prueba.csv', index=False)
joblib.dump(model, 'modelo_prueba.pkl')
print('✅ Datos generados')
"

REM Probar evaluador
echo 🧪 Probando evaluador...
python evaluate.py --model modelo_prueba.pkl --data datos_prueba.csv --target temperatura --threshold 1.5

echo ✅ Prueba completada!
pause
