# Evaluador de Modelo ML - Centro de Ski

Evaluador simple para modelos de machine learning de predicción climática. Determina automáticamente si un modelo necesita reentrenamiento basado en métricas de rendimiento.

## 🌨️ Características

- **Funciona local y en AWS Lambda**
- **Soporte S3**: Carga modelos y datos desde S3 automáticamente
- **Métricas completas**: MAE, RMSE, MAPE
- **Decisión automática**: Compara métricas con umbrales para decidir reentrenamiento
- **Salida JSON**: Formato estándar para integración MLOps

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 💻 Uso Local

### Ejemplo básico:
```bash
python evaluate.py --model modelo.pkl --data datos.csv --target temperatura
```

### Ejemplo completo:
```bash
python evaluate.py \
  --model s3://bucket/modelo.pkl \
  --data s3://bucket/datos.csv \
  --target temperatura \
  --features "humedad,viento,presion" \
  --metric mae \
  --threshold 2.0 \
  --output metricas.json \
  --s3-output s3://bucket/metricas.json
```

## ☁️ AWS Lambda

### Evento de entrada:
```json
{
  "model_path": "s3://bucket/modelo.pkl",
  "data_path": "s3://bucket/datos.csv",
  "target_col": "temperatura",
  "feature_cols": ["humedad", "viento", "presion"],
  "primary_metric": "mae",
  "threshold": 2.0,
  "s3_output": "s3://bucket/metricas.json"
}
```

### Deploy Lambda:
```bash
# Crear package
pip install -r requirements.txt -t package/
cp evaluate.py package/
cd package && zip -r ../lambda.zip .

# Subir a Lambda con runtime python3.11
```

## 📊 Salida

```json
{
  "timestamp": "2025-09-07T21:10:00Z",
  "model_path": "modelo.pkl",
  "data_path": "datos.csv",
  "target_col": "temperatura",
  "feature_cols": ["humedad", "viento", "presion"],
  "metrics": {
    "mae": 1.82,
    "rmse": 2.41,
    "mape": 6.3,
    "n_samples": 720
  },
  "primary_metric": "mae",
  "threshold": 2.0,
  "should_retrain": false
}
```

## 🔧 Parámetros

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--model` | Ruta al modelo | `modelo.pkl` o `s3://bucket/modelo.pkl` |
| `--data` | Datos CSV | `datos.csv` o `s3://bucket/datos.csv` |
| `--target` | Columna objetivo | `temperatura` |
| `--features` | Features (opcional) | `"col1,col2,col3"` |
| `--metric` | Métrica principal | `mae`, `rmse`, `mape` |
| `--threshold` | Umbral reentrenamiento | `2.0` |
| `--output` | Archivo local | `metricas.json` |
| `--s3-output` | S3 para métricas | `s3://bucket/metricas.json` |

## 🚨 Exit Codes

- `0`: Modelo OK (no reentrenar)
- `1`: **REENTRENAR** (métrica > umbral)
- `2`: Error en evaluación

## 🧪 Generar Datos de Prueba

```python
# examples/generar_datos.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Datos sintéticos
np.random.seed(42)
data = pd.DataFrame({
    'humedad': np.random.uniform(40, 90, 200),
    'viento': np.random.uniform(5, 25, 200),
    'presion': np.random.uniform(990, 1020, 200),
})

data['temperatura'] = (
    -0.1 * data['humedad'] + 
    -0.2 * data['viento'] + 
    0.05 * data['presion'] + 
    np.random.normal(0, 1.5, 200)
)

# Entrenar modelo
X = data[['humedad', 'viento', 'presion']]
y = data['temperatura']
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Guardar
data.to_csv('datos.csv', index=False)
joblib.dump(model, 'modelo.pkl')
```

## � AWS Configuración

### IAM Policy mínima:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::tu-bucket/*"
    }
  ]
}
```

---

**MLOps simplificado para centro de ski** ⛷️
