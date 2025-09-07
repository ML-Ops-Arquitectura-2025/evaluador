# Evaluador de Modelo de Predicción Climática

Este proyecto contiene un evaluador robusto para modelos de machine learning de predicción climática en centros de ski, diseñado para funcionar tanto localmente como en AWS Lambda.

## 🌨️ Características Principales

- **Compatibilidad dual**: Ejecución local (CLI) y AWS Lambda
- **Soporte S3**: Carga/descarga automática de modelos y datasets desde S3
- **Métricas completas**: MAE, RMSE, MAPE con decisión automática de reentrenamiento
- **Formato estándar**: Salida JSON estructurada para integración MLOps
- **Robusto**: Validación de datos, manejo de errores, limpieza automática

## 📁 Estructura del Proyecto

```
evaluador/
├── evaluate.py           # Script principal del evaluador
├── requirements.txt      # Dependencias Python
├── README.md            # Documentación
├── examples/            # Ejemplos de uso
│   ├── test_data.csv    # Datos de ejemplo
│   └── lambda_event.json
└── aws/                 # Configuración AWS
    ├── iam_policy.json  # Política IAM mínima
    └── lambda_config.json
```

## 🚀 Instalación

### Local

```bash
# Clonar o descargar el proyecto
cd evaluador

# Crear entorno virtual (recomendado)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### AWS Lambda

#### Opción 1: Deployment Package
```bash
# Crear package para Lambda
pip install -r requirements.txt -t package/
cp evaluate.py package/
cd package && zip -r ../lambda_function.zip .
```

#### Opción 2: Container Image (Recomendado)
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

COPY evaluate.py ${LAMBDA_TASK_ROOT}

CMD ["evaluate.lambda_handler"]
```

## 💻 Uso Local

### Ejemplo Básico
```bash
python evaluate.py \
  --model_path model.pkl \
  --data_path data.csv \
  --target_col temperature
```

### Ejemplo Completo con S3
```bash
python evaluate.py \
  --model_path s3://mi-bucket/modelos/modelo_clima_v1.pkl \
  --data_path s3://mi-bucket/datasets/datos_recientes.csv \
  --target_col temperature \
  --feature_cols "humidity,wind_speed,pressure,altitude" \
  --primary_metric rmse \
  --threshold 2.5 \
  --metrics_local_path ./resultados/metrics.json \
  --metrics_s3_uri s3://mi-bucket/metricas/metrics_2025-09-07.json \
  --verbose
```

### Parámetros CLI

| Parámetro | Requerido | Descripción | Ejemplo |
|-----------|-----------|-------------|---------|
| `--model_path` | ✅ | Ruta al modelo (.pkl) | `model.pkl` o `s3://bucket/model.pkl` |
| `--data_path` | ✅ | Ruta a datos CSV | `data.csv` o `s3://bucket/data.csv` |
| `--target_col` | ✅ | Columna objetivo | `temperature` |
| `--feature_cols` | ❌ | Features (opcional) | `"col1,col2,col3"` |
| `--primary_metric` | ❌ | Métrica principal | `mae`, `rmse`, `mape` (default: `mae`) |
| `--threshold` | ❌ | Umbral reentrenamiento | `2.0` (default) |
| `--metrics_local_path` | ❌ | Ruta local métricas | `metrics.json` (default) |
| `--metrics_s3_uri` | ❌ | S3 para métricas | `s3://bucket/metrics.json` |
| `--verbose` | ❌ | Logs detallados | Flag |

## ☁️ Uso en AWS Lambda

### Evento de Entrada
```json
{
  "model_path": "s3://mi-bucket/modelos/modelo_clima.pkl",
  "data_path": "s3://mi-bucket/datasets/evaluacion_diaria.csv",
  "target_col": "temperature",
  "feature_cols": ["humidity", "wind_speed", "pressure", "altitude"],
  "primary_metric": "mae",
  "threshold": 2.0,
  "metrics_local_path": "/tmp/metrics.json",
  "metrics_s3_uri": "s3://mi-bucket/metricas/metrics_2025-09-07.json"
}
```

### Respuesta Lambda
```json
{
  "statusCode": 200,
  "body": {
    "timestamp_utc": "2025-09-07T21:10:00Z",
    "model_path": "s3://mi-bucket/modelos/modelo_clima.pkl",
    "data_path": "s3://mi-bucket/datasets/evaluacion_diaria.csv",
    "target_col": "temperature",
    "feature_cols": ["humidity", "wind_speed", "pressure", "altitude"],
    "metrics": {
      "mae": 1.82,
      "rmse": 2.41,
      "mape": 6.3,
      "n_samples": 720
    },
    "primary_metric": "mae",
    "threshold": 2.0,
    "should_retrain": false,
    "tag": "eval_2025-09-07"
  }
}
```

## 🔧 Configuración AWS

### IAM Policy Mínima
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::mi-bucket/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

### Variables de Entorno Lambda
```bash
# Configuración opcional
PYTHONPATH=/var/runtime
AWS_DEFAULT_REGION=us-east-1
```

## 🔄 Integración MLOps

### GitHub Actions
```yaml
name: Model Evaluation
on:
  schedule:
    - cron: '0 6 * * *'  # Diario a las 6 AM

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run evaluation
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python evaluate.py \
            --model_path s3://bucket/model.pkl \
            --data_path s3://bucket/daily_data.csv \
            --target_col temperature \
            --threshold 2.0 \
            --metrics_s3_uri s3://bucket/metrics/$(date +%Y-%m-%d).json
      
      - name: Check if retraining needed
        run: |
          if [ $? -eq 1 ]; then
            echo "Model needs retraining!"
            # Trigger retraining pipeline
          fi
```

### CloudWatch Events (EventBridge)
```json
{
  "Rules": [
    {
      "Name": "DailyModelEvaluation",
      "ScheduleExpression": "cron(0 6 * * ? *)",
      "State": "ENABLED",
      "Targets": [
        {
          "Id": "ModelEvaluatorLambda",
          "Arn": "arn:aws:lambda:region:account:function:model-evaluator",
          "Input": "{\"model_path\": \"s3://bucket/model.pkl\", \"data_path\": \"s3://bucket/daily_data.csv\", \"target_col\": \"temperature\"}"
        }
      ]
    }
  ]
}
```

### S3 Event Trigger
```json
{
  "Rules": [
    {
      "Name": "NewDataEvaluation",
      "EventPattern": {
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
          "bucket": {"name": ["mi-bucket"]},
          "object": {"key": [{"prefix": "datasets/new/"}]}
        }
      },
      "Targets": [
        {
          "Id": "TriggerEvaluation",
          "Arn": "arn:aws:lambda:region:account:function:model-evaluator"
        }
      ]
    }
  ]
}
```

## 📊 Formato de Salida

### Métricas JSON
```json
{
  "timestamp_utc": "2025-09-07T21:10:00Z",
  "model_path": "s3://mi-bucket/modelos/model.pkl",
  "data_path": "s3://mi-bucket/datasets/val_reciente.csv",
  "target_col": "temperature",
  "feature_cols": ["humidity", "wind_speed", "pressure"],
  "metrics": {
    "mae": 1.82,
    "rmse": 2.41,
    "mape": 6.3,
    "n_samples": 720
  },
  "primary_metric": "mae",
  "threshold": 2.0,
  "should_retrain": false,
  "tag": "eval_2025-09-07"
}
```

### Exit Codes (CLI)
- `0`: Evaluación exitosa, no reentrenar
- `1`: Evaluación exitosa, **REENTRENAR MODELO**
- `2`: Error en evaluación

## 🧪 Testing Local

### Crear Datos de Prueba
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generar datos sintéticos
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'humidity': np.random.uniform(30, 90, n_samples),
    'wind_speed': np.random.uniform(0, 25, n_samples),
    'pressure': np.random.uniform(980, 1020, n_samples),
    'altitude': np.random.uniform(1000, 3000, n_samples)
})

# Target con ruido
data['temperature'] = (
    -0.01 * data['altitude'] + 
    0.1 * data['humidity'] - 
    0.2 * data['wind_speed'] + 
    0.05 * data['pressure'] + 
    np.random.normal(0, 2, n_samples)
)

# Guardar datos
data.to_csv('test_data.csv', index=False)

# Entrenar modelo simple
X = data[['humidity', 'wind_speed', 'pressure', 'altitude']]
y = data['temperature']

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)
joblib.dump(model, 'test_model.pkl')

print("Datos y modelo de prueba creados!")
```

### Ejecutar Prueba
```bash
python evaluate.py \
  --model_path test_model.pkl \
  --data_path test_data.csv \
  --target_col temperature \
  --threshold 1.5 \
  --verbose
```

## 📈 Monitoreo y Alertas

### CloudWatch Metrics
El evaluador genera logs que pueden monitorearse:
- Métricas de rendimiento (MAE, RMSE, MAPE)
- Flags de reentrenamiento
- Errores y excepciones
- Tiempo de ejecución

### SNS Notifications
```python
# Ejemplo de notificación cuando should_retrain=true
import boto3

def send_retrain_alert(metrics_data):
    sns = boto3.client('sns')
    
    message = f"""
    🚨 REENTRENAMIENTO REQUERIDO
    
    Modelo: {metrics_data['model_path']}
    Métrica: {metrics_data['primary_metric']} = {metrics_data['metrics'][metrics_data['primary_metric']]}
    Umbral: {metrics_data['threshold']}
    
    Revisar pipeline de reentrenamiento.
    """
    
    sns.publish(
        TopicArn='arn:aws:sns:region:account:model-alerts',
        Message=message,
        Subject='Modelo requiere reentrenamiento'
    )
```

## 🔍 Troubleshooting

### Errores Comunes

1. **Credenciales AWS**: Configurar AWS CLI o variables de entorno
2. **Memoria Lambda**: Aumentar memoria si datasets son grandes
3. **Timeout Lambda**: Ajustar timeout para modelos complejos
4. **Formato datos**: Verificar encoding CSV (UTF-8)

### Logs y Debugging
```bash
# Modo verbose local
python evaluate.py --verbose ...

# CloudWatch Logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/model-evaluator \
  --start-time $(date -d '1 hour ago' +%s)000
```

## 🚀 Próximos Pasos

1. **Métricas personalizadas**: Agregar métricas específicas del dominio
2. **A/B Testing**: Comparar múltiples modelos
3. **Data Drift**: Detectar cambios en distribución de datos
4. **Alertas avanzadas**: Integración con Slack/Teams
5. **Dashboard**: Visualización de métricas en tiempo real

---

**Desarrollado para MLOps en Centro de Ski** 🎿  
*Fecha: Septiembre 2025*
