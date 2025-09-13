# 🌡️ Sistema MLOps Local para Predicción Climática

Un sistema MLOps automatizado y completo en Python que monitorea, evalúa y reentrena automáticamente un modelo Random Forest de predicción climática. Funciona completamente en local sin necesidad de contenedores.

## 🎯 Características Principales

- ✅ **Monitoreo automático**: Detecta solo archivos nuevos (ignora existentes)
- ✅ **Evaluación inteligente**: Compara rendimiento contra baseline y umbrales
- ✅ **Reentrenamiento automático**: Se ejecuta cuando la performance baja
- ✅ **Versionado de modelos**: Guarda modelos con timestamp automáticamente
- ✅ **Logging completo**: Visibilidad total de todas las operaciones
- ✅ **Gestión de archivos**: Salida unificada en un solo directorio
- ✅ **Feature engineering avanzado**: Lag features, rolling windows, características temporales

## 📁 Estructura del Proyecto

```
evaluador2/
├── main.py                    # Orquestador principal del sistema MLOps
├── api_client.py             # 🌐 Cliente para Open-Meteo API
├── models/
│   └── model.py              # Modelo Random Forest con métodos MLOps y API
├── src/
│   ├── data_monitor.py       # Monitoreo de archivos con detección de completitud
│   ├── model_evaluator.py    # Evaluación de performance y decisiones de reentrenamiento
│   ├── model_trainer.py      # Entrenamiento y guardado de modelos
│   └── model_manager.py      # Gestión y versionado de modelos
├── data/
│   ├── input/               # Archivos CSV de entrada (monitoreo automático)
│   └── output/              # Salida unificada del modelo
├── config/
│   └── config.yaml          # Configuración del sistema
├── logs/                    # Logs del sistema (auto-generados)
├── requirements.txt         # Dependencias Python
├── API_INTEGRATION.md       # 📚 Documentación detallada de API
└── README.md               # Este archivo
```

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/ML-Ops-Arquitectura-2025/evaluador.git
cd evaluador
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `pandas`, `numpy`: Manipulación de datos
- `scikit-learn`: Machine Learning
- `watchdog`: Monitoreo de archivos
- `PyYAML`: Configuración
- `requests`: **Cliente HTTP para Open-Meteo API**

### 3. Ejecutar el sistema
```bash
python main.py
```

## ⚙️ Configuración

El archivo `config/config.yaml` permite configurar:

```yaml
target_variable: "temperature_2m"        # Variable a predecir
mae_threshold: 1.5                       # Umbral para reentrenamiento
check_interval: 300                      # Intervalo de monitoreo (segundos)
model_params:                           # Parámetros del Random Forest
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 2
```

## 🔄 Flujo de Trabajo MLOps

1. **📂 Detección**: El sistema monitorea `data/input/` para archivos CSV nuevos
2. **📊 Evaluación**: Compara el modelo actual contra baseline y umbrales
3. **🤖 Decisión**: Determina si es necesario reentrenar basado en métricas
4. **🔄 Reentrenamiento**: Entrena nuevo modelo si la performance ha bajado
5. **💾 Versionado**: Guarda modelos con timestamp automático
6. **📈 Comparación**: Valida que el nuevo modelo sea mejor
7. **📤 Salida**: Genera predicciones en `data/output/`

## 📊 Métricas y Evaluación

El sistema evalúa modelos usando:

- **MAE (Mean Absolute Error)**: Métrica principal de evaluación
- **RMSE (Root Mean Square Error)**: Error cuadrático medio
- **R² Score**: Coeficiente de determinación
- **Baseline Comparison**: Comparación contra modelo de persistencia

### Criterios de Reentrenamiento

- MAE supera el umbral configurado (default: 1.5)
- Degradación significativa respecto al baseline
- Validación manual mediante logging detallado

## 🖥️ Salida del Sistema

El sistema proporciona logging completo con emojis para facilitar el seguimiento:

```
[MLOps] 🚀 Iniciando sistema MLOps...
[Watcher] 🔍 Ignorando archivos existentes...
[Watcher] ⏳ Esperando SOLO archivos nuevos en: data/input
[Pipeline] 📁 Procesando nuevo archivo: data/input/climate_data.csv
[Pipeline] 🔍 EVALUACIÓN DE MODELO EXISTENTE
[Pipeline] ✅ Modelo actual cumple umbrales (MAE: 0.85 <= 1.5)
[Pipeline] 📊 Generando predicciones...
```

## 🎯 Características del Modelo

### Feature Engineering Automático

- **Lag Features**: Variables con retardo temporal (1-24 horas)
- **Rolling Windows**: Medias móviles (3, 6, 12, 24 horas)
- **Características Temporales**: Hora del día, día del año, estacionalidad
- **Variables Derivadas**: Wind chill, wet bulb temperature, índices calculados

### Variables Climáticas Soportadas

- Temperatura 2m, punto de rocío, humedad relativa
- Velocidad del viento, presión atmosférica
- Radiación solar, cobertura nubosa
- Precipitación, profundidad de nieve (para predicción de nevadas)

## 🌐 Integración con Open-Meteo API

El sistema incluye integración completa con la API de Open-Meteo para obtener datos climáticos reales en tiempo real.

### 🚀 Características de la API

- ✅ **Conexión directa**: directo a Open-Meteo
- ✅ **Datos reales**: Información meteorológica actualizada
- ✅ **Cobertura global**: Cualquier ubicación del mundo
- ✅ **Datos completos**: 7 días de pronóstico horario
- ✅ **Múltiples variables**: Temperatura, humedad, viento, presión, etc.
- ✅ **Integración automática**: Compatible con sistema MLOps existente

### 📊 Datos Disponibles

La API obtiene automáticamente:
- **temperature_2m**: Temperatura a 2 metros (°C)
- **dew_point_2m**: Punto de rocío (°C)
- **relative_humidity_2m**: Humedad relativa (%)
- **wind_speed_10m**: Velocidad del viento (km/h)
- **pressure_msl**: Presión atmosférica (hPa)
- **cloud_cover**: Cobertura nubosa (%)
- **shortwave_radiation**: Radiación solar (W/m²)

### 🔧 Uso de la API

#### **1. Sistema principal con menú**
```bash
python main.py
# Seleccionar opción 2: "Obtener datos desde Open-Meteo API"
# O opción 3: "Ejecutar análisis completo con Open-Meteo API"
```

#### **2. Uso programático**
```python
from models.model import ClimatePredictor

# Crear modelo
modelo = ClimatePredictor()

# Obtener datos de Berlin (default)
df = modelo.load_data_from_api()

# O especificar otra ubicación
df = modelo.load_data_from_api(
    latitude=-34.61,  # Buenos Aires
    longitude=-58.38,
    save_to_file="datos_bsas.csv"
)

# Entrenar modelo con datos reales
modelo.train(df)
predictions = modelo.predict(df.tail(24))
```

#### **3. Cliente API directo**
```python
from api_client import get_climate_data_from_api

# Obtener datos directamente
df = get_climate_data_from_api(
    latitude=40.4,    # Madrid
    longitude=-3.7
)
```

### 🌍 Ubicaciones Populares

```python
# Principales ciudades (ejemplos)
ciudades = {
    "Berlin": (52.52, 13.41),
    "Buenos Aires": (-34.61, -58.38),
    "Madrid": (40.4, -3.7),
    "Ciudad de México": (19.43, -99.13),
    "Londres": (51.51, -0.13),
    "Nueva York": (40.71, -74.01),
    "Tokio": (35.68, 139.69),
    "Sídney": (-33.87, 151.21)
}

# Usar cualquier ciudad
lat, lon = ciudades["Buenos Aires"]
df = modelo.load_data_from_api(latitude=lat, longitude=lon)
```

### 🔄 Integración con MLOps

La API se integra perfectamente con el sistema MLOps:

1. **Datos automáticos**: Opción 2 del menú obtiene datos y los guarda en `data/input/`
2. **Monitoreo automático**: El sistema detecta y procesa los nuevos datos API
3. **Reentrenamiento**: Si la performance baja, se reentrena con datos actualizados
4. **Versionado**: Modelos entrenados con datos API se versionan automáticamente

### 📁 Archivos de API

- **`api_client.py`**: Cliente principal para Open-Meteo
- **`API_INTEGRATION.md`**: Documentación detallada

### 🛠️ Troubleshooting API

#### **Error: No se puede conectar**
```bash
# Verificar conectividad
python api_client.py
```

#### **Campos sintéticos**
Si ves advertencias sobre "campos sintéticos", significa que algunos datos opcionales no están disponibles en Open-Meteo y se generan valores realistas automáticamente.

#### **Timeout**
Para ubicaciones remotas o conexiones lentas:
```python
df = modelo.load_data_from_api(timeout=60)  # 60 segundos
```

## 🧪 Testing y Simulación

Para probar el sistema, puedes:

1. **Generar datos de prueba**:
```python
python -c "from models.model import generate_sample_data; import pandas as pd; df = generate_sample_data(1000); df.to_csv('data/input/test_data.csv', index=False)"
```

2. **Simular degradación**: Modificar el umbral en `config.yaml` para forzar reentrenamiento

3. **Monitorear logs**: Revisar `logs/` para seguimiento detallado

## 🔧 Solución de Problemas

### Problemas Comunes

1. **Error de importación**:
   - Verificar que todas las dependencias estén instaladas
   - Ejecutar desde el directorio raíz del proyecto

2. **No detecta archivos**:
   - Asegurar que los archivos CSV tengan extensión `.csv`
   - Verificar permisos de escritura en `data/input/`

3. **Modelo no reentrena**:
   - Revisar umbrales en `config.yaml`
   - Verificar logs para entender decisiones del sistema

### Logs Detallados

Todos los eventos se registran en `logs/` con timestamps y categorías:
- `INFO`: Operaciones normales
- `WARNING`: Situaciones que requieren atención
- `ERROR`: Errores que requieren intervención

## 🤝 Contribuciones

Este proyecto es parte del curso de Ingeniería Informática - MLOps 2025.

## 📄 Licencia

Proyecto académico para demostración de conceptos MLOps.
