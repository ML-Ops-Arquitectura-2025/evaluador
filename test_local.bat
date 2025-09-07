@echo off
REM Script de prueba local para Windows PowerShell
REM Genera datos de prueba y ejecuta evaluaciones de ejemplo

echo 🌨️  MLOps - Evaluador de Modelos Climáticos
echo =============================================

REM Verificar que Python esté instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Instalar Python 3.11+
    exit /b 1
)

echo ✅ Python encontrado
python --version

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias
echo 📥 Instalando dependencias...
pip install -r requirements.txt

REM Generar datos de prueba
echo 🎲 Generando datos de prueba...
python examples\generate_test_data.py

REM Ejecutar evaluaciones de ejemplo
echo.
echo 🧪 PRUEBA 1: Evaluación básica local
echo ======================================
python evaluate.py --model_path examples\model.pkl --data_path examples\test_data.csv --target_col temperature --threshold 2.0

echo.
echo 🧪 PRUEBA 2: Evaluación con todas las opciones
echo ==============================================
python evaluate.py --model_path examples\model.pkl --data_path examples\test_data.csv --target_col temperature --feature_cols "humidity,wind_speed,pressure,altitude,hour,month,cloud_cover" --primary_metric rmse --threshold 3.0 --metrics_local_path examples\metrics_test.json --verbose

echo.
echo 🧪 PRUEBA 3: Evaluación con umbral bajo (debería activar reentrenamiento)
echo ========================================================================
python evaluate.py --model_path examples\model.pkl --data_path examples\test_data.csv --target_col temperature --primary_metric mae --threshold 0.5 --verbose

echo.
echo ✅ Pruebas completadas!
echo 📋 Revisar archivos generados en examples/
echo 📊 Métricas guardadas en examples/metrics_test.json

pause
