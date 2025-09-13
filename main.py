import os
import sys
import time
import logging
import yaml
import pandas as pd
# Add models directory to Python path
models_path = os.path.join(os.path.dirname(__file__), 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)
from model import ClimatePredictor
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer
from src.data_monitor import start_monitoring
from src.model_manager import ModelManager

# Import API client
from api_client import OpenMeteoAPIClient

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'system.log')),
            logging.StreamHandler(sys.stdout)  # Agregar salida a consola
        ]
    )

def process_new_data(csv_path, config, model_manager):
    print(f"[PIPELINE] Procesando nuevo archivo: {csv_path}")
    logging.info(f"Procesando nuevo archivo de datos: {csv_path}")
    
    # Crear una instancia temporal de ClimatePredictor para cargar y procesar los datos
    temp_predictor = ClimatePredictor(target_variable=config['target_variable'])
    df = temp_predictor.load_data(csv_path)
    print(f"[PIPELINE] ✅ Archivo cargado exitosamente")
    print(f"[PIPELINE] 📊 Datos: {len(df)} filas, {len(df.columns)} columnas")
    print(f"[PIPELINE] 🎯 Variable objetivo: {config['target_variable']}")
    
    trainer = ModelTrainer('config/config.yaml')
    best_model_path = model_manager.get_best_model()
    
    if not best_model_path:
        # No hay modelo previo, entrenar y guardar uno base
        print("\n" + "="*60)
        print("[PIPELINE] 🚀 ENTRENAMIENTO INICIAL - No existe modelo previo")
        print("="*60)
        logging.info("No existe modelo previo. Entrenando modelo base...")
        
        model_path, metrics = trainer.train_and_save(df)
        model_manager.save_metrics(model_path, metrics)
        
        print(f"[PIPELINE] ✅ Modelo base entrenado exitosamente")
        print(f"[PIPELINE] 📁 Ruta: {model_path}")
        print(f"[PIPELINE] 📈 Métricas finales:")
        print(f"[PIPELINE]    • MAE:  {metrics['mae']:.4f}")
        print(f"[PIPELINE]    • RMSE: {metrics['rmse']:.4f}")
        print(f"[PIPELINE]    • R²:   {metrics['r2']:.4f}")
        print("="*60)
        logging.info(f"Modelo base entrenado y guardado: {model_path}")
        return
    
    # Evaluación del modelo existente
    print("\n" + "="*60)
    print("[PIPELINE] 🔍 EVALUACIÓN DE MODELO EXISTENTE")
    print("="*60)
    print(f"[PIPELINE] 📁 Modelo actual: {best_model_path}")
    
    evaluator = ModelEvaluator('config/config.yaml', best_model_path)
    metrics = evaluator.evaluate(df)
    
    print(f"[PIPELINE] 📊 Métricas actuales:")
    print(f"[PIPELINE]    • MAE:  {metrics['mae']:.4f}")
    print(f"[PIPELINE]    • RMSE: {metrics['rmse']:.4f}")
    print(f"[PIPELINE]    • R²:   {metrics['r2']:.4f}")
    print(f"[PIPELINE] 🎯 Umbral MAE para reentrenamiento: {config['training']['retrain_threshold_mae']}")
    
    if evaluator.needs_retraining(metrics):
        print(f"[PIPELINE] ⚠️  MAE actual ({metrics['mae']:.4f}) > Umbral ({config['training']['retrain_threshold_mae']})")
        print("[PIPELINE] 🔄 INICIANDO REENTRENAMIENTO...")
        print("="*60)
        logging.info("Performance degradada. Iniciando reentrenamiento...")
        
        model_path, new_metrics = trainer.train_and_save(df)
        model_manager.save_metrics(model_path, new_metrics)
        
        print(f"[PIPELINE] ✅ Reentrenamiento completado")
        print(f"[PIPELINE] 📁 Nuevo modelo: {model_path}")
        print(f"[PIPELINE] 📊 Nuevas métricas:")
        print(f"[PIPELINE]    • MAE:  {new_metrics['mae']:.4f}")
        print(f"[PIPELINE]    • RMSE: {new_metrics['rmse']:.4f}")
        print(f"[PIPELINE]    • R²:   {new_metrics['r2']:.4f}")
        
        # Comparar y decidir si mantener el nuevo modelo
        mae_improvement = metrics['mae'] - new_metrics['mae']
        print(f"\n[PIPELINE] 🔄 COMPARACIÓN DE MODELOS:")
        print(f"[PIPELINE]    • MAE anterior: {metrics['mae']:.4f}")
        print(f"[PIPELINE]    • MAE nuevo:    {new_metrics['mae']:.4f}")
        print(f"[PIPELINE]    • Mejora:       {mae_improvement:.4f}")
        
        if new_metrics['mae'] < metrics['mae']:
            print(f"[PIPELINE] ✅ MODELO MEJORADO - Manteniendo nuevo modelo")
            print(f"[PIPELINE] 📈 Mejora de {mae_improvement:.4f} en MAE")
            logging.info(f"Nuevo modelo mejorado guardado: {model_path}")
        else:
            rollback_path = model_manager.rollback()
            print(f"[PIPELINE] ❌ MODELO EMPEORADO - Realizando rollback")
            print(f"[PIPELINE] 📉 Empeoramiento de {abs(mae_improvement):.4f} en MAE")
            print(f"[PIPELINE] 🔙 Rollback a: {rollback_path}")
            logging.warning(f"El nuevo modelo es peor. Rollback a: {rollback_path}")
        print("="*60)
    else:
        print(f"[PIPELINE] ✅ MODELO EN BUEN ESTADO")
        print(f"[PIPELINE] 📊 MAE actual ({metrics['mae']:.4f}) ≤ Umbral ({config['training']['retrain_threshold_mae']})")
        print(f"[PIPELINE] 🎯 No se requiere reentrenamiento")
        print("="*60)
        logging.info("Performance dentro de los umbrales. No se requiere reentrenamiento.")

def fetch_api_data_to_monitoring():
    """
    Obtener datos de Open-Meteo API y guardarlos en la carpeta de monitoreo.
    """
    print("\n🌐 OBTENER DATOS DESDE OPEN-METEO API")
    print("=" * 40)
    
    # Probar conexión con API
    api_client = OpenMeteoAPIClient()
    if not api_client.test_connection():
        print("❌ No se puede conectar con Open-Meteo API")
        print("   Verifica tu conexión a internet")
        return False
    
    try:
        # Obtener datos de la API
        df = api_client.fetch_climate_data()
        
        # Guardar en carpeta de monitoreo para trigger automático
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/input/openmeteo_data_{timestamp}.csv"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Guardar datos
        df.to_csv(output_file, index=False)
        
        print(f"✅ Datos de Open-Meteo guardados en: {output_file}")
        print(f"📊 Registros: {len(df)}")
        print(f"📅 Rango: {df['datetime'].min()} a {df['datetime'].max()}")
        print(f"🌡️ Temperatura promedio: {df['temperature_2m'].mean():.1f}°C")
        print("🔍 El sistema de monitoreo procesará estos datos automáticamente...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al obtener datos de Open-Meteo API: {str(e)}")
        return False

def show_menu():
    """Mostrar menú de opciones."""
    print("\n" + "=" * 60)
    print("🤖 SISTEMA MLOps LOCAL - PREDICCIÓN CLIMÁTICA")
    print("=" * 60)
    print("Opciones disponibles:")
    print("1. 🔍 Monitorear carpeta de datos (automático)")
    print("2. 🌐 Obtener datos desde Open-Meteo API")
    print("3. 📊 Ejecutar demo completo con Open-Meteo API")
    print("4. ❌ Salir")
    print("=" * 60)

def run_api_demo():
    """Ejecutar demo completo con API."""
    print("\n🚀 EJECUTANDO DEMO CON API...")
    
    try:
        # Importar y ejecutar demo
        from demo_api_integration import ejecutar_demo_api
        ejecutar_demo_api()
    except ImportError:
        print("❌ Demo de API no disponible")
    except Exception as e:
        print(f"❌ Error en demo: {str(e)}")

def main():
    setup_logging()
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_manager = ModelManager()
    
    # Usar ruta absoluta para el directorio de monitoreo
    watch_dir = config['monitoring']['watch_directory']
    if not os.path.isabs(watch_dir):
        watch_dir = os.path.join(os.path.dirname(__file__), watch_dir)
    watch_dir = os.path.abspath(watch_dir)
    
    check_interval = config['monitoring']['check_interval']
    
    while True:
        show_menu()
        
        opcion = input("\nSelecciona una opción (1-4): ").strip()
        
        if opcion == "1":
            print(f"📁 Carpeta monitoreada: {watch_dir}")
            print(f"⏱️  Intervalo de chequeo: {check_interval} segundos")
            print(f"🎯 Variable objetivo: {config['target_variable']}")
            print(f"📊 Umbral MAE reentrenamiento: {config['training']['retrain_threshold_mae']}")
            print("=" * 60)
            
            def callback(csv_path):
                process_new_data(csv_path, config, model_manager)
            
            logging.info("Sistema MLOps iniciado. Monitoreando carpeta de datos...")
            print("🔍 Sistema iniciado. Monitoreando cambios en archivos...")
            print("Presiona Ctrl+C para detener...")
            
            try:
                start_monitoring(watch_dir, callback, check_interval)
            except KeyboardInterrupt:
                print("\n🛑 Monitoreo detenido por el usuario.")
                break
        
        elif opcion == "2":
            fetch_api_data_to_monitoring()
        
        elif opcion == "3":
            run_api_demo()
        
        elif opcion == "4":
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida. Intenta de nuevo.")

def main_legacy():
    """Función legacy para monitoreo directo (sin menú)."""

if __name__ == '__main__':
    main()
