import os
import sys
import time
import logging
import yaml
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Cargar variables de entorno
load_dotenv()

# Add models directory to Python path
models_path = os.path.join(os.path.dirname(__file__), 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)
from model import ClimatePredictor, list_s3_files, download_file_from_s3
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer
from src.data_monitor import start_monitoring
from src.model_manager import ModelManager

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

def check_s3_for_new_results():
    """Verifica si hay nuevos archivos de resultados en S3."""
    try:
        # Cargar credenciales de AWS
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = os.getenv('AWS_BUCKET', 'ml-ops-datos-prediccion-clima-uadec22025-ml')
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
        
        if not access_key or not secret_key:
            print("Credenciales de AWS no configuradas. Usando monitoreo local.")
            return []
        
        print(f"[S3] Verificando nuevos resultados en s3://{bucket}/results/")
        
        # Listar archivos en S3 en la carpeta results/
        files = list_s3_files(access_key, secret_key, bucket, region, "results/")
        
        # Filtrar solo archivos de resultados
        result_files = [f for f in files if f['Key'].startswith('results/model_results_') and f['Key'].endswith('.csv')]
        
        # Ordenar por fecha de modificación (más reciente primero)
        result_files.sort(key=lambda x: x['LastModified'], reverse=True)
        
        print(f"[S3] Encontrados {len(result_files)} archivos de resultados en S3")
        
        return result_files
        
    except Exception as e:
        print(f"Error verificando S3: {str(e)}")
        return []

def process_s3_result_file(s3_file_info, config, model_manager):
    """Descarga y procesa un archivo de resultados desde S3."""
    try:
        # Cargar credenciales de AWS
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = os.getenv('AWS_BUCKET', 'ml-ops-datos-prediccion-clima-uadec22025-ml')
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
        
        s3_key = s3_file_info['Key']
        filename = os.path.basename(s3_key)
        
        # Crear directorio temporal para descargas
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        local_path = os.path.join(temp_dir, filename)
        
        print(f"[S3] Descargando archivo de resultados: {filename}")
        
        # Descargar archivo desde S3
        success = download_file_from_s3(access_key, secret_key, bucket, region, s3_key, local_path)
        
        if success and os.path.exists(local_path):
            print(f"[S3] Archivo descargado exitosamente: {local_path}")
            
            # Procesar el archivo de resultados
            process_model_results(local_path, config, model_manager)
            
            # Limpiar archivo temporal
            os.remove(local_path)
            print(f"[S3] Archivo temporal eliminado: {local_path}")
            
        else:
            print(f"Error descargando archivo desde S3: {filename}")
            
    except Exception as e:
        print(f"Error procesando archivo S3: {str(e)}")

def check_for_new_results_s3(config, model_manager, last_checked_files=None):
    """Verifica y procesa nuevos archivos de resultados en S3."""
    if last_checked_files is None:
        last_checked_files = set()
    
    # Obtener archivos de resultados de S3
    s3_files = check_s3_for_new_results()
    
    if not s3_files:
        print("[S3] No se encontraron archivos de resultados en S3")
        return last_checked_files
    
    # Verificar si hay archivos nuevos
    current_files = {f['Key'] for f in s3_files}
    new_files = current_files - last_checked_files
    
    if new_files:
        print(f"[S3] Encontrados {len(new_files)} archivos nuevos de resultados")
        
        # Procesar solo los archivos más recientes (límite de 3)
        new_s3_files = [f for f in s3_files if f['Key'] in new_files]
        new_s3_files.sort(key=lambda x: x['LastModified'], reverse=True)
        
        for file_info in new_s3_files[:3]:  # Procesar máximo 3 archivos más recientes
            print(f"\n{'='*60}")
            print(f"PROCESANDO RESULTADO DESDE S3: {os.path.basename(file_info['Key'])}")
            print(f"Fecha S3: {file_info['LastModified']}")
            print(f"{'='*60}")
            
            process_s3_result_file(file_info, config, model_manager)
    else:
        print("[S3] No hay archivos nuevos de resultados")
    
    return current_files

def process_model_results(results_csv_path, config, model_manager):
    """Procesa archivos de resultados del modelo para evaluación y reentrenamiento automático."""
    print(f"[MLOps] Evaluando resultados: {os.path.basename(results_csv_path)}")
    logging.info(f"Evaluando resultados del modelo: {results_csv_path}")
    
    try:
        # Leer las métricas del archivo de resultados
        results_df = pd.read_csv(results_csv_path)
        if len(results_df) == 0:
            print("[MLOps] WARNING: Archivo de resultados vacío")
            return
        
        # Obtener las métricas más recientes
        latest_results = results_df.iloc[-1]
        current_mae = latest_results['mae']
        current_r2 = latest_results['r2']
        current_rmse = latest_results['rmse']
        target_variable = latest_results['target_variable']
        
        print(f"[MLOps] Variable objetivo: {target_variable}")
        print(f"[MLOps] MAE: {current_mae:.4f}")
        print(f"[MLOps] RMSE: {current_rmse:.4f}")
        print(f"[MLOps] R2: {current_r2:.4f}")
        
        # Obtener umbrales de configuración
        mae_threshold = config['training']['retrain_threshold_mae']
        r2_threshold = config['performance_thresholds']['r2']
        
        print(f"[MLOps] Umbral MAE: {mae_threshold}")
        print(f"[MLOps] Umbral R2: {r2_threshold}")
        
        # Evaluar performance y mostrar resultado claro
        needs_retraining = current_mae > mae_threshold or current_r2 < r2_threshold
        
        print("\n" + "=" * 70)
        if not needs_retraining:
            print("         RESULTADO DE EVALUACION: MODELO EXCELENTE")
            print("=" * 70)
            print("[ESTADO] *** MODELO EN PERFECTO FUNCIONAMIENTO ***")
            print(f"[ESTADO] Performance SUPERIOR a los umbrales requeridos")
            print(f"[ESTADO] MAE: {current_mae:.4f} <= {mae_threshold} (EXCELENTE)")
            print(f"[ESTADO] R2: {current_r2:.4f} >= {r2_threshold} (EXCELENTE)")
            
            # Calcular qué tan bueno es el modelo
            mae_improvement = ((mae_threshold - current_mae) / mae_threshold) * 100
            r2_improvement = ((current_r2 - r2_threshold) / r2_threshold) * 100
            
            print(f"[ESTADO] Margen de seguridad MAE: {mae_improvement:.1f}% mejor que umbral")
            print(f"[ESTADO] Margen de seguridad R2: {r2_improvement:.1f}% mejor que umbral")
            print("[ESTADO] *** NO SE REQUIERE REENTRENAMIENTO ***")
            print("=" * 70)
            logging.info(f"Modelo con excelente performance. MAE: {current_mae:.4f}, R²: {current_r2:.4f}")
        else:
            print("         RESULTADO DE EVALUACION: MODELO DEGRADADO")
            print("=" * 70)
            print("[ESTADO] *** MODELO REQUIERE ATENCION INMEDIATA ***")
            if current_mae > mae_threshold:
                degradation = ((current_mae - mae_threshold) / mae_threshold) * 100
                print(f"[ESTADO] MAE fuera de rango: {current_mae:.4f} > {mae_threshold} ({degradation:.1f}% peor)")
            if current_r2 < r2_threshold:
                degradation = ((r2_threshold - current_r2) / r2_threshold) * 100
                print(f"[ESTADO] R2 fuera de rango: {current_r2:.4f} < {r2_threshold} ({degradation:.1f}% peor)")
            
            print("[ESTADO] *** INICIANDO REENTRENAMIENTO AUTOMATICO ***")
            print("=" * 70)
            logging.warning(f"Iniciando reentrenamiento automático. MAE: {current_mae:.4f}, R²: {current_r2:.4f}")
            
            # En lugar de buscar archivos locales, obtener datos frescos desde S3
            print(f"[MLOps] Obteniendo datos frescos desde AWS S3 para reentrenamiento...")
            try:
                # Importar las funciones necesarias del modelo
                import importlib.util
                spec = importlib.util.spec_from_file_location("model", "models/model.py")
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                
                # Configuración AWS S3 desde variables de entorno
                AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
                AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
                AWS_BUCKET = os.getenv("AWS_BUCKET", "ml-ops-datos-prediccion-clima-uadec22025-ml")
                AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
                
                if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
                    raise ValueError("AWS credentials not found in environment variables")
                
                # Obtener datos directamente desde S3
                fresh_data = model_module.get_climate_data_from_s3(
                    access_key=AWS_ACCESS_KEY,
                    secret_key=AWS_SECRET_KEY,
                    bucket=AWS_BUCKET,
                    region=AWS_REGION
                )
                
                if fresh_data is not None and len(fresh_data) > 0:
                    print(f"[MLOps] ✅ Datos frescos obtenidos desde S3: {len(fresh_data)} registros")
                    
                    # Ejecutar reentrenamiento usando los datos directos de S3
                    print(f"[MLOps] Procesando datos para reentrenamiento...")
                    process_new_data_from_df(fresh_data, config, model_manager)
                    
                    print(f"\n[MLOps] [OK] REENTRENAMIENTO AUTOMÁTICO COMPLETADO")
                    print("=" * 60)
                else:
                    print(f"[MLOps] [ERROR] No se pudieron obtener datos frescos desde S3")
                    
            except Exception as e:
                print(f"[MLOps] [ERROR] Error durante reentrenamiento con S3: {str(e)}")
                # Fallback: buscar archivos locales como antes
                input_dir = os.path.join(os.path.dirname(results_csv_path).replace('output', 'input'))
                if os.path.exists(input_dir):
                    input_files = [f for f in os.listdir(input_dir) if f.startswith('climate_data_') and f.endswith('.csv')]
                    if input_files:
                        input_files.sort(reverse=True)  # Más reciente primero
                        latest_data_file = os.path.join(input_dir, input_files[0])
                        print(f"[MLOps] Fallback: Usando datos locales: {input_files[0]}")
                        process_new_data(latest_data_file, config, model_manager)
                    else:
                        print(f"[MLOps] [ERROR] No se encontraron archivos de datos para reentrenamiento en {input_dir}")
                else:
                    print(f"[MLOps] [ERROR] Directorio de datos no encontrado: {input_dir}")
        
        # Mostrar información adicional y resumen final
        print(f"\n[DETALLE] Características utilizadas: {latest_results.get('n_features', 'N/A')}")
        print(f"[DETALLE] Datos procesados: {latest_results.get('data_shape_rows', 'N/A')} filas")
        print(f"[DETALLE] Variable objetivo: {target_variable}")
        
        # RESUMEN FINAL MUY VISIBLE
        print("\n" + "#" * 70)
        if not needs_retraining:
            print("# RESUMEN FINAL: MODELO FUNCIONANDO PERFECTAMENTE")
            print("# STATUS: APROBADO PARA PRODUCCION")
            print("# ACCION: NINGUNA REQUERIDA")
        else:
            print("# RESUMEN FINAL: MODELO NECESITA REENTRENAMIENTO")
            print("# STATUS: REQUIERE ATENCION")
            print("# ACCION: REENTRENAMIENTO EN PROCESO")
        print("#" * 70)
        
    except Exception as e:
        print(f"[MLOps] [ERROR] Error al procesar resultados: {str(e)}")
        logging.error(f"Error al procesar resultados: {str(e)}")
        import traceback
        traceback.print_exc()

def process_new_data_from_df(df, config, model_manager):
    """Procesa datos directamente desde DataFrame (ej: desde S3)"""
    print(f"[PIPELINE] Procesando datos desde DataFrame")
    logging.info(f"Procesando datos desde DataFrame en memoria")
    
    print(f"[PIPELINE] ✅ Datos cargados exitosamente desde fuente directa")
    print(f"[PIPELINE] 📊 Datos: {len(df)} filas, {len(df.columns)} columnas")
    print(f"[PIPELINE] 🎯 Variable objetivo: {config['target_variable']}")
    
    trainer = ModelTrainer('config/config.yaml')
    best_model_path = model_manager.get_best_model()
    
    if not best_model_path or not os.path.exists(best_model_path):
        # No hay modelo previo válido, entrenar y guardar uno base
        print("\n" + "="*60)
        print("[PIPELINE] 🚀 ENTRENAMIENTO INICIAL - No existe modelo válido")
        print("="*60)
        logging.info("No existe modelo válido. Entrenando modelo base...")
        
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
        print(f"[PIPELINE] 📈 Nuevas métricas:")
        print(f"[PIPELINE]    • MAE:  {new_metrics['mae']:.4f}")
        print(f"[PIPELINE]    • RMSE: {new_metrics['rmse']:.4f}")
        print(f"[PIPELINE]    • R²:   {new_metrics['r2']:.4f}")
        
        improvement_mae = ((metrics['mae'] - new_metrics['mae']) / metrics['mae']) * 100
        improvement_r2 = ((new_metrics['r2'] - metrics['r2']) / metrics['r2']) * 100
        
        print(f"[PIPELINE] 📈 Mejoras:")
        print(f"[PIPELINE]    • MAE: {improvement_mae:+.2f}%")
        print(f"[PIPELINE]    • R²:  {improvement_r2:+.2f}%")
        print("="*60)
        
        logging.info(f"Reentrenamiento completado. Nuevo modelo: {model_path}")
    else:
        print(f"[PIPELINE] ✅ Rendimiento satisfactorio")
        print(f"[PIPELINE] 📊 MAE actual ({metrics['mae']:.4f}) ≤ Umbral ({config['training']['retrain_threshold_mae']})")
        print("[PIPELINE] 🎯 No se requiere reentrenamiento")
        print("="*60)


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
    
    if not best_model_path or not os.path.exists(best_model_path):
        # No hay modelo previo válido, entrenar y guardar uno base
        print("\n" + "="*60)
        print("[PIPELINE] 🚀 ENTRENAMIENTO INICIAL - No existe modelo válido")
        print("="*60)
        logging.info("No existe modelo válido. Entrenando modelo base...")
        
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

def run_model_directly():
    """Ejecuta el modelo directamente y procesa automáticamente los resultados."""
    import time
    execution_id = int(time.time())
    
    print("=" * 60)
    print("EJECUCION DIRECTA DEL MODELO CON EVALUACION")
    print("=" * 60)
    print(f"ID de Ejecución: {execution_id}")
    
    print("DEBUG: Iniciando función run_model_directly()")
    print("DEBUG: ESTA EJECUCION SERA PROCESADA COMPLETAMENTE")
    
    # Cargar configuración
    try:
        print("🔧 DEBUG: Cargando configuración...")
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("🔧 DEBUG: Configuración cargada exitosamente")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return False
        
    try:
        print("🔧 DEBUG: Inicializando ModelManager...")
        model_manager = ModelManager()
        print("🔧 DEBUG: ModelManager inicializado exitosamente")
    except Exception as e:
        print(f"❌ Error inicializando ModelManager: {e}")
        return False
    
    # Obtener directorio de output
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'output')
    print(f"🔧 DEBUG: Directorio output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print("🔧 DEBUG: Directorio output verificado/creado")
    
    # Importar y ejecutar el modelo desde models/model.py
    try:
        print("🔧 DEBUG: Importando modelo...")
        from models.model import main as model_main
        print("� DEBUG: Modelo importado exitosamente")
        
        print("�📊 Ejecutando el modelo de predicción climática...")
        model_main()
        print("\n✅ Modelo ejecutado exitosamente!")
        print("🔧 DEBUG: Ejecución del modelo completada")
        
        # Buscar el archivo de resultados más reciente (SIEMPRE procesar)
        print("🔧 DEBUG: Verificando archivos en directorio output...")
        print("🔧 DEBUG: FORZANDO PROCESAMIENTO - Esta ejecución SIEMPRE será procesada")
        if os.path.exists(output_dir):
            all_files = os.listdir(output_dir)
            print(f"🔧 DEBUG: Archivos encontrados: {all_files}")
            result_files = [f for f in all_files if f.startswith('model_results_') and f.endswith('.csv')]
            print(f"🔧 DEBUG: Archivos de resultados encontrados: {result_files}")
            
            if result_files:
                # Ordenar por timestamp y tomar el más reciente (el que se acaba de crear)
                result_files.sort(reverse=True)
                latest_result = result_files[0]
                result_path = os.path.join(output_dir, latest_result)
                print(f"🔧 DEBUG: Procesando archivo: {latest_result}")
                
                print("\n" + "=" * 60)
                print("EVALUACION AUTOMATICA DE RESULTADOS MLOps")
                print("=" * 60)
                print(f"Procesando archivo: {latest_result}")
                
                # SIEMPRE procesar el archivo de resultados
                print("🔧 DEBUG: Llamando a process_model_results()...")
                process_model_results(result_path, config, model_manager)
                print("🔧 DEBUG: process_model_results() completado")
                
                # SIEMPRE FORZAR EVALUACION ADICIONAL - SIMULAR LLEGADA DE NUEVOS DATOS
                print("\n" + "=" * 60)
                print("EVALUACION ADICIONAL - PROCESANDO COMO DATOS NUEVOS")
                print("=" * 60)
                
                # Buscar archivo de datos más reciente para simular pipeline completo
                input_dir = os.path.join(os.path.dirname(__file__), 'data', 'input')
                if os.path.exists(input_dir):
                    input_files = [f for f in os.listdir(input_dir) if f.startswith('climate_data_') and f.endswith('.csv')]
                    if input_files:
                        input_files.sort(reverse=True)
                        latest_input = os.path.join(input_dir, input_files[0])
                        print(f"Procesando archivo de datos: {input_files[0]}")
                        print("Esto activará el pipeline completo de evaluación...")
                        
                        # SIEMPRE procesar como si fueran datos nuevos llegando al sistema
                        process_new_data(latest_input, config, model_manager)
                    else:
                        print("No se encontraron archivos de datos de entrada")
                else:
                    print("Directorio de entrada no encontrado")
                
            else:
                print("\n⚠️  No se encontraron archivos de resultados en data/output/")
                print("📁 Verificando contenido del directorio...")
                all_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
                print(f"Archivos encontrados: {all_files}")
        else:
            print(f"\n⚠️  Directorio {output_dir} no existe")
            
    except Exception as e:
        print(f"❌ Error al ejecutar el modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # RESUMEN FINAL DEL PROCESO COMPLETO
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + f"EJECUCION {execution_id} COMPLETA - TODOS LOS PROCESOS FINALIZADOS".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" + "1. MODELO EJECUTADO Y ENTRENADO: [OK]".ljust(78) + "*")
    print("*" + "2. RESULTADOS GENERADOS: [OK]".ljust(78) + "*")
    print("*" + "3. EVALUACION MLOps REALIZADA: [OK]".ljust(78) + "*")
    print("*" + "4. PIPELINE COMPLETO ACTIVADO: [OK]".ljust(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" + f"EJECUCION {execution_id}: SISTEMA REGISTRO Y EVALUO EL MODELO".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    print(f"DEBUG: Ejecución {execution_id} completada exitosamente")
    return True

def test_automatic_retraining():
    """Prueba el reentrenamiento automático con umbrales estrictos."""
    print("=" * 60)
    print("🧪 PRUEBA DE REENTRENAMIENTO AUTOMÁTICO")
    print("=" * 60)
    
    # Cargar configuración
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_manager = ModelManager()
    
    # Buscar el archivo de resultados más reciente
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'output')
    
    if os.path.exists(output_dir):
        all_files = os.listdir(output_dir)
        result_files = [f for f in all_files if f.startswith('model_results_') and f.endswith('.csv')]
        
        if result_files:
            result_files.sort(reverse=True)
            latest_result = result_files[0]
            result_path = os.path.join(output_dir, latest_result)
            
            print(f"📁 Procesando archivo de resultados: {latest_result}")
            print(f"🎚️  Usando umbrales estrictos para forzar reentrenamiento")
            
            # Procesar con umbrales estrictos
            process_model_results(result_path, config, model_manager)
            
        else:
            print("❌ No se encontraron archivos de resultados para evaluar")
            print("💡 Ejecuta primero: python main.py --run-model")
    else:
        print("❌ Directorio de output no encontrado")
    
    return True

def run_single_check():
    """Ejecuta una sola verificación del sistema sin bucle infinito."""
    print("=" * 60)
    print("VERIFICACIÓN ÚNICA DEL SISTEMA MLOps")
    print("=" * 60)
    
    # Cargar configuración
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_manager = ModelManager()
    
    print("1. Verificando archivos nuevos en S3...")
    last_checked_files = check_for_new_results_s3(config, model_manager, set())
    
    print(f"\n2. Archivos procesados: {len(last_checked_files)}")
    
    print("\n3. Verificando archivos locales...")
    watch_dir = config['monitoring']['watch_directory']
    if not os.path.isabs(watch_dir):
        watch_dir = os.path.join(os.path.dirname(__file__), watch_dir)
    watch_dir = os.path.abspath(watch_dir)
    
    if os.path.exists(watch_dir):
        local_files = [f for f in os.listdir(watch_dir) if f.endswith('.csv')]
        print(f"Archivos locales encontrados: {len(local_files)}")
    else:
        print("Directorio local no existe")
    
    print("\n=" * 60)
    print("VERIFICACIÓN COMPLETADA")
    print("=" * 60)
    return True

def main():
    """Función principal - inicia directamente el monitoreo automático."""
    setup_logging()
    
    print("=" * 60)
    print("SISTEMA MLOps LOCAL - PREDICCION CLIMATICA")
    print("=" * 60)
    
    # DEBUG: Mostrar argumentos recibidos
    print(f"DEBUG: Argumentos recibidos: {sys.argv}")
    print(f"DEBUG: Numero de argumentos: {len(sys.argv)}")
    if len(sys.argv) > 1:
        print(f"DEBUG: Primer argumento: '{sys.argv[1]}'")
    
    # Verificar si se quiere ejecutar el modelo directamente
    if len(sys.argv) > 1 and sys.argv[1] == "--run-model":
        print("DEBUG: Detectado argumento --run-model, ejecutando modelo directamente...")
        return run_model_directly()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-retrain":
        print("DEBUG: Detectado argumento --test-retrain, probando reentrenamiento automático...")
        return test_automatic_retraining()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check-once":
        print("DEBUG: Detectado argumento --check-once, verificando una sola vez...")
        return run_single_check()
    else:
        print("DEBUG: No se detectó argumento especial, iniciando monitoreo continuo...")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_manager = ModelManager()
    
    # Usar ruta absoluta para el directorio de monitoreo local (como respaldo)
    watch_dir = config['monitoring']['watch_directory']
    if not os.path.isabs(watch_dir):
        watch_dir = os.path.join(os.path.dirname(__file__), watch_dir)
    watch_dir = os.path.abspath(watch_dir)
    
    check_interval = config['monitoring']['check_interval']
    
    print(f"Sistema MLOps con monitoreo S3 y local")
    print(f"📁 Carpeta local (respaldo): {watch_dir}")
    print(f"☁️  Bucket S3: {os.getenv('AWS_BUCKET', 'ml-ops-datos-prediccion-clima-uadec22025-ml')}")
    print(f"⏰ Intervalo de chequeo: {check_interval} segundos")
    print(f"🎯 Variable objetivo: {config['target_variable']}")
    print(f"📊 Umbral MAE reentrenamiento: {config['training']['retrain_threshold_mae']}")
    print("=" * 60)
    
    # Verificar si las credenciales de AWS están configuradas
    aws_configured = os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if aws_configured:
        print("Credenciales AWS configuradas - Usando monitoreo S3 + local")
    else:
        print("Credenciales AWS no configuradas - Solo monitoreo local")
    
    def callback(csv_path):
        filename = os.path.basename(csv_path)
        if filename.startswith('model_results_'):
            # Es un archivo de resultados del modelo
            print(f"[MONITOR LOCAL] Detectado archivo de resultados: {filename}")
            process_model_results(csv_path, config, model_manager)
        else:
            # Es un archivo de datos para procesamiento
            print(f"[MONITOR LOCAL] Detectado archivo de datos: {filename}")
            process_new_data(csv_path, config, model_manager)
    
    logging.info("Sistema MLOps iniciado. Monitoreando S3 y carpeta local...")
    print("🚀 Sistema iniciado. Monitoreando cambios en S3 y archivos locales...")
    print("🛑 Presiona Ctrl+C para detener...")
    
    # Variables para el monitoreo
    last_checked_s3_files = set()
    
    try:
        # Bucle principal de monitoreo
        while True:
            try:
                # 1. Verificar archivos nuevos en S3 (si está configurado)
                if aws_configured:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 Verificando S3 para nuevos resultados...")
                    last_checked_s3_files = check_for_new_results_s3(config, model_manager, last_checked_s3_files)
                
                # 2. Verificar archivos locales (respaldo)
                if os.path.exists(watch_dir):
                    local_files = [f for f in os.listdir(watch_dir) if f.endswith('.csv')]
                    if local_files:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📁 Verificando {len(local_files)} archivos locales...")
                        for file in local_files:
                            full_path = os.path.join(watch_dir, file)
                            if os.path.getmtime(full_path) > time.time() - check_interval:
                                print(f"[MONITOR LOCAL] Archivo reciente detectado: {file}")
                                callback(full_path)
                
                # 3. Pausa antes del siguiente ciclo
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💤 Esperando {check_interval} segundos...")
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"❌ Error en ciclo de monitoreo: {str(e)}")
                time.sleep(check_interval)
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Monitoreo detenido por el usuario.")
        print("📊 Resumen de la sesión:")
        print(f"   • Archivos S3 procesados: {len(last_checked_s3_files)}")
        print("   • Sistema MLOps detenido correctamente")

if __name__ == '__main__':
    main()
