#!/usr/bin/env python3
"""
Evaluador de Modelo de Predicción Climática para Centro de Ski

Script simplificado para evaluar modelos ML y decidir si necesitan reentrenamiento.
Funciona localmente y en AWS Lambda.

Autor: MLOps Team - Centro de Ski
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importar boto3 solo si está disponible (opcional para S3)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_from_s3(s3_path: str, local_path: str) -> None:
    """Descarga archivo desde S3 si boto3 está disponible"""
    if not S3_AVAILABLE:
        raise RuntimeError("boto3 no disponible para descargar desde S3")
    
    # Parsear ruta S3
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Ruta S3 inválida: {s3_path}")
    
    path_parts = s3_path[5:].split('/', 1)
    if len(path_parts) != 2:
        raise ValueError(f"Formato S3 inválido: {s3_path}")
    
    bucket, key = path_parts
    
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Descargado: {s3_path} → {local_path}")
    except Exception as e:
        raise RuntimeError(f"Error descargando {s3_path}: {e}")


def upload_to_s3(local_path: str, s3_path: str) -> None:
    """Sube archivo a S3 si boto3 está disponible"""
    if not S3_AVAILABLE:
        logger.warning("boto3 no disponible, saltando subida a S3")
        return
    
    path_parts = s3_path[5:].split('/', 1)
    bucket, key = path_parts
    
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"Subido: {local_path} → {s3_path}")
    except Exception as e:
        logger.error(f"Error subiendo a {s3_path}: {e}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de evaluación"""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # MAPE evitando división por cero
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float('inf')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(y_true)
    }


def load_file(file_path: str, is_model: bool = False):
    """Carga archivo desde local o S3"""
    temp_file = None
    
    try:
        # Si es S3, descargar primero
        if file_path.startswith('s3://'):
            suffix = '_model.pkl' if is_model else '_data.csv'
            temp_file = tempfile.mktemp(suffix=suffix)
            download_from_s3(file_path, temp_file)
            local_path = temp_file
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            local_path = file_path
        
        # Cargar archivo
        if is_model:
            result = joblib.load(local_path)
            logger.info("Modelo cargado exitosamente")
        else:
            result = pd.read_csv(local_path)
            logger.info(f"Datos cargados: {len(result)} filas")
        
        return result
        
    finally:
        # Limpiar archivo temporal
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def evaluate_model(model_path: str, data_path: str, target_col: str,
                  feature_cols: Optional[List[str]] = None,
                  primary_metric: str = 'mae', threshold: float = 2.0,
                  output_path: str = 'metrics.json',
                  s3_output: Optional[str] = None) -> Dict[str, Any]:
    """
    Evalúa un modelo y determina si necesita reentrenamiento
    
    Args:
        model_path: Ruta al modelo (local o S3)
        data_path: Ruta a los datos CSV (local o S3)
        target_col: Columna objetivo
        feature_cols: Lista de features (opcional)
        primary_metric: Métrica principal ('mae', 'rmse', 'mape')
        threshold: Umbral para reentrenamiento
        output_path: Ruta local para guardar métricas
        s3_output: Ruta S3 para subir métricas (opcional)
    
    Returns:
        Dict con métricas y decisión de reentrenamiento
    """
    
    logger.info("=== Iniciando Evaluación del Modelo ===")
    logger.info(f"Modelo: {model_path}")
    logger.info(f"Datos: {data_path}")
    
    try:
        # Cargar modelo y datos
        model = load_file(model_path, is_model=True)
        data = load_file(data_path, is_model=False)
        
        # Validar columnas
        if target_col not in data.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
        
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
            logger.info(f"Usando todas las columnas como features: {feature_cols}")
        
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Limpiar datos
        clean_data = data.dropna(subset=[target_col] + feature_cols)
        if len(clean_data) == 0:
            raise ValueError("No hay datos válidos después de limpieza")
        
        if len(clean_data) < len(data):
            logger.warning(f"Removidas {len(data) - len(clean_data)} filas con datos faltantes")
        
        # Preparar datos y predecir
        X = clean_data[feature_cols]
        y_true = clean_data[target_col].values
        
        y_pred = model.predict(X)
        logger.info(f"Predicciones realizadas para {len(y_pred)} muestras")
        
        # Calcular métricas
        metrics = calculate_metrics(y_true, y_pred)
        
        # Decisión de reentrenamiento
        if primary_metric not in metrics:
            raise ValueError(f"Métrica '{primary_metric}' no válida")
        
        metric_value = metrics[primary_metric]
        should_retrain = metric_value > threshold
        
        logger.info(f"Métricas: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, MAPE={metrics['mape']:.1f}%")
        logger.info(f"Decisión: {primary_metric}={metric_value:.3f} {'>' if should_retrain else '<='} {threshold} → {'REENTRENAR' if should_retrain else 'OK'}")
        
        # Resultado
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_path': model_path,
            'data_path': data_path,
            'target_col': target_col,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'primary_metric': primary_metric,
            'threshold': threshold,
            'should_retrain': should_retrain
        }
        
        # Guardar métricas
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Métricas guardadas: {output_path}")
        
        # Subir a S3 si se especifica
        if s3_output:
            upload_to_s3(output_path, s3_output)
        
        return result
        
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        raise


def parse_feature_cols(feature_cols_str: Optional[str]) -> Optional[List[str]]:
    """Parsea la cadena de columnas de features"""
    if not feature_cols_str:
        return None
    
    # Soportar formato: "col1,col2,col3"
    return [col.strip() for col in feature_cols_str.split(',') if col.strip()]


def main():
    """Función principal para ejecución CLI"""
    parser = argparse.ArgumentParser(
        description='Evaluador de Modelo de Predicción Climática',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación básica
  python evaluate.py --model model.pkl --data data.csv --target temperature

  # Con S3 y configuración completa
  python evaluate.py --model s3://bucket/model.pkl --data s3://bucket/data.csv --target temperature --features "humidity,wind_speed" --metric rmse --threshold 3.0
        """
    )
    
    parser.add_argument('--model', required=True, help='Ruta al modelo (local o S3)')
    parser.add_argument('--data', required=True, help='Ruta a los datos (local o S3)')
    parser.add_argument('--target', required=True, help='Columna objetivo')
    parser.add_argument('--features', help='Features separadas por comas (opcional)')
    parser.add_argument('--metric', default='mae', choices=['mae', 'rmse', 'mape'],
                       help='Métrica principal')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Umbral para reentrenamiento')
    parser.add_argument('--output', default='metrics.json',
                       help='Archivo local para métricas')
    parser.add_argument('--s3-output', help='URI S3 para subir métricas')
    parser.add_argument('--verbose', '-v', action='store_true', help='Logs detallados')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    feature_cols = parse_feature_cols(args.features)
    
    try:
        result = evaluate_model(
            model_path=args.model,
            data_path=args.data,
            target_col=args.target,
            feature_cols=feature_cols,
            primary_metric=args.metric,
            threshold=args.threshold,
            output_path=args.output,
            s3_output=args.s3_output
        )
        
        # Imprimir resultado JSON
        print(json.dumps(result, indent=2))
        
        # Exit code para CI/CD
        sys.exit(1 if result['should_retrain'] else 0)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(json.dumps({
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        sys.exit(2)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handler para AWS Lambda"""
    logger.info("Lambda handler iniciado")
    
    try:
        # Extraer parámetros del evento
        result = evaluate_model(
            model_path=event['model_path'],
            data_path=event['data_path'],
            target_col=event['target_col'],
            feature_cols=event.get('feature_cols'),
            primary_metric=event.get('primary_metric', 'mae'),
            threshold=float(event.get('threshold', 2.0)),
            output_path=event.get('output_path', '/tmp/metrics.json'),
            s3_output=event.get('s3_output')
        )
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except KeyError as e:
        error_msg = f"Parámetro requerido faltante: {e}"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': {'error': error_msg, 'timestamp': datetime.now(timezone.utc).isoformat()}
        }
    
    except Exception as e:
        error_msg = f"Error en evaluación: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': {'error': error_msg, 'timestamp': datetime.now(timezone.utc).isoformat()}
        }


if __name__ == '__main__':
    main()
