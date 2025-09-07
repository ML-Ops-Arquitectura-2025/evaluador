#!/usr/bin/env python3
"""
Evaluador de Modelo de Predicción Climática para Centro de Ski
==============================================================

Este script evalúa el rendimiento de modelos de machine learning para predicción 
de variables climáticas y determina si es necesario reentrenar el modelo.

Características:
- Compatible con ejecución local (CLI) y AWS Lambda
- Soporte para archivos en local y S3
- Cálculo de métricas MAE, RMSE, MAPE
- Decisión automática de reentrenamiento basada en umbrales
- Salida estructurada en JSON

Autor: Sistema MLOps - Centro de Ski
Fecha: 2025-09-07
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Handler:
    """Maneja operaciones con AWS S3"""
    
    def __init__(self):
        try:
            self.s3_client = boto3.client('s3')
            logger.info("Cliente S3 inicializado correctamente")
        except NoCredentialsError:
            logger.warning("Credenciales AWS no encontradas. Funcionará solo con archivos locales.")
            self.s3_client = None
    
    def is_s3_path(self, path: str) -> bool:
        """Verifica si una ruta es de S3"""
        return path.startswith('s3://')
    
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parsea una ruta S3 y retorna bucket y key"""
        if not self.is_s3_path(s3_path):
            raise ValueError(f"Ruta inválida de S3: {s3_path}")
        
        # Remover s3:// y dividir en bucket/key
        path_without_protocol = s3_path[5:]
        parts = path_without_protocol.split('/', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Formato de ruta S3 inválido: {s3_path}")
        
        return parts[0], parts[1]
    
    def download_file(self, s3_path: str, local_path: str) -> None:
        """Descarga un archivo desde S3"""
        if not self.s3_client:
            raise RuntimeError("Cliente S3 no disponible")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Descargando {s3_path} a {local_path}")
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Descarga completada: {local_path}")
        except ClientError as e:
            logger.error(f"Error descargando {s3_path}: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_path: str) -> None:
        """Sube un archivo a S3"""
        if not self.s3_client:
            raise RuntimeError("Cliente S3 no disponible")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Subiendo {local_path} a {s3_path}")
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Subida completada: {s3_path}")
        except ClientError as e:
            logger.error(f"Error subiendo a {s3_path}: {e}")
            raise


class MetricsCalculator:
    """Calcula métricas de evaluación para modelos de regresión"""
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Mean Absolute Error"""
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Root Mean Squared Error"""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Mean Absolute Percentage Error"""
        # Evitar división por cero
        mask = y_true != 0
        if not mask.any():
            logger.warning("Todos los valores reales son cero, MAPE no se puede calcular")
            return float('inf')
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)
    
    @classmethod
    def calculate_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas las métricas disponibles"""
        return {
            'mae': cls.calculate_mae(y_true, y_pred),
            'rmse': cls.calculate_rmse(y_true, y_pred),
            'mape': cls.calculate_mape(y_true, y_pred),
            'n_samples': len(y_true)
        }


class ModelEvaluator:
    """Evaluador principal del modelo"""
    
    def __init__(self):
        self.s3_handler = S3Handler()
        self.metrics_calculator = MetricsCalculator()
        self.temp_files = []  # Para limpieza de archivos temporales
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Limpia archivos temporales"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Archivo temporal eliminado: {temp_file}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {temp_file}: {e}")
    
    def _get_temp_file(self, suffix: str = '') -> str:
        """Crea un archivo temporal y lo registra para limpieza"""
        temp_file = tempfile.mktemp(suffix=suffix)
        self.temp_files.append(temp_file)
        return temp_file
    
    def _download_if_needed(self, file_path: str, suffix: str = '') -> str:
        """Descarga archivo si es S3, retorna ruta local"""
        if self.s3_handler.is_s3_path(file_path):
            local_path = self._get_temp_file(suffix=suffix)
            self.s3_handler.download_file(file_path, local_path)
            return local_path
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo local no encontrado: {file_path}")
            return file_path
    
    def load_model(self, model_path: str):
        """Carga el modelo desde local o S3"""
        logger.info(f"Cargando modelo desde: {model_path}")
        local_model_path = self._download_if_needed(model_path, suffix='_model.pkl')
        
        try:
            model = joblib.load(local_model_path)
            logger.info("Modelo cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Carga datos desde local o S3"""
        logger.info(f"Cargando datos desde: {data_path}")
        local_data_path = self._download_if_needed(data_path, suffix='_data.csv')
        
        try:
            data = pd.read_csv(local_data_path)
            logger.info(f"Datos cargados: {len(data)} filas, {len(data.columns)} columnas")
            return data
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame, target_col: str, 
                     feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Valida y prepara los datos para evaluación"""
        
        # Verificar columna objetivo
        if target_col not in data.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en los datos")
        
        # Determinar columnas de features
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
            logger.info(f"Usando todas las columnas como features (excepto target): {feature_cols}")
        else:
            # Verificar que todas las features existan
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Columnas de features no encontradas: {missing_cols}")
        
        # Verificar datos faltantes
        missing_target = data[target_col].isna().sum()
        missing_features = data[feature_cols].isna().sum().sum()
        
        if missing_target > 0:
            logger.warning(f"Datos faltantes en columna objetivo: {missing_target}")
        
        if missing_features > 0:
            logger.warning(f"Datos faltantes en features: {missing_features}")
        
        # Remover filas con datos faltantes
        clean_data = data.dropna(subset=[target_col] + feature_cols)
        removed_rows = len(data) - len(clean_data)
        
        if removed_rows > 0:
            logger.info(f"Filas removidas por datos faltantes: {removed_rows}")
        
        if len(clean_data) == 0:
            raise ValueError("No quedan datos válidos después de la limpieza")
        
        return clean_data, feature_cols
    
    def evaluate_model(self, model, data: pd.DataFrame, target_col: str, 
                      feature_cols: List[str]) -> Dict[str, float]:
        """Evalúa el modelo y calcula métricas"""
        
        logger.info("Iniciando evaluación del modelo...")
        
        # Preparar datos
        X = data[feature_cols].values
        y_true = data[target_col].values
        
        # Realizar predicciones
        try:
            y_pred = model.predict(X)
            logger.info(f"Predicciones realizadas para {len(y_pred)} muestras")
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
        
        # Calcular métricas
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred)
        
        logger.info(f"Métricas calculadas: MAE={metrics['mae']:.4f}, "
                   f"RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        return metrics
    
    def should_retrain(self, metrics: Dict[str, float], primary_metric: str, 
                      threshold: float) -> bool:
        """Determina si el modelo debe ser reentrenado"""
        
        if primary_metric not in metrics:
            raise ValueError(f"Métrica primaria '{primary_metric}' no encontrada en las métricas")
        
        metric_value = metrics[primary_metric]
        should_retrain = metric_value > threshold
        
        logger.info(f"Evaluación de reentrenamiento: {primary_metric}={metric_value:.4f}, "
                   f"umbral={threshold}, reentrenar={should_retrain}")
        
        return should_retrain
    
    def save_metrics(self, metrics_data: Dict[str, Any], local_path: str, 
                    s3_uri: Optional[str] = None) -> None:
        """Guarda las métricas localmente y opcionalmente en S3"""
        
        # Guardar localmente
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas guardadas localmente: {local_path}")
        
        # Subir a S3 si se especifica
        if s3_uri and self.s3_handler.s3_client:
            try:
                self.s3_handler.upload_file(local_path, s3_uri)
                logger.info(f"Métricas subidas a S3: {s3_uri}")
            except Exception as e:
                logger.error(f"Error subiendo métricas a S3: {e}")
                # No fallar si la subida a S3 falla
    
    def run_evaluation(self, model_path: str, data_path: str, target_col: str,
                      feature_cols: Optional[List[str]] = None,
                      primary_metric: str = 'mae', threshold: float = 2.0,
                      metrics_local_path: str = 'metrics.json',
                      metrics_s3_uri: Optional[str] = None) -> Dict[str, Any]:
        """Ejecuta la evaluación completa del modelo"""
        
        logger.info("=== Iniciando Evaluación del Modelo ===")
        logger.info(f"Modelo: {model_path}")
        logger.info(f"Datos: {data_path}")
        logger.info(f"Métrica primaria: {primary_metric}, Umbral: {threshold}")
        
        try:
            # Cargar modelo y datos
            model = self.load_model(model_path)
            data = self.load_data(data_path)
            
            # Validar y preparar datos
            clean_data, final_feature_cols = self.validate_data(data, target_col, feature_cols)
            
            # Evaluar modelo
            metrics = self.evaluate_model(model, clean_data, target_col, final_feature_cols)
            
            # Determinar si reentrenar
            should_retrain_flag = self.should_retrain(metrics, primary_metric, threshold)
            
            # Crear resultado
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            tag = f"eval_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            
            result = {
                'timestamp_utc': timestamp,
                'model_path': model_path,
                'data_path': data_path,
                'target_col': target_col,
                'feature_cols': final_feature_cols,
                'metrics': metrics,
                'primary_metric': primary_metric,
                'threshold': threshold,
                'should_retrain': should_retrain_flag,
                'tag': tag
            }
            
            # Guardar métricas
            self.save_metrics(result, metrics_local_path, metrics_s3_uri)
            
            logger.info("=== Evaluación Completada Exitosamente ===")
            logger.info(f"Resultado: should_retrain = {should_retrain_flag}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            raise


def parse_feature_cols(feature_cols_str: Optional[str]) -> Optional[List[str]]:
    """Parsea la cadena de columnas de features"""
    if not feature_cols_str:
        return None
    
    # Soportar formato: "col1,col2,col3" o "['col1','col2','col3']"
    if feature_cols_str.startswith('[') and feature_cols_str.endswith(']'):
        # Formato JSON-like
        try:
            return json.loads(feature_cols_str.replace("'", '"'))
        except json.JSONDecodeError:
            # Fallback: dividir por comas
            pass
    
    # Formato simple separado por comas
    return [col.strip() for col in feature_cols_str.split(',') if col.strip()]


def main():
    """Función principal para ejecución CLI"""
    parser = argparse.ArgumentParser(
        description='Evaluador de Modelo de Predicción Climática',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación básica local
  python evaluate.py --model_path model.pkl --data_path data.csv --target_col temperature

  # Evaluación con S3
  python evaluate.py --model_path s3://bucket/model.pkl --data_path s3://bucket/data.csv --target_col temperature --metrics_s3_uri s3://bucket/metrics.json

  # Evaluación con configuración completa
  python evaluate.py --model_path model.pkl --data_path data.csv --target_col temperature --feature_cols "humidity,wind_speed,pressure" --primary_metric rmse --threshold 3.0
        """
    )
    
    parser.add_argument('--model_path', required=True,
                       help='Ruta al modelo (local o S3)')
    parser.add_argument('--data_path', required=True,
                       help='Ruta a los datos de evaluación (local o S3)')
    parser.add_argument('--target_col', required=True,
                       help='Nombre de la columna objetivo')
    parser.add_argument('--feature_cols',
                       help='Columnas de features separadas por comas (opcional)')
    parser.add_argument('--primary_metric', default='mae',
                       choices=['mae', 'rmse', 'mape'],
                       help='Métrica primaria para decisión de reentrenamiento')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Umbral para la métrica primaria')
    parser.add_argument('--metrics_local_path', default='metrics.json',
                       help='Ruta local para guardar métricas')
    parser.add_argument('--metrics_s3_uri',
                       help='URI S3 para subir métricas (opcional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Modo verbose para logs detallados')
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parsear feature columns
    feature_cols = parse_feature_cols(args.feature_cols)
    
    try:
        with ModelEvaluator() as evaluator:
            result = evaluator.run_evaluation(
                model_path=args.model_path,
                data_path=args.data_path,
                target_col=args.target_col,
                feature_cols=feature_cols,
                primary_metric=args.primary_metric,
                threshold=args.threshold,
                metrics_local_path=args.metrics_local_path,
                metrics_s3_uri=args.metrics_s3_uri
            )
            
            # Imprimir resultado JSON
            print(json.dumps(result, indent=2))
            
            # Exit code basado en should_retrain para uso en CI/CD
            sys.exit(1 if result['should_retrain'] else 0)
            
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        print(json.dumps({
            'error': str(e),
            'timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        }, indent=2))
        sys.exit(2)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handler para AWS Lambda"""
    
    logger.info("=== Lambda Handler Iniciado ===")
    logger.info(f"Event recibido: {json.dumps(event, default=str)}")
    
    try:
        # Extraer parámetros del evento
        model_path = event['model_path']
        data_path = event['data_path']
        target_col = event['target_col']
        
        # Parámetros opcionales con valores por defecto
        feature_cols = event.get('feature_cols')
        if isinstance(feature_cols, str):
            feature_cols = parse_feature_cols(feature_cols)
        
        primary_metric = event.get('primary_metric', 'mae')
        threshold = float(event.get('threshold', 2.0))
        metrics_local_path = event.get('metrics_local_path', '/tmp/metrics.json')
        metrics_s3_uri = event.get('metrics_s3_uri')
        
        # Ejecutar evaluación
        with ModelEvaluator() as evaluator:
            result = evaluator.run_evaluation(
                model_path=model_path,
                data_path=data_path,
                target_col=target_col,
                feature_cols=feature_cols,
                primary_metric=primary_metric,
                threshold=threshold,
                metrics_local_path=metrics_local_path,
                metrics_s3_uri=metrics_s3_uri
            )
        
        logger.info("=== Lambda Handler Completado Exitosamente ===")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except KeyError as e:
        error_msg = f"Parámetro requerido faltante: {e}"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': {
                'error': error_msg,
                'timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        }
    
    except Exception as e:
        error_msg = f"Error en evaluación: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        }


if __name__ == '__main__':
    main()
