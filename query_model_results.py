#!/usr/bin/env python3
"""
Funciones para consultar resultados del modelo desde S3

Este módulo proporciona funciones para:
- Consultar el último resultado del modelo
- Obtener historial de modelos
- Formatear resultados de manera legible

Uso:
    from query_model_results import get_latest_model_result, format_model_result
    
    latest = get_latest_model_result()
    print(format_model_result(latest))
"""

# Importar funciones del modelo principal
from models.model import get_latest_model_result, format_model_result, _evaluate_model_quality
import os
import boto3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import io

# Cargar variables de entorno
load_dotenv()

def get_model_history(limit=5):
    """
    Obtiene el historial de los últimos modelos
    
    Args:
        limit: Número máximo de resultados a retornar
    
    Returns:
        list: Lista de diccionarios con resultados históricos
    """
    try:
        # Configuración AWS
        AWS_ACCESS_KEY = os.getenv("AWS_JOBLIB_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_KEY = os.getenv("AWS_JOBLIB_SECRET_KEY") or os.getenv("AWS_SECRET_KEY")
        AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
        BUCKET_NAME = "ml-ops-datos-prediccion-clima-uadec22025-ml"
        
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            raise Exception("Credenciales AWS no encontradas")
        
        # Conectar a S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        # Listar archivos en la carpeta results/
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, 
            Prefix='results/model_results_'
        )
        
        if 'Contents' not in response:
            return []
            
        # Recopilar todos los archivos
        result_files = []
        for obj in response['Contents']:
            if obj['Key'].startswith('results/model_results_') and obj['Key'].endswith('.csv'):
                result_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified']
                })
        
        # Ordenar por fecha (más recientes primero) y limitar
        latest_files = sorted(result_files, key=lambda x: x['last_modified'], reverse=True)[:limit]
        
        history = []
        for file_info in latest_files:
            try:
                # Descargar archivo
                response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_info['key'])
                file_content = response['Body'].read()
                df = pd.read_csv(io.BytesIO(file_content))
                
                if not df.empty:
                    result = df.iloc[0]
                    history.append({
                        'file_name': os.path.basename(file_info['key']),
                        'timestamp': pd.to_datetime(result['timestamp']),
                        'mae': float(result['mae']),
                        'r2': float(result['r2']),
                        'quality': _evaluate_model_quality(result['mae'], result['r2'])
                    })
            except Exception as e:
                print(f"Error procesando {file_info['key']}: {e}")
                continue
        
        return history
        
    except Exception as e:
        print(f"Error al obtener historial: {str(e)}")
        return []

def query_model_api():
    """
    Función principal para consultar el estado del modelo (simula un endpoint)
    
    Returns:
        dict: Respuesta con el estado actual del modelo
    """
    latest = get_latest_model_result()
    
    if latest is None:
        return {
            'status': 'error',
            'message': 'No se encontraron resultados del modelo',
            'data': None
        }
    
    return {
        'status': 'success',
        'message': 'Resultado obtenido exitosamente',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'model_timestamp': latest['timestamp'].isoformat(),
            'model_file': latest['file_name'],
            'target_variable': latest['target_variable'],
            'metrics': {
                'mae': latest['mae'],
                'rmse': latest['rmse'],
                'r2': latest['r2'],
                'baseline_mae': latest['baseline_mae'],
                'improvement_percent': latest['improvement_over_baseline']
            },
            'quality': latest['model_quality'],
            'features_count': latest['n_features'],
            'data_size': latest['data_shape_rows'],
            's3_path': latest['file_path_s3']
        }
    }

# Función principal para demostrar uso
if __name__ == "__main__":
    print("🔍 CONSULTANDO ÚLTIMO RESULTADO DEL MODELO...")
    print()
    
    # Simular endpoint API
    api_response = query_model_api()
    
    if api_response['status'] == 'success':
        data = api_response['data']
        print("✅ API Response:")
        print(f"   Status: {api_response['status']}")
        print(f"   Timestamp: {api_response['timestamp']}")
        print(f"   Model File: {data['model_file']}")
        print(f"   MAE: {data['metrics']['mae']:.6f}")
        print(f"   Quality: {data['quality']}")
        print()
        
        # Formato legible
        latest = get_latest_model_result()
        print(format_model_result(latest))
        
        print("\n" + "=" * 50)
        print("📈 HISTORIAL RECIENTE:")
        print()
        
        # Obtener historial
        history = get_model_history(3)
        for i, model in enumerate(history, 1):
            print(f"{i}. {model['file_name']}")
            print(f"   📅 {model['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   📊 MAE: {model['mae']:.4f} | R²: {model['r2']:.4f}")
            print(f"   ⭐ {model['quality']}")
            print()
    else:
        print(f"❌ Error: {api_response['message']}")