"""
API Client para Open-Meteo
==========================

Módulo para obtener datos climáticos desde la API REST que conecta con Open-Meteo.
Proporciona funciones para consultar datos y transformarlos al formato requerido.

Autor: Climate ML Team
Fecha: Septiembre 2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
import json

# Configurar logging
logger = logging.getLogger(__name__)

class OpenMeteoAPIClient:
    """Cliente para obtener datos climáticos directamente desde Open-Meteo."""
    
    def __init__(self, latitude: float = 52.52, longitude: float = 13.41):
        """
        Inicializar cliente de API.
        
        Args:
            latitude: Latitud de la ubicación (default: Berlin)
            longitude: Longitud de la ubicación (default: Berlin)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
        # Construir URL con parámetros
        self.api_endpoint = (
            f"{self.base_url}?"
            f"latitude={latitude}&longitude={longitude}&"
            f"hourly=temperature_2m,dew_point_2m,relative_humidity_2m,"
            f"wind_speed_10m,pressure_msl,cloud_cover,shortwave_radiation"
        )
        
    def test_connection(self, timeout: int = 5) -> bool:
        """
        Probar conexión con Open-Meteo API.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            True si la conexión es exitosa, False si hay error
        """
        try:
            logger.info(f"🔗 Probando conexión con Open-Meteo: {self.base_url}")
            response = requests.get(self.api_endpoint, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if 'hourly' in data and 'time' in data['hourly']:
                logger.info("✅ Open-Meteo API responde correctamente")
                return True
            else:
                logger.warning("⚠️ Open-Meteo API responde pero formato inválido")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ No se puede conectar con Open-Meteo API")
            return False
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout al conectar con Open-Meteo API")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado: {str(e)}")
            return False
    
    def fetch_climate_data(self, timeout: int = 30) -> pd.DataFrame:
        """
        Obtener datos climáticos directamente desde Open-Meteo API.
        
        Args:
            timeout: Timeout en segundos para la petición
            
        Returns:
            DataFrame con datos climáticos procesados
            
        Raises:
            requests.RequestException: Error en la petición HTTP
            ValueError: Error en el formato de datos
        """
        try:
            logger.info(f"🌐 Consultando datos climáticos desde Open-Meteo...")
            logger.info(f"📍 Ubicación: {self.latitude}°N, {self.longitude}°E")
            
            # Realizar petición GET directa a Open-Meteo
            response = requests.get(self.api_endpoint, timeout=timeout)
            response.raise_for_status()
            
            # Parsear respuesta JSON
            api_data = response.json()
            logger.info("✅ Datos recibidos de Open-Meteo API")
            
            # Validar estructura de datos
            if 'hourly' not in api_data:
                raise ValueError("Formato de API inválido: falta campo 'hourly'")
            
            hourly_data = api_data['hourly']
            
            # Validar campos requeridos
            if 'time' not in hourly_data or 'temperature_2m' not in hourly_data:
                raise ValueError("Datos faltantes: se requieren 'time' y 'temperature_2m'")
            
            # Procesar datos
            df = self._process_api_data(hourly_data)
            
            logger.info(f"📊 Datos procesados: {len(df)} registros de {df['datetime'].min()} a {df['datetime'].max()}")
            
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout al conectar con Open-Meteo API")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Error de conexión con Open-Meteo API")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code}: {e}")
            raise
        except json.JSONDecodeError:
            logger.error("❌ Error al parsear respuesta JSON")
            raise
        except Exception as e:
            logger.error(f"❌ Error inesperado: {str(e)}")
            raise
    
    def _process_api_data(self, hourly_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Procesar datos de la API Open-Meteo al formato requerido.
        
        Args:
            hourly_data: Datos horarios de la API
            
        Returns:
            DataFrame procesado
        """
        # Crear DataFrame base con datos disponibles
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly_data['time']),
            'temperature_2m': hourly_data['temperature_2m']
        })
        
        # Mapeo de campos adicionales disponibles en Open-Meteo
        field_mapping = {
            'dew_point_2m': ['dewpoint_2m', 'dew_point_2m'],
            'relative_humidity_2m': ['relativehumidity_2m', 'relative_humidity_2m'], 
            'wind_speed_10m': ['windspeed_10m', 'wind_speed_10m'],
            'pressure_msl': ['pressure_msl', 'surface_pressure'],
            'cloud_cover': ['cloudcover', 'cloud_cover'],
            'shortwave_radiation': ['shortwave_radiation', 'solar_radiation']
        }
        
        # Agregar campos disponibles o generar sintéticos
        for target_field, api_variants in field_mapping.items():
            field_added = False
            
            # Buscar el campo en las variantes de la API
            for api_field in api_variants:
                if api_field in hourly_data:
                    df[target_field] = hourly_data[api_field]
                    field_added = True
                    logger.debug(f"✅ Campo '{target_field}' obtenido de API como '{api_field}'")
                    break
            
            # Si no está disponible, generar valores sintéticos realistas
            if not field_added:
                df[target_field] = self._generate_synthetic_field(target_field, df)
                logger.warning(f"⚠️ Campo '{target_field}' no disponible, generando valores sintéticos")
        
        # Agregar elevation (dato fijo para la ubicación de Berlin)
        df['elevation'] = 34.0  # Elevación real de Berlin en metros
        
        # Limpiar datos
        df = self._clean_data(df)
        
        return df
    
    def _generate_synthetic_field(self, field_name: str, df: pd.DataFrame) -> np.ndarray:
        """
        Generar valores sintéticos realistas para campos faltantes.
        
        Args:
            field_name: Nombre del campo a generar
            df: DataFrame con datos base
            
        Returns:
            Array con valores sintéticos
        """
        n_samples = len(df)
        hours = df['datetime'].dt.hour.values
        temp = df['temperature_2m'].values
        
        if field_name == 'dew_point_2m':
            # Punto de rocío = Temperatura - diferencia típica
            return temp - 5 + np.random.normal(0, 1, n_samples)
            
        elif field_name == 'relative_humidity_2m':
            # Humedad relativa con patrón diario
            base_humidity = 70 + 15 * np.sin(2 * np.pi * hours / 24)
            return np.clip(base_humidity + np.random.normal(0, 10, n_samples), 20, 100)
            
        elif field_name == 'wind_speed_10m':
            # Velocidad del viento con distribución exponencial
            return np.clip(np.random.exponential(5, n_samples), 0, 30)
            
        elif field_name == 'pressure_msl':
            # Presión atmosférica con variación típica
            return 1013.25 + np.random.normal(0, 8, n_samples)
            
        elif field_name == 'cloud_cover':
            # Cobertura nubosa aleatoria
            return np.clip(np.random.uniform(0, 100, n_samples), 0, 100)
            
        elif field_name == 'shortwave_radiation':
            # Radiación solar basada en hora del día
            solar_pattern = np.maximum(0, 800 * np.sin(np.pi * hours / 12))
            return solar_pattern + np.random.normal(0, 50, n_samples)
            
        else:
            # Campo desconocido, generar valores neutros
            return np.zeros(n_samples)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y validar datos.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame limpio
        """
        # Eliminar valores nulos
        df = df.dropna()
        
        # Asegurar rangos válidos
        df['relative_humidity_2m'] = np.clip(df['relative_humidity_2m'], 0, 100)
        df['cloud_cover'] = np.clip(df['cloud_cover'], 0, 100)
        df['wind_speed_10m'] = np.clip(df['wind_speed_10m'], 0, 50)
        df['shortwave_radiation'] = np.maximum(0, df['shortwave_radiation'])
        
        # Ordenar por fecha
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df

def get_climate_data_from_api(latitude: float = 52.52, longitude: float = 13.41, timeout: int = 30) -> pd.DataFrame:
    """
    Función simplificada para obtener datos climáticos directamente desde Open-Meteo.
    
    Args:
        latitude: Latitud de la ubicación (default: Berlin)
        longitude: Longitud de la ubicación (default: Berlin)
        timeout: Timeout en segundos
        
    Returns:
        DataFrame con datos climáticos
        
    Raises:
        Exception: Si hay algún error en la obtención de datos
    """
    client = OpenMeteoAPIClient(latitude, longitude)
    
    # Probar conexión primero
    if not client.test_connection():
        raise ConnectionError(f"No se puede conectar con Open-Meteo API")
    
    # Obtener datos
    return client.fetch_climate_data(timeout)

def save_api_data_to_csv(output_path: str, latitude: float = 52.52, longitude: float = 13.41, 
                        timeout: int = 30) -> str:
    """
    Obtener datos de Open-Meteo API y guardarlos como CSV.
    
    Args:
        output_path: Ruta donde guardar el archivo CSV
        latitude: Latitud de la ubicación (default: Berlin)
        longitude: Longitud de la ubicación (default: Berlin)
        timeout: Timeout en segundos
        
    Returns:
        Ruta del archivo guardado
    """
    # Obtener datos
    df = get_climate_data_from_api(latitude, longitude, timeout)
    
    # Guardar como CSV
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Datos guardados en: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Prueba del módulo
    try:
        print("🧪 Probando módulo de API...")
        print("🌐 Conectando directamente con Open-Meteo API...")
        df = get_climate_data_from_api()
        print(f"✅ Datos obtenidos: {len(df)} registros")
        print(f"📊 Columnas: {list(df.columns)}")
        print(f"📅 Rango: {df['datetime'].min()} a {df['datetime'].max()}")
        print(f"🌡️ Temperatura promedio: {df['temperature_2m'].mean():.1f}°C")
    except Exception as e:
        print(f"❌ Error: {str(e)}")