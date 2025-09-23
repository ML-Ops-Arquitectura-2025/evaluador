"""
Climate Prediction Model using Random Forest Regressor
======================================================

This script implements a machine learning pipeline for predicting climate variables
(temperature or snowfall) using Random Forest Regressor with comprehensive feature engineering.

Features:
- Lag features (1-24 hours)
- Rolling mean windows (3, 6, 12, 24 hours)
- Temporal features (hour, day, season)
- Baseline persistence model comparison
- Comprehensive evaluation metrics

Author: Climate ML Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import sys
import os
import boto3
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
from botocore.exceptions import NoCredentialsError, ClientError
import io

# Add parent directory to path for api_client import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import our API client
try:
    from api_client import get_climate_data_from_api, save_api_data_to_csv
except ImportError:
    # Fallback for when API client is not available
    get_climate_data_from_api = None
    save_api_data_to_csv = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ClimatePredictor:
    """
    A comprehensive climate prediction system using Random Forest Regressor.
    Supports both temperature and snowfall prediction with automated feature engineering.
    """
    
    def __init__(self, target_variable: str = 'snow_probability', random_state: int = None):
        """
        Initialize the ClimatePredictor.
        
        Args:
            target_variable: The climate variable to predict ('snow_probability', 'rain_probability', or 'temperature_2m')
            random_state: Random state for reproducibility (None for timestamp-based)
        """
        self.target_variable = target_variable
        # Use timestamp-based random state for variability in retraining
        if random_state is None:
            import time
            self.random_state = int(time.time()) % 10000  # Keep it reasonable
        else:
            self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.baseline_mae = None
        self.model_metrics = {}
        
        # Define variables for feature engineering based on target
        if target_variable == 'temperature_2m':
            self.base_features = [
                'dew_point_2m (°C)', 'relative_humidity_2m (%)', 'cloudcover (%)',
                'shortwave_radiation (W/m²)', 'wind_speed_10m (km/h)', 'pressure_msl (hPa)'
            ]
        elif target_variable in ['snow_probability', 'rain_probability']:
            self.base_features = [
                'temperature_2m (°C)', 'dew_point_2m (°C)', 'relative_humidity_2m (%)', 'cloudcover (%)',
                'shortwave_radiation (W/m²)', 'wind_speed_10m (km/h)', 'pressure_msl (hPa)',
                'precipitation (mm)', 'snowfall (cm)'
            ]
        elif target_variable == 'snowfall':
            self.base_features = [
                'temperature_2m (°C)', 'dew_point_2m (°C)', 'relative_humidity_2m (%)', 'cloudcover (%)',
                'shortwave_radiation (W/m²)', 'wind_speed_10m (km/h)', 'pressure_msl (hPa)',
                'precipitation (mm)', 'snow_depth (m)'
            ]
        else:
            raise ValueError(f"Unsupported target variable: {target_variable}. Use 'snow_probability', 'rain_probability', or 'temperature_2m'")
    
    def _get_column_name(self, base_name: str, df: pd.DataFrame) -> str:
        """
        Get the actual column name from the dataframe, handling different naming conventions.
        
        Args:
            base_name: The base column name (e.g., 'temperature_2m')
            df: The dataframe to search in
            
        Returns:
            The actual column name found in the dataframe
        """
        # If the base name exists, use it
        if base_name in df.columns:
            return base_name
            
        # Try with common unit suffixes
        variations = [
            f"{base_name} (°C)",
            f"{base_name} (cm)", 
            f"{base_name} (mm)",
            f"{base_name} (%)",
            f"{base_name} (W/m²)",
            f"{base_name} (km/h)",
            f"{base_name} (hPa)",
            f"{base_name} (m)"
        ]
        
        for variation in variations:
            if variation in df.columns:
                return variation
                
        # If no variation found, return the base name (will cause KeyError if used)
        return base_name
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the climate dataset.
        
        Args:
            file_path: Path to the CSV file containing climate data
            
        Returns:
            Loaded and validated DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Try to read the file normally first
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # If parsing fails, try skipping first few lines (metadata)
                logger.info("Normal CSV parsing failed, trying to skip metadata lines...")
                df = pd.read_csv(file_path, skiprows=3)  # Skip first 3 lines
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns found: {list(df.columns)}")
            
            # Validate required columns
            required_columns = self.base_features + [self.target_variable]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                # Try to find target variable with different naming (e.g., with units)
                target_variations = [
                    f"{self.target_variable} (°C)",
                    f"{self.target_variable} (cm)",
                    f"{self.target_variable} (mm)", 
                    f"{self.target_variable} (%)"
                ]
                
                target_found = None
                for variation in target_variations:
                    if variation in df.columns:
                        target_found = variation
                        logger.info(f"Found target variable with units: {target_found}")
                        break
                
                if not target_found:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                    
                # Update the target variable to the one found
                self.target_variable = target_found
            
            # Ensure datetime column exists or can be created
            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                elif 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                else:
                    logger.warning("No datetime column found. Creating index-based datetime.")
                    df['datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime to ensure temporal order
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_data_from_api(self, latitude: float = 52.52, longitude: float = 13.41, save_to_file: str = None) -> pd.DataFrame:
        """
        Load climate data directly from Open-Meteo API.
        
        Args:
            latitude: Latitud de la ubicación (default: Berlin 52.52)
            longitude: Longitud de la ubicación (default: Berlin 13.41)
            save_to_file: Opcional - ruta donde guardar los datos como CSV
            
        Returns:
            DataFrame con datos climáticos desde Open-Meteo API
        """
        if get_climate_data_from_api is None:
            raise ImportError("API client not available. Make sure api_client.py is in the parent directory.")
        
        logger.info(f"Cargando datos desde Open-Meteo API")
        logger.info(f"Ubicación: {latitude}°N, {longitude}°E")
        
        try:
            # Obtener datos desde la API usando nuestro cliente
            df = get_climate_data_from_api(latitude, longitude)
            
            logger.info(f"Datos de Open-Meteo cargados exitosamente. Shape: {df.shape}")
            logger.info(f"Rango temporal: {df['datetime'].min()} a {df['datetime'].max()}")
            
            # Guardar opcionalmente a archivo
            if save_to_file:
                df.to_csv(save_to_file, index=False)
                logger.info(f"Datos guardados en: {save_to_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar datos desde Open-Meteo API: {str(e)}")
            raise
    
    def fetch_and_save_api_data(self, output_path: str, latitude: float = 52.52, longitude: float = 13.41) -> str:
        """
        Obtener datos de Open-Meteo API y guardarlos en archivo CSV.
        
        Args:
            output_path: Ruta donde guardar el archivo CSV
            latitude: Latitud de la ubicación (default: Berlin)
            longitude: Longitud de la ubicación (default: Berlin)
            
        Returns:
            Ruta del archivo guardado
        """
        if save_api_data_to_csv is None:
            raise ImportError("API client not available. Make sure api_client.py is in the parent directory.")
        
        logger.info(f"Obteniendo y guardando datos de Open-Meteo API en: {output_path}")
        logger.info(f"Ubicación: {latitude}°N, {longitude}°E")
        
        try:
            # Usar la función del cliente API
            saved_path = save_api_data_to_csv(output_path, latitude, longitude)
            
            logger.info(f"Datos de Open-Meteo API guardados exitosamente en: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error al obtener y guardar datos de Open-Meteo API: {str(e)}")
            raise
    
    def create_probability_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create probability targets for snow and rain based on meteorological conditions.
        
        Args:
            df: Input DataFrame with meteorological data
            
        Returns:
            DataFrame with added probability columns
        """
        df = df.copy()
        
        # Crear probabilidad de nieve basada en condiciones meteorológicas
        if 'snow_probability' not in df.columns:
            # Get actual column names
            temp_col = self._get_column_name('temperature_2m', df)
            humidity_col = self._get_column_name('relative_humidity_2m', df)
            precip_col = self._get_column_name('precipitation', df)
            snowfall_col = self._get_column_name('snowfall', df)
            
            # Condiciones para nieve: temp baja, alta humedad, precipitación
            temp_condition = (df[temp_col] <= 4.0).astype(float) if temp_col in df.columns else 0
            humidity_condition = (df[humidity_col] >= 75).astype(float) if humidity_col in df.columns else 0
            
            # Si tenemos datos de precipitación y snowfall
            if precip_col in df.columns and snowfall_col in df.columns:
                precip_condition = (df[precip_col] > 0).astype(float)  # Hay precipitación
                snow_condition = (df[snowfall_col] > 0).astype(float)  # Hay snowfall registrado
                
                # Probabilidad de nieve: combinación de factores
                df['snow_probability'] = (
                    0.4 * temp_condition +  # 40% peso a temperatura
                    0.3 * humidity_condition +  # 30% peso a humedad
                    0.2 * precip_condition +  # 20% peso a precipitación
                    0.1 * snow_condition  # 10% peso a snowfall real
                ).clip(0, 1)
            else:
                # Sin datos de precipitación, usar temp, humedad y otros factores
                cloud_col = self._get_column_name('cloudcover', df)
                pressure_col = self._get_column_name('pressure_msl', df)
                cloud_condition = (df[cloud_col] >= 60).astype(float) if cloud_col in df.columns else 0
                pressure_condition = (df[pressure_col] <= 1015).astype(float) if pressure_col in df.columns else 0
                
                # Hacer más probabilística: añadir algo de aleatoriedad controlada
                base_prob = (
                    0.5 * temp_condition +  # 50% peso a temperatura
                    0.3 * humidity_condition +  # 30% peso a humedad
                    0.15 * cloud_condition +  # 15% peso a nubosidad
                    0.05 * pressure_condition  # 5% peso a presión baja
                )
                
                # Añadir variabilidad estacional (más probable en invierno)
                if 'datetime' in df.columns:
                    month = pd.to_datetime(df['datetime']).dt.month
                    winter_boost = ((month <= 3) | (month >= 11)).astype(float) * 0.2
                    base_prob = base_prob + winter_boost
                
                # Añadir un poco de ruido para crear variabilidad
                np.random.seed(42)  # Reproducible
                noise = np.random.normal(0, 0.1, len(df))
                
                df['snow_probability'] = (base_prob + noise).clip(0, 1)
        
        # Crear probabilidad de lluvia basada en condiciones meteorológicas
        if 'rain_probability' not in df.columns:
            # Get actual column names  
            temp_col = self._get_column_name('temperature_2m', df)
            humidity_col = self._get_column_name('relative_humidity_2m', df)
            precip_col = self._get_column_name('precipitation', df)
            snowfall_col = self._get_column_name('snowfall', df)
            
            # Condiciones para lluvia: temp moderada, alta humedad, precipitación
            temp_condition = ((df[temp_col] > 2.0) & (df[temp_col] <= 35.0)).astype(float) if temp_col in df.columns else 0
            humidity_condition = (df[humidity_col] >= 70).astype(float) if humidity_col in df.columns else 0
            
            if precip_col in df.columns:
                precip_condition = (df[precip_col] > 0).astype(float)
                no_snow_condition = (df[snowfall_col] == 0).astype(float) if snowfall_col in df.columns else 1.0
                
                df['rain_probability'] = (
                    0.3 * temp_condition +  # 30% peso a temperatura
                    0.3 * humidity_condition +  # 30% peso a humedad
                    0.3 * precip_condition +  # 30% peso a precipitación
                    0.1 * no_snow_condition  # 10% peso a no-nieve
                ).clip(0, 1)
            else:
                cloud_col = self._get_column_name('cloudcover', df)
                pressure_col = self._get_column_name('pressure_msl', df)
                cloud_condition = (df[cloud_col] >= 60).astype(float) if cloud_col in df.columns else 0
                pressure_condition = (df[pressure_col] <= 1015).astype(float) if pressure_col in df.columns else 0
                
                df['rain_probability'] = (
                    0.4 * temp_condition +  # 40% peso a temperatura
                    0.4 * humidity_condition +  # 40% peso a humedad
                    0.15 * cloud_condition +  # 15% peso a nubosidad
                    0.05 * pressure_condition  # 5% peso a presión
                ).clip(0, 1)
        
        logger.info(f"Creadas variables objetivo de probabilidad: snow_probability (avg: {df['snow_probability'].mean():.3f}), rain_probability (avg: {df['rain_probability'].mean():.3f})")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime column.
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Creating temporal features")
        
        df = df.copy()
        
        # Extract temporal components
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        
        # Create cyclical features for better model understanding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Fall
        
        df['season'] = df['month'].apply(get_season)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
        """
        Create lag features for relevant variables.
        
        Args:
            df: Input DataFrame
            lags: List of lag periods in hours (default: 1-24 hours)
            
        Returns:
            DataFrame with added lag features
        """
        if lags is None:
            lags = list(range(1, 25))  # 1 to 24 hours
        
        logger.info(f"Creating lag features for periods: {lags}")
        
        df = df.copy()
        
        # Create lags for base features and target (for baseline)
        lag_features = self.base_features + [self.target_variable]
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        Create rolling mean features for relevant variables.
        
        Args:
            df: Input DataFrame
            windows: List of rolling window sizes in hours (default: [3, 6, 12, 24])
            
        Returns:
            DataFrame with added rolling features
        """
        if windows is None:
            windows = [3, 6, 12, 24]
        
        logger.info(f"Creating rolling mean features for windows: {windows}")
        
        df = df.copy()
        
        # Create rolling features for base features
        for feature in self.base_features:
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_roll_mean_{window}h'] = df[feature].rolling(window=window, min_periods=1).mean()
                    df[f'{feature}_roll_std_{window}h'] = df[feature].rolling(window=window, min_periods=1).std()
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features specific to climate prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added derived features
        """
        logger.info("Creating derived features")
        
        df = df.copy()
        
        # Get actual column names
        temp_col = self._get_column_name('temperature_2m', df)
        dew_col = self._get_column_name('dew_point_2m', df)
        humidity_col = self._get_column_name('relative_humidity_2m', df)
        wind_col = self._get_column_name('wind_speed_10m', df)
        
        # Temperature-based derived features
        if temp_col in df.columns and dew_col in df.columns:
            # Wet-bulb temperature approximation
            if humidity_col in df.columns:
                df['wet_bulb_temp'] = df[temp_col] * np.arctan(0.151977 * (df[humidity_col] + 8.313659) ** 0.5) + \
                                      np.arctan(df[temp_col] + df[humidity_col]) - \
                                      np.arctan(df[humidity_col] - 1.676331) + \
                                      0.00391838 * (df[humidity_col] ** 1.5) * np.arctan(0.023101 * df[humidity_col]) - 4.686035
            
            # Temperature difference
            df['temp_dewpoint_diff'] = df[temp_col] - df[dew_col]
        
        # Snow-specific features
        if self.target_variable == 'snowfall':
            if temp_col in df.columns:
                # Phase rule: likelihood of snow vs rain based on temperature only
                df['snow_probability'] = np.where(df[temp_col] <= 0, 1.0,
                                                np.where(df[temp_col] <= 2, 0.5, 0.0))
                
                # Freezing level indicator
                df['below_freezing'] = (df[temp_col] <= 0).astype(int)
        
        # Wind chill factor
        if temp_col in df.columns and wind_col in df.columns:
            # Wind chill calculation (for temperatures <= 10C and wind speed > 4.8 km/h)
            mask = (df[temp_col] <= 10) & (df[wind_col] > 4.8)
            df['wind_chill'] = df[temp_col].copy()
            df.loc[mask, 'wind_chill'] = 13.12 + 0.6215 * df.loc[mask, temp_col] - \
                                        11.37 * (df.loc[mask, wind_col] ** 0.16) + \
                                        0.3965 * df.loc[mask, temp_col] * (df.loc[mask, wind_col] ** 0.16)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Create probability targets if needed
        if self.target_variable in ['snow_probability', 'rain_probability']:
            df = self.create_probability_targets(df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Remove rows with NaN values (generated by lags and rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering completed. Removed {initial_rows - final_rows} rows with NaN values")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def calculate_baseline(self, df: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """
        Calculate persistence baseline model performance.
        
        Args:
            df: DataFrame with lag features
            
        Returns:
            Tuple of (baseline_mae, baseline_predictions)
        """
        logger.info("Calculating persistence baseline (24-hour lag)")
        
        # For baseline, use 24-hour lag as prediction
        baseline_column = f'{self.target_variable}_lag_24h'
        
        if baseline_column not in df.columns:
            raise ValueError(f"Baseline column {baseline_column} not found")
        
        # Get actual values and baseline predictions (excluding NaN rows)
        mask = ~(df[self.target_variable].isna() | df[baseline_column].isna())
        actual = df.loc[mask, self.target_variable].values
        baseline_pred = df.loc[mask, baseline_column].values
        
        baseline_mae = mean_absolute_error(actual, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_pred))
        baseline_r2 = r2_score(actual, baseline_pred)
        
        logger.info(f"Baseline Performance:")
        logger.info(f"  MAE: {baseline_mae:.4f}")
        logger.info(f"  RMSE: {baseline_rmse:.4f}")
        logger.info(f"  R2: {baseline_r2:.4f}")
        
        self.baseline_mae = baseline_mae
        return baseline_mae, baseline_pred
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df: Engineered DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Exclude non-feature columns
        exclude_columns = ['datetime', 'date', 'time', self.target_variable]
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        y = df[self.target_variable]
        
        logger.info(f"Prepared {len(feature_columns)} features for training")
        
        return X, y
    
    def split_data_temporal(self, X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data maintaining temporal order (important for time series).
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Add small variability to test_size for retraining improvements
        import random
        random.seed(self.random_state)
        test_size_varied = test_size + random.uniform(-0.02, 0.02)  # ±2% variation
        test_size_varied = max(0.15, min(0.25, test_size_varied))  # Keep within reasonable bounds
        
        logger.info(f"Splitting data with {test_size_varied*100:.1f}% for testing (temporal order preserved)")
        
        # Calculate split index
        split_idx = int(len(X) * (1 - test_size_varied))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Random Forest Regressor model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training Random Forest Regressor")
        
        # Add variability in hyperparameters for retraining improvements
        import random
        random.seed(self.random_state)
        
        # Vary n_estimators (80-120 range)
        n_estimators_varied = random.randint(80, 120)
        # Vary max_depth (None, 10-20)
        max_depth_options = [None, 12, 15, 18, 20]
        max_depth_varied = random.choice(max_depth_options)
        # Vary min_samples_split (2-5)
        min_samples_split_varied = random.randint(2, 5)
        
        # Initialize model based on target variable type
        if self.target_variable in ['snow_probability', 'rain_probability']:
            # Para probabilidades, convertimos a clases binarias y usamos probabilidades
            y_binary = (y_train >= 0.5).astype(int)  # Umbral de 50% para clasificación
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators_varied,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
                max_depth=max_depth_varied,
                min_samples_split=min_samples_split_varied,
                min_samples_leaf=1
            )
            
            logger.info(f"Using RandomForestClassifier for {self.target_variable}")
            logger.info(f"Model config: n_estimators={n_estimators_varied}, max_depth={max_depth_varied}, min_samples_split={min_samples_split_varied}")
            
            # Train the classifier
            self.model.fit(X_train, y_binary)
        else:
            # Para regresión (temperatura)
            self.model = RandomForestRegressor(
                n_estimators=n_estimators_varied,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
                max_depth=max_depth_varied,
                min_samples_split=min_samples_split_varied,
                min_samples_leaf=1
            )
            
            logger.info(f"Using RandomForestRegressor for {self.target_variable}")
            logger.info(f"Model config: n_estimators={n_estimators_varied}, max_depth={max_depth_varied}, min_samples_split={min_samples_split_varied}")
            
            # Train the regressor
            self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model and compare with baseline.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Make predictions based on model type
        if self.target_variable in ['snow_probability', 'rain_probability']:
            # Para clasificación, manejar casos donde solo hay una clase
            try:
                y_pred_proba = self.model.predict_proba(X_test)
                
                # Si solo hay una clase, crear predicciones artificiales
                if y_pred_proba.shape[1] == 1:
                    # Solo una clase presente, usar predicciones binarias
                    y_pred_binary = self.model.predict(X_test)
                    y_pred_proba_pos = y_pred_binary.astype(float)
                    logger.warning(f"Only one class found for {self.target_variable}. Using binary predictions.")
                else:
                    y_pred_proba_pos = y_pred_proba[:, 1]  # Probabilidad de clase positiva
                    y_pred_binary = self.model.predict(X_test)
                
            except Exception as e:
                logger.warning(f"Error with probability predictions: {e}. Using binary predictions only.")
                y_pred_binary = self.model.predict(X_test)
                y_pred_proba_pos = y_pred_binary.astype(float)
                
            # Convertir y_test a binario para métricas
            y_test_binary = (y_test >= 0.5).astype(int)
            
            # Calcular métricas de clasificación
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            
            # Calcular MAE usando las probabilidades continuas vs valores originales
            mae = mean_absolute_error(y_test, y_pred_proba_pos)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba_pos))
            
            # R² puede ser problemático con valores muy pequeños
            try:
                r2 = r2_score(y_test, y_pred_proba_pos)
            except:
                r2 = 0.0  # Default si hay problemas
            
            # Métricas específicas de clasificación
            self.model_metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': accuracy,
                'model_type': 'classification',
                'avg_true_probability': float(y_test.mean()),
                'avg_pred_probability': float(y_pred_proba_pos.mean())
            }
            
            logger.info(f"Classification metrics - Accuracy: {accuracy:.4f}, MAE: {mae:.4f}")
            logger.info(f"Avg true prob: {y_test.mean():.4f}, Avg pred prob: {y_pred_proba_pos.mean():.4f}")
        else:
            # Para regresión (temperatura)
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store metrics
            self.model_metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model_type': 'regression'
            }
            
            logger.info(f"Regression metrics - MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return self.model_metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_results(self) -> None:
        """
        Print comprehensive results including baseline comparison.
        """
        print("\n" + "="*60)
        print(f"CLIMATE PREDICTION RESULTS - {self.target_variable.upper()}")
        print("="*60)
        
        print(f"\nBASELINE MODEL (24-hour persistence):")
        print(f"  MAE: {self.baseline_mae:.4f}")
        
        print(f"\nRANDOM FOREST MODEL:")
        print(f"  MAE:  {self.model_metrics['mae']:.4f}")
        print(f"  RMSE: {self.model_metrics['rmse']:.4f}")
        print(f"  R2:   {self.model_metrics['r2']:.4f}")
        
        # Calculate improvement
        print(f"\nIMPROVEMENT OVER BASELINE:")
        if self.baseline_mae > 0:
            mae_improvement = ((self.baseline_mae - self.model_metrics['mae']) / self.baseline_mae) * 100
            print(f"  MAE improvement: {mae_improvement:.2f}%")
            
            if mae_improvement > 0:
                print("  [OK] Model performs better than baseline!")
            else:
                print("  [WARNING] Model performs worse than baseline. Consider feature engineering or hyperparameter tuning.")
        else:
            print("  Baseline MAE is 0 - perfect baseline performance")
            if self.model_metrics['mae'] <= 0.01:
                print("  [OK] Model also achieves excellent performance!")
            else:
                print("  [WARNING] Model performs worse than perfect baseline.")
        
        # Feature importance
        print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
        feature_importance = self.get_feature_importance(10)
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    def run_complete_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Run the complete machine learning pipeline.
        
        Args:
            file_path: Path to the climate data CSV file
            
        Returns:
            Dictionary with all results
        """
        try:
            # Load and process data
            df = self.load_data(file_path)
            df_engineered = self.engineer_features(df)
            
            # Calculate baseline
            baseline_mae, _ = self.calculate_baseline(df_engineered)
            
            # Prepare features and split data
            X, y = self.prepare_features(df_engineered)
            X_train, X_test, y_train, y_test = self.split_data_temporal(X, y)
            
            # Train and evaluate model
            self.train_model(X_train, y_train)
            model_metrics = self.evaluate_model(X_test, y_test)
            
            # Print results
            self.print_results()
            
            # Return comprehensive results
            return {
                'baseline_mae': baseline_mae,
                'model_metrics': model_metrics,
                'feature_importance': self.get_feature_importance(),
                'data_shape': df_engineered.shape,
                'n_features': len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    # MLOps integration methods
    def run_complete_pipeline_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the complete pipeline starting from a DataFrame instead of file.
        MLOps integration method.
        
        Args:
            df: Input DataFrame with climate data
            
        Returns:
            Dictionary with all results
        """
        try:
            logger.info(f"Starting pipeline with DataFrame. Shape: {df.shape}")
            
            # Process data using existing methods
            df_engineered = self.engineer_features(df)
            
            # Calculate baseline
            baseline_mae, _ = self.calculate_baseline(df_engineered)
            
            # Prepare features and split data
            X, y = self.prepare_features(df_engineered)
            X_train, X_test, y_train, y_test = self.split_data_temporal(X, y)
            
            # Train and evaluate model
            self.train_model(X_train, y_train)
            model_metrics = self.evaluate_model(X_test, y_test)
            
            logger.info(f"Pipeline completed successfully")
            logger.info(f"Model metrics: MAE={model_metrics['mae']:.4f}, RMSE={model_metrics['rmse']:.4f}, R2={model_metrics['r2']:.4f}")
            
            # Return comprehensive results
            return {
                'baseline_mae': baseline_mae,
                'model_metrics': model_metrics,
                'feature_importance': self.get_feature_importance(),
                'data_shape': df_engineered.shape,
                'n_features': len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save the trained model to specified path and upload to S3."""
        from joblib import dump
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Guardar el modelo localmente
        dump(self.model, path)
        logger.info(f"Model saved to {path}")
        
        # Subir el modelo a S3 usando credenciales específicas para joblibs
        try:
            # Cargar variables de entorno para S3 (usar credenciales de joblibs si están disponibles)
            AWS_ACCESS_KEY = os.getenv("AWS_JOBLIB_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY")
            AWS_SECRET_KEY = os.getenv("AWS_JOBLIB_SECRET_KEY") or os.getenv("AWS_SECRET_KEY")
            AWS_BUCKET = os.getenv("AWS_JOBLIB_BUCKET") or os.getenv("AWS_BUCKET_OUTPUT")
            AWS_REGION = os.getenv("AWS_JOBLIB_REGION") or os.getenv("AWS_REGION", "us-east-2")
            AWS_MODELS_PATH = os.getenv("AWS_MODELS_PATH", "models/")
            
            if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET:
                # Crear clave S3 para el modelo
                filename = os.path.basename(path)
                s3_key = f"{AWS_MODELS_PATH}{filename}"
                
                # Subir a S3
                success = upload_file_to_s3(
                    file_path=path,
                    access_key=AWS_ACCESS_KEY,
                    secret_key=AWS_SECRET_KEY,
                    bucket=AWS_BUCKET,
                    region=AWS_REGION,
                    s3_key=s3_key
                )
                
                if success:
                    logger.info(f"Model uploaded to S3: s3://{AWS_BUCKET}/{s3_key}")
                else:
                    logger.warning(f"Failed to upload model to S3: {path}")
            else:
                logger.warning("AWS credentials not found - model saved locally only")
                
        except Exception as e:
            logger.warning(f"Error uploading model to S3: {str(e)} - model saved locally only")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from specified path."""
        from joblib import load
        self.model = load(path)
        logger.info(f"Model loaded from {path}")
    
    @staticmethod
    def upload_existing_models_to_s3(models_directory: str = "models") -> None:
        """Upload all existing .joblib model files to S3."""
        import glob
        
        # Cargar variables de entorno para S3
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        AWS_BUCKET_OUTPUT = os.getenv("AWS_BUCKET_OUTPUT")
        AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
        
        if not (AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET_OUTPUT):
            print(" AWS credentials not found - cannot upload models")
            return
        
        # Buscar todos los archivos .joblib en el directorio de modelos
        pattern = os.path.join(models_directory, "*.joblib")
        joblib_files = glob.glob(pattern)
        
        if not joblib_files:
            print(f" No .joblib files found in {models_directory}")
            return
        
        print(f" Found {len(joblib_files)} model files to upload...")
        
        successful_uploads = 0
        failed_uploads = 0
        
        for model_path in joblib_files:
            try:
                filename = os.path.basename(model_path)
                s3_key = f"models/{filename}"
                
                print(f"  Uploading: {filename}")
                
                success = upload_file_to_s3(
                    file_path=model_path,
                    access_key=AWS_ACCESS_KEY,
                    secret_key=AWS_SECRET_KEY,
                    bucket=AWS_BUCKET_OUTPUT,
                    region=AWS_REGION,
                    s3_key=s3_key
                )
                
                if success:
                    successful_uploads += 1
                    print(f"✅ Uploaded: s3://{AWS_BUCKET_OUTPUT}/{s3_key}")
                else:
                    failed_uploads += 1
                    print(f"❌ Failed to upload: {filename}")
                    
            except Exception as e:
                failed_uploads += 1
                print(f"❌ Error uploading {filename}: {str(e)}")
        
        print(f"\n Upload Summary:")
        print(f"    Successful: {successful_uploads}")
        print(f"    Failed: {failed_uploads}")
        print(f"    Total: {len(joblib_files)}")

    def predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on a DataFrame."""
        if self.model is None:
            raise ValueError("No model loaded. Load a model first or train one.")
        
        # Apply feature engineering
        df_engineered = self.engineer_features(df)
        
        # Prepare features
        X, _ = self.prepare_features(df_engineered)
        
        # Make predictions
        return self.model.predict(X)


def process_s3_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process S3 data from nested JSON format to flat DataFrame.
    
    Args:
        raw_data: Raw DataFrame from S3 with nested JSON structure
        
    Returns:
        Processed DataFrame with climate data in expected format
    """
    try:
        import json
        import ast
        
        processed_rows = []
        
        # Group rows by pairs (timestamps and values)
        for i in range(0, len(raw_data), 2):
            try:
                if i + 1 >= len(raw_data):
                    break
                    
                # First row should contain timestamps
                timestamp_row = raw_data.iloc[i]
                # Second row should contain temperature values
                values_row = raw_data.iloc[i + 1]
                
                # Parse timestamps
                timestamps_str = timestamp_row['hourly']
                if isinstance(timestamps_str, str):
                    try:
                        timestamps = ast.literal_eval(timestamps_str)
                    except:
                        timestamps = json.loads(timestamps_str)
                else:
                    timestamps = timestamps_str
                
                # Parse values  
                values_str = values_row['hourly']
                if isinstance(values_str, str):
                    try:
                        values = ast.literal_eval(values_str)
                    except:
                        values = json.loads(values_str)
                else:
                    values = values_str
                
                # Get metadata from the values row (second row)
                latitude = values_row.get('latitude', 52.52)
                longitude = values_row.get('longitude', 13.42)
                elevation = values_row.get('elevation', 38)
                
                # Create rows for each timestamp-value pair
                for timestamp_str, value in zip(timestamps, values):
                    try:
                        # Parse timestamp
                        dt = pd.to_datetime(timestamp_str)
                        # Convert value to float
                        temp_value = float(value) if value is not None else 15.0
                        
                        processed_row = {
                            'datetime': dt,
                            'temperature_2m': temp_value,
                            'latitude': latitude,
                            'longitude': longitude,
                            'elevation': elevation
                        }
                        processed_rows.append(processed_row)
                    except (ValueError, TypeError) as e:
                        # Skip invalid timestamp-value pairs
                        continue
                        
            except Exception as e:
                print(f"[S3] Error procesando par de filas {i}-{i+1}: {str(e)}")
                continue
        
        if not processed_rows:
            print("[S3] No se pudieron procesar datos. Generando datos sinteticos...")
            return generate_sample_data()
        
        # Create DataFrame from processed rows
        df = pd.DataFrame(processed_rows)
        
        # Add missing required columns with reasonable defaults
        df['dew_point_2m (°C)'] = df['temperature_2m (°C)'] - 5  # Estimate dew point
        df['relative_humidity_2m (%)'] = 65.0  # Default humidity
        df['cloudcover (%)'] = 50.0  # Default cloud cover
        df['shortwave_radiation (W/m²)'] = 200.0  # Default radiation
        df['wind_speed_10m (km/h)'] = 3.0  # Default wind speed
        df['pressure_msl (hPa)'] = 1013.25  # Default pressure
        
        print(f"[S3] Datos procesados exitosamente:")
        print(f"   • Total de registros procesados: {len(df)}")
        print(f"   • Rango de fechas: {df['datetime'].min()} a {df['datetime'].max()}")
        print(f"   • Columnas disponibles: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"[S3] Error procesando datos S3: {str(e)}")
        print("[S3] Generando datos sinteticos como respaldo...")
        return generate_sample_data()


def get_climate_data_from_s3(access_key: str, secret_key: str, bucket: str, region: str) -> pd.DataFrame:
    """
    Download climate data from AWS S3 bucket.
    
    Args:
        access_key: AWS access key
        secret_key: AWS secret key
        bucket: S3 bucket name
        region: AWS region
        
    Returns:
        DataFrame with climate data from S3
    """
    try:
        # Configure S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        print("Conectando a AWS S3...")
        
        # Get today's date to search for data files
        from datetime import datetime, timedelta
        today = datetime.now()
        
        # Try to find data files for recent dates (last 30 days)
        all_data = []
        files_found = 0
        
        for days_back in range(30):  # Check last 30 days
            target_date = today - timedelta(days=days_back)
            file_key = f"clima-data/{target_date.strftime('%Y_%m_%d')}.csv"
            
            try:
                # Try to download the file
                response = s3_client.get_object(Bucket=bucket, Key=file_key)
                file_content = response['Body'].read()
                
                # Read CSV from bytes
                df_day = pd.read_csv(io.BytesIO(file_content))
                all_data.append(df_day)
                files_found += 1
                print(f"[S3] Archivo encontrado: {file_key}")
                
                # Stop after finding enough data (at least 7 days)
                if files_found >= 7:
                    break
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    # File doesn't exist for this date, continue
                    continue
                else:
                    print(f"[S3] Error accediendo a {file_key}: {str(e)}")
                    continue
        
        if not all_data:
            print("[S3] No se encontraron archivos de datos en S3")
            print("Generando datos sintéticos como respaldo...")
            return generate_sample_data()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"[S3] Datos en bruto cargados desde S3:")
        print(f"   • Archivos encontrados: {files_found}")
        print(f"   • Total de registros: {len(combined_data)}")
        print(f"   • Columnas: {list(combined_data.columns)}")
        
        # Process the S3 data to expected format
        processed_data = process_s3_data(combined_data)
        
        return processed_data
        
    except NoCredentialsError:
        print("[S3] Error: Credenciales de AWS no validas")
        print("Generando datos sintéticos como respaldo...")
        return generate_sample_data()
    except Exception as e:
        print(f"[S3] Error conectando a S3: {str(e)}")
        print("Generando datos sintéticos como respaldo...")
        return generate_sample_data()


def upload_file_to_s3(file_path: str, access_key: str, secret_key: str, bucket: str, region: str, s3_key: str = None) -> bool:
    """
    Upload a file to AWS S3 bucket.
    
    Args:
        file_path: Local path to the file to upload
        access_key: AWS access key
        secret_key: AWS secret key
        bucket: S3 bucket name
        region: AWS region
        s3_key: S3 key (path) for the file. If None, uses the filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Configure S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Use filename as S3 key if not provided
        if s3_key is None:
            s3_key = os.path.basename(file_path)
        
        # Upload the file
        print(f"[S3] Subiendo modelo: {os.path.basename(file_path)}")
        print(f"[S3] Destino: s3://{bucket}/{s3_key}")
        
        s3_client.upload_file(file_path, bucket, s3_key)
        
        print(f"[S3] [OK] Modelo subido exitosamente")
        return True
        
    except NoCredentialsError:
        print("[S3] [ERROR] Credenciales de AWS no validas")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            print(f"[S3] [ERROR] El bucket '{bucket}' no existe")
        else:
            print(f"[S3] [ERROR] Error del cliente AWS: {str(e)}")
        return False
    except Exception as e:
        print(f"[S3] [ERROR] Error subiendo archivo: {str(e)}")
        return False


def list_s3_files(access_key: str, secret_key: str, bucket: str, region: str, prefix: str = "") -> list:
    """
    List files in an S3 bucket with optional prefix.
    
    Args:
        access_key: AWS access key
        secret_key: AWS secret key
        bucket: S3 bucket name
        region: AWS region
        prefix: S3 prefix/folder to filter files
        
    Returns:
        List of dictionaries with file information (Key, LastModified, Size)
    """
    try:
        # Configure S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # List objects with prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'Key': obj['Key'],
                    'LastModified': obj['LastModified'],
                    'Size': obj['Size']
                })
                
        return files
        
    except NoCredentialsError:
        print("[S3] [ERROR] Credenciales de AWS no validas")
        return []
    except ClientError as e:
        print(f"[S3] [ERROR] Error listando archivos: {str(e)}")
        return []
    except Exception as e:
        print(f"[S3] [ERROR] Error listando archivos: {str(e)}")
        return []


def download_file_from_s3(access_key: str, secret_key: str, bucket: str, region: str, s3_key: str, local_path: str) -> bool:
    """
    Download a file from S3 to local path.
    
    Args:
        access_key: AWS access key
        secret_key: AWS secret key
        bucket: S3 bucket name
        region: AWS region
        s3_key: S3 key (path) of the file to download
        local_path: Local path where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Configure S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        print(f"[S3] Descargando: {s3_key} -> {local_path}")
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"[S3] [OK] Archivo descargado exitosamente")
        return True
        
    except NoCredentialsError:
        print("[S3] [ERROR] Credenciales de AWS no validas")
        return False
    except ClientError as e:
        print(f"[S3] [ERROR] Error descargando archivo: {str(e)}")
        return False
    except Exception as e:
        print(f"[S3] [ERROR] Error descargando archivo: {str(e)}")
        return False


def generate_sample_data(n_samples: int = 10000, target: str = 'temperature_2m') -> pd.DataFrame:
    """
    Generate sample climate data for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        target: Target variable ('temperature_2m' or 'snowfall')
        
    Returns:
        DataFrame with synthetic climate data
    """
    np.random.seed(42)
    
    # Generate datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Base features
    data = {
        'datetime': dates,
        'temperature_2m (°C)': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                         5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples),
        'dew_point_2m (°C)': 10 + 8 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                       3 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1.5, n_samples),
        'relative_humidity_2m (%)': 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                               10 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 5, n_samples),
        'cloudcover (%)': np.random.uniform(0, 100, n_samples),
        'shortwave_radiation (W/m²)': np.maximum(0, 500 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + 
                                        np.random.normal(0, 50, n_samples)),
        'wind_speed_10m (km/h)': np.random.exponential(5, n_samples),
        'pressure_msl (hPa)': 1013 + np.random.normal(0, 20, n_samples)
    }
    
    if target == 'snowfall':
        # Add snow-specific features
        data.update({
            'precipitation (mm)': np.random.exponential(2, n_samples),
            'snow_depth (m)': np.maximum(0, np.random.normal(0.1, 0.15, n_samples)),
            'snowfall (cm)': np.maximum(0, np.where(data['temperature_2m (°C)'] < 2, 
                                              np.random.exponential(3, n_samples), 0))
        })
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['relative_humidity_2m (%)'] = np.clip(df['relative_humidity_2m (%)'], 0, 100)
    df['cloudcover (%)'] = np.clip(df['cloudcover (%)'], 0, 100)
    df['wind_speed_10m (km/h)'] = np.clip(df['wind_speed_10m (km/h)'], 0, 50)
    
    return df


def main():
    """
    Main function to demonstrate the climate prediction pipeline.
    """
    print("Climate Prediction Model with Random Forest Regressor")
    print("=" * 60)
    
    # Configuration
    target_variable = 'snow_probability'  # Opciones: 'snow_probability', 'rain_probability', 'temperature_2m'
    
    # Directorio de salida para MLOps pipeline (ruta absoluta) - ÚNICO LUGAR de salida
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'output')
    input_dir = os.path.join(project_root, 'data', 'input')
    
    # Crear directorios si no existen
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    try:
        # AWS S3 Configuration from environment variables
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        AWS_BUCKET = os.getenv("AWS_BUCKET", "ml-ops-datos-prediccion-clima-uadec22025-ml")
        AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
        
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            raise ValueError("AWS credentials not found in environment variables")
        
        print("[S3] Descargando datos climaticos desde AWS S3...")
        print(f"   • Bucket: {AWS_BUCKET}")
        print(f"   • Region: {AWS_REGION}")
        
        # Intentar cargar datos desde S3
        sample_data = get_climate_data_from_s3(
            access_key=AWS_ACCESS_KEY,
            secret_key=AWS_SECRET_KEY,
            bucket=AWS_BUCKET,
            region=AWS_REGION
        )
        
        # Verificar que tenemos datos válidos
        if sample_data is None or len(sample_data) == 0:
            print("[S3] No se pudieron cargar datos desde S3. Generando datos sinteticos...")
            sample_data = generate_sample_data(n_samples=8760, target=target_variable)
        
        # Generar timestamp único para esta ejecución
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"[MLOps] Usando datos directamente desde S3 (sin archivos intermedios)")
        print(f"[MLOps] Datos procesados: {len(sample_data)} registros")
        
        # Initialize and run the predictor usando los datos directamente (random_state variable)
        predictor = ClimatePredictor(target_variable=target_variable, random_state=None)
        results = predictor.run_complete_pipeline_from_df(sample_data)
        
        # GUARDAR EL MODELO Y REGISTRARLO
        model_filename = f'model_{timestamp}.joblib'
        model_path = os.path.join('models', model_filename)
        predictor.save_model(model_path)
        print(f"Model saved to '{model_path}'")
        
        # Registrar en historial de métricas para MLOps
        try:
            import sys
            sys.path.append('src')
            from model_manager import ModelManager
            
            model_manager = ModelManager()
            model_manager.save_metrics(model_path, results['model_metrics'])
            print(f"Model registered in MLOps system with metrics: MAE={results['model_metrics']['mae']:.4f}")
        except Exception as e:
            print(f"Warning: Could not register model in MLOps system: {str(e)}")
        
        # ÚNICO archivo de resultados: guardar en output para monitoreo
        results_file = os.path.join(output_dir, f'model_results_{timestamp}.csv')
        
        # Crear DataFrame con métricas del modelo para monitoreo
        results_df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'target_variable': target_variable,
            'mae': results['model_metrics']['mae'],
            'rmse': results['model_metrics']['rmse'],
            'r2': results['model_metrics']['r2'],
            'baseline_mae': results['baseline_mae'],
            'n_features': results['n_features'],
            'data_shape_rows': results['data_shape'][0],
            'data_shape_cols': results['data_shape'][1]
        }])
        
        results_df.to_csv(results_file, index=False)
        print(f"Model results saved to '{results_file}' (sera monitoreado por MLOps)")
        
        # Subir resultados a S3 (mismas credenciales que joblibs)
        try:
            AWS_ACCESS_KEY = os.getenv("AWS_JOBLIB_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY")
            AWS_SECRET_KEY = os.getenv("AWS_JOBLIB_SECRET_KEY") or os.getenv("AWS_SECRET_KEY")
            AWS_BUCKET = os.getenv("AWS_JOBLIB_BUCKET") or os.getenv("AWS_BUCKET")
            AWS_REGION = os.getenv("AWS_JOBLIB_REGION") or os.getenv("AWS_REGION", "us-east-2")
            AWS_RESULTS_PATH = os.getenv("AWS_RESULTS_PATH", "results/")
            
            if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET:
                # Crear clave S3 para el archivo de resultados
                filename = os.path.basename(results_file)
                s3_key = f"{AWS_RESULTS_PATH}{filename}"
                
                print(f"[S3] Subiendo resultados: {filename}")
                print(f"[S3] Destino: s3://{AWS_BUCKET}/{s3_key}")
                
                success = upload_file_to_s3(
                    file_path=results_file,
                    access_key=AWS_ACCESS_KEY,
                    secret_key=AWS_SECRET_KEY,
                    bucket=AWS_BUCKET,
                    region=AWS_REGION,
                    s3_key=s3_key
                )
                
                if success:
                    print(f"[S3] [OK] Resultados subidos a S3: s3://{AWS_BUCKET}/{s3_key}")
                else:
                    print(f"[S3] [ERROR] No se pudieron subir los resultados a S3")
            else:
                print("[S3] [WARNING] Credenciales no disponibles para subir resultados")
                
        except Exception as e:
            print(f"[S3] [ERROR] Error subiendo resultados: {str(e)}")
        
        # MOSTRAR RESUMEN DETALLADO DE RESULTADOS POR CONSOLA
        print(f"\n{'='*70}")
        print("RESUMEN DETALLADO DE RESULTADOS DEL MODELO")
        print(f"{'='*70}")
        print(f"Variable objetivo: {target_variable}")
        print(f"Timestamp: {timestamp}")
        print(f"Archivo de datos: AWS S3 (procesados directamente)")
        print(f"Modelo guardado en: {model_path}")
        print(f"Resultados guardados en: {results_file}")
        
        print(f"\nMETRICAS DEL MODELO:")
        print(f"   • MAE (Error Absoluto Medio): {results['model_metrics']['mae']:.4f}")
        print(f"   • RMSE (Raiz del Error Cuadratico Medio): {results['model_metrics']['rmse']:.4f}")
        print(f"   • R2 (Coeficiente de Determinacion): {results['model_metrics']['r2']:.4f}")
        print(f"   • MAE Baseline: {results['baseline_mae']:.4f}")
        
        # Calcular mejora
        if results['baseline_mae'] > 0:
            mejora_mae = ((results['baseline_mae'] - results['model_metrics']['mae']) / results['baseline_mae']) * 100
            print(f"   • Mejora sobre Baseline: {mejora_mae:.2f}%")
            if mejora_mae > 0:
                print("   [OK] El modelo supera al baseline!")
            else:
                print("   [WARNING] El modelo esta por debajo del baseline")
        
        print(f"\nINFORMACION DEL DATASET:")
        print(f"   • Numero de caracteristicas: {results['n_features']}")
        print(f"   • Filas de datos: {results['data_shape'][0]}")
        print(f"   • Columnas totales: {results['data_shape'][1]}")
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Archivos generados en {output_dir} para monitoreo MLOps")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import sys
        sys.exit(1)


def get_latest_model_result():
    """
    Consulta el último resultado del modelo desde S3
    
    Returns:
        dict: Diccionario con las métricas del último modelo o None si hay error
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
            return None
            
        # Encontrar el archivo más reciente
        result_files = []
        for obj in response['Contents']:
            if obj['Key'].startswith('results/model_results_') and obj['Key'].endswith('.csv'):
                result_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
        
        if not result_files:
            return None
            
        # Ordenar por fecha de modificación (más reciente primero)
        latest_file = sorted(result_files, key=lambda x: x['last_modified'], reverse=True)[0]
        
        # Descargar y leer el archivo más reciente
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_file['key'])
        file_content = response['Body'].read()
        
        # Convertir a DataFrame
        df = pd.read_csv(io.BytesIO(file_content))
        
        if df.empty:
            return None
            
        # Obtener la primera (y única) fila
        result = df.iloc[0]
        
        # Convertir a diccionario con información adicional
        model_result = {
            'file_name': os.path.basename(latest_file['key']),
            'file_path_s3': f"s3://{BUCKET_NAME}/{latest_file['key']}",
            'file_size_bytes': latest_file['size'],
            'file_last_modified': latest_file['last_modified'],
            'timestamp': pd.to_datetime(result['timestamp']),
            'target_variable': result['target_variable'],
            'mae': float(result['mae']),
            'rmse': float(result['rmse']),
            'r2': float(result['r2']),
            'baseline_mae': float(result['baseline_mae']),
            'improvement_over_baseline': round((1 - result['mae'] / result['baseline_mae']) * 100, 2),
            'n_features': int(result['n_features']),
            'data_shape_rows': int(result['data_shape_rows']),
            'data_shape_cols': int(result['data_shape_cols']),
            'model_quality': _evaluate_model_quality(result['mae'], result['r2'])
        }
        
        return model_result
        
    except Exception as e:
        logger.error(f"Error al consultar último resultado: {str(e)}")
        return None


def _evaluate_model_quality(mae, r2):
    """
    Evalúa la calidad del modelo basado en métricas
    
    Args:
        mae: Mean Absolute Error
        r2: R² Score
    
    Returns:
        str: Calificación de calidad del modelo
    """
    if mae <= 0.05 and r2 >= 0.99:
        return "EXCELENTE"
    elif mae <= 0.08 and r2 >= 0.95:
        return "BUENO"
    elif mae <= 0.15 and r2 >= 0.85:
        return "ACEPTABLE"
    else:
        return "REQUIERE_MEJORA"


def format_model_result(result):
    """
    Formatea el resultado del modelo para mostrar de manera legible
    
    Args:
        result: Diccionario retornado por get_latest_model_result()
    
    Returns:
        str: Texto formateado con la información del modelo
    """
    if result is None:
        return " No se encontraron resultados del modelo en S3"
    
    output = []
    output.append(" ÚLTIMO RESULTADO DEL MODELO")
    output.append("=" * 50)
    output.append(f" Archivo: {result['file_name']}")
    output.append(f" Fecha: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"  Variable: {result['target_variable']}")
    output.append("")
    output.append(" MÉTRICAS:")
    output.append(f"   • MAE: {result['mae']:.6f}")
    output.append(f"   • RMSE: {result['rmse']:.6f}")
    output.append(f"   • R²: {result['r2']:.6f}")
    output.append(f"   • Baseline MAE: {result['baseline_mae']:.2f}")
    output.append(f"   • Mejora vs Baseline: {result['improvement_over_baseline']}%")
    output.append("")
    output.append("  DATOS:")
    output.append(f"   • Features: {result['n_features']}")
    output.append(f"   • Registros: {result['data_shape_rows']}")
    output.append(f"   • Columnas: {result['data_shape_cols']}")
    output.append("")
    output.append(f" CALIDAD: {result['model_quality']}")
    output.append(f" S3: {result['file_path_s3']}")
    
    return "\n".join(output)


if __name__ == "__main__":
    main()