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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import sys
import os

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
    
    def __init__(self, target_variable: str = 'temperature_2m', random_state: int = 42):
        """
        Initialize the ClimatePredictor.
        
        Args:
            target_variable: The climate variable to predict ('temperature_2m' or 'snowfall')
            random_state: Random state for reproducibility
        """
        self.target_variable = target_variable
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.baseline_mae = None
        self.model_metrics = {}
        
        # Define variables for feature engineering based on target
        if target_variable == 'temperature_2m':
            self.base_features = [
                'dew_point_2m', 'relative_humidity_2m', 'cloud_cover',
                'shortwave_radiation', 'wind_speed_10m', 'pressure_msl', 'elevation'
            ]
        elif target_variable == 'snowfall':
            self.base_features = [
                'temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 'cloud_cover',
                'shortwave_radiation', 'wind_speed_10m', 'pressure_msl', 'elevation',
                'precipitation', 'snow_depth', 'freezing_level_height'
            ]
        else:
            raise ValueError(f"Unsupported target variable: {target_variable}")
    
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
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            required_columns = self.base_features + [self.target_variable]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Ensure datetime column exists or can be created
            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
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
        
        logger.info(f"🌐 Cargando datos desde Open-Meteo API")
        logger.info(f"📍 Ubicación: {latitude}°N, {longitude}°E")
        
        try:
            # Obtener datos desde la API usando nuestro cliente
            df = get_climate_data_from_api(latitude, longitude)
            
            logger.info(f"✅ Datos de Open-Meteo cargados exitosamente. Shape: {df.shape}")
            logger.info(f"📅 Rango temporal: {df['datetime'].min()} a {df['datetime'].max()}")
            
            # Guardar opcionalmente a archivo
            if save_to_file:
                df.to_csv(save_to_file, index=False)
                logger.info(f"💾 Datos guardados en: {save_to_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error al cargar datos desde Open-Meteo API: {str(e)}")
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
        
        logger.info(f"📡 Obteniendo y guardando datos de Open-Meteo API en: {output_path}")
        logger.info(f"📍 Ubicación: {latitude}°N, {longitude}°E")
        
        try:
            # Usar la función del cliente API
            saved_path = save_api_data_to_csv(output_path, latitude, longitude)
            
            logger.info(f"✅ Datos de Open-Meteo API guardados exitosamente en: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"❌ Error al obtener y guardar datos de Open-Meteo API: {str(e)}")
            raise
    
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
        
        # Temperature-based derived features
        if 'temperature_2m' in df.columns and 'dew_point_2m' in df.columns:
            # Wet-bulb temperature approximation
            df['wet_bulb_temp'] = df['temperature_2m'] * np.arctan(0.151977 * (df['relative_humidity_2m'] + 8.313659) ** 0.5) + \
                                  np.arctan(df['temperature_2m'] + df['relative_humidity_2m']) - \
                                  np.arctan(df['relative_humidity_2m'] - 1.676331) + \
                                  0.00391838 * (df['relative_humidity_2m'] ** 1.5) * np.arctan(0.023101 * df['relative_humidity_2m']) - 4.686035
            
            # Temperature difference
            df['temp_dewpoint_diff'] = df['temperature_2m'] - df['dew_point_2m']
        
        # Snow-specific features
        if self.target_variable == 'snowfall':
            if 'temperature_2m' in df.columns and 'freezing_level_height' in df.columns:
                # Phase rule: likelihood of snow vs rain
                df['snow_probability'] = np.where(df['temperature_2m'] <= 0, 1.0,
                                                np.where(df['temperature_2m'] <= 2, 0.5, 0.0))
                
                # Freezing level indicator
                df['below_freezing'] = (df['temperature_2m'] <= 0).astype(int)
        
        # Wind chill factor
        if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
            # Wind chill calculation (for temperatures <= 10C and wind speed > 4.8 km/h)
            mask = (df['temperature_2m'] <= 10) & (df['wind_speed_10m'] > 4.8)
            df['wind_chill'] = df['temperature_2m'].copy()
            df.loc[mask, 'wind_chill'] = 13.12 + 0.6215 * df.loc[mask, 'temperature_2m'] - \
                                        11.37 * (df.loc[mask, 'wind_speed_10m'] ** 0.16) + \
                                        0.3965 * df.loc[mask, 'temperature_2m'] * (df.loc[mask, 'wind_speed_10m'] ** 0.16)
        
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
        logger.info(f"Splitting data with {test_size*100}% for testing (temporal order preserved)")
        
        # Calculate split index
        split_idx = int(len(X) * (1 - test_size))
        
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
        
        # Initialize model with basic hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        # Train the model
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
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        self.model_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
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
        """Save the trained model to specified path."""
        from joblib import dump
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from specified path."""
        from joblib import load
        self.model = load(path)
        logger.info(f"Model loaded from {path}")
    
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
        'temperature_2m': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                         5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples),
        'dew_point_2m': 10 + 8 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                       3 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1.5, n_samples),
        'relative_humidity_2m': 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                               10 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 5, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
        'shortwave_radiation': np.maximum(0, 500 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + 
                                        np.random.normal(0, 50, n_samples)),
        'wind_speed_10m': np.random.exponential(5, n_samples),
        'pressure_msl': 1013 + np.random.normal(0, 20, n_samples),
        'elevation': 500 + np.random.normal(0, 100, n_samples)
    }
    
    if target == 'snowfall':
        # Add snow-specific features
        data.update({
            'precipitation': np.random.exponential(2, n_samples),
            'snow_depth': np.maximum(0, np.random.normal(10, 15, n_samples)),
            'freezing_level_height': 2000 + np.random.normal(0, 500, n_samples),
            'snowfall': np.maximum(0, np.where(data['temperature_2m'] < 2, 
                                              np.random.exponential(3, n_samples), 0))
        })
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['relative_humidity_2m'] = np.clip(df['relative_humidity_2m'], 0, 100)
    df['cloud_cover'] = np.clip(df['cloud_cover'], 0, 100)
    df['wind_speed_10m'] = np.clip(df['wind_speed_10m'], 0, 50)
    
    return df


def main():
    """
    Main function to demonstrate the climate prediction pipeline.
    """
    print("Climate Prediction Model with Random Forest Regressor")
    print("=" * 60)
    
    # Configuration
    target_variable = 'temperature_2m'  # Change to 'snowfall' for snow prediction
    
    # Directorio de salida para MLOps pipeline (ruta absoluta) - ÚNICO LUGAR de salida
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'output')
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Generar o cargar datos
        sample_data = None
        
        # Buscar datos existentes en ubicaciones conocidas
        possible_files = ['../climate_data.csv', 'climate_data.csv', '../data/climate_data.csv']
        data_found = False
        
        for possible_file in possible_files:
            if os.path.exists(possible_file):
                print(f"Loading existing data from: {possible_file}")
                sample_data = pd.read_csv(possible_file)
                data_found = True
                break
        
        if not data_found:
            print("No existing data found. Generating new sample data...")
            sample_data = generate_sample_data(n_samples=8760, target=target_variable)
        
        # Generar timestamp único para esta ejecución
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ÚNICO archivo de datos: guardar en output para MLOps
        output_file = os.path.join(output_dir, f'climate_data_{timestamp}.csv')
        sample_data.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}' (sera detectado por el pipeline MLOps)")
        
        # Initialize and run the predictor usando el archivo en output
        predictor = ClimatePredictor(target_variable=target_variable, random_state=42)
        results = predictor.run_complete_pipeline(output_file)
        
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
            model_manager.save_model_with_metrics(model_path, results['model_metrics'])
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
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Archivos generados en {output_dir} para monitoreo MLOps")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()