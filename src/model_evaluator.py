import os
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import logging
from datetime import datetime

class ModelEvaluator:
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = load(model_path)
        self.target_variable = self.config['target_variable']
        self.thresholds = self.config['performance_thresholds']
        self.log_path = os.path.join('logs', 'evaluation.log')
        logging.basicConfig(filename=self.log_path, level=logging.INFO)

    def evaluate(self, df):
        print(f"[EVALUATOR] 🔍 Iniciando evaluación de modelo")
        print(f"[EVALUATOR] 📊 Datos de entrada: {len(df)} filas, {len(df.columns)} columnas")
        
        # Aplicar feature engineering primero (para tener las mismas columnas que en entrenamiento)
        import sys
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        from model import ClimatePredictor
        
        print(f"[EVALUATOR] ⚙️  Aplicando feature engineering...")
        temp_predictor = ClimatePredictor(target_variable=self.target_variable)
        df = temp_predictor.engineer_features(df)
        print(f"[EVALUATOR] ✅ Feature engineering completado: {len(df)} filas, {len(df.columns)} columnas")
        
        print(f"[EVALUATOR] 🎯 Preparando datos para predicción...")
        y_true = df[self.target_variable]
        # Excluir columnas no numéricas y target
        drop_cols = [self.target_variable, 'datetime', 'date']
        X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        print(f"[EVALUATOR] 📈 Realizando predicciones en {len(X)} muestras...")
        y_pred = self.model.predict(X)
        
        print(f"[EVALUATOR] 📊 Calculando métricas...")
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"[EVALUATOR] ✅ Evaluación completada:")
        print(f"[EVALUATOR]    • MAE:  {mae:.4f}")
        print(f"[EVALUATOR]    • RMSE: {rmse:.4f}")
        print(f"[EVALUATOR]    • R²:   {r2:.4f}")
        
        self.log_metrics(metrics)
        return metrics

    def log_metrics(self, metrics):
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}"
        logging.info(log_entry)

    def needs_retraining(self, metrics):
        threshold = self.config['training']['retrain_threshold_mae']
        needs_retrain = metrics['mae'] > threshold
        
        print(f"[EVALUATOR] 🎯 Verificando umbral de reentrenamiento:")
        print(f"[EVALUATOR]    • MAE actual: {metrics['mae']:.4f}")
        print(f"[EVALUATOR]    • Umbral:     {threshold:.4f}")
        print(f"[EVALUATOR]    • Resultado:  {'REENTRENAR' if needs_retrain else 'MANTENER'}")
        
        return needs_retrain
