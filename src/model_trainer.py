import os
import sys
import pandas as pd
import importlib.util
from joblib import dump
from datetime import datetime
import yaml

# Añadir el directorio models al path para importación directa
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

from model import ClimatePredictor

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.model_dir = 'models'
        self.target_variable = None
        self.model_params = None
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.target_variable = config['target_variable']
        self.model_params = config.get('model_params', None)

    def train_and_save(self, df):
        """
        Entrenar un modelo usando importación directa
        """
        print(f"[TRAINER] 🤖 Iniciando entrenamiento de modelo")
        print(f"[TRAINER] 🎯 Variable objetivo: {self.target_variable}")
        print(f"[TRAINER] 📊 Datos de entrada: {len(df)} filas, {len(df.columns)} columnas")
        
        # Crear instancia del predictor directamente
        print(f"[TRAINER] 🏗️  Creando instancia de ClimatePredictor...")
        predictor = ClimatePredictor(target_variable=self.target_variable, random_state=42)
        print(f"[TRAINER] ✅ Modelo cargado exitosamente")
        
        # Verificar si el método existe
        if hasattr(predictor, 'run_complete_pipeline_from_df'):
            print(f"[TRAINER] 🔄 Ejecutando pipeline con DataFrame directo...")
            results = predictor.run_complete_pipeline_from_df(df)
        else:
            print(f"[TRAINER] 🔄 Ejecutando pipeline con archivo temporal...")
            temp_file = f"temp_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(temp_file, index=False)
            try:
                results = predictor.run_complete_pipeline(temp_file)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"[TRAINER] ✅ Pipeline completado exitosamente")
        print(f"[TRAINER] 📊 Métricas obtenidas:")
        print(f"[TRAINER]    • MAE:  {results['model_metrics']['mae']:.4f}")
        print(f"[TRAINER]    • RMSE: {results['model_metrics']['rmse']:.4f}")
        print(f"[TRAINER]    • R²:   {results['model_metrics']['r2']:.4f}")
        
        # Guardar el modelo entrenado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f"model_{timestamp}.joblib")
        print(f"[TRAINER] 💾 Guardando modelo en: {model_path}")
        predictor.save_model(model_path)
        print(f"[TRAINER] ✅ Modelo guardado exitosamente")
        
        return model_path, results['model_metrics']
