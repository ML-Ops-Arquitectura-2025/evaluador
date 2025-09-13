import os
import json
import glob
import shutil
from joblib import load

class ModelManager:
    def __init__(self, model_dir='models', metrics_file='models/metrics_history.json'):
        self.model_dir = model_dir
        self.metrics_file = metrics_file
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump([], f)

    def save_metrics(self, model_path, metrics):
        with open(self.metrics_file, 'r') as f:
            history = json.load(f)
        entry = {'model_path': model_path, 'metrics': metrics}
        history.append(entry)
        with open(self.metrics_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_best_model(self):
        with open(self.metrics_file, 'r') as f:
            history = json.load(f)
        if not history:
            return None
        best = max(history, key=lambda x: x['metrics']['r2'])
        return best['model_path']

    def rollback(self):
        with open(self.metrics_file, 'r') as f:
            history = json.load(f)
        if len(history) < 2:
            return None
        last = history[-2]
        return last['model_path']
