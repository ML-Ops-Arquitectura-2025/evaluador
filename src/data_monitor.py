import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
    
    def should_process_file(self, file_path):
        """Determinar si el archivo debe ser procesado por el pipeline MLOps"""
        filename = os.path.basename(file_path)
        # Procesar archivos de resultados del modelo para evaluación MLOps
        if filename.startswith('model_results_'):
            return True
        if filename.startswith('climate_data_') or filename == 'climate_data.csv':
            return True
        # Procesar otros CSV que no sean archivos de configuración
        return not filename.startswith('config_') and filename.endswith('.csv')
    
    def wait_for_file_complete(self, file_path, timeout=10):
        """Esperar a que el archivo esté completamente escrito"""
        import time
        start_time = time.time()
        last_size = 0
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    # El archivo no ha cambiado de tamaño y no está vacío
                    time.sleep(0.5)  # Esperar un poco más para asegurar
                    final_size = os.path.getsize(file_path)
                    if final_size == current_size:
                        return True
                last_size = current_size
                time.sleep(0.5)
            except (OSError, FileNotFoundError):
                time.sleep(0.5)
        
        return False
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            if self.should_process_file(event.src_path):
                print(f"[Watcher] 📁 Archivo creado: {event.src_path}")
                print(f"[Watcher] ⏳ Esperando que el archivo se complete...")
                
                if self.wait_for_file_complete(event.src_path):
                    print(f"[Watcher] ✅ Archivo listo para procesar")
                    self.callback(event.src_path)
                else:
                    print(f"[Watcher] ⚠️ Timeout esperando archivo completo: {event.src_path}")
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            if self.should_process_file(event.src_path):
                print(f"[Watcher] ✏️  Archivo modificado: {event.src_path}")
                print(f"[Watcher] ⏳ Esperando que el archivo se complete...")
                
                if self.wait_for_file_complete(event.src_path):
                    print(f"[Watcher] ✅ Archivo listo para procesar")
                    self.callback(event.src_path)
                else:
                    print(f"[Watcher] ⚠️ Timeout esperando archivo completo: {event.src_path}")

def start_monitoring(watch_directory, callback, check_interval=300):
    logging.info(f"Iniciando monitoreo en {watch_directory}")
    print(f"[Watcher] Escuchando cambios en: {watch_directory}")
    
    # Crear handler para reutilizar la lógica de filtrado
    event_handler = DataChangeHandler(callback)
    
    # Procesar archivos existentes al iniciar
    print(f"[Watcher] Verificando archivos existentes...")
    if os.path.exists(watch_directory):
        for filename in os.listdir(watch_directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(watch_directory, filename)
                if event_handler.should_process_file(file_path):
                    print(f"[Watcher] 📁 Archivo existente encontrado: {file_path}")
                    if event_handler.wait_for_file_complete(file_path):
                        print(f"[Watcher] ✅ Procesando archivo existente")
                        callback(file_path)
                    else:
                        print(f"[Watcher] ⚠️ Archivo no disponible: {file_path}")
        
    # Iniciar monitoreo de nuevos archivos
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    observer.start()
    try:
        while True:
            print("[Watcher] Esperando eventos...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
