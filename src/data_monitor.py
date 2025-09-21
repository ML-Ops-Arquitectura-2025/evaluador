import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def check_local_directory(watch_directory, callback, max_files=5):
    """
    Función auxiliar para verificar archivos locales sin usar watchdog.
    Usada como respaldo cuando el monitoreo S3 es el principal.
    
    Args:
        watch_directory: Directorio local a verificar
        callback: Función a llamar cuando se encuentren archivos
        max_files: Máximo número de archivos a procesar por verificación
    
    Returns:
        List of processed files
    """
    processed_files = []
    
    if not os.path.exists(watch_directory):
        return processed_files
    
    try:
        # Buscar archivos CSV
        csv_files = [f for f in os.listdir(watch_directory) if f.endswith('.csv')]
        
        if not csv_files:
            return processed_files
            
        # Filtrar archivos relevantes
        relevant_files = []
        for filename in csv_files:
            file_path = os.path.join(watch_directory, filename)
            if should_process_file_static(file_path):
                # Verificar si el archivo fue modificado recientemente (últimos 10 minutos)
                mod_time = os.path.getmtime(file_path)
                if time.time() - mod_time < 600:  # 10 minutos
                    relevant_files.append((filename, file_path, mod_time))
        
        # Ordenar por fecha de modificación (más reciente primero)
        relevant_files.sort(key=lambda x: x[2], reverse=True)
        
        # Procesar archivos (máximo max_files)
        for filename, file_path, mod_time in relevant_files[:max_files]:
            try:
                print(f"[MONITOR LOCAL] Archivo reciente detectado: {filename}")
                callback(file_path)
                processed_files.append(filename)
            except Exception as e:
                print(f"[MONITOR LOCAL] Error procesando {filename}: {str(e)}")
                
    except Exception as e:
        print(f"[MONITOR LOCAL] Error verificando directorio local: {str(e)}")
    
    return processed_files

def should_process_file_static(file_path):
    """Versión estática de should_process_file para uso independiente"""
    filename = os.path.basename(file_path)
    # Procesar archivos de resultados del modelo para evaluación MLOps
    if filename.startswith('model_results_'):
        return True
    if filename.startswith('climate_data_') or filename == 'climate_data.csv':
        return True
    # Procesar otros CSV que no sean archivos de configuración
    return not filename.startswith('config_') and filename.endswith('.csv')

class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
    
    def should_process_file(self, file_path):
        """Determinar si el archivo debe ser procesado por el pipeline MLOps"""
        return should_process_file_static(file_path)
    
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
                print(f"[MONITOR LOCAL] Archivo creado: {os.path.basename(event.src_path)}")
                print(f"[MONITOR LOCAL] Esperando que el archivo se complete...")
                
                if self.wait_for_file_complete(event.src_path):
                    print(f"[MONITOR LOCAL] Archivo listo para procesar")
                    self.callback(event.src_path)
                else:
                    print(f"[MONITOR LOCAL] Timeout esperando archivo completo: {os.path.basename(event.src_path)}")
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            if self.should_process_file(event.src_path):
                print(f"[MONITOR LOCAL] Archivo modificado: {os.path.basename(event.src_path)}")
                print(f"[MONITOR LOCAL] Esperando que el archivo se complete...")
                
                if self.wait_for_file_complete(event.src_path):
                    print(f"[MONITOR LOCAL] Archivo listo para procesar")
                    self.callback(event.src_path)
                else:
                    print(f"[MONITOR LOCAL] Timeout esperando archivo completo: {os.path.basename(event.src_path)}")

def start_monitoring(watch_directory, callback, check_interval=300):
    """
    Inicia monitoreo de archivos locales como respaldo al sistema S3.
    Este monitoreo funciona en paralelo con el monitoreo S3 del main.py.
    """
    logging.info(f"Iniciando monitoreo LOCAL de respaldo en {watch_directory}")
    print(f"[MONITOR LOCAL] Escuchando cambios en: {watch_directory}")
    print(f"[MONITOR LOCAL] Nota: Este es el sistema de respaldo. El monitoreo principal es S3.")
    
    # Crear handler para reutilizar la lógica de filtrado
    event_handler = DataChangeHandler(callback)
    
    # Procesar archivos existentes al iniciar
    print(f"[MONITOR LOCAL] Verificando archivos existentes...")
    if os.path.exists(watch_directory):
        existing_files = [f for f in os.listdir(watch_directory) if f.endswith('.csv')]
        if existing_files:
            print(f"[MONITOR LOCAL] Encontrados {len(existing_files)} archivos CSV existentes")
            for filename in existing_files:
                file_path = os.path.join(watch_directory, filename)
                if event_handler.should_process_file(file_path):
                    print(f"[MONITOR LOCAL] Archivo válido encontrado: {filename}")
                    if event_handler.wait_for_file_complete(file_path):
                        print(f"[MONITOR LOCAL] Procesando archivo existente")
                        callback(file_path)
                    else:
                        print(f"[MONITOR LOCAL] Archivo no disponible: {filename}")
        else:
            print(f"[MONITOR LOCAL] No se encontraron archivos CSV existentes")
    else:
        print(f"[MONITOR LOCAL] Directorio no existe: {watch_directory}")
        print(f"[MONITOR LOCAL] Creando directorio...")
        os.makedirs(watch_directory, exist_ok=True)
        
    # Iniciar monitoreo de nuevos archivos
    print(f"[MONITOR LOCAL] Iniciando vigilancia de archivos nuevos...")
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    observer.start()
    
    try:
        print(f"[MONITOR LOCAL] Monitoreo local activo - Intervalo: {check_interval}s")
        while True:
            print(f"[MONITOR LOCAL] Esperando eventos locales...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print(f"[MONITOR LOCAL] Deteniendo monitoreo local...")
        observer.stop()
    observer.join()
    print(f"[MONITOR LOCAL] Monitoreo local detenido")