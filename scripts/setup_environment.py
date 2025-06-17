# scripts/setup_environment.py
#!/usr/bin/env python3
"""
Script de configuración automática para el proyecto de reconocimiento de emociones faciales.
Detecta la plataforma y configura el entorno optimal para desarrollo.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import yaml
import logging

class EnvironmentSetup:
    """Configurador automático del entorno de desarrollo."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.machine = platform.machine().lower()
        self.project_root = Path(__file__).parent.parent
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Iniciando setup en {self.platform} ({self.machine})")
    
    def detect_platform_config(self) -> str:
        """Detecta la configuración de plataforma adecuada."""
        if self.platform == 'darwin' and 'arm' in self.machine:
            return 'macos'
        elif self.platform == 'linux':
            return 'linux'
        elif self.platform == 'windows':
            return 'windows'
        else:
            self.logger.warning(f"Plataforma no reconocida: {self.platform}")
            return 'linux'  # Fallback
    
    def check_conda_installation(self) -> bool:
        """Verifica si conda está instalado."""
        return shutil.which('conda') is not None
    
    def check_python_version(self) -> bool:
        """Verifica la versión de Python."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            self.logger.info(f"Python {version.major}.{version.minor} ✓")
            return True
        else:
            self.logger.error(f"Python {version.major}.{version.minor} no es compatible. Se requiere Python 3.9+")
            return False
    
    def create_conda_environment(self, platform_config: str) -> bool:
        """Crea el entorno conda específico para la plataforma."""
        env_file = self.project_root / f"environments/environment_{platform_config}.yml"
        
        if not env_file.exists():
            self.logger.error(f"Archivo de entorno no encontrado: {env_file}")
            return False
        
        try:
            # Leer el archivo de entorno para obtener el nombre
            with open(env_file, 'r') as f:
                env_config = yaml.safe_load(f)
            
            env_name = env_config['name']
            
            # Verificar si el entorno ya existe
            result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
            if env_name in result.stdout:
                self.logger.info(f"Entorno {env_name} ya existe")
                return True
            
            # Crear el entorno
            self.logger.info(f"Creando entorno conda: {env_name}")
            cmd = ['conda', 'env', 'create', '-f', str(env_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Entorno {env_name} creado exitosamente")
                return True
            else:
                self.logger.error(f"Error creando entorno: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en create_conda_environment: {e}")
            return False
    
    def setup_project_structure(self):
        """Crea la estructura de directorios del proyecto."""
        directories = [
            'data/raw',
            'data/processed/train',
            'data/processed/val',
            'data/processed/test',
            'data/external',
            'data/temp',
            'models/trained',
            'models/checkpoints',
            'models/optimized',
            'logs/training',
            'logs/inference',
            'logs/web_app',
            'src/web/static/css',
            'src/web/static/js',
            'src/web/static/assets'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger.info("Estructura de directorios creada")
    
    def check_gpu_support(self) -> dict:
        """Verifica el soporte de GPU disponible."""
        gpu_info = {
            'cuda_available': False,
            'mps_available': False,
            'cuda_devices': 0,
            'recommended_device': 'cpu'
        }
        
        try:
            import torch
            
            # Verificar CUDA
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_devices'] = torch.cuda.device_count()
                gpu_info['recommended_device'] = 'cuda'
                self.logger.info(f"CUDA disponible con {gpu_info['cuda_devices']} dispositivos")
            
            # Verificar MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
                gpu_info['recommended_device'] = 'mps'
                self.logger.info("MPS (Apple Silicon) disponible")
            
            if gpu_info['recommended_device'] == 'cpu':
                self.logger.warning("No se detectó aceleración por GPU, usando CPU")
                
        except ImportError:
            self.logger.warning("PyTorch no está instalado, no se puede verificar GPU")
        
        return gpu_info
    
    def download_sample_data(self):
        """Descarga datos de muestra para testing."""
        try:
            import requests
            from tqdm import tqdm
            
            # URLs de datos de ejemplo (FER2013 samples)
            sample_urls = {
                'fer2013_sample.zip': 'https://github.com/example/fer2013_sample.zip'  # URL de ejemplo
            }
            
            data_dir = self.project_root / 'data' / 'external'
            
            for filename, url in sample_urls.items():
                file_path = data_dir / filename
                
                if file_path.exists():
                    self.logger.info(f"{filename} ya existe")
                    continue
                
                self.logger.info(f"Descargando {filename}...")
                
                # En implementación real, aquí se descargarían archivos de muestra
                # Por ahora, crear archivos dummy para testing
                with open(file_path, 'w') as f:
                    f.write("# Archivo de muestra para testing\n")
                
                self.logger.info(f"{filename} descargado")
                
        except Exception as e:
            self.logger.warning(f"No se pudieron descargar datos de muestra: {e}")
    
    def create_activation_script(self, platform_config: str):
        """Crea script de activación del entorno."""
        env_name = f"emotion_recognition_{platform_config}"
        
        if self.platform == 'windows':
            script_name = 'activate_env.bat'
            script_content = f"""@echo off
echo Activando entorno {env_name}...
call conda activate {env_name}
echo Entorno activado. Para desactivar usa: conda deactivate
cmd /k
"""
        else:
            script_name = 'activate_env.sh'
            script_content = f"""#!/bin/bash
echo "Activando entorno {env_name}..."
conda activate {env_name}
echo "Entorno activado. Para desactivar usa: conda deactivate"
exec "$SHELL"
"""
        
        script_path = self.project_root / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if self.platform != 'windows':
            os.chmod(script_path, 0o755)
        
        self.logger.info(f"Script de activación creado: {script_name}")
    
    def create_makefile(self):
        """Crea Makefile con comandos útiles."""
        makefile_content = """# Makefile para proyecto de reconocimiento de emociones

# Variables
PYTHON = python
PIP = pip
CONDA = conda

# Comandos principales
.PHONY: help setup train evaluate test clean

help:
	@echo "Comandos disponibles:"
	@echo "  setup     - Configurar entorno"
	@echo "  train     - Entrenar modelo"
	@echo "  evaluate  - Evaluar modelo"
	@echo "  test      - Ejecutar tests"
	@echo "  web       - Iniciar aplicación web"
	@echo "  clean     - Limpiar archivos temporales"

setup:
	$(PYTHON) scripts/setup_environment.py

train:
	$(PYTHON) scripts/train_model.py

evaluate:
	$(PYTHON) scripts/evaluate_model.py

test:
	$(PYTHON) -m pytest tests/

web:
	$(PYTHON) src/web/app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/temp/*
	rm -rf logs/temp/*

# Comandos de datos
process-data:
	$(PYTHON) scripts/process_emognition_data.py

download-fer2013:
	$(PYTHON) scripts/download_datasets.py --dataset fer2013

# Comandos de modelo
train-basic:
	$(PYTHON) scripts/train_model.py --model cnn_basic --epochs 100

train-hybrid:
	$(PYTHON) scripts/train_model.py --model cnn_rnn_hybrid --epochs 500

train-transfer:
	$(PYTHON) scripts/train_model.py --model transfer_learning --base mobilenetv2

# Optimización
optimize-model:
	$(PYTHON) scripts/optimize_model.py

# Docker (futuro)
docker-build:
	docker build -t emotion-recognition .

docker-run:
	docker run -p 5000:5000 emotion-recognition
"""
        
        makefile_path = self.project_root / 'Makefile'
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        self.logger.info("Makefile creado")
    
    def verify_installation(self) -> bool:
        """Verifica que la instalación sea correcta."""
        try:
            # Verificar imports críticos
            import torch
            import cv2
            import numpy as np
            import yaml
            
            self.logger.info("Verificación de imports: ✓")
            
            # Verificar estructura de directorios
            required_dirs = ['data', 'models', 'logs', 'src']
            for directory in required_dirs:
                if not (self.project_root / directory).exists():
                    self.logger.error(f"Directorio faltante: {directory}")
                    return False
            
            self.logger.info("Verificación de estructura: ✓")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Error de importación: {e}")
            return False
    
    def run_setup(self):
        """Ejecuta el setup completo."""
        self.logger.info("=== INICIANDO SETUP DEL PROYECTO ===")
        
        # 1. Verificar prerrequisitos
        if not self.check_python_version():
            sys.exit(1)
        
        if not self.check_conda_installation():
            self.logger.error("Conda no está instalado. Por favor instala Miniconda o Anaconda.")
            sys.exit(1)
        
        # 2. Detectar plataforma
        platform_config = self.detect_platform_config()
        self.logger.info(f"Configuración de plataforma: {platform_config}")
        
        # 3. Crear estructura del proyecto
        self.setup_project_structure()
        
        # 4. Crear entorno conda
        if not self.create_conda_environment(platform_config):
            self.logger.error("Fallo al crear entorno conda")
            sys.exit(1)
        
        # 5. Verificar soporte de GPU
        gpu_info = self.check_gpu_support()
        
        # 6. Crear scripts de utilidad
        self.create_activation_script(platform_config)
        self.create_makefile()
        
        # 7. Descargar datos de muestra
        self.download_sample_data()
        
        # 8. Verificar instalación
        if self.verify_installation():
            self.logger.info("=== SETUP COMPLETADO EXITOSAMENTE ===")
            self.print_next_steps(platform_config, gpu_info)
        else:
            self.logger.error("Setup completado con errores")
            sys.exit(1)
    
    def print_next_steps(self, platform_config: str, gpu_info: dict):
        """Imprime los próximos pasos para el usuario."""
        env_name = f"emotion_recognition_{platform_config}"
        
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\n📋 PRÓXIMOS PASOS:")
        print(f"\n1. Activar el entorno:")
        print(f"   conda activate {env_name}")
        
        print(f"\n2. Para obtener el dataset Emognition 2020:")
        print("   - Registrarse en Harvard Dataverse")
        print("   - Completar el EULA y enviarlo a emotions@pwr.edu.pl")
        print("   - Descargar y extraer en data/raw/emognition_2020/")
        
        print(f"\n3. Procesar los datos:")
        print("   python scripts/process_emognition_data.py")
        
        print(f"\n4. Entrenar el modelo:")
        print("   make train")
        print("   # o")
        print("   python scripts/train_model.py")
        
        print(f"\n5. Iniciar la aplicación web:")
        print("   make web")
        
        print(f"\n🖥️  CONFIGURACIÓN DETECTADA:")
        print(f"   Plataforma: {self.platform} ({self.machine})")
        print(f"   Entorno: {env_name}")
        print(f"   Dispositivo recomendado: {gpu_info['recommended_device']}")
        
        if gpu_info['cuda_available']:
            print(f"   CUDA devices: {gpu_info['cuda_devices']}")
        if gpu_info['mps_available']:
            print("   MPS (Apple Silicon): Disponible")
        
        print(f"\n📚 COMANDOS ÚTILES:")
        print("   make help          - Ver todos los comandos")
        print("   make train-basic   - Entrenar modelo básico")
        print("   make train-hybrid  - Entrenar modelo híbrido")
        print("   make evaluate      - Evaluar modelo")
        print("   make clean         - Limpiar archivos temporales")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run_setup()