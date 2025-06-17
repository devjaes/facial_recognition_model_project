# src/config_manager.py
import yaml
import os
import platform
import torch
from typing import Dict, Any, Optional
import logging
from pathlib import Path

class ConfigManager:
    """
    Gestor centralizado de configuraciones parametrizables.
    Maneja automáticamente la detección de plataforma y configuración óptima.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/project_config.yaml"
        self.platform_info = self._detect_platform()
        self.config = self._load_and_process_config()
        
        # Configurar logging
        self._setup_logging()
        
        logging.info(f"ConfigManager inicializado para plataforma: {self.platform_info['platform']}")
        logging.info(f"Dispositivo detectado: {self.config['device']}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detecta la plataforma y hardware disponible."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        platform_info = {
            'platform': system,
            'machine': machine,
            'python_version': platform.python_version(),
            'has_cuda': torch.cuda.is_available(),
            'has_mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # Detectar configuración específica
        if system == 'darwin' and 'arm' in machine:  # macOS Apple Silicon
            platform_info['config_key'] = 'macos_m3'
            platform_info['optimal_device'] = 'mps'
        elif system == 'linux' and platform_info['has_cuda']:
            platform_info['config_key'] = 'linux_cuda'
            platform_info['optimal_device'] = 'cuda'
        elif system == 'windows' and platform_info['has_cuda']:
            platform_info['config_key'] = 'windows_cuda'
            platform_info['optimal_device'] = 'cuda'
        else:
            platform_info['config_key'] = 'cpu_fallback'
            platform_info['optimal_device'] = 'cpu'
        
        return platform_info
    
    def _load_and_process_config(self) -> Dict[str, Any]:
        """Carga y procesa la configuración principal."""
        # Cargar configuración base
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Procesar configuración de emociones
        config = self._process_emotion_config(config)
        
        # Aplicar configuración específica de plataforma
        config = self._apply_platform_config(config)
        
        # Procesar paths
        config = self._process_paths(config)
        
        return config
    
    def _process_emotion_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la configuración de emociones basada en el modo seleccionado."""
        emotion_mode = config['emotions']['mode']
        
        if emotion_mode == 'basic':
            active_emotions = config['emotions']['basic_emotions']
        elif emotion_mode == 'extended':
            active_emotions = config['emotions']['extended_emotions']
        else:
            raise ValueError(f"Modo de emociones no válido: {emotion_mode}")
        
        # Actualizar configuración
        config['emotions']['active_emotions'] = active_emotions
        config['emotions']['num_classes'] = len(active_emotions)
        
        # Crear mapeo de emociones a índices
        config['emotions']['emotion_to_idx'] = {emotion: idx for idx, emotion in enumerate(active_emotions)}
        config['emotions']['idx_to_emotion'] = {idx: emotion for idx, emotion in enumerate(active_emotions)}
        
        return config
    
    def _apply_platform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica configuración específica de la plataforma detectada."""
        platform_key = self.platform_info['config_key']
        platform_config = config['devices'][platform_key]
        
        # Configurar dispositivo
        if config['devices']['auto_detect']:
            config['device'] = self.platform_info['optimal_device']
        else:
            config['device'] = platform_config['device']
        
        # Ajustar batch size según la plataforma
        base_batch_size = config['training']['batch_size']
        multiplier = platform_config['batch_size_multiplier']
        config['training']['effective_batch_size'] = int(base_batch_size * multiplier)
        
        # Configurar mixed precision
        config['training']['mixed_precision'] = platform_config.get('mixed_precision', False)
        
        # Configuraciones específicas para MPS (Apple Silicon)
        if config['device'] == 'mps':
            # MPS tiene algunas limitaciones conocidas
            config['training']['mixed_precision'] = False
            config['model']['cnn_rnn_hybrid']['channels'] = 3  # MPS prefiere RGB
        
        return config
    
    def _process_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa y crea los paths necesarios."""
        base_path = Path(config['paths']['project_root'])
        
        # Convertir todos los paths a Path objects y crear directorios si no existen
        for key, path_str in config['paths'].items():
            if key == 'project_root':
                continue
            
            full_path = base_path / path_str
            config['paths'][key] = str(full_path)
            
            # Crear directorio si no existe
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
        
        return config
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        log_config = self.config['logging']
        
        # Crear directorio de logs si no existe
        log_dir = Path(self.config['paths']['logs_root'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        log_level = getattr(logging, log_config['level'])
        log_format = log_config['format']
        
        handlers = []
        
        if log_config['console_logging']:
            handlers.append(logging.StreamHandler())
        
        if log_config['file_logging']:
            log_file = log_dir / 'application.log'
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True
        )
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Obtiene la configuración específica para un tipo de modelo."""
        if model_type not in self.config['model']:
            raise ValueError(f"Tipo de modelo no válido: {model_type}")
        
        model_config = self.config['model'][model_type].copy()
        
        # Agregar configuración global necesaria
        model_config['num_emotions'] = self.config['emotions']['num_classes']
        model_config['device'] = self.config['device']
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de entrenamiento optimizada para la plataforma."""
        training_config = self.config['training'].copy()
        
        # Usar batch size efectivo según la plataforma
        training_config['batch_size'] = training_config['effective_batch_size']
        
        # Configuraciones adicionales para diferentes dispositivos
        if self.config['device'] == 'mps':
            # Apple Silicon optimizations
            training_config['num_workers'] = min(4, os.cpu_count())
            training_config['pin_memory'] = False  # No soportado en MPS
        elif self.config['device'] == 'cuda':
            # CUDA optimizations
            training_config['num_workers'] = min(8, os.cpu_count())
            training_config['pin_memory'] = True
        else:
            # CPU fallback
            training_config['num_workers'] = min(2, os.cpu_count())
            training_config['pin_memory'] = False
        
        return training_config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de datos."""
        data_config = {
            'paths': self.config['paths'],
            'emotions': self.config['emotions'],
            'input': self.config['input'],
            'augmentation': self.config['training']['augmentation'],
            'model_input_size': self.config['model'][self.config['model']['architecture']]['input_size']
        }
        
        return data_config
    
    def update_config(self, updates: Dict[str, Any]):
        """Actualiza la configuración dinámicamente."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        
        # Re-procesar configuraciones dependientes
        self.config = self._process_emotion_config(self.config)
        
        logging.info("Configuración actualizada dinámicamente")
    
    def save_config(self, path: Optional[str] = None):
        """Guarda la configuración actual."""
        save_path = path or f"{self.config_path}.processed"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logging.info(f"Configuración guardada en: {save_path}")
    
    def get_full_config(self) -> Dict[str, Any]:
        """Retorna la configuración completa procesada."""
        return self.config.copy()
    
    def print_config_summary(self):
        """Imprime un resumen de la configuración actual."""
        print("\n" + "="*50)
        print("RESUMEN DE CONFIGURACIÓN")
        print("="*50)
        print(f"Plataforma: {self.platform_info['platform']} ({self.platform_info['machine']})")
        print(f"Dispositivo: {self.config['device']}")
        print(f"Modo de emociones: {self.config['emotions']['mode']}")
        print(f"Número de emociones: {self.config['emotions']['num_classes']}")
        print(f"Emociones activas: {self.config['emotions']['active_emotions']}")
        print(f"Arquitectura del modelo: {self.config['model']['architecture']}")
        print(f"Tamaño de batch efectivo: {self.config['training']['effective_batch_size']}")
        print(f"Mixed precision: {self.config['training']['mixed_precision']}")
        print(f"Tipo de entrada: {self.config['input']['source_type']}")
        print("="*50)


# Función de conveniencia para crear el gestor de configuración
def create_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Función de conveniencia para crear un ConfigManager.
    
    Args:
        config_path: Path al archivo de configuración (opcional)
        
    Returns:
        ConfigManager instanciado y configurado
    """
    return ConfigManager(config_path)


# Decorador para funciones que necesitan configuración
def with_config(config_path: Optional[str] = None):
    """
    Decorador que inyecta la configuración en funciones.
    
    Usage:
        @with_config()
        def my_function(config, other_args):
            # La configuración estará disponible como primer argumento
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            config_manager = create_config_manager(config_path)
            return func(config_manager.get_full_config(), *args, **kwargs)
        return wrapper
    return decorator