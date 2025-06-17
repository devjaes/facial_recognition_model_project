# src/utils/logger.py
"""
Sistema de logging centralizado para el proyecto de reconocimiento de emociones.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
import json
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Formatter personalizado con colores para diferentes niveles de log."""
    
    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Verde
        'WARNING': '\033[33m',  # Amarillo
        'ERROR': '\033[31m',    # Rojo
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Agregar color para terminal
        if hasattr(record, 'no_color') and record.no_color:
            # Sin color para archivos
            return super().format(record)
        
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class EmotionLogger:
    """
    Sistema de logging centralizado para el proyecto.
    Maneja múltiples handlers y formatos según el contexto.
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logger(cls, 
                     name: str = "emotion_recognition",
                     log_dir: Optional[str] = None,
                     level: str = "INFO",
                     console_output: bool = True,
                     file_output: bool = True,
                     config: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """
        Configura un logger para el proyecto.
        
        Args:
            name: Nombre del logger
            log_dir: Directorio para archivos de log
            level: Nivel de logging
            console_output: Si mostrar en consola
            file_output: Si guardar en archivo
            config: Configuración del proyecto
            
        Returns:
            logging.Logger: Logger configurado
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Crear logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Limpiar handlers existentes
        logger.handlers.clear()
        
        # Configurar directorio de logs
        if log_dir is None:
            log_dir = "logs"
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Formato para archivos (sin color)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Formato para consola (con color)
        console_formatter = CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, level.upper()))
            logger.addHandler(console_handler)
        
        # Handler para archivo principal
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                log_path / f"{name}_{timestamp}.log"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        cls._initialized = True
        
        logger.info(f"Logger '{name}' inicializado correctamente")
        
        return logger
    
    @classmethod
    def get_logger(cls, name: str = "emotion_recognition") -> logging.Logger:
        """Obtiene un logger existente o crea uno básico."""
        if name in cls._loggers:
            return cls._loggers[name]
        return cls.setup_logger(name)

# Funciones de conveniencia
def setup_logger(name: str = "emotion_recognition", **kwargs) -> logging.Logger:
    """Función de conveniencia para configurar un logger."""
    return EmotionLogger.setup_logger(name, **kwargs)

def get_logger(name: str = "emotion_recognition") -> logging.Logger:
    """Función de conveniencia para obtener un logger."""
    return EmotionLogger.get_logger(name)

if __name__ == "__main__":
    # Test básico
    logger = setup_logger("test")
    logger.info("✅ Logger funcionando correctamente")
