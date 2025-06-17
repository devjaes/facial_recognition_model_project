# src/utils/device_manager.py
"""
Gestor de dispositivos para el sistema de reconocimiento de emociones.
Detecta automáticamente el dispositivo óptimo según la plataforma.
"""

import torch
import platform
import logging
from typing import Dict, Any, Optional
import subprocess

class DeviceManager:
    """
    Gestor inteligente de dispositivos que detecta automáticamente
    la configuración óptima según la plataforma y hardware disponible.
    """
    
    _instance = None
    _device_info = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_optimal_device(cls, config: Optional[Dict[str, Any]] = None) -> torch.device:
        """
        Detecta y retorna el dispositivo óptimo disponible.
        
        Args:
            config: Configuración del proyecto (opcional)
            
        Returns:
            torch.device: Dispositivo óptimo detectado
        """
        if config and not config.get('devices', {}).get('auto_detect', True):
            # Usar dispositivo manual si está especificado
            manual_device = config['devices'].get('manual_device', 'cpu')
            return torch.device(manual_device)
        
        # Detección automática
        device_info = cls._detect_hardware()
        
        if device_info['mps_available']:
            return torch.device('mps')
        elif device_info['cuda_available']:
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    @classmethod
    def _detect_hardware(cls) -> Dict[str, Any]:
        """Detecta el hardware disponible en el sistema."""
        if cls._device_info is not None:
            return cls._device_info
        
        device_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': False,
            'cuda_version': None,
            'cuda_device_count': 0,
            'cuda_devices': [],
            'mps_available': False,
            'cpu_cores': None,
            'memory_gb': None
        }
        
        # Detectar CUDA
        if torch.cuda.is_available():
            device_info['cuda_available'] = True
            device_info['cuda_version'] = torch.version.cuda
            device_info['cuda_device_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info['cuda_devices'].append({
                    'name': device_props.name,
                    'memory_gb': device_props.total_memory / (1024**3),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
        
        # Detectar Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['mps_available'] = True
        
        # Información de CPU
        try:
            import psutil
            device_info['cpu_cores'] = psutil.cpu_count(logical=False)
            device_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback si psutil no está disponible
            import os
            device_info['cpu_cores'] = os.cpu_count()
        
        cls._device_info = device_info
        return device_info
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Retorna información detallada del hardware."""
        return cls._detect_hardware()
    
    @classmethod
    def print_device_info(cls):
        """Imprime información detallada del hardware disponible."""
        info = cls._detect_hardware()
        
        print("🖥️  INFORMACIÓN DEL DISPOSITIVO")
        print("=" * 50)
        print(f"Plataforma: {info['platform']} ({info['architecture']})")
        print(f"Python: {info['python_version']}")
        print(f"PyTorch: {info['pytorch_version']}")
        
        if info['cpu_cores']:
            print(f"CPU: {info['cpu_cores']} cores")
        if info['memory_gb']:
            print(f"RAM: {info['memory_gb']:.1f} GB")
        
        print()
        print("🚀 ACELERACIÓN DISPONIBLE:")
        
        # MPS (Apple Silicon)
        if info['mps_available']:
            print("✅ MPS (Apple Silicon) - DISPONIBLE")
        else:
            print("❌ MPS - NO DISPONIBLE")
        
        # CUDA
        if info['cuda_available']:
            print(f"✅ CUDA {info['cuda_version']} - DISPONIBLE")
            print(f"   Dispositivos CUDA: {info['cuda_device_count']}")
            for i, device in enumerate(info['cuda_devices']):
                print(f"   GPU {i}: {device['name']}")
                print(f"           Memoria: {device['memory_gb']:.1f} GB")
                print(f"           Compute: {device['compute_capability']}")
        else:
            print("❌ CUDA - NO DISPONIBLE")
        
        # Dispositivo óptimo
        optimal_device = cls.get_optimal_device()
        print(f"\n🎯 DISPOSITIVO ÓPTIMO: {optimal_device}")
    
    @classmethod
    def get_batch_size_multiplier(cls, config: Optional[Dict[str, Any]] = None) -> float:
        """
        Calcula el multiplicador de batch size basado en el dispositivo.
        
        Args:
            config: Configuración del proyecto
            
        Returns:
            float: Multiplicador para el batch size base
        """
        device = cls.get_optimal_device(config)
        device_info = cls._detect_hardware()
        
        if device.type == 'mps':
            # Apple Silicon - moderado
            return config.get('devices', {}).get('macos_m3', {}).get('batch_size_multiplier', 1.5)
        elif device.type == 'cuda':
            # CUDA GPU - agresivo
            gpu_memory = 0
            if device_info['cuda_devices']:
                gpu_memory = device_info['cuda_devices'][0]['memory_gb']
            
            if gpu_memory >= 16:
                return 3.0  # GPU de alta gama
            elif gpu_memory >= 8:
                return 2.0  # GPU de gama media
            else:
                return 1.5  # GPU de gama baja
        else:
            # CPU - conservador
            return config.get('devices', {}).get('cpu_fallback', {}).get('batch_size_multiplier', 0.5)
    
    @classmethod
    def get_effective_batch_size(cls, base_batch_size: int, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Calcula el batch size efectivo basado en el dispositivo.
        
        Args:
            base_batch_size: Batch size base de la configuración
            config: Configuración del proyecto
            
        Returns:
            int: Batch size efectivo para el dispositivo
        """
        multiplier = cls.get_batch_size_multiplier(config)
        effective_size = int(base_batch_size * multiplier)
        
        # Asegurar que sea al menos 1 y potencia de 2 para eficiencia
        effective_size = max(1, effective_size)
        
        # Redondear a la potencia de 2 más cercana para eficiencia de GPU
        import math
        power_of_2 = 2 ** math.floor(math.log2(effective_size))
        if effective_size - power_of_2 > power_of_2 * 0.5:
            power_of_2 *= 2
        
        return max(1, power_of_2)
    
    @classmethod
    def should_use_mixed_precision(cls, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determina si se debe usar mixed precision según el dispositivo.
        
        Args:
            config: Configuración del proyecto
            
        Returns:
            bool: True si se debe usar mixed precision
        """
        device = cls.get_optimal_device(config)
        device_info = cls._detect_hardware()
        
        if device.type == 'cuda':
            # Verificar que la GPU soporte mixed precision (Compute Capability >= 7.0)
            if device_info['cuda_devices']:
                compute_capability = device_info['cuda_devices'][0]['compute_capability']
                major_version = float(compute_capability.split('.')[0])
                return major_version >= 7.0
        
        # MPS y CPU no soportan mixed precision actualmente
        return False
    
    @classmethod
    def optimize_torch_settings(cls, config: Optional[Dict[str, Any]] = None):
        """
        Optimiza la configuración de PyTorch según el dispositivo.
        
        Args:
            config: Configuración del proyecto
        """
        device = cls.get_optimal_device(config)
        device_info = cls._detect_hardware()
        
        # Configuraciones generales
        torch.backends.cudnn.benchmark = True  # Optimizar para inputs de tamaño fijo
        
        if device.type == 'cuda':
            # Optimizaciones específicas para CUDA
            torch.backends.cudnn.deterministic = False  # Mejor performance
            
            # Configurar memory management
            if device_info['cuda_devices']:
                gpu_memory = device_info['cuda_devices'][0]['memory_gb']
                if gpu_memory < 8:
                    # GPU con poca memoria - ser más conservador
                    torch.cuda.empty_cache()
        
        elif device.type == 'mps':
            # Optimizaciones específicas para Apple Silicon
            # MPS actualmente tiene limitaciones, configurar conservadoramente
            pass
        
        else:
            # Optimizaciones para CPU
            # Configurar threads para mejor performance en CPU
            cpu_cores = device_info.get('cpu_cores', 4)
            torch.set_num_threads(min(cpu_cores, 8))  # No más de 8 threads
        
        logging.info(f"PyTorch optimizado para dispositivo: {device}")


def test_device_manager():
    """Función de test para DeviceManager."""
    print("🧪 TESTING DEVICE MANAGER")
    print("=" * 40)
    
    # Test básico
    device = DeviceManager.get_optimal_device()
    print(f"Dispositivo óptimo: {device}")
    
    # Información detallada
    DeviceManager.print_device_info()
    
    # Test de batch size
    base_batch = 32
    effective_batch = DeviceManager.get_effective_batch_size(base_batch)
    print(f"\nBatch size: {base_batch} → {effective_batch}")
    
    # Test de mixed precision
    use_mp = DeviceManager.should_use_mixed_precision()
    print(f"Mixed precision: {'✅' if use_mp else '❌'}")
    
    # Optimizar configuración
    DeviceManager.optimize_torch_settings()
    print("✅ Configuración de PyTorch optimizada")


if __name__ == "__main__":
    test_device_manager()
