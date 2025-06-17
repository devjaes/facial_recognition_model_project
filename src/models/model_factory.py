# src/models/model_factory.py
"""
Factory para crear modelos de reconocimiento de emociones según configuración.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

# Importar modelos disponibles
from .cnn_rnn_hybrid import EmotionCNNRNN

class ModelFactory:
    """
    Factory para crear modelos de reconocimiento de emociones.
    """
    
    AVAILABLE_MODELS = {
        'cnn_rnn_hybrid': EmotionCNNRNN,
        'emotion_cnn_rnn': EmotionCNNRNN,  # Alias
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> nn.Module:
        """
        Crea un modelo según la configuración especificada.
        
        Args:
            config: Configuración completa del proyecto
            
        Returns:
            nn.Module: Modelo inicializado
        """
        model_config = config.get('model', {})
        architecture = model_config.get('architecture', 'cnn_rnn_hybrid')
        
        logging.info(f"Creando modelo: {architecture}")
        
        if architecture not in cls.AVAILABLE_MODELS:
            available = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Arquitectura '{architecture}' no disponible. "
                f"Arquitecturas disponibles: {available}"
            )
        
        model_class = cls.AVAILABLE_MODELS[architecture]
        
        try:
            # Crear modelo con configuración completa
            model = model_class(config)
            
            # Log de información del modelo
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logging.info(f"Modelo creado exitosamente:")
            logging.info(f"  - Arquitectura: {architecture}")
            logging.info(f"  - Total parámetros: {total_params:,}")
            logging.info(f"  - Parámetros entrenables: {trainable_params:,}")
            logging.info(f"  - Clases: {config['emotions']['num_classes']}")
            
            return model
            
        except Exception as e:
            logging.error(f"Error creando modelo {architecture}: {e}")
            raise
    
    @classmethod
    def get_model_summary(cls, model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un resumen detallado del modelo.
        
        Args:
            model: Modelo a analizar
            config: Configuración del modelo
            
        Returns:
            Dict con resumen del modelo
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calcular tamaño aproximado en MB
        model_size_mb = total_params * 4 / (1024 ** 2)  # Asumiendo float32
        
        summary = {
            'architecture': config['model']['architecture'],
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'input_shape': config['model'].get('cnn_rnn_hybrid', {}).get('input_size', [224, 224]),
            'num_classes': config['emotions']['num_classes'],
            'class_names': config['emotions']['active_emotions']
        }
        
        return summary
    
    @classmethod
    def print_model_summary(cls, model: nn.Module, config: Dict[str, Any]):
        """
        Imprime un resumen detallado del modelo.
        
        Args:
            model: Modelo a mostrar
            config: Configuración del modelo
        """
        summary = cls.get_model_summary(model, config)
        
        print("\n🧠 RESUMEN DEL MODELO")
        print("=" * 50)
        print(f"Arquitectura: {summary['architecture']}")
        print(f"Clase: {summary['model_class']}")
        print(f"Total de parámetros: {summary['total_parameters']:,}")
        print(f"Parámetros entrenables: {summary['trainable_parameters']:,}")
        print(f"Tamaño del modelo: {summary['model_size_mb']:.2f} MB")
        print(f"Número de clases: {summary['num_classes']}")
        
        print(f"\n🎭 Emociones reconocidas:")
        for i, emotion in enumerate(summary['class_names']):
            print(f"  {i}: {emotion}")
        
        print("=" * 50)


def test_model_factory():
    """Función de test para ModelFactory."""
    print("🧪 TESTING MODEL FACTORY")
    print("=" * 40)
    
    # Configuración de prueba
    test_config = {
        'model': {
            'architecture': 'cnn_rnn_hybrid',
            'cnn_rnn_hybrid': {
                'input_size': [224, 224],
                'cnn_filters': [128, 64],
                'lstm_hidden': [512, 128],
                'dropout': 0.5,
                'sequence_length': 10
            }
        },
        'emotions': {
            'num_classes': 7,
            'active_emotions': ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
        },
        'devices': {
            'auto_detect': True
        }
    }
    
    # Test creación de modelo
    try:
        print("🏗️ Creando modelo...")
        model = ModelFactory.create_model(test_config)
        print("✅ Modelo creado exitosamente")
        
        # Mostrar resumen
        ModelFactory.print_model_summary(model, test_config)
        
        print("\n✅ Test de ModelFactory completado")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_model_factory()
