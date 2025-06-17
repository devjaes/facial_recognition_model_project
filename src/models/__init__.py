# src/models/__init__.py
"""
Módulo de modelos para reconocimiento de emociones faciales.
"""

from .cnn_rnn_hybrid import EmotionCNNRNN

# Alias para compatibilidad
CNNRNNHybrid = EmotionCNNRNN

__all__ = ['EmotionCNNRNN', 'CNNRNNHybrid']
