# src/models/cnn_rnn_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

class EmotionCNNRNN(nn.Module):
    """
    Modelo híbrido CNN-RNN para reconocimiento de emociones faciales.
    Arquitectura basada en el paper de Manalu & Rifai (2024).
    
    Completamente parametrizable a través de configuraciones.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EmotionCNNRNN, self).__init__()
        
        self.config = config
        self.num_emotions = config['emotions']['num_classes']
        self.input_size = config['model']['cnn_rnn_hybrid']['input_size']
        self.sequence_length = config['model']['cnn_rnn_hybrid']['sequence_length']
        
        # Configuración de dispositivo
        self.device = self._get_device(config)
        
        # CNN Feature Extractor (basado en el paper)
        self._build_cnn_layers()
        
        # RNN Temporal Modeling
        self._build_rnn_layers()
        
        # Clasificador final
        self._build_classifier()
        
        # Inicialización de pesos
        self._initialize_weights()
        
        logging.info(f"Modelo CNN-RNN creado para {self.num_emotions} emociones")
        logging.info(f"Dispositivo: {self.device}")
    
    def _get_device(self, config: Dict[str, Any]) -> torch.device:
        """Detecta y configura el dispositivo óptimo según la plataforma."""
        if config['devices']['auto_detect']:
            if torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(config['devices']['manual_device'])
    
    def _build_cnn_layers(self):
        """Construye las capas CNN basadas en la configuración."""
        cnn_config = self.config['model']['cnn_rnn_hybrid']
        
        # Primera capa convolucional (128 filtros como en el paper)
        # Calcular padding para mantener dimensiones compatibles
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # Cambio: stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Reducir dimensiones aquí
        )
        
        # Segunda capa convolucional (64 filtros como en el paper)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Cambio: stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Reducir dimensiones aquí
            nn.Dropout(cnn_config['dropout'])
        )
        
        # Tercera capa convolucional para mayor reducción de dimensiones
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Forzar salida a 4x4
            nn.Dropout(cnn_config['dropout'])
        )
        
        # El tamaño de salida será 32 * 4 * 4 = 512
        self.cnn_output_size = 32 * 4 * 4
        
        # Capa de aplanamiento para RNN
        self.flatten = nn.Flatten()
        
    def _build_rnn_layers(self):
        """Construye las capas RNN/LSTM basadas en la configuración."""
        rnn_config = self.config['model']['cnn_rnn_hybrid']
        
        # Time Distributed wrapper para procesar secuencias de frames
        self.time_distributed = True
        
        # Primea capa LSTM (512 unidades como en el paper)
        self.lstm1 = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=rnn_config['lstm_hidden'][0],  # 512
            batch_first=True,
            dropout=rnn_config['dropout'] if len(rnn_config['lstm_hidden']) > 1 else 0
        )
        
        # Segunda capa LSTM (128 unidades como en el paper)
        self.lstm2 = nn.LSTM(
            input_size=rnn_config['lstm_hidden'][0],
            hidden_size=rnn_config['lstm_hidden'][1],  # 128
            batch_first=True
        )
        
    def _build_classifier(self):
        """Construye el clasificador final."""
        rnn_config = self.config['model']['cnn_rnn_hybrid']
        
        # Capas densas finales
        self.classifier = nn.Sequential(
            nn.Linear(rnn_config['lstm_hidden'][1], 128),  # 128 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(rnn_config['dropout']),
            nn.Linear(128, self.num_emotions),  # 128 -> num_emotions
            nn.Softmax(dim=1) if self.config['training']['loss_function'] != 'cross_entropy' else nn.Identity()
        )
    
    def _initialize_weights(self):
        """Inicializa los pesos del modelo."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada [batch_size, channels, height, width] o 
               [batch_size, sequence_length, channels, height, width]
            
        Returns:
            Tensor de predicciones [batch_size, num_emotions]
        """
        # Manejar tanto imágenes individuales como secuencias
        if x.dim() == 4:  # Imagen individual [batch_size, channels, height, width]
            batch_size, channels, height, width = x.shape
            # Simular secuencia repitiendo la imagen
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1, 1, 1)
            seq_len = self.sequence_length
        elif x.dim() == 5:  # Secuencia [batch_size, sequence_length, channels, height, width]
            batch_size, seq_len, channels, height, width = x.shape
        else:
            raise ValueError(f"Entrada inesperada con {x.dim()} dimensiones. Esperado 4 o 5.")
        
        # Procesar cada frame a través de CNN
        # Reshape para procesar todos los frames como un batch
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # Nueva capa convolucional
        x = self.flatten(x)
        
        # Reshape de vuelta para RNN [batch_size, seq_len, features]
        x = x.view(batch_size, seq_len, -1)
        
        # RNN Temporal Modeling
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Tomar solo la última salida temporal
        last_output = lstm2_out[:, -1, :]
        
        # Clasificación final
        predictions = self.classifier(last_output)
        
        return predictions
    
    def predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Predicción para un solo frame (modo tiempo real).
        
        Args:
            frame: Tensor de un frame [channels, height, width] o [1, channels, height, width]
            
        Returns:
            Tensor de predicciones [num_emotions]
        """
        # Asegurar que tenga batch dimension
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)  # [1, channels, height, width]
        
        with torch.no_grad():
            prediction = self.forward(frame)
        
        return prediction.squeeze(0)  # Remover batch dimension
    
    def get_emotion_probabilities(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Obtiene las probabilidades por emoción con nombres.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Diccionario con probabilidades por emoción
        """
        predictions = self.forward(x)
        probabilities = F.softmax(predictions, dim=1)
        
        emotion_names = self.config['emotions'][f"{self.config['emotions']['mode']}_emotions"]
        
        result = {}
        for i, emotion in enumerate(emotion_names):
            result[emotion] = float(probabilities[0, i])
        
        return result
    
    def save_model(self, path: str, include_config: bool = True):
        """Guarda el modelo y su configuración."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
        }
        
        if include_config:
            save_dict['config'] = self.config
            
        torch.save(save_dict, path)
        logging.info(f"Modelo guardado en: {path}")
    
    @classmethod
    def load_model(cls, path: str, config: Optional[Dict[str, Any]] = None):
        """Carga un modelo desde archivo."""
        checkpoint = torch.load(path, map_location='cpu')
        
        if config is None:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("No se encontró configuración en el checkpoint y no se proporcionó una")
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logging.info(f"Modelo cargado desde: {path}")
        return model