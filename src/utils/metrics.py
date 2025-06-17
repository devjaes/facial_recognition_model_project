# src/utils/metrics.py
"""
Calculador de métricas para evaluación de modelos de reconocimiento de emociones.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
from datetime import datetime
from pathlib import Path

class MetricsCalculator:
    """
    Calculador de métricas completo para modelos de reconocimiento de emociones.
    """
    
    def __init__(self, num_classes: int, class_names: List[str]):
        """
        Inicializa el calculador de métricas.
        
        Args:
            num_classes: Número de clases (emociones)
            class_names: Nombres de las clases/emociones
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reinicia las métricas acumuladas."""
        self.all_predictions = []
        self.all_labels = []
        self.batch_accuracies = []
        self.batch_losses = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, loss: Optional[float] = None):
        """
        Actualiza las métricas con un nuevo batch.
        
        Args:
            predictions: Predicciones del modelo [batch_size, num_classes]
            labels: Etiquetas reales [batch_size]
            loss: Loss del batch (opcional)
        """
        # Convertir a numpy si es necesario
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Acumular predicciones y etiquetas
        self.all_predictions.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())
        
        # Calcular accuracy del batch
        batch_acc = accuracy_score(labels, predictions)
        self.batch_accuracies.append(batch_acc)
        
        if loss is not None:
            self.batch_losses.append(loss)
    
    def calculate_metrics(self, y_true: Optional[List] = None, y_pred: Optional[List] = None) -> Dict[str, Any]:
        """
        Calcula todas las métricas disponibles.
        
        Args:
            y_true: Etiquetas reales (usa las acumuladas si no se especifica)
            y_pred: Predicciones (usa las acumuladas si no se especifica)
            
        Returns:
            Dict con todas las métricas calculadas
        """
        if y_true is None:
            y_true = self.all_labels
        if y_pred is None:
            y_pred = self.all_predictions
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return self._empty_metrics()
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métricas por clase
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        
        # Crear diccionario de métricas
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'precision_weighted': float(precision_weighted),
            'recall_macro': float(recall_macro),
            'recall_weighted': float(recall_weighted),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {}
        }
        
        # Métricas por clase
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics['per_class_metrics'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(np.sum(cm[i]))
                }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Retorna métricas vacías cuando no hay datos."""
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_macro': 0.0,
            'recall_weighted': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'confusion_matrix': [[0] * self.num_classes for _ in range(self.num_classes)],
            'per_class_metrics': {name: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0} 
                                for name in self.class_names}
        }
    
    def get_classification_report(self, y_true: Optional[List] = None, y_pred: Optional[List] = None) -> str:
        """
        Genera un reporte de clasificación detallado.
        
        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones
            
        Returns:
            str: Reporte de clasificación formateado
        """
        if y_true is None:
            y_true = self.all_labels
        if y_pred is None:
            y_pred = self.all_predictions
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return "No hay datos para generar el reporte"
        
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def save_metrics(self, save_path: str, metrics: Optional[Dict] = None):
        """
        Guarda las métricas en un archivo JSON.
        
        Args:
            save_path: Path donde guardar las métricas
            metrics: Métricas a guardar (calcula si no se especifica)
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        # Agregar metadatos
        metrics_with_metadata = {
            'metadata': {
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'total_samples': len(self.all_labels),
                'timestamp': datetime.now().isoformat()
            },
            'metrics': metrics,
            'classification_report': self.get_classification_report()
        }
        
        # Crear directorio si no existe
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar en JSON
        with open(save_path, 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2, default=str)
    
    def get_current_accuracy(self) -> float:
        """Retorna la accuracy actual basada en las predicciones acumuladas."""
        if len(self.all_labels) == 0 or len(self.all_predictions) == 0:
            return 0.0
        return accuracy_score(self.all_labels, self.all_predictions)
    
    def get_average_loss(self) -> float:
        """Retorna el loss promedio de los batches."""
        if len(self.batch_losses) == 0:
            return 0.0
        return np.mean(self.batch_losses)
    
    def print_summary(self):
        """Imprime un resumen de las métricas actuales."""
        metrics = self.calculate_metrics()
        
        print("\n📊 RESUMEN DE MÉTRICAS")
        print("=" * 40)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"Total de muestras: {len(self.all_labels)}")
        
        print("\n📈 Métricas por Emoción:")
        print("-" * 40)
        for emotion, emotion_metrics in metrics['per_class_metrics'].items():
            print(f"{emotion.capitalize():12}: F1={emotion_metrics['f1']:.3f}, "
                  f"Precision={emotion_metrics['precision']:.3f}, "
                  f"Recall={emotion_metrics['recall']:.3f}")


def test_metrics():
    """Función de test para el calculador de métricas."""
    print("🧪 TESTING METRICS CALCULATOR")
    print("=" * 40)
    
    # Crear datos de prueba
    emotions = ['happy', 'sad', 'angry', 'neutral']
    num_classes = len(emotions)
    
    # Simular predicciones y etiquetas
    np.random.seed(42)
    y_true = np.random.randint(0, num_classes, 100)
    y_pred = np.random.randint(0, num_classes, 100)
    
    # Crear calculador
    calc = MetricsCalculator(num_classes, emotions)
    
    # Simular actualización por batches
    for i in range(0, 100, 10):
        batch_true = y_true[i:i+10]
        batch_pred = y_pred[i:i+10]
        calc.update(torch.tensor(batch_pred), torch.tensor(batch_true), loss=np.random.random())
    
    # Calcular métricas
    metrics = calc.calculate_metrics()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_macro']:.4f}")
    
    # Imprimir resumen
    calc.print_summary()
    
    print("\n✅ Test de métricas completado")


if __name__ == "__main__":
    test_metrics()
