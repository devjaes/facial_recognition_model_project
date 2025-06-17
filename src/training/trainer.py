# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json
from tqdm import tqdm
import wandb

from ..config_manager import ConfigManager
from ..models.model_factory import ModelFactory
from ..data.data_preprocessor import create_data_loaders
from ..utils.metrics import MetricsCalculator
from ..utils.device_manager import DeviceManager

class EmotionTrainer:
    """
    Entrenador principal para modelos de reconocimiento de emociones.
    Completamente parametrizable y optimizado para múltiples plataformas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = DeviceManager.get_optimal_device(config)
        
        # Configuración de entrenamiento
        self.train_config = config['training']
        self.model_config = config['model']
        
        # Inicializar componentes
        self._setup_logging()
        self._setup_directories()
        self._setup_model()
        self._setup_optimizer_and_loss()
        self._setup_metrics()
        self._setup_monitoring()
        
        # Estado del entrenamiento
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        logging.info(f"EmotionTrainer inicializado en dispositivo: {self.device}")
        logging.info(f"Modelo: {self.model_config['architecture']}")
        logging.info(f"Emociones: {config['emotions']['num_classes']}")
    
    def _setup_logging(self):
        """Configura el sistema de logging específico para entrenamiento."""
        self.train_log_dir = Path(self.config['paths']['logs_root']) / 'training'
        self.train_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger específico para entrenamiento
        self.logger = logging.getLogger('trainer')
        
        # Handler para archivo de entrenamiento
        train_log_file = self.train_log_dir / f"training_{int(time.time())}.log"
        file_handler = logging.FileHandler(train_log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_directories(self):
        """Configura los directorios necesarios."""
        self.model_save_dir = Path(self.config['paths']['trained_models'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoints'])
        
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_model(self):
        """Inicializa el modelo."""
        self.model = ModelFactory.create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Modelo creado - Total params: {total_params:,}")
        self.logger.info(f"Parámetros entrenables: {trainable_params:,}")
        
        # Configurar mixed precision si está habilitado
        self.use_mixed_precision = self.train_config.get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision habilitado")
    
    def _setup_optimizer_and_loss(self):
        """Configura el optimizador y función de pérdida."""
        # Optimizador
        optimizer_name = self.train_config['optimizer'].lower()
        lr = self.train_config['learning_rate']
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer_name}")
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Función de pérdida
        loss_name = self.train_config['loss_function'].lower()
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            from ..training.loss_functions import FocalLoss
            self.criterion = FocalLoss(alpha=1, gamma=2)
        else:
            raise ValueError(f"Función de pérdida no soportada: {loss_name}")
        
        self.logger.info(f"Optimizador: {optimizer_name}, LR: {lr}")
        self.logger.info(f"Función de pérdida: {loss_name}")
    
    def _setup_metrics(self):
        """Configura el calculador de métricas."""
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config['emotions']['num_classes'],
            class_names=self.config['emotions']['active_emotions']
        )
    
    def _setup_monitoring(self):
        """Configura el monitoreo con TensorBoard y opcionalmente W&B."""
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=self.train_log_dir / f"tensorboard_{int(time.time())}"
        )
        
        # Weights & Biases (opcional)
        if self.config['monitoring'].get('enable_wandb', False):
            wandb.init(
                project="facial-emotion-recognition",
                config=self.config,
                name=f"training_{self.model_config['architecture']}_{int(time.time())}"
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """
        Entrena el modelo por una época.
        
        Returns:
            Tuple con (loss_promedio, accuracy_promedio)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Época {self.current_epoch}")
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass con mixed precision si está habilitado
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass con scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Actualizar progress bar
            current_acc = 100 * correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # Log cada N batches
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), 
                                     self.current_epoch * len(dataloader) + batch_idx)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader) -> Tuple[float, float, Dict[str, Any]]:
        """
        Valida el modelo.
        
        Returns:
            Tuple con (loss_promedio, accuracy_promedio, métricas_detalladas)
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validación"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Calcular métricas detalladas
        detailed_metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions
        )
        
        return epoch_loss, epoch_acc, detailed_metrics
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Guarda un checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'training_history': self.training_history,
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_model_path = self.model_save_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            self.logger.info(f"Nuevo mejor modelo guardado con accuracy: {val_acc:.2f}%")
        
        # Mantener solo los últimos 5 checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Limpia checkpoints antiguos manteniendo solo los últimos N."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_last:
            # Ordenar por número de época
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            # Eliminar los más antiguos
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Carga un checkpoint para continuar entrenamiento."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_accuracy = checkpoint['best_val_accuracy']
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"Checkpoint cargado desde época {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando checkpoint: {e}")
            return False
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Entrena el modelo completo.
        
        Args:
            num_epochs: Número de épocas (usa config si no se especifica)
            
        Returns:
            Diccionario con historia de entrenamiento y estadísticas
        """
        if num_epochs is None:
            num_epochs = self.train_config['epochs']
        
        # Crear data loaders
        self.logger.info("Creando data loaders...")
        dataloaders = create_data_loaders(self.config)
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        self.logger.info(f"Iniciando entrenamiento por {num_epochs} épocas")
        self.logger.info(f"Batches por época - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Entrenamiento
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validación
            val_loss, val_acc, detailed_metrics = self.validate_epoch(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar historia
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            # Determinar si es el mejor modelo
            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc
            
            # Guardar checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Época {epoch}/{num_epochs-1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - Tiempo: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr,
                    **detailed_metrics
                })
            
            # Early stopping check
            if self._should_early_stop():
                self.logger.info("Early stopping activado")
                break
        
        # Finalizar entrenamiento
        total_time = time.time() - start_time
        
        # Guardar modelo final
        final_checkpoint_path = self.model_save_dir / "final_model.pth"
        self.save_checkpoint(epoch, val_acc, is_best=False)
        
        # Guardar historia de entrenamiento
        history_path = self.model_save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Cerrar writers
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        # Retornar estadísticas finales
        training_stats = {
            'total_epochs': epoch + 1,
            'total_time_hours': total_time / 3600,
            'best_val_accuracy': self.best_val_accuracy,
            'final_val_accuracy': val_acc,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            'training_history': self.training_history
        }
        
        self.logger.info("=== ENTRENAMIENTO COMPLETADO ===")
        self.logger.info(f"Mejor accuracy de validación: {self.best_val_accuracy:.2f}%")
        self.logger.info(f"Tiempo total: {total_time/3600:.2f} horas")
        
        return training_stats
    
    def _should_early_stop(self, patience: int = 20) -> bool:
        """Determina si se debe hacer early stopping."""
        if len(self.training_history['val_acc']) < patience:
            return False
        
        recent_accuracies = self.training_history['val_acc'][-patience:]
        max_recent = max(recent_accuracies)
        
        # Si no ha mejorado en las últimas 'patience' épocas
        return max_recent <= self.best_val_accuracy
    
    def evaluate_test_set(self) -> Dict[str, Any]:
        """Evalúa el mejor modelo en el conjunto de test."""
        if self.best_model_path is None:
            self.logger.error("No hay modelo entrenado para evaluar")
            return {}
        
        # Cargar el mejor modelo
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Crear data loader de test
        dataloaders = create_data_loaders(self.config)
        test_loader = dataloaders['test']
        
        self.logger.info("Evaluando en conjunto de test...")
        
        # Evaluar
        test_loss, test_acc, detailed_metrics = self.validate_epoch(test_loader)
        
        # Crear reporte de evaluación
        evaluation_report = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'detailed_metrics': detailed_metrics,
            'model_path': str(self.best_model_path),
            'config_used': self.config
        }
        
        # Guardar reporte
        report_path = self.model_save_dir / "test_evaluation.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        self.logger.info(f"Accuracy en test: {test_acc:.2f}%")
        self.logger.info(f"Reporte guardado en: {report_path}")
        
        return evaluation_report


# Función de conveniencia para entrenar desde script
def train_model_from_config(config_path: Optional[str] = None, 
                          resume_from: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para entrenar un modelo desde configuración.
    
    Args:
        config_path: Path al archivo de configuración
        resume_from: Path a checkpoint para continuar entrenamiento
        
    Returns:
        Estadísticas de entrenamiento
    """
    # Crear config manager
    config_manager = ConfigManager(config_path)
    config = config_manager.get_full_config()
    
    # Imprimir resumen de configuración
    config_manager.print_config_summary()
    
    # Crear entrenador
    trainer = EmotionTrainer(config)
    
    # Cargar checkpoint si se especifica
    if resume_from:
        if not trainer.load_checkpoint(resume_from):
            logging.error(f"No se pudo cargar checkpoint: {resume_from}")
            return {}
    
    # Entrenar
    try:
        training_stats = trainer.train()
        
        # Evaluar en test set
        test_results = trainer.evaluate_test_set()
        training_stats['test_results'] = test_results
        
        return training_stats
        
    except KeyboardInterrupt:
        logging.info("Entrenamiento interrumpido por el usuario")
        # Guardar checkpoint de emergencia
        trainer.save_checkpoint(trainer.current_epoch, 0.0, is_best=False)
        return trainer.training_history
    
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")
        raise