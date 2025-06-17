#!/usr/bin/env python3
"""
Script de entrenamiento simplificado para FER2013.
Versión que funciona con la estructura actual del proyecto.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Importaciones locales
from config_manager import ConfigManager
from data.fer2013_processor import create_fer2013_data_loaders
from models.cnn_rnn_hybrid import EmotionCNNRNN

def setup_logging():
    """Configura logging para entrenamiento."""
    log_dir = Path('logs/training')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    log_file = log_dir / f'fer2013_training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para FER2013')
    
    parser.add_argument('--model', type=str, default='cnn_rnn_hybrid',
                       choices=['cnn_basic', 'cnn_rnn_hybrid', 'transfer_learning'],
                       help='Tipo de modelo a entrenar')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño de batch')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--config', type=str, default='config/fer2013_config.yaml',
                       help='Path al archivo de configuración')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Entrenamiento rápido para testing (5 épocas)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Dispositivo para entrenamiento')
    
    return parser.parse_args()

def get_device(preferred_device='auto'):
    """Detecta y retorna el mejor dispositivo disponible."""
    if preferred_device != 'auto':
        return torch.device(preferred_device)
    
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_model(config, device):
    """Crea el modelo según la configuración."""
    # CNN-RNN híbrido usando la configuración completa
    model = EmotionCNNRNN(config)
    
    return model.to(device)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Época {epoch}")
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
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
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validación"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, val_acc, is_best=False):
    """Guarda un checkpoint del modelo."""
    models_dir = Path('models/trained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc
    }
    
    # Guardar checkpoint
    checkpoint_path = models_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Guardar mejor modelo
    if is_best:
        best_model_path = models_dir / "best_model.pth"
        torch.save(checkpoint, best_model_path)
        return str(best_model_path)
    
    return str(checkpoint_path)

def print_training_summary(config, args, device):
    """Imprime resumen de la configuración de entrenamiento."""
    print("\n🚀 INICIANDO ENTRENAMIENTO FER2013")
    print("=" * 60)
    print(f"🧠 Modelo: {args.model}")
    print(f"📊 Dataset: FER2013 ({config['emotions']['num_classes']} emociones)")
    print(f"💻 Dispositivo: {device}")
    print(f"⚙️  Configuración:")
    print(f"   • Épocas: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Learning rate: {args.learning_rate}")
    
    print(f"\n🎭 Emociones a detectar:")
    for i, emotion in enumerate(config['emotions']['active_emotions']):
        print(f"   {i}: {emotion}")
    
    print("\n📈 Objetivos:")
    print(f"   • Baseline FER2013: 65%")
    print(f"   • Target: 75%")
    print(f"   • Excelente: 80%+")

def main():
    """Función principal."""
    # Setup
    logger = setup_logging()
    args = parse_arguments()
    
    # Ajustes para quick test
    if args.quick_test:
        args.epochs = 5
        args.batch_size = 16
        logger.info("🏃‍♂️ Modo quick test activado: 5 épocas, batch size reducido")
    
    # Cargar configuración
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.get_full_config()
        logger.info("✅ Configuración cargada correctamente")
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        sys.exit(1)
    
    # Actualizar configuración con argumentos
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    
    # Detectar dispositivo
    device = get_device(args.device)
    logger.info(f"Dispositivo detectado: {device}")
    
    # Verificar dataset
    fer2013_path = Path(config['paths']['data_root']) / 'raw' / 'fer2013'
    if not fer2013_path.exists():
        logger.error(f"❌ Dataset FER2013 no encontrado en: {fer2013_path}")
        sys.exit(1)
    logger.info("✅ Dataset FER2013 encontrado")
    
    # Mostrar resumen
    print_training_summary(config, args, device)
    
    # Confirmar inicio (solo si no es quick test)
    if not args.quick_test:
        response = input("\\n¿Continuar con el entrenamiento? (y/N): ")
        if response.lower() not in ['y', 'yes', 'sí', 's']:
            print("Entrenamiento cancelado")
            sys.exit(0)
    
    # Crear data loaders
    try:
        logger.info("📚 Creando data loaders...")
        dataloaders = create_fer2013_data_loaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        logger.info(f"✅ Data loaders creados:")
        logger.info(f"  Train: {len(train_loader.dataset)} imágenes, {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_loader.dataset)} imágenes, {len(val_loader)} batches")
        
    except Exception as e:
        logger.error(f"❌ Error creando data loaders: {e}")
        sys.exit(1)
    
    # Crear modelo
    try:
        logger.info("🧠 Creando modelo...")
        model = create_model(config, device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Modelo creado:")
        logger.info(f"  Total parámetros: {total_params:,}")
        logger.info(f"  Parámetros entrenables: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"❌ Error creando modelo: {e}")
        sys.exit(1)
    
    # Configurar optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Configurar TensorBoard
    log_dir = Path('logs/training') / f"tensorboard_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Variables de entrenamiento
    best_val_accuracy = 0.0
    best_model_path = None
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\\n🏁 INICIANDO ENTRENAMIENTO...")
    print(f"⏰ Hora de inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Loop de entrenamiento
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            
            # Entrenamiento
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validación
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # Actualizar scheduler
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar historia
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Determinar si es el mejor modelo
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                best_model_path = save_checkpoint(model, optimizer, epoch, val_acc, is_best=True)
                logger.info(f"🏆 Nuevo mejor modelo guardado con accuracy: {val_acc:.2f}%")
            
            # Guardar checkpoint cada 10 épocas
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, val_acc, is_best=False)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Época {epoch}/{args.epochs-1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - Tiempo: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Mostrar progreso cada 10 épocas
            if epoch % 10 == 0:
                print(f"\\n📊 Progreso Época {epoch}:")
                print(f"   🎯 Mejor val accuracy: {best_val_accuracy:.2f}%")
                print(f"   ⏱️ Tiempo transcurrido: {(time.time() - start_time)/3600:.2f}h")
        
        # Finalizar entrenamiento
        total_time = time.time() - start_time
        
        # Guardar modelo final
        final_checkpoint_path = save_checkpoint(model, optimizer, args.epochs-1, val_acc, is_best=False)
        
        # Guardar historia de entrenamiento
        history_path = Path('models/trained') / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Cerrar writer
        writer.close()
        
        # Resultados finales
        print(f"\\n🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"⏰ Tiempo total: {total_time/3600:.2f} horas")
        print(f"🎯 Mejor accuracy validación: {best_val_accuracy:.2f}%")
        print(f"🏆 Mejor modelo guardado en: {best_model_path}")
        
        # Evaluación del resultado
        if best_val_accuracy >= 80:
            print(f"🏆 ¡EXCELENTE! Superaste el objetivo de 80%")
        elif best_val_accuracy >= 75:
            print(f"🎉 ¡MUY BUENO! Alcanzaste el objetivo de 75%")
        elif best_val_accuracy >= 65:
            print(f"✅ BUENO! Superaste el baseline de 65%")
        else:
            print(f"⚠️ Por debajo del baseline, considera ajustar hiperparámetros")
        
        # Próximos pasos
        print(f"\\n🚀 PRÓXIMOS PASOS:")
        print(f"   1. 📊 Ver métricas: tensorboard --logdir {log_dir}")
        print(f"   2. 🌐 Probar en web app: python src/web/app.py")
        print(f"   3. 📱 Test tiempo real: python scripts/test_realtime.py")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'total_time_hours': total_time / 3600,
            'best_model_path': best_model_path,
            'training_history': training_history
        }
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ Entrenamiento interrumpido por el usuario")
        # Guardar checkpoint de emergencia
        emergency_path = save_checkpoint(model, optimizer, epoch, val_acc, is_best=False)
        print(f"💾 Checkpoint de emergencia guardado en: {emergency_path}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        print(f"❌ Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()
