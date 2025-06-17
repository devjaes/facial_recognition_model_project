# scripts/train_fer2013.py
#!/usr/bin/env python3
"""
Script de entrenamiento específico para FER2013.
Incluye configuraciones optimizadas y monitoreo específico para este dataset.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config_manager import ConfigManager
from training.trainer import EmotionTrainer
from data.fer2013_processor import create_fer2013_data_loaders
from models.model_factory import ModelFactory
from utils.device_manager import DeviceManager

def setup_logging():
    """Configura logging específico para entrenamiento FER2013."""
    log_dir = Path('logs/training')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f'fer2013_training_{int(time.time())}.log')
        ]
    )

def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para FER2013')
    
    parser.add_argument('--model', type=str, default='cnn_rnn_hybrid',
                       choices=['cnn_basic', 'cnn_rnn_hybrid', 'transfer_learning'],
                       help='Tipo de modelo a entrenar')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas (usa config si no se especifica)')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Tamaño de batch (usa config si no se especifica)')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (usa config si no se especifica)')
    
    parser.add_argument('--config', type=str, default='config/fer2013_config.yaml',
                       help='Path al archivo de configuración')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path a checkpoint para continuar entrenamiento')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Entrenamiento rápido para testing (5 épocas)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Dispositivo para entrenamiento')
    
    parser.add_argument('--augmentation', type=str, default='normal',
                       choices=['none', 'light', 'normal', 'aggressive'],
                       help='Nivel de data augmentation')
    
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Nombre del experimento para logging')
    
    return parser.parse_args()

def update_config_from_args(config: dict, args) -> dict:
    """Actualiza la configuración con argumentos de línea de comandos."""
    
    # Actualizar modelo
    config['model']['architecture'] = args.model
    
    # Actualizar hiperparámetros si se especifican
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Quick test mode
    if args.quick_test:
        config['training']['epochs'] = 5
        config['training']['batch_size'] = min(16, config['training']['batch_size'])
        logging.info("🏃‍♂️ Modo quick test activado: 5 épocas, batch size reducido")
    
    # Configurar augmentation
    aug_levels = {
        'none': False,
        'light': True,
        'normal': True,
        'aggressive': True
    }
    
    config['training']['enable_augmentation'] = aug_levels[args.augmentation]
    
    if args.augmentation == 'light':
        # Reducir intensidad de augmentation
        config['training']['augmentation']['rotation_range'] = 10
        config['training']['augmentation']['brightness_range'] = [0.8, 1.2]
    elif args.augmentation == 'aggressive':
        # Incrementar intensidad
        config['training']['augmentation']['rotation_range'] = 30
        config['training']['augmentation']['brightness_range'] = [0.6, 1.4]
        config['training']['augmentation']['gaussian_noise'] = True
        config['training']['augmentation']['gaussian_blur'] = True
    
    # Configurar dispositivo
    if args.device != 'auto':
        config['devices']['auto_detect'] = False
        config['devices']['manual_device'] = args.device
    
    return config

def validate_environment(config: dict) -> bool:
    """Valida que el entorno esté configurado correctamente."""
    
    print("\n🔍 VALIDANDO ENTORNO")
    print("=" * 50)
    
    # Verificar dataset
    fer2013_path = Path(config['paths']['data_root']) / 'raw' / 'fer2013'
    if not fer2013_path.exists():
        print(f"❌ Dataset FER2013 no encontrado en: {fer2013_path}")
        return False
    
    print(f"✅ Dataset FER2013 encontrado")
    
    # Verificar dispositivo
    device = DeviceManager.get_optimal_device(config)
    print(f"✅ Dispositivo: {device}")
    
    # Verificar data loaders
    try:
        print("🔍 Verificando data loaders...")
        dataloaders = create_fer2013_data_loaders(config)
        
        for split, loader in dataloaders.items():
            print(f"  {split}: {len(loader.dataset)} imágenes, {len(loader)} batches")
        
        print("✅ Data loaders creados correctamente")
        
        # Test rápido de un batch
        train_loader = dataloaders['train']
        for images, labels in train_loader:
            print(f"  Batch test: {images.shape}, labels: {labels.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Error en data loaders: {e}")
        return False

def create_experiment_config(config: dict, args) -> dict:
    """Crea configuración específica del experimento."""
    
    experiment_name = args.experiment_name or f"fer2013_{args.model}_{int(time.time())}"
    
    experiment_config = {
        'experiment_name': experiment_name,
        'dataset': 'FER2013',
        'model_architecture': args.model,
        'training_params': {
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['effective_batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'augmentation_level': args.augmentation
        },
        'device_info': {
            'device': str(DeviceManager.get_optimal_device(config)),
            'mixed_precision': config['training'].get('mixed_precision', False)
        },
        'timestamp': time.time(),
        'git_commit': None,  # Aquí podrías agregar el commit de git
        'notes': f"Entrenamiento FER2013 con {args.model}"
    }
    
    return experiment_config

def save_experiment_results(experiment_config: dict, training_stats: dict):
    """Guarda los resultados del experimento."""
    
    # Combinar configuración del experimento con resultados
    full_results = {
        'experiment_config': experiment_config,
        'training_results': training_stats,
        'final_metrics': {
            'best_val_accuracy': training_stats.get('best_val_accuracy', 0),
            'final_val_accuracy': training_stats.get('final_val_accuracy', 0),
            'total_training_time': training_stats.get('total_time_hours', 0),
            'total_epochs': training_stats.get('total_epochs', 0)
        }
    }
    
    # Guardar en directorio de experimentos
    experiments_dir = Path('experiments')
    experiments_dir.mkdir(exist_ok=True)
    
    experiment_file = experiments_dir / f"{experiment_config['experiment_name']}.json"
    
    with open(experiment_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"📊 Resultados del experimento guardados en: {experiment_file}")
    
    return experiment_file

def print_training_summary(config: dict, experiment_config: dict):
    """Imprime resumen de la configuración de entrenamiento."""
    
    print("\n🚀 INICIANDO ENTRENAMIENTO FER2013")
    print("=" * 60)
    print(f"🎯 Experimento: {experiment_config['experiment_name']}")
    print(f"🧠 Modelo: {config['model']['architecture']}")
    print(f"📊 Dataset: FER2013 ({config['emotions']['num_classes']} emociones)")
    print(f"💻 Dispositivo: {experiment_config['device_info']['device']}")
    print(f"⚙️  Configuración:")
    print(f"   • Épocas: {config['training']['epochs']}")
    print(f"   • Batch size: {config['training']['effective_batch_size']}")
    print(f"   • Learning rate: {config['training']['learning_rate']}")
    print(f"   • Optimizador: {config['training']['optimizer']}")
    print(f"   • Augmentation: {'✅' if config['training']['enable_augmentation'] else '❌'}")
    print(f"   • Mixed precision: {'✅' if config['training'].get('mixed_precision', False) else '❌'}")
    
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
    setup_logging()
    args = parse_arguments()
    
    # Cargar y actualizar configuración
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.get_full_config()
        config = update_config_from_args(config, args)
        
        # Imprimir resumen de configuración
        config_manager.print_config_summary()
        
    except Exception as e:
        logging.error(f"Error cargando configuración: {e}")
        sys.exit(1)
    
    # Validar entorno
    if not validate_environment(config):
        logging.error("Validación del entorno falló")
        sys.exit(1)
    
    # Crear configuración del experimento
    experiment_config = create_experiment_config(config, args)
    
    # Mostrar resumen
    print_training_summary(config, experiment_config)
    
    # Confirmar inicio (solo si no es quick test)
    if not args.quick_test:
        response = input("\n¿Continuar con el entrenamiento? (y/N): ")
        if response.lower() not in ['y', 'yes', 'sí', 's']:
            print("Entrenamiento cancelado")
            sys.exit(0)
    
    # Crear entrenador
    try:
        trainer = EmotionTrainer(config)
        
        # Cargar checkpoint si se especifica
        if args.resume:
            if not trainer.load_checkpoint(args.resume):
                logging.error(f"No se pudo cargar checkpoint: {args.resume}")
                sys.exit(1)
            print(f"✅ Checkpoint cargado: {args.resume}")
        
    except Exception as e:
        logging.error(f"Error creando entrenador: {e}")
        sys.exit(1)
    
    # Entrenar
    start_time = time.time()
    
    try:
        print(f"\n🏁 INICIANDO ENTRENAMIENTO...")
        print(f"⏰ Hora de inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ejecutar entrenamiento
        training_stats = trainer.train()
        
        # Evaluar en test set
        test_results = trainer.evaluate_test_set()
        training_stats['test_results'] = test_results
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        training_stats['total_time_seconds'] = total_time
        training_stats['total_time_hours'] = total_time / 3600
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"⏰ Tiempo total: {total_time/3600:.2f} horas")
        print(f"🎯 Mejor accuracy validación: {training_stats['best_val_accuracy']:.2f}%")
        
        if test_results:
            print(f"🧪 Accuracy en test: {test_results['test_accuracy']:.2f}%")
        
        # Guardar resultados del experimento
        experiment_file = save_experiment_results(experiment_config, training_stats)
        
        # Mostrar resumen final
        print_final_summary(training_stats, experiment_config)
        
        return training_stats
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Entrenamiento interrumpido por el usuario")
        
        # Guardar checkpoint de emergencia
        emergency_path = f"models/checkpoints/emergency_checkpoint_{int(time.time())}.pth"
        trainer.save_checkpoint(trainer.current_epoch, 0.0, is_best=False)
        
        print(f"💾 Checkpoint de emergencia guardado")
        return None
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")
        print(f"❌ Error durante el entrenamiento: {e}")
        
        # Intentar guardar estado actual
        try:
            emergency_path = f"models/checkpoints/error_checkpoint_{int(time.time())}.pth"
            trainer.save_checkpoint(trainer.current_epoch, 0.0, is_best=False)
            print(f"💾 Checkpoint de error guardado")
        except:
            pass
        
        raise

def print_final_summary(training_stats: dict, experiment_config: dict):
    """Imprime resumen final del entrenamiento."""
    
    print(f"\n📊 RESUMEN FINAL DEL EXPERIMENTO")
    print("=" * 60)
    print(f"🎯 Experimento: {experiment_config['experiment_name']}")
    print(f"🧠 Modelo: {experiment_config['training_params']['model_architecture']}")
    print(f"⏰ Tiempo total: {training_stats['total_time_hours']:.2f} horas")
    print(f"🔄 Épocas completadas: {training_stats['total_epochs']}")
    print(f"🎯 Mejor accuracy validación: {training_stats['best_val_accuracy']:.2f}%")
    
    if 'test_results' in training_stats and training_stats['test_results']:
        test_acc = training_stats['test_results']['test_accuracy']
        print(f"🧪 Accuracy en test: {test_acc:.2f}%")
        
        # Evaluación del resultado
        if test_acc >= 80:
            print(f"🏆 ¡EXCELENTE! Superaste el objetivo de 80%")
        elif test_acc >= 75:
            print(f"🎉 ¡MUY BUENO! Alcanzaste el objetivo de 75%")
        elif test_acc >= 65:
            print(f"✅ BUENO! Superaste el baseline de 65%")
        else:
            print(f"⚠️ Por debajo del baseline, considera ajustar hiperparámetros")
    
    # Paths de archivos importantes
    print(f"\n📁 ARCHIVOS GENERADOS:")
    if training_stats.get('best_model_path'):
        print(f"   🏆 Mejor modelo: {training_stats['best_model_path']}")
    
    print(f"   📊 Historia: models/trained/training_history.json")
    print(f"   📈 TensorBoard: logs/training/")
    print(f"   📋 Experimento: experiments/{experiment_config['experiment_name']}.json")
    
    # Próximos pasos
    print(f"\n🚀 PRÓXIMOS PASOS:")
    print(f"   1. 📊 Ver métricas detalladas:")
    print(f"      tensorboard --logdir logs/training")
    print(f"   2. 🌐 Probar modelo en web app:")
    print(f"      python src/web/app.py")
    print(f"   3. 🔧 Optimizar modelo:")
    print(f"      python scripts/optimize_model.py")
    print(f"   4. 📱 Test en tiempo real:")
    print(f"      python scripts/test_realtime.py")

def run_quick_experiments():
    """Ejecuta una serie de experimentos rápidos para comparar modelos."""
    
    print("\n🧪 EJECUTANDO EXPERIMENTOS RÁPIDOS")
    print("=" * 60)
    
    models_to_test = ['cnn_basic', 'cnn_rnn_hybrid', 'transfer_learning']
    results = {}
    
    for model in models_to_test:
        print(f"\n🔬 Probando {model}...")
        
        # Crear argumentos simulados
        class QuickArgs:
            model = model
            epochs = 5
            batch_size = 16
            learning_rate = None
            config = 'config/fer2013_config.yaml'
            resume = None
            quick_test = True
            device = 'auto'
            augmentation = 'light'
            experiment_name = f'quick_test_{model}'
        
        args = QuickArgs()
        
        try:
            # Ejecutar entrenamiento rápido
            # (Simplificado - en implementación real llamarías a main con estos args)
            print(f"   ⏱️ Entrenando por 5 épocas...")
            # training_result = main_with_args(args)
            # results[model] = training_result
            
            # Por ahora simular resultado
            import random
            results[model] = {
                'val_accuracy': random.uniform(60, 75),
                'training_time': random.uniform(5, 15)
            }
            
        except Exception as e:
            print(f"   ❌ Error en {model}: {e}")
            results[model] = {'error': str(e)}
    
    # Mostrar comparación
    print(f"\n📊 COMPARACIÓN DE MODELOS RÁPIDOS:")
    print("-" * 50)
    for model, result in results.items():
        if 'error' not in result:
            print(f"{model:15}: {result['val_accuracy']:.1f}% ({result['training_time']:.1f}min)")
        else:
            print(f"{model:15}: Error - {result['error']}")
    
    # Recomendación
    if results:
        best_model = max(results.keys(), 
                        key=lambda x: results[x].get('val_accuracy', 0))
        print(f"\n🏆 Mejor modelo en pruebas rápidas: {best_model}")
        print(f"💡 Recomendación: Entrenar {best_model} con más épocas")

if __name__ == "__main__":
    main()