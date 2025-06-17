# scripts/setup_fer2013.py
#!/usr/bin/env python3
"""
Script para setup y análisis del dataset FER2013.
Procesa la estructura específica que tienes: data/raw/fer2013/images/{train,validation,images}/
"""

import os
import sys
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config_manager import ConfigManager
from data.fer2013_processor import FER2013Processor, create_fer2013_data_loaders

def setup_logging():
    """Configura logging para el script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/fer2013_setup.log')
        ]
    )

def verify_fer2013_structure(data_path: Path) -> bool:
    """Verifica la estructura del dataset FER2013."""
    
    print("\n🔍 VERIFICANDO ESTRUCTURA DEL DATASET FER2013")
    print("=" * 60)
    
    fer2013_path = data_path / 'raw' / 'fer2013'
    images_path = fer2013_path / 'images'
    
    if not fer2013_path.exists():
        print(f"❌ No se encontró el directorio FER2013: {fer2013_path}")
        return False
    
    if not images_path.exists():
        print(f"❌ No se encontró el directorio images: {images_path}")
        return False
    
    print(f"✅ Directorio FER2013 encontrado: {fer2013_path}")
    print(f"✅ Directorio images encontrado: {images_path}")
    
    # Verificar subdirectorios
    expected_dirs = ['train', 'validation']
    found_dirs = []
    
    for dir_name in expected_dirs:
        dir_path = images_path / dir_name
        if dir_path.exists():
            found_dirs.append(dir_name)
            print(f"✅ Subdirectorio encontrado: {dir_name}")
        else:
            print(f"❌ Subdirectorio faltante: {dir_name}")
    
    # Verificar directorio 'images' adicional si existe
    additional_images = images_path / 'images'
    if additional_images.exists():
        print(f"ℹ️  Directorio 'images' adicional encontrado: {additional_images}")
        # Verificar qué contiene
        content = list(additional_images.iterdir())
        if content:
            print(f"   Contiene {len(content)} elementos")
    
    # Contar imágenes en cada directorio
    print("\n📊 CONTEO DE IMÁGENES:")
    total_images = 0
    
    for dir_name in found_dirs:
        dir_path = images_path / dir_name
        emotion_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
        
        print(f"\n  📁 {dir_name.upper()}:")
        dir_total = 0
        
        for emotion_dir in emotion_dirs:
            emotion_name = emotion_dir.name
            images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
            count = len(images)
            dir_total += count
            
            print(f"    {emotion_name}: {count:,} imágenes")
        
        print(f"    TOTAL {dir_name}: {dir_total:,} imágenes")
        total_images += dir_total
    
    print(f"\n📈 TOTAL GENERAL: {total_images:,} imágenes")
    
    return len(found_dirs) >= 2 and total_images > 0

def analyze_fer2013_distribution(config: dict) -> dict:
    """Analiza la distribución del dataset FER2013."""
    
    print("\n📊 ANALIZANDO DISTRIBUCIÓN DEL DATASET")
    print("=" * 60)
    
    try:
        # Crear procesador
        processor = FER2013Processor(config)
        
        # Ejecutar análisis
        analysis = processor.analyze_dataset()
        
        print(f"✅ Análisis completado")
        print(f"📈 Total de imágenes: {analysis['total_images']:,}")
        print(f"🎭 Emociones encontradas: {list(analysis['distribution'].keys())}")
        
        # Mostrar distribución
        print("\n📊 DISTRIBUCIÓN POR EMOCIÓN:")
        for emotion, count in analysis['distribution'].items():
            percentage = (count / analysis['total_images']) * 100
            print(f"  {emotion:12}: {count:5,} imágenes ({percentage:5.1f}%)")
        
        # Mostrar distribución por split
        print("\n📂 DISTRIBUCIÓN POR SPLIT:")
        for split, data in analysis['splits'].items():
            print(f"  {split:12}: {data['total']:5,} imágenes")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        logging.error(f"Error en análisis: {e}")
        return {}

def create_sample_training(config: dict):
    """Crea un entrenamiento de muestra para verificar que todo funciona."""
    
    print("\n🚀 CREANDO ENTRENAMIENTO DE MUESTRA")
    print("=" * 60)
    
    try:
        # Crear data loaders
        print("📚 Creando data loaders...")
        dataloaders = create_fer2013_data_loaders(config)
        
        print(f"✅ Data loaders creados:")
        for split, loader in dataloaders.items():
            print(f"  {split:5}: {len(loader)} batches, {len(loader.dataset)} imágenes")
        
        # Test de un batch
        print("\n🧪 Testing un batch de entrenamiento...")
        train_loader = dataloaders['train']
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Image dtype: {images.dtype}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Sample labels: {labels[:5].tolist()}")
            break
        
        print("✅ Data loaders funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error creando data loaders: {e}")
        logging.error(f"Error en data loaders: {e}")
        return False

def generate_fer2013_report(config: dict, analysis: dict):
    """Genera un reporte completo del dataset FER2013."""
    
    print("\n📋 GENERANDO REPORTE COMPLETO")
    print("=" * 60)
    
    report = {
        'dataset_info': {
            'name': 'FER2013',
            'total_images': analysis.get('total_images', 0),
            'num_emotions': len(analysis.get('distribution', {})),
            'emotions': list(analysis.get('distribution', {}).keys()),
            'image_info': analysis.get('image_info', {})
        },
        'distribution': analysis.get('distribution', {}),
        'splits': analysis.get('splits', {}),
        'recommendations': []
    }
    
    # Análisis de balance de clases
    if 'distribution' in analysis:
        counts = list(analysis['distribution'].values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            report['class_balance'] = {
                'max_count': max_count,
                'min_count': min_count,
                'imbalance_ratio': imbalance_ratio,
                'is_balanced': imbalance_ratio < 3.0
            }
            
            # Recomendaciones
            if imbalance_ratio > 3.0:
                report['recommendations'].append(
                    "El dataset está desbalanceado. Considerar usar class weights o técnicas de balanceo."
                )
            
            if analysis['total_images'] < 10000:
                report['recommendations'].append(
                    "Dataset relativamente pequeño. Usar data augmentation agresiva."
                )
            
            # Recomendaciones de modelo
            if analysis['total_images'] < 30000:
                report['recommendations'].append(
                    "Para este tamaño de dataset, transfer learning puede ser más efectivo que entrenar desde cero."
                )
    
    # Guardar reporte
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / 'fer2013_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Reporte guardado en: {report_path}")
    
    # Mostrar resumen
    print("\n📊 RESUMEN DEL ANÁLISIS:")
    print(f"  📈 Total imágenes: {report['dataset_info']['total_images']:,}")
    print(f"  🎭 Emociones: {report['dataset_info']['num_emotions']}")
    print(f"  ⚖️  Balance de clases: {'✅ Balanceado' if report.get('class_balance', {}).get('is_balanced', False) else '⚠️ Desbalanceado'}")
    
    if report['recommendations']:
        print(f"\n💡 RECOMENDACIONES:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report

def main():
    """Función principal del script."""
    
    print("🎭 SETUP Y ANÁLISIS DE FER2013")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Cargar configuración
    print("⚙️  Cargando configuración...")
    try:
        config_manager = ConfigManager('config/fer2013_config.yaml')
        config = config_manager.get_full_config()
        print("✅ Configuración cargada")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        # Usar configuración por defecto
        config = {
            'paths': {'data_root': './data', 'processed_data': './data/processed'},
            'emotions': {'active_emotions': ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']},
            'model': {'cnn_rnn_hybrid': {'input_size': [224, 224]}},
            'training': {'enable_augmentation': True, 'effective_batch_size': 32, 'num_workers': 4, 'pin_memory': True}
        }
    
    # Verificar estructura del dataset
    data_path = Path(config['paths']['data_root'])
    if not verify_fer2013_structure(data_path):
        print("\n❌ La estructura del dataset no es correcta.")
        print("\n🔧 ESTRUCTURA ESPERADA:")
        print("data/")
        print("└── raw/")
        print("    └── fer2013/")
        print("        └── images/")
        print("            ├── train/")
        print("            │   ├── angry/")
        print("            │   ├── disgust/")
        print("            │   ├── fear/")
        print("            │   ├── happy/")
        print("            │   ├── sad/")
        print("            │   ├── surprise/")
        print("            │   └── neutral/")
        print("            └── validation/")
        print("                ├── angry/")
        print("                ├── disgust/")
        print("                ├── fear/")
        print("                ├── happy/")
        print("                ├── sad/")
        print("                ├── surprise/")
        print("                └── neutral/")
        sys.exit(1)
    
    # Analizar distribución
    analysis = analyze_fer2013_distribution(config)
    if not analysis:
        print("❌ No se pudo completar el análisis del dataset")
        sys.exit(1)
    
    # Crear entrenamiento de muestra
    if not create_sample_training(config):
        print("❌ Error en la creación de data loaders")
        sys.exit(1)
    
    # Generar reporte
    report = generate_fer2013_report(config, analysis)
    
    print("\n🎉 SETUP COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. 🏃 Entrenar modelo básico:")
    print("   python scripts/train_model.py --model cnn_basic --epochs 10")
    print("\n2. 🧠 Entrenar modelo híbrido:")
    print("   python scripts/train_model.py --model cnn_rnn_hybrid --epochs 50")
    print("\n3. 🔄 Entrenar con transfer learning:")
    print("   python scripts/train_model.py --model transfer_learning --epochs 30")
    print("\n4. 🌐 Lanzar aplicación web:")
    print("   python src/web/app.py")
    print("\n5. 📊 Ver métricas:")
    print("   tensorboard --logdir logs/training")
    
    print(f"\n🎯 OBJETIVOS DE ACCURACY PARA FER2013:")
    print(f"   • Baseline: 65%")
    print(f"   • Target: 75%")
    print(f"   • Excelente: 80%+")

if __name__ == "__main__":
    main()