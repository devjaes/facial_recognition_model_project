# src/data/fer2013_processor.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class FER2013Processor:
    """
    Procesador especializado para el dataset FER2013.
    Maneja la estructura específica: data/raw/fer2013/images/{train,validation,images}/
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Mapeo de emociones FER2013 (índices estándar)
        self.fer2013_emotions = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # Configurar mapeo según configuración del proyecto
        self._setup_emotion_mapping()
        
        # Paths del dataset
        self.fer2013_root = Path(config['paths']['data_root']) / 'raw' / 'fer2013'
        self.images_dir = self.fer2013_root / 'images'
        self.processed_dir = Path(config['paths']['processed_data'])
        
        # Verificar estructura
        self._verify_dataset_structure()
        
        # Configurar transformaciones
        self._setup_transforms()
        
        logging.info("FER2013Processor inicializado")
        logging.info(f"Dataset path: {self.fer2013_root}")
        logging.info(f"Emociones configuradas: {list(self.emotion_mapping.values())}")
    
    def _setup_emotion_mapping(self):
        """Configura el mapeo de emociones según la configuración del proyecto."""
        active_emotions = self.config['emotions']['active_emotions']
        
        # Mapeo de FER2013 a emociones configuradas en el proyecto
        fer_to_project = {
            'angry': 'anger',      # FER2013 -> Proyecto
            'disgust': 'disgust',
            'fear': 'fear', 
            'happy': 'happy',
            'sad': 'sadness',      # FER2013 -> Proyecto
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        # Crear mapeo final solo para emociones activas
        self.emotion_mapping = {}
        self.reverse_mapping = {}
        
        new_idx = 0
        for fer_idx, fer_emotion in self.fer2013_emotions.items():
            project_emotion = fer_to_project.get(fer_emotion, fer_emotion)
            
            if project_emotion in active_emotions:
                self.emotion_mapping[fer_idx] = new_idx
                self.reverse_mapping[new_idx] = project_emotion
                new_idx += 1
        
        # Actualizar configuración con el número real de clases
        self.num_classes = len(self.reverse_mapping)
        self.config['emotions']['num_classes'] = self.num_classes
        
        logging.info(f"Mapeo de emociones FER2013 -> Proyecto: {self.reverse_mapping}")
    
    def _verify_dataset_structure(self):
        """Verifica que la estructura del dataset sea correcta."""
        required_dirs = ['train', 'validation']
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.images_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            raise FileNotFoundError(f"Directorios faltantes en FER2013: {missing_dirs}")
        
        # Verificar que hay imágenes
        train_images = list((self.images_dir / 'train').rglob('*.jpg')) + \
                      list((self.images_dir / 'train').rglob('*.png'))
        val_images = list((self.images_dir / 'validation').rglob('*.jpg')) + \
                    list((self.images_dir / 'validation').rglob('*.png'))
        
        logging.info(f"Imágenes encontradas - Train: {len(train_images)}, Validation: {len(val_images)}")
        
        if len(train_images) == 0:
            raise ValueError("No se encontraron imágenes en el directorio train")
    
    def _setup_transforms(self):
        """Configura las transformaciones de datos."""
        # Tamaño de entrada según configuración del modelo
        input_size = self.config['model'][self.config['model']['architecture']]['input_size']
        
        # Transformaciones base (sin augmentation)
        self.base_transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Transformaciones con augmentation para entrenamiento
        if self.config['training']['enable_augmentation']:
            aug_config = self.config['training']['augmentation']
            self.train_transform = A.Compose([
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Rotate(limit=aug_config['rotation_range'], p=0.6),
                A.HorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomResizedCrop(
                    height=input_size[0], 
                    width=input_size[1], 
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                    p=0.5
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.train_transform = self.base_transform
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analiza la distribución del dataset FER2013."""
        analysis = {
            'total_images': 0,
            'distribution': {},
            'splits': {},
            'image_info': {'min_size': None, 'max_size': None, 'avg_size': None}
        }
        
        splits = ['train', 'validation']
        
        for split in splits:
            split_dir = self.images_dir / split
            
            # Contar imágenes por emoción
            emotion_counts = {}
            total_split = 0
            image_sizes = []
            
            # Buscar en subdirectorios (estructura típica de FER2013)
            for emotion_dir in split_dir.iterdir():
                if emotion_dir.is_dir():
                    emotion_name = emotion_dir.name
                    
                    # Contar imágenes en este directorio de emoción
                    images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
                    count = len(images)
                    
                    if count > 0:
                        emotion_counts[emotion_name] = count
                        total_split += count
                        
                        # Analizar tamaños de imagen (muestra)
                        for i, img_path in enumerate(images[:10]):  # Solo primeras 10
                            try:
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    h, w = img.shape[:2]
                                    image_sizes.append((w, h))
                            except Exception as e:
                                logging.warning(f"Error leyendo {img_path}: {e}")
            
            analysis['splits'][split] = {
                'total': total_split,
                'emotions': emotion_counts
            }
            analysis['total_images'] += total_split
            
            # Información de tamaños
            if image_sizes:
                widths = [size[0] for size in image_sizes]
                heights = [size[1] for size in image_sizes]
                
                analysis['image_info'] = {
                    'sample_sizes': image_sizes,
                    'width_range': (min(widths), max(widths)),
                    'height_range': (min(heights), max(heights))
                }
        
        # Distribución global
        for split_data in analysis['splits'].values():
            for emotion, count in split_data['emotions'].items():
                if emotion in analysis['distribution']:
                    analysis['distribution'][emotion] += count
                else:
                    analysis['distribution'][emotion] = count
        
        # Guardar análisis
        analysis_path = self.processed_dir / 'fer2013_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logging.info(f"Análisis del dataset guardado en: {analysis_path}")
        return analysis
    
    def visualize_dataset(self, analysis: Dict[str, Any]):
        """Crea visualizaciones del dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('FER2013 Dataset Analysis', fontsize=16)
        
        # 1. Distribución global de emociones
        emotions = list(analysis['distribution'].keys())
        counts = list(analysis['distribution'].values())
        
        axes[0, 0].bar(emotions, counts)
        axes[0, 0].set_title('Distribución Global de Emociones')
        axes[0, 0].set_xlabel('Emoción')
        axes[0, 0].set_ylabel('Número de Imágenes')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución por split
        splits_data = analysis['splits']
        emotions_in_splits = {}
        
        for split, data in splits_data.items():
            for emotion, count in data['emotions'].items():
                if emotion not in emotions_in_splits:
                    emotions_in_splits[emotion] = {}
                emotions_in_splits[emotion][split] = count
        
        # Crear dataframe para seaborn
        split_df = []
        for emotion in emotions_in_splits:
            for split in splits_data.keys():
                count = emotions_in_splits[emotion].get(split, 0)
                split_df.append({'Emotion': emotion, 'Split': split, 'Count': count})
        
        if split_df:
            import pandas as pd
            df = pd.DataFrame(split_df)
            sns.barplot(data=df, x='Emotion', y='Count', hue='Split', ax=axes[0, 1])
            axes[0, 1].set_title('Distribución por Split')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Porcentajes
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        
        axes[1, 0].pie(percentages, labels=emotions, autopct='%1.1f%%')
        axes[1, 0].set_title('Distribución Porcentual')
        
        # 4. Estadísticas generales
        axes[1, 1].text(0.1, 0.8, f'Total de imágenes: {analysis["total_images"]:,}', fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Número de emociones: {len(emotions)}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Splits disponibles: {list(splits_data.keys())}', fontsize=12)
        
        if analysis['image_info']['sample_sizes']:
            sample_size = analysis['image_info']['sample_sizes'][0]
            axes[1, 1].text(0.1, 0.5, f'Tamaño típico: {sample_size[0]}x{sample_size[1]}', fontsize=12)
        
        axes[1, 1].text(0.1, 0.4, f'Clases balanceadas: {"Sí" if max(counts)/min(counts) < 2 else "No"}', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Estadísticas Generales')
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = self.processed_dir / 'fer2013_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info(f"Visualización guardada en: {viz_path}")
    
    def process_fer2013_dataset(self) -> Dict[str, Any]:
        """
        Procesa el dataset FER2013 completo y crea datasets de entrenamiento.
        """
        logging.info("Iniciando procesamiento del dataset FER2013...")
        
        # Analizar dataset primero
        analysis = self.analyze_dataset()
        
        # Crear visualizaciones
        self.visualize_dataset(analysis)
        
        # Procesar imágenes
        processed_data = {
            'train': {'images': [], 'labels': []},
            'val': {'images': [], 'labels': []},
            'test': {'images': [], 'labels': []}  # Usaremos parte de validation como test
        }
        
        stats = {
            'processed_images': 0,
            'skipped_images': 0,
            'class_distribution': {split: {} for split in ['train', 'val', 'test']},
            'original_analysis': analysis
        }
        
        # Procesar train split
        train_dir = self.images_dir / 'train'
        train_images, train_labels = self._process_split_directory(train_dir, 'train')
        processed_data['train']['images'] = train_images
        processed_data['train']['labels'] = train_labels
        
        # Procesar validation split (dividir en val y test)
        val_dir = self.images_dir / 'validation'
        val_images, val_labels = self._process_split_directory(val_dir, 'validation')
        
        # Dividir validation en val (70%) y test (30%)
        total_val = len(val_images)
        val_split_idx = int(total_val * 0.7)
        
        processed_data['val']['images'] = val_images[:val_split_idx]
        processed_data['val']['labels'] = val_labels[:val_split_idx]
        processed_data['test']['images'] = val_images[val_split_idx:]
        processed_data['test']['labels'] = val_labels[val_split_idx:]
        
        # Calcular estadísticas
        for split in ['train', 'val', 'test']:
            stats['processed_images'] += len(processed_data[split]['images'])
            
            # Distribución de clases
            label_counts = Counter(processed_data[split]['labels'])
            for label, count in label_counts.items():
                emotion_name = self.reverse_mapping[label]
                stats['class_distribution'][split][emotion_name] = count
        
        # Guardar datos procesados
        self._save_processed_data(processed_data)
        
        # Calcular pesos de clase para entrenamiento balanceado
        class_weights = self._calculate_class_weights(processed_data['train']['labels'])
        stats['class_weights'] = class_weights
        
        # Guardar estadísticas
        stats_path = self.processed_dir / 'fer2013_processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logging.info(f"Procesamiento completado: {stats['processed_images']} imágenes procesadas")
        logging.info(f"Distribución final - Train: {len(processed_data['train']['images'])}, "
                    f"Val: {len(processed_data['val']['images'])}, Test: {len(processed_data['test']['images'])}")
        
        return stats
    
    def _process_split_directory(self, split_dir: Path, split_name: str) -> Tuple[List[np.ndarray], List[int]]:
        """Procesa un directorio de split específico."""
        images = []
        labels = []
        
        logging.info(f"Procesando split: {split_name}")
        
        # Buscar subdirectorios de emociones
        emotion_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for emotion_dir in emotion_dirs:
            emotion_name = emotion_dir.name.lower()
            
            # Mapear emoción FER2013 a índice del proyecto
            fer_emotion_idx = None
            for idx, fer_emotion in self.fer2013_emotions.items():
                if fer_emotion == emotion_name:
                    fer_emotion_idx = idx
                    break
            
            if fer_emotion_idx is None or fer_emotion_idx not in self.emotion_mapping:
                logging.warning(f"Emoción {emotion_name} no encontrada en mapeo, omitiendo...")
                continue
            
            # Obtener índice mapeado
            mapped_label = self.emotion_mapping[fer_emotion_idx]
            
            # Procesar imágenes en este directorio
            image_files = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
            
            for img_path in image_files:
                try:
                    # Leer imagen
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    # Convertir BGR a RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Redimensionar a tamaño estándar (48x48 como FER2013 original)
                    image = cv2.resize(image, (48, 48))
                    
                    images.append(image)
                    labels.append(mapped_label)
                    
                except Exception as e:
                    logging.warning(f"Error procesando {img_path}: {e}")
                    continue
            
            logging.info(f"  {emotion_name}: {len([l for l in labels if l == mapped_label])} imágenes")
        
        return images, labels
    
    def _calculate_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calcula pesos de clase para entrenamiento balanceado."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        
        weight_dict = {label: weight for label, weight in zip(unique_labels, class_weights)}
        
        logging.info("Pesos de clase calculados:")
        for label, weight in weight_dict.items():
            emotion_name = self.reverse_mapping[label]
            logging.info(f"  {emotion_name}: {weight:.3f}")
        
        return weight_dict
    
    def _save_processed_data(self, processed_data: Dict[str, Any]):
        """Guarda los datos procesados."""
        for split in ['train', 'val', 'test']:
            split_path = self.processed_dir / split
            split_path.mkdir(parents=True, exist_ok=True)
            
            # Convertir listas a arrays numpy
            images_array = np.array(processed_data[split]['images'], dtype=np.uint8)
            labels_array = np.array(processed_data[split]['labels'], dtype=np.int64)
            
            # Guardar
            np.save(split_path / 'images.npy', images_array)
            np.save(split_path / 'labels.npy', labels_array)
            
            logging.info(f"Split {split} guardado: {len(images_array)} imágenes")
        
        # Guardar metadatos
        metadata = {
            'emotion_mapping': self.reverse_mapping,
            'num_classes': self.num_classes,
            'image_shape': (48, 48, 3),
            'dataset_name': 'FER2013',
            'processing_date': str(pd.Timestamp.now())
        }
        
        metadata_path = self.processed_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class FER2013Dataset(Dataset):
    """Dataset de PyTorch para FER2013 procesado."""
    
    def __init__(self, split: str, config: Dict[str, Any], transform=None):
        self.split = split
        self.config = config
        self.transform = transform
        
        # Cargar datos
        self._load_data()
        
        logging.info(f"FER2013Dataset {split} cargado: {len(self.images)} imágenes")
    
    def _load_data(self):
        """Carga los datos preprocesados."""
        processed_path = Path(self.config['paths']['processed_data'])
        split_path = processed_path / self.split
        
        images_path = split_path / 'images.npy'
        labels_path = split_path / 'labels.npy'
        
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Datos procesados no encontrados en {split_path}")
        
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        
        # Cargar metadatos
        metadata_path = processed_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtiene un item del dataset."""
        image = self.images[idx]
        label = self.labels[idx]
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
        else:
            # Transformación básica si no se especifica
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


def create_fer2013_data_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders para FER2013.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Diccionario con DataLoaders para cada split
    """
    # Crear procesador y procesar datos si es necesario
    processor = FER2013Processor(config)
    
    # Verificar si ya están procesados
    processed_path = Path(config['paths']['processed_data'])
    if not (processed_path / 'train' / 'images.npy').exists():
        logging.info("Datos no procesados, procesando FER2013...")
        processor.process_fer2013_dataset()
    
    # Configuración de entrenamiento
    train_config = config['training']
    
    # Crear transforms
    train_transform = processor.train_transform
    val_transform = processor.base_transform
    
    # Crear datasets
    datasets = {
        'train': FER2013Dataset('train', config, transform=train_transform),
        'val': FER2013Dataset('val', config, transform=val_transform),
        'test': FER2013Dataset('test', config, transform=val_transform)
    }
    
    # Crear DataLoaders
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        batch_size = train_config['effective_batch_size'] if split == 'train' else max(1, train_config['effective_batch_size'] // 2)
        
        dataloaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=train_config.get('pin_memory', True),
            drop_last=(split == 'train')
        )
    
    logging.info("DataLoaders FER2013 creados exitosamente")
    return dataloaders