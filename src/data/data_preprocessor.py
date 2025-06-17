# src/data/data_preprocessor.py
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
import mediapipe as mp

class EmognitionDataProcessor:
    """
    Procesador especializado para el dataset Emognition Wearable 2020.
    Maneja la extracción de frames de videos y preprocessamiento para el modelo.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config
        
        # Configuración de emociones
        self.emotions = config['emotions']['active_emotions']
        self.emotion_to_idx = config['emotions']['emotion_to_idx']
        self.num_classes = config['emotions']['num_classes']
        
        # Configuración de entrada del modelo
        self.input_size = config['model_input_size']
        self.sequence_length = config['model'][config['model']['architecture']].get('sequence_length', 1)
        
        # Inicializar detectores de cara
        self._init_face_detectors()
        
        # Configurar transformaciones
        self._setup_transforms()
        
        # Paths
        self.raw_data_path = Path(config['paths']['emognition_dataset'])
        self.processed_data_path = Path(config['paths']['processed_data'])
        
        logging.info(f"EmognitionDataProcessor inicializado para {self.num_classes} emociones")
        logging.info(f"Tamaño de entrada: {self.input_size}")
    
    def _init_face_detectors(self):
        """Inicializa los detectores de cara (OpenCV + MediaPipe)."""
        # OpenCV Haar Cascade (rápido, para fallback)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # MediaPipe Face Detection (más preciso)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector_mp = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        logging.info("Detectores de cara inicializados (OpenCV + MediaPipe)")
    
    def _setup_transforms(self):
        """Configura las transformaciones de datos."""
        # Transformaciones base (sin augmentation)
        self.base_transform = A.Compose([
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet norm
            ToTensorV2()
        ])
        
        # Transformaciones con augmentation para entrenamiento
        if self.config['training']['enable_augmentation']:
            aug_config = self.config['training']['augmentation']
            self.train_transform = A.Compose([
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
                A.Rotate(limit=aug_config['rotation_range'], p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=aug_config['width_shift_range'],
                    scale_limit=0.0,
                    rotate_limit=0,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0.0),
                A.RandomResizedCrop(
                    height=self.input_size[0], 
                    width=self.input_size[1], 
                    scale=(0.9, 1.1),
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.train_transform = self.base_transform
    
    def detect_face_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detecta cara usando OpenCV Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Tomar la cara más grande
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        return None
    
    def detect_face_mediapipe(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detecta cara usando MediaPipe (más preciso)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector_mp.process(rgb_frame)
        
        if results.detections:
            # Tomar la primera detección (más confiable)
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convertir coordenadas relativas a absolutas
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            return (x, y, width, height)
        return None
    
    def extract_face(self, frame: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extrae la región facial del frame con padding.
        
        Args:
            frame: Frame de entrada
            padding: Padding adicional alrededor de la cara
            
        Returns:
            Frame con solo la cara extraída o None si no se detecta
        """
        # Intentar con MediaPipe primero (más preciso)
        face_coords = self.detect_face_mediapipe(frame)
        
        # Fallback a OpenCV si MediaPipe falla
        if face_coords is None:
            face_coords = self.detect_face_opencv(frame)
        
        if face_coords is None:
            return None
        
        x, y, w, h = face_coords
        
        # Aplicar padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Calcular coordenadas con padding
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        # Extraer región facial
        face_region = frame[y1:y2, x1:x2]
        
        return face_region
    
    def process_video_file(self, video_path: Path, emotion: str, 
                          max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Procesa un archivo de video y extrae frames con caras.
        
        Args:
            video_path: Path al archivo de video
            emotion: Emoción correspondiente al video
            max_frames: Número máximo de frames a extraer
            
        Returns:
            Lista de frames procesados
        """
        if not video_path.exists():
            logging.warning(f"Video no encontrado: {video_path}")
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning(f"No se pudo abrir el video: {video_path}")
            return []
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular intervalo para extraer frames uniformemente
        if max_frames and total_frames > max_frames:
            interval = total_frames // max_frames
        else:
            interval = 1
            max_frames = total_frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraer frames a intervalos regulares
            if frame_count % interval == 0:
                face_frame = self.extract_face(frame)
                if face_frame is not None:
                    frames.append(face_frame)
                
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        logging.info(f"Extraídos {len(frames)} frames de {video_path}")
        return frames
    
    def process_emognition_dataset(self, max_frames_per_video: int = 30) -> Dict[str, Any]:
        """
        Procesa todo el dataset Emognition y crea datasets de entrenamiento.
        
        Args:
            max_frames_per_video: Máximo de frames por video
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado en: {self.raw_data_path}")
        
        processed_data = {
            'train': {'frames': [], 'labels': []},
            'val': {'frames': [], 'labels': []},
            'test': {'frames': [], 'labels': []}
        }
        
        stats = {
            'total_videos': 0,
            'total_frames': 0,
            'frames_per_emotion': {emotion: 0 for emotion in self.emotions},
            'failed_videos': []
        }
        
        # Buscar archivos de video en el dataset
        video_files = []
        for emotion in self.emotions:
            emotion_path = self.raw_data_path / emotion
            if emotion_path.exists():
                videos = list(emotion_path.glob("*.mp4")) + list(emotion_path.glob("*.avi"))
                for video in videos:
                    video_files.append((video, emotion))
        
        if not video_files:
            # Estructura alternativa del dataset Emognition
            # Buscar en subdirectorios de participantes
            participant_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]
            for participant_dir in participant_dirs:
                for video_file in participant_dir.glob("*.mp4"):
                    # Inferir emoción del nombre del archivo o metadatos
                    emotion = self._infer_emotion_from_filename(video_file.name)
                    if emotion and emotion in self.emotions:
                        video_files.append((video_file, emotion))
        
        logging.info(f"Encontrados {len(video_files)} videos para procesar")
        
        # Procesar cada video
        for i, (video_path, emotion) in enumerate(video_files):
            try:
                frames = self.process_video_file(video_path, emotion, max_frames_per_video)
                
                if frames:
                    # Dividir en train/val/test
                    split = self._determine_split(i, len(video_files))
                    
                    for frame in frames:
                        processed_data[split]['frames'].append(frame)
                        processed_data[split]['labels'].append(self.emotion_to_idx[emotion])
                    
                    stats['total_frames'] += len(frames)
                    stats['frames_per_emotion'][emotion] += len(frames)
                else:
                    stats['failed_videos'].append(str(video_path))
                
                stats['total_videos'] += 1
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Procesados {i + 1}/{len(video_files)} videos")
                
            except Exception as e:
                logging.error(f"Error procesando {video_path}: {e}")
                stats['failed_videos'].append(str(video_path))
        
        # Guardar datos procesados
        self._save_processed_data(processed_data)
        
        # Guardar estadísticas
        self._save_stats(stats)
        
        logging.info(f"Procesamiento completado: {stats['total_frames']} frames de {stats['total_videos']} videos")
        return stats
    
    def _infer_emotion_from_filename(self, filename: str) -> Optional[str]:
        """Infiere la emoción del nombre del archivo."""
        filename_lower = filename.lower()
        for emotion in self.emotions:
            if emotion.lower() in filename_lower:
                return emotion
        return None
    
    def _determine_split(self, index: int, total: int) -> str:
        """Determina el split (train/val/test) para un índice dado."""
        train_split = self.config['training']['train_split']
        val_split = self.config['training']['val_split']
        
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        if index < train_end:
            return 'train'
        elif index < val_end:
            return 'val'
        else:
            return 'test'
    
    def _save_processed_data(self, processed_data: Dict[str, Any]):
        """Guarda los datos procesados."""
        for split in ['train', 'val', 'test']:
            split_path = self.processed_data_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            
            # Guardar frames como arrays numpy
            frames_path = split_path / 'frames.npy'
            labels_path = split_path / 'labels.npy'
            
            np.save(frames_path, processed_data[split]['frames'])
            np.save(labels_path, processed_data[split]['labels'])
            
            logging.info(f"Guardados {len(processed_data[split]['frames'])} frames en {split_path}")
    
    def _save_stats(self, stats: Dict[str, Any]):
        """Guarda las estadísticas del procesamiento."""
        stats_path = self.processed_data_path / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Estadísticas guardadas en {stats_path}")


class EmognitionDataset(Dataset):
    """
    Dataset de PyTorch para datos procesados de Emognition.
    Maneja secuencias de frames para el modelo CNN-RNN.
    """
    
    def __init__(self, split: str, config: Dict[str, Any], transform=None):
        self.split = split
        self.config = config
        self.transform = transform
        
        # Configuración del modelo
        self.sequence_length = config['model'][config['model']['architecture']].get('sequence_length', 1)
        
        # Cargar datos
        self._load_data()
        
        logging.info(f"Dataset {split} cargado: {len(self.frames)} frames")
    
    def _load_data(self):
        """Carga los datos preprocesados."""
        processed_path = Path(self.config['paths']['processed_data'])
        split_path = processed_path / self.split
        
        frames_path = split_path / 'frames.npy'
        labels_path = split_path / 'labels.npy'
        
        if not frames_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Datos procesados no encontrados en {split_path}")
        
        self.frames = np.load(frames_path, allow_pickle=True)
        self.labels = np.load(labels_path)
        
        # Convertir a lista si es necesario
        if isinstance(self.frames, np.ndarray) and self.frames.dtype == object:
            self.frames = self.frames.tolist()
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene un item del dataset.
        
        Returns:
            Tuple con (secuencia_frames, label)
        """
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convertir frame a formato adecuado
        if isinstance(frame, np.ndarray):
            # Asegurar que el frame esté en formato RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=frame)
            frame_tensor = transformed['image']
        else:
            # Transformación básica si no se especifica
            frame = cv2.resize(frame, (224, 224))
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Crear secuencia de frames para el modelo CNN-RNN
        if self.sequence_length > 1:
            # Duplicar el frame para crear una secuencia temporal
            # En implementación futura, aquí se cargarían frames consecutivos
            frame_sequence = frame_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        else:
            frame_sequence = frame_tensor.unsqueeze(0)
        
        return frame_sequence, torch.tensor(label, dtype=torch.long)


def create_data_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Crea los DataLoaders para entrenamiento, validación y test.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Diccionario con DataLoaders para cada split
    """
    # Crear procesador de datos
    processor = EmognitionDataProcessor(config)
    
    # Configuración de entrenamiento
    train_config = config['training']
    
    # Crear transforms
    train_transform = processor.train_transform
    val_transform = processor.base_transform
    
    # Crear datasets
    datasets = {
        'train': EmognitionDataset('train', config, transform=train_transform),
        'val': EmognitionDataset('val', config, transform=val_transform),
        'test': EmognitionDataset('test', config, transform=val_transform)
    }
    
    # Crear DataLoaders
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        batch_size = train_config['effective_batch_size'] if split == 'train' else train_config['effective_batch_size'] // 2
        
        dataloaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=train_config.get('pin_memory', True),
            drop_last=(split == 'train')
        )
    
    logging.info("DataLoaders creados exitosamente")
    return dataloaders