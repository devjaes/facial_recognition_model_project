# 🎭 Reconocimiento de Emociones Faciales con FER2013

Sistema avanzado de reconocimiento de emociones faciales usando Deep Learning, optimizado específicamente para el dataset FER2013.

## 🎯 Características Principales

- ✅ **7 Emociones**: Anger, Disgust, Fear, Happy, Sadness, Surprise, Neutral
- ✅ **Múltiples Arquitecturas**: CNN básico, CNN-RNN híbrido, Transfer Learning
- ✅ **Multiplataforma**: macOS M3, Linux, Windows (CUDA/CPU)
- ✅ **Tiempo Real**: Reconocimiento via webcam
- ✅ **Web App**: Interface web para demostración
- ✅ **Completamente Parametrizable**: Configuración YAML

## 🚀 Quick Start

### 1. Setup del Entorno

```bash
# Clonar o crear directorio del proyecto
mkdir facial_emotion_recognition_fer2013
cd facial_emotion_recognition_fer2013

# Ejecutar setup automático
python scripts/setup_environment.py

# Activar entorno (según tu plataforma)
conda activate emotion_recognition_macos    # macOS
conda activate emotion_recognition_linux    # Linux  
conda activate emotion_recognition_windows  # Windows
```

### 2. Configurar Dataset FER2013

Si ya tienes FER2013 descargado de Kaggle:

```bash
# Tu estructura actual debe ser:
data/raw/fer2013/images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### 3. Analizar y Procesar Dataset

```bash
# Analizar estructura y distribución del dataset
python scripts/setup_fer2013.py

# Esto creará:
# - Análisis de distribución de emociones
# - Visualizaciones del dataset
# - Datos procesados listos para entrenamiento
```

### 4. Entrenar Modelo

```bash
# Entrenamiento rápido para testing (5 épocas)
python scripts/train_fer2013.py --model cnn_basic --quick-test

# Entrenamiento completo - CNN básico
python scripts/train_fer2013.py --model cnn_basic --epochs 100

# Entrenamiento completo - CNN-RNN híbrido (recomendado)
python scripts/train_fer2013.py --model cnn_rnn_hybrid --epochs 150

# Transfer Learning con MobileNetV2
python scripts/train_fer2013.py --model transfer_learning --epochs 50
```

### 5. Lanzar Aplicación Web

```bash
# Aplicación web con interfaz completa
python src/web/app.py

# Abrir navegador en: http://localhost:5000
# Funciona también en móviles desde la red local
```

## 📊 Objetivos de Performance

| Métrica | Baseline | Target | Excelente |
|---------|----------|--------|-----------|
| **Accuracy** | 65% | 75% | 80%+ |
| **FPS (Tiempo Real)** | 10 | 15 | 20+ |
| **Latencia** | <100ms | <50ms | <30ms |

## 🧠 Arquitecturas Disponibles

### 1. CNN Básico (`cnn_basic`)
- **Rápido y ligero**
- Ideal para testing y baseline
- Input: 48x48 (tamaño original FER2013)
- ~500K parámetros

### 2. CNN-RNN Híbrido (`cnn_rnn_hybrid`) ⭐ Recomendado
- **Mejor accuracy**
- Combina CNN para features + LSTM para temporalidad
- Input: 224x224 (upscaled)
- ~2M parámetros

### 3. Transfer Learning (`transfer_learning`)
- **Pre-entrenado en ImageNet**
- MobileNetV2 backbone + clasificador custom
- Excelente para datasets pequeños
- ~3M parámetros

## ⚙️ Configuración Avanzada

### Modificar Emociones

```yaml
# config/fer2013_config.yaml
emotions:
  active_emotions: ["anger", "happy", "neutral"]  # Solo 3 emociones
  num_classes: 3
```

### Ajustar Hiperparámetros

```yaml
training:
  batch_size: 64        # Incrementar si tienes GPU potente
  learning_rate: 0.0005 # Reducir para mejor convergencia
  epochs: 200          # Más épocas para mejor accuracy
```

### Data Augmentation

```bash
# Sin augmentation (rápido)
python scripts/train_fer2013.py --augmentation none

# Augmentation ligero
python scripts/train_fer2013.py --augmentation light

# Augmentation agresivo (mejor accuracy)
python scripts/train_fer2013.py --augmentation aggressive
```

## 🛠️ Comandos Útiles

```bash
# Ver todos los comandos disponibles
make help

# Monitorear entrenamiento
tensorboard --logdir logs/training

# Evaluar modelo entrenado
python scripts/evaluate_model.py --model models/trained/best_model.pth

# Test de performance en tiempo real
python scripts/benchmark.py --model models/trained/best_model.pth

# Optimizar modelo para producción
python scripts/optimize_model.py --method quantization

# Limpiar archivos temporales
make clean
```

## 📈 Monitoreo y Debugging

### TensorBoard

```bash
# Ver métricas de entrenamiento
tensorboard --logdir logs/training

# Métricas disponibles:
# - Loss (train/validation)
# - Accuracy (train/validation) 
# - Learning Rate
# - Confusion Matrices
# - Distribución de pesos
```

### Logs

```bash
# Ver logs de entrenamiento
tail -f logs/training/fer2013_training_*.log

# Ver logs de aplicación web
tail -f logs/web_app/web_app.log

# Ver todos los logs
find logs/ -name "*.log" -exec tail -n 20 {} \;
```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Error "Dataset no encontrado"
```bash
# Verificar estructura
ls -la data/raw/fer2013/images/

# Debe mostrar: train/ validation/
# Si no, revisar extracción del ZIP de Kaggle
```

#### 2. Error de GPU/CUDA
```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Forzar CPU si hay problemas
python scripts/train_fer2013.py --device cpu
```

#### 3. Out of Memory
```bash
# Reducir batch size
python scripts/train_fer2013.py --batch-size 16

# O usar modelo más pequeño
python scripts/train_fer2013.py --model cnn_basic
```

#### 4. Accuracy muy baja (<50%)
- Verificar que las etiquetas estén correctas
- Incrementar data augmentation
- Probar diferentes learning rates
- Verificar distribución de clases

### Configuración por Dispositivo

#### macOS M3 Pro
```bash
# Configuración optimizada
python scripts/train_fer2013.py \
  --model cnn_rnn_hybrid \
  --batch-size 32 \
  --device mps
```

#### Linux/Windows con GPU
```bash
# Configuración optimizada
python scripts/train_fer2013.py \
  --model cnn_rnn_hybrid \
  --batch-size 64 \
  --device cuda
```

#### CPU (cualquier plataforma)
```bash
# Configuración para CPU
python scripts/train_fer2013.py \
  --model cnn_basic \
  --batch-size 16 \
  --device cpu
```

## 📚 Estructura del Proyecto

```
facial_emotion_recognition_fer2013/
├── 📁 config/                    # Configuraciones
│   └── fer2013_config.yaml       # Config principal FER2013
├── 📁 data/
│   ├── raw/fer2013/              # Dataset original
│   └── processed/                # Datos procesados
├── 📁 src/
│   ├── data/fer2013_processor.py # Procesador específico FER2013
│   ├── models/                   # Arquitecturas de modelos
│   ├── training/                 # Sistema de entrenamiento
│   ├── inference/                # Inferencia y predicción
│   └── web/                      # Aplicación web
├── 📁 scripts/
│   ├── setup_fer2013.py          # Setup específico FER2013
│   ├── train_fer2013.py          # Entrenamiento FER2013
│   └── evaluate_model.py         # Evaluación
├── 📁 models/trained/             # Modelos entrenados
├── 📁 logs/                      # Logs del sistema
└── 📁 experiments/               # Resultados de experimentos
```

## 🎯 Roadmap de Desarrollo

### Fase 1: Setup y Baseline ✅
- [x] Configuración del entorno
- [x] Procesamiento de FER2013
- [x] Modelo básico funcionando

### Fase 2: Optimización 🔄
- [ ] CNN-RNN híbrido optimizado
- [ ] Transfer learning fine-tuning
- [ ] Hyperparameter tuning

### Fase 3: Aplicación 📱
- [ ] Web app completa
- [ ] Tiempo real optimizado
- [ ] Interface mobile-friendly

### Fase 4: Producción 🚀
- [ ] Model quantization
- [ ] Docker deployment
- [ ] API REST

## 🤝 Contribución

1. **Fork** el proyecto
2. **Clone** tu fork
3. **Crear branch** para tu feature
4. **Commit** tus cambios
5. **Push** y crear **Pull Request**

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **FER2013 Dataset**: [Kaggle FER2013](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- **PyTorch**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes
- **MediaPipe**: Detección facial avanzada

---

## 📞 Soporte

¿Problemas o preguntas?

1. 📖 Revisar este README
2. 🔍 Buscar en logs: `logs/`
3. 🧪 Probar modo quick-test
4. 💬 Crear issue en GitHub

**¡Happy coding! 🎭✨**