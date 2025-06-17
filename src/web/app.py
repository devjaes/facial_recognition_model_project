# src/web/app.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import torch
import json
import logging
from pathlib import Path
import threading
import time
from typing import Dict, Any, Optional

from ..config_manager import ConfigManager
from ..models.model_factory import ModelFactory
from ..inference.realtime_processor import RealTimeEmotionProcessor
from ..utils.device_manager import DeviceManager

class EmotionWebApp:
    """
    Aplicación web Flask para reconocimiento de emociones en tiempo real.
    Soporta tanto carga de imágenes como streaming desde webcam.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Cargar configuración
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_full_config()
        
        # Configuración de la app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'emotion_recognition_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar procesador de emociones
        self._setup_emotion_processor()
        
        # Configurar rutas
        self._setup_routes()
        
        # Estado de la aplicación
        self.active_sessions = {}
        self.stats = {
            'total_predictions': 0,
            'realtime_sessions': 0,
            'uploaded_images': 0
        }
        
        self.logger.info("EmotionWebApp inicializada")
    
    def _setup_logging(self):
        """Configura logging específico para la web app."""
        log_dir = Path(self.config['paths']['logs_root']) / 'web_app'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('web_app')
        handler = logging.FileHandler(log_dir / 'web_app.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def _setup_emotion_processor(self):
        """Inicializa el procesador de emociones en tiempo real."""
        try:
            # Buscar el mejor modelo entrenado
            models_dir = Path(self.config['paths']['trained_models'])
            best_model_path = models_dir / 'best_model.pth'
            
            if best_model_path.exists():
                self.emotion_processor = RealTimeEmotionProcessor(
                    model_path=str(best_model_path),
                    config=self.config
                )
                self.model_loaded = True
                self.logger.info(f"Modelo cargado desde: {best_model_path}")
            else:
                self.model_loaded = False
                self.logger.warning("No se encontró modelo entrenado")
                
        except Exception as e:
            self.model_loaded = False
            self.logger.error(f"Error cargando modelo: {e}")
    
    def _setup_routes(self):
        """Configura las rutas de la aplicación."""
        
        @self.app.route('/')
        def index():
            """Página principal."""
            return render_template('index.html', 
                                 model_loaded=self.model_loaded,
                                 emotions=self.config['emotions']['active_emotions'])
        
        @self.app.route('/upload')
        def upload_page():
            """Página de carga de imágenes."""
            return render_template('upload.html',
                                 model_loaded=self.model_loaded,
                                 emotions=self.config['emotions']['active_emotions'])
        
        @self.app.route('/realtime')
        def realtime_page():
            """Página de reconocimiento en tiempo real."""
            return render_template('realtime.html',
                                 model_loaded=self.model_loaded,
                                 emotions=self.config['emotions']['active_emotions'])
        
        @self.app.route('/api/predict_image', methods=['POST'])
        def predict_image():
            """API para predicción en imagen estática."""
            if not self.model_loaded:
                return jsonify({'error': 'Modelo no disponible'}), 503
            
            try:
                # Obtener imagen del request
                if 'image' not in request.files:
                    return jsonify({'error': 'No se proporcionó imagen'}), 400
                
                image_file = request.files['image']
                
                # Leer y procesar imagen
                image_bytes = image_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Imagen inválida'}), 400
                
                # Predecir emoción
                result = self.emotion_processor.predict_frame(image)
                
                # Actualizar estadísticas
                self.stats['uploaded_images'] += 1
                self.stats['total_predictions'] += 1
                
                return jsonify({
                    'success': True,
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'face_detected': result['face_detected']
                })
                
            except Exception as e:
                self.logger.error(f"Error en predict_image: {e}")
                return jsonify({'error': 'Error procesando imagen'}), 500
        
        @self.app.route('/api/stats')
        def get_stats():
            """API para obtener estadísticas de la aplicación."""
            return jsonify({
                'stats': self.stats,
                'model_info': {
                    'loaded': self.model_loaded,
                    'architecture': self.config['model']['architecture'],
                    'emotions': self.config['emotions']['active_emotions'],
                    'device': str(self.emotion_processor.device) if self.model_loaded else 'N/A'
                },
                'config': {
                    'realtime_enabled': self.config['input']['realtime']['enable_webcam'],
                    'max_concurrent_sessions': self.config['deployment']['performance']['max_concurrent_requests']
                }
            })
        
        # WebSocket events para tiempo real
        @self.socketio.on('connect')
        def handle_connect():
            """Maneja conexión de cliente WebSocket."""
            self.logger.info(f"Cliente conectado: {request.sid}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Maneja desconexión de cliente WebSocket."""
            if request.sid in self.active_sessions:
                del self.active_sessions[request.sid]
                self.stats['realtime_sessions'] = len(self.active_sessions)
            
            self.logger.info(f"Cliente desconectado: {request.sid}")
        
        @self.socketio.on('start_realtime')
        def handle_start_realtime():
            """Inicia sesión de reconocimiento en tiempo real."""
            if not self.model_loaded:
                emit('error', {'message': 'Modelo no disponible'})
                return
            
            max_sessions = self.config['deployment']['performance']['max_concurrent_requests']
            if len(self.active_sessions) >= max_sessions:
                emit('error', {'message': 'Demasiadas sesiones activas'})
                return
            
            self.active_sessions[request.sid] = {
                'start_time': time.time(),
                'predictions_count': 0
            }
            self.stats['realtime_sessions'] = len(self.active_sessions)
            
            emit('realtime_started', {'session_id': request.sid})
            self.logger.info(f"Sesión en tiempo real iniciada: {request.sid}")
        
        @self.socketio.on('stop_realtime')
        def handle_stop_realtime():
            """Detiene sesión de reconocimiento en tiempo real."""
            if request.sid in self.active_sessions:
                session_info = self.active_sessions[request.sid]
                duration = time.time() - session_info['start_time']
                
                del self.active_sessions[request.sid]
                self.stats['realtime_sessions'] = len(self.active_sessions)
                
                emit('realtime_stopped', {
                    'duration': duration,
                    'predictions_made': session_info['predictions_count']
                })
                
                self.logger.info(f"Sesión en tiempo real terminada: {request.sid}")
        
        @self.socketio.on('process_frame')
        def handle_process_frame(data):
            """Procesa frame de video en tiempo real."""
            if not self.model_loaded or request.sid not in self.active_sessions:
                emit('error', {'message': 'Sesión no válida'})
                return
            
            try:
                # Decodificar imagen base64
                image_data = data['image'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    emit('error', {'message': 'Frame inválido'})
                    return
                
                # Procesar frame
                result = self.emotion_processor.predict_frame(frame)
                
                # Actualizar estadísticas de sesión
                self.active_sessions[request.sid]['predictions_count'] += 1
                self.stats['total_predictions'] += 1
                
                # Enviar resultado
                emit('prediction_result', {
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'face_detected': result['face_detected'],
                    'processing_time_ms': result['processing_time_ms']
                })
                
            except Exception as e:
                self.logger.error(f"Error procesando frame: {e}")
                emit('error', {'message': 'Error procesando frame'})
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Ejecuta la aplicación web.
        
        Args:
            host: Host de la aplicación
            port: Puerto de la aplicación  
            debug: Modo debug
        """
        # Usar configuración si no se especifican parámetros
        if host is None:
            host = self.config['deployment']['local_web']['host']
        if port is None:
            port = self.config['deployment']['local_web']['port']
        if debug is None:
            debug = self.config['deployment']['local_web']['debug']
        
        self.logger.info(f"Iniciando aplicación web en http://{host}:{port}")
        
        # Información de inicio
        print("\n" + "="*60)
        print("🎭 APLICACIÓN WEB DE RECONOCIMIENTO DE EMOCIONES")
        print("="*60)
        print(f"🌐 URL: http://{host}:{port}")
        print(f"🤖 Modelo cargado: {'✓' if self.model_loaded else '✗'}")
        if self.model_loaded:
            print(f"🎯 Emociones detectables: {len(self.config['emotions']['active_emotions'])}")
            print(f"📋 Lista: {', '.join(self.config['emotions']['active_emotions'])}")
        print(f"💻 Dispositivo: {self.emotion_processor.device if self.model_loaded else 'N/A'}")
        print("\n📱 Funcionalidades disponibles:")
        print("   • Carga de imágenes estáticas")
        print("   • Reconocimiento en tiempo real con webcam")
        print("   • API REST para integración")
        print("   • Estadísticas en tiempo real")
        print("="*60)
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False  # Evitar problemas con threading
            )
        except KeyboardInterrupt:
            self.logger.info("Aplicación web detenida por el usuario")
        except Exception as e:
            self.logger.error(f"Error ejecutando aplicación web: {e}")
            raise


# Template HTML base
HTML_TEMPLATES = {
    'base.html': '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Reconocimiento de Emociones{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .emotion-card {
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        .emotion-card:hover {
            transform: translateY(-5px);
        }
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
        }
        .video-container {
            position: relative;
            display: inline-block;
        }
        .emotion-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-smile"></i> Reconocimiento de Emociones
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Inicio</a>
                <a class="nav-link" href="/upload">Subir Imagen</a>
                <a class="nav-link" href="/realtime">Tiempo Real</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% if not model_loaded %}
        <div class="alert alert-warning">
            <i class="fas fa-exclamation-triangle"></i>
            El modelo no está cargado. Por favor entrena un modelo primero.
        </div>
        {% endif %}
        
        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
    ''',
    
    'index.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="jumbotron bg-light p-5 rounded">
            <h1 class="display-4">🎭 Reconocimiento de Emociones Faciales</h1>
            <p class="lead">Sistema avanzado de detección de emociones usando Deep Learning</p>
            
            {% if model_loaded %}
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card emotion-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-upload fa-3x text-primary mb-3"></i>
                            <h5>Subir Imagen</h5>
                            <p>Analiza emociones en imágenes estáticas</p>
                            <a href="/upload" class="btn btn-primary">Comenzar</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card emotion-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-video fa-3x text-success mb-3"></i>
                            <h5>Tiempo Real</h5>
                            <p>Reconocimiento en vivo desde webcam</p>
                            <a href="/realtime" class="btn btn-success">Comenzar</a>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-4">
                <h5>Emociones Detectables:</h5>
                <div class="row">
                    {% for emotion in emotions %}
                    <div class="col-md-3 mb-2">
                        <span class="badge bg-secondary">{{ emotion|title }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-12">
        <h3>Estadísticas del Sistema</h3>
        <div id="stats-container">
            <div class="d-flex justify-content-center">
                <div class="spinner-border" role="status"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Cargar estadísticas
function loadStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('stats-container');
            container.innerHTML = `
                <div class="row">
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="text-primary">${data.stats.total_predictions}</h4>
                                <p>Predicciones Totales</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="text-success">${data.stats.realtime_sessions}</h4>
                                <p>Sesiones Activas</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="text-info">${data.stats.uploaded_images}</h4>
                                <p>Imágenes Procesadas</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="text-warning">${data.model_info.loaded ? '✓' : '✗'}</h4>
                                <p>Modelo Cargado</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error cargando estadísticas:', error);
        });
}

// Cargar estadísticas al iniciar y cada 30 segundos
loadStats();
setInterval(loadStats, 30000);
</script>
{% endblock %}
    ''',
    
    'realtime.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">
                    <i class="fas fa-video"></i> Reconocimiento en Tiempo Real
                </h5>
                <div>
                    <button id="startBtn" class="btn btn-success" {% if not model_loaded %}disabled{% endif %}>
                        <i class="fas fa-play"></i> Iniciar
                    </button>
                    <button id="stopBtn" class="btn btn-danger" disabled>
                        <i class="fas fa-stop"></i> Detener
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="video-container">
                    <video id="videoElement" width="640" height="480" autoplay muted></video>
                    <canvas id="canvasElement" width="640" height="480" style="display: none;"></canvas>
                    <div id="emotionOverlay" class="emotion-overlay" style="display: none;">
                        <div class="fw-bold" id="currentEmotion">-</div>
                        <div id="currentConfidence">0%</div>
                    </div>
                </div>
                
                {% if not model_loaded %}
                <div class="alert alert-warning mt-3">
                    <i class="fas fa-exclamation-triangle"></i>
                    El modelo no está disponible. Entrena un modelo primero.
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-chart-bar"></i> Probabilidades
                </h5>
            </div>
            <div class="card-body" id="probabilitiesContainer">
                <p class="text-muted">Inicia el reconocimiento para ver las probabilidades</p>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-info-circle"></i> Estado de la Sesión
                </h5>
            </div>
            <div class="card-body">
                <div class="mb-2">
                    <strong>Estado:</strong> <span id="sessionStatus" class="badge bg-secondary">Desconectado</span>
                </div>
                <div class="mb-2">
                    <strong>Predicciones:</strong> <span id="predictionCount">0</span>
                </div>
                <div class="mb-2">
                    <strong>FPS:</strong> <span id="fpsCounter">0</span>
                </div>
                <div>
                    <strong>Cara detectada:</strong> <span id="faceStatus" class="badge bg-warning">No</span>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
class RealtimeEmotionApp {
    constructor() {
        this.socket = io();
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('canvasElement');
        this.ctx = this.canvas.getContext('2d');
        this.isStreaming = false;
        this.predictionCount = 0;
        this.lastFrameTime = 0;
        this.fps = 0;
        
        this.setupSocketListeners();
        this.setupUI();
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Conectado al servidor');
            this.updateSessionStatus('Conectado', 'bg-success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Desconectado del servidor');
            this.updateSessionStatus('Desconectado', 'bg-secondary');
        });
        
        this.socket.on('realtime_started', (data) => {
            console.log('Sesión en tiempo real iniciada');
            this.isStreaming = true;
            this.updateSessionStatus('Streaming', 'bg-primary');
            this.startVideoProcessing();
        });
        
        this.socket.on('realtime_stopped', (data) => {
            console.log('Sesión terminada:', data);
            this.isStreaming = false;
            this.updateSessionStatus('Detenido', 'bg-warning');
        });
        
        this.socket.on('prediction_result', (data) => {
            this.handlePredictionResult(data);
        });
        
        this.socket.on('error', (data) => {
            console.error('Error:', data.message);
            alert('Error: ' + data.message);
        });
    }
    
    setupUI() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        startBtn.addEventListener('click', () => {
            this.startStreaming();
        });
        
        stopBtn.addEventListener('click', () => {
            this.stopStreaming();
        });
    }
    
    async startStreaming() {
        try {
            // Solicitar acceso a la cámara
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            this.video.srcObject = stream;
            
            // Notificar al servidor
            this.socket.emit('start_realtime');
            
            // Actualizar UI
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('emotionOverlay').style.display = 'block';
            
        } catch (error) {
            console.error('Error accediendo a la cámara:', error);
            alert('No se pudo acceder a la cámara. Verifica los permisos.');
        }
    }
    
    stopStreaming() {
        // Detener stream
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        // Notificar al servidor
        this.socket.emit('stop_realtime');
        
        // Actualizar UI
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('emotionOverlay').style.display = 'none';
        
        this.isStreaming = false;
    }
    
    startVideoProcessing() {
        const processFrame = () => {
            if (!this.isStreaming) return;
            
            // Capturar frame
            this.ctx.drawImage(this.video, 0, 0, 640, 480);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Enviar frame al servidor
            this.socket.emit('process_frame', { image: imageData });
            
            // Calcular FPS
            const now = performance.now();
            if (this.lastFrameTime) {
                this.fps = Math.round(1000 / (now - this.lastFrameTime));
                document.getElementById('fpsCounter').textContent = this.fps;
            }
            this.lastFrameTime = now;
            
            // Procesar siguiente frame
            setTimeout(processFrame, 100); // 10 FPS
        };
        
        processFrame();
    }
    
    handlePredictionResult(data) {
        this.predictionCount++;
        
        // Actualizar emoción principal
        document.getElementById('currentEmotion').textContent = 
            data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
        document.getElementById('currentConfidence').textContent = 
            Math.round(data.confidence * 100) + '%';
        
        // Actualizar estado de cara
        const faceStatus = document.getElementById('faceStatus');
        if (data.face_detected) {
            faceStatus.textContent = 'Sí';
            faceStatus.className = 'badge bg-success';
        } else {
            faceStatus.textContent = 'No';
            faceStatus.className = 'badge bg-warning';
        }
        
        // Actualizar contador
        document.getElementById('predictionCount').textContent = this.predictionCount;
        
        // Actualizar probabilidades
        this.updateProbabilities(data.probabilities);
    }
    
    updateProbabilities(probabilities) {
        const container = document.getElementById('probabilitiesContainer');
        
        let html = '';
        for (const [emotion, probability] of Object.entries(probabilities)) {
            const percentage = Math.round(probability * 100);
            const isActive = percentage === Math.max(...Object.values(probabilities).map(p => Math.round(p * 100)));
            
            html += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span class="${isActive ? 'fw-bold' : ''}">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                        <span class="${isActive ? 'fw-bold' : ''}">${percentage}%</span>
                    </div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar ${isActive ? 'bg-primary' : 'bg-secondary'}" 
                             style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
    
    updateSessionStatus(status, badgeClass) {
        const statusElement = document.getElementById('sessionStatus');
        statusElement.textContent = status;
        statusElement.className = `badge ${badgeClass}`;
    }
}

// Inicializar aplicación cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    new RealtimeEmotionApp();
});
</script>
{% endblock %}
    '''
}


def create_web_templates():
    """Crea los templates HTML en el directorio correspondiente."""
    templates_dir = Path("src/web/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in HTML_TEMPLATES.items():
        template_path = templates_dir / filename
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Templates HTML creados en {templates_dir}")


# Función principal para ejecutar la aplicación
def run_web_app(config_path: Optional[str] = None, 
                host: Optional[str] = None, 
                port: Optional[int] = None,
                debug: Optional[bool] = None):
    """
    Función principal para ejecutar la aplicación web.
    
    Args:
        config_path: Path al archivo de configuración
        host: Host para la aplicación
        port: Puerto para la aplicación
        debug: Modo debug
    """
    # Crear templates si no existen
    create_web_templates()
    
    # Crear y ejecutar aplicación
    app = EmotionWebApp(config_path)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web_app()