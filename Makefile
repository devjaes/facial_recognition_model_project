# Makefile para proyecto de reconocimiento de emociones

# Variables
PYTHON = python
PIP = pip
CONDA = conda

# Comandos principales
.PHONY: help setup train evaluate test clean

help:
	@echo "Comandos disponibles:"
	@echo "  setup     - Configurar entorno"
	@echo "  train     - Entrenar modelo"
	@echo "  evaluate  - Evaluar modelo"
	@echo "  test      - Ejecutar tests"
	@echo "  web       - Iniciar aplicación web"
	@echo "  clean     - Limpiar archivos temporales"

setup:
	$(PYTHON) scripts/setup_environment.py

train:
	$(PYTHON) scripts/train_model.py

evaluate:
	$(PYTHON) scripts/evaluate_model.py

test:
	$(PYTHON) -m pytest tests/

web:
	$(PYTHON) src/web/app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/temp/*
	rm -rf logs/temp/*

# Comandos de datos
process-data:
	$(PYTHON) scripts/process_emognition_data.py

download-fer2013:
	$(PYTHON) scripts/download_datasets.py --dataset fer2013

# Comandos de modelo
train-basic:
	$(PYTHON) scripts/train_model.py --model cnn_basic --epochs 100

train-hybrid:
	$(PYTHON) scripts/train_model.py --model cnn_rnn_hybrid --epochs 500

train-transfer:
	$(PYTHON) scripts/train_model.py --model transfer_learning --base mobilenetv2

# Optimización
optimize-model:
	$(PYTHON) scripts/optimize_model.py

# Docker (futuro)
docker-build:
	docker build -t emotion-recognition .

docker-run:
	docker run -p 5000:5000 emotion-recognition
