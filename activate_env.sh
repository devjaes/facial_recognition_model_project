#!/bin/bash
echo "Activando entorno emotion_recognition_linux..."
conda activate emotion_recognition_linux
echo "Entorno activado. Para desactivar usa: conda deactivate"
exec "$SHELL"
