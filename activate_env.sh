#!/bin/bash
echo "Activando entorno emotion_recognition_macos..."
conda activate emotion_recognition_macos
echo "Entorno activado. Para desactivar usa: conda deactivate"
exec "$SHELL"
