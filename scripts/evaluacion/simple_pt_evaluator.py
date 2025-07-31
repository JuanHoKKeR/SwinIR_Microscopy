#!/usr/bin/env python3
"""
Evaluador Simple para Modelos SwinIR Optimizados (.pt)
Evalúa un modelo optimizado con una imagen individual

Uso:
    python simple_pt_evaluator.py --model optimized_models/swinir_64to128_optimized.pt --input imagen_lr.png --output imagen_sr.png
"""

import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import time

def load_optimized_model(model_path, device):
    """
    Carga un modelo SwinIR optimizado (.pt)
    
    Args:
        model_path: Ruta al archivo .pt del modelo
        device: Dispositivo (CPU/GPU)
    
    Returns:
        Modelo cargado y listo para inferencia
    """
    print(f"📦 Cargando modelo optimizado desde: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    try:
        # Cargar modelo optimizado (TorchScript)
        # Usar weights_only=False para modelos TorchScript
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        
        print("✅ Modelo TorchScript cargado correctamente")
        return model
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise

def load_image(image_path):
    """
    Carga una imagen y la prepara para el modelo
    
    Args:
        image_path: Ruta a la imagen de entrada
    
    Returns:
        Tensor de imagen normalizado [1, 3, H, W]
    """
    print(f"🖼️  Cargando imagen: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    # Cargar imagen con OpenCV (BGR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convertir a float32 y normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convertir a tensor PyTorch y cambiar dimensiones a CHW
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    # Agregar dimensión de batch
    image_tensor = image_tensor.unsqueeze(0)
    
    print(f"✅ Imagen cargada - Tamaño: {image_tensor.shape}")
    return image_tensor

def save_image(image_tensor, output_path):
    """
    Guarda un tensor de imagen como archivo PNG
    
    Args:
        image_tensor: Tensor de imagen [1, 3, H, W] en rango [0, 1]
        output_path: Ruta de salida
    """
    print(f"💾 Guardando imagen en: {output_path}")
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convertir tensor a numpy
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Clampear a [0, 1] y convertir a uint8
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    # Convertir RGB a BGR para OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Guardar imagen
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"✅ Imagen guardada exitosamente")
    else:
        print(f"❌ Error guardando imagen")

def infer_model(model, input_tensor, device):
    """
    Realiza inferencia con el modelo
    
    Args:
        model: Modelo SwinIR optimizado
        input_tensor: Tensor de entrada [1, 3, H, W]
        device: Dispositivo (CPU/GPU)
    
    Returns:
        Tensor de salida [1, 3, H*scale, W*scale]
    """
    print("🚀 Iniciando inferencia...")
    
    # Mover tensor a dispositivo
    input_tensor = input_tensor.to(device)
    
    # Realizar inferencia
    with torch.no_grad():
        start_time = time.time()
        output_tensor = model(input_tensor)
        inference_time = time.time() - start_time
    
    print(f"✅ Inferencia completada en {inference_time:.3f} segundos")
    print(f"   Input: {input_tensor.shape} -> Output: {output_tensor.shape}")
    
    return output_tensor

def main():
    parser = argparse.ArgumentParser(description='Evaluador simple para modelos SwinIR optimizados (.pt)')
    parser.add_argument('--model', type=str, required=True, 
                       help='Ruta al modelo optimizado (.pt)')
    parser.add_argument('--input', type=str, required=True,
                       help='Ruta a la imagen de entrada')
    parser.add_argument('--output', type=str, required=True,
                       help='Ruta de salida para la imagen procesada')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Dispositivo a usar (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Usando dispositivo: {device}")
    
    try:
        # Cargar modelo
        model = load_optimized_model(args.model, device)
        
        # Cargar imagen
        input_tensor = load_image(args.input)
        
        # Realizar inferencia
        output_tensor = infer_model(model, input_tensor, device)
        
        # Guardar resultado
        save_image(output_tensor, args.output)
        
        print("🎉 Procesamiento completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())