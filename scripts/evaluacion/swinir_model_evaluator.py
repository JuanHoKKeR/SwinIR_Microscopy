#!/usr/bin/env python3
"""
Script SIMPLE para comparar UNA imagen generada por SwinIR vs su ground truth
Calcula PSNR, SSIM, MS-SSIM, MSE + Índice Perceptual usando DenseNet+KimiaNet
Adaptado específicamente para el proyecto SwinIR
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json

# Importar las librerías necesarias para KimiaNet (si están disponibles)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow no disponible. Índice perceptual no estará disponible.")
    TF_AVAILABLE = False

# Importar pytorch-msssim si está disponible
try:
    from pytorch_msssim import ssim, ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-msssim no disponible. MS-SSIM no estará disponible.")
    MSSSIM_AVAILABLE = False


class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet+KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no está disponible")
            
        print("🧠 Cargando DenseNet121 con pesos KimiaNet...")
        
        # Cargar DenseNet121 sin la capa final
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,  # Sin pesos de ImageNet
            input_shape=(None, None, 3)
        )
        
        # Cargar pesos KimiaNet si existe el archivo
        if kimianet_weights_path and os.path.exists(kimianet_weights_path):
            try:
                self.densenet.load_weights(kimianet_weights_path)
                print(f"✅ Pesos KimiaNet cargados desde: {kimianet_weights_path}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos KimiaNet: {e}")
                print("    Usando DenseNet121 sin preentrenar")
        else:
            print("⚠️  No se encontraron pesos KimiaNet, usando DenseNet121 sin preentrenar")
        
        # Usar una capa intermedia para extraer características
        try:
            feature_layer = self.densenet.get_layer('conv4_block6_concat')
        except:
            try:
                feature_layer = self.densenet.get_layer('conv4_block24_concat')
            except:
                feature_layer = self.densenet.layers[-2]
        
        self.feature_extractor = tf.keras.Model(
            inputs=self.densenet.input,
            outputs=feature_layer.output
        )
        
        # Congelar el modelo  
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
    def calculate_perceptual_distance(self, img1, img2):
        """
        Calcula distancia perceptual entre dos imágenes usando KimiaNet
        
        Args:
            img1, img2: Imágenes en formato [H, W, 3] con valores [0, 255]
            
        Returns:
            Distancia perceptual (más bajo = más similar)
        """
        # Asegurar que sean tensores float32
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)
        
        # Normalizar para DenseNet
        img1_norm = (img1 - 127.5) / 127.5  # [-1, 1]
        img2_norm = (img2 - 127.5) / 127.5  # [-1, 1]
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Calcular distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return float(perceptual_distance.numpy())


def load_image(image_path):
    """Carga una imagen usando OpenCV y la convierte a tensor PyTorch"""
    try:
        # Leer archivo con OpenCV (BGR)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir a float32 y normalizar a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convertir a tensor PyTorch
        image_tensor = torch.from_numpy(image)
        
        return image_tensor
    except Exception as e:
        print(f"❌ Error cargando {image_path}: {e}")
        return None


def calculate_psnr(img1, img2):
    """Calcula PSNR entre dos imágenes usando PyTorch"""
    # Asegurar que las imágenes tengan el mismo tamaño
    if img1.shape != img2.shape:
        print(f"⚠️  Redimensionando para PSNR: {img1.shape} vs {img2.shape}")
        img2 = F.interpolate(img2.unsqueeze(0).permute(2, 0, 1).unsqueeze(0), 
                           size=img1.shape[:2], mode='bicubic').squeeze(0).permute(1, 2, 0)
    
    # Calcular MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # Calcular PSNR
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return float(psnr)


def calculate_ssim_metrics(img1, img2):
    """
    Calcula SSIM y MS-SSIM usando pytorch-msssim
    
    Args:
        img1, img2: Tensores de imagen [H, W, 3] en rango [0, 1]
    
    Returns:
        dict con valores de SSIM y MS-SSIM
    """
    if not MSSSIM_AVAILABLE:
        return {'ssim': None, 'ms_ssim': None}
    
    # Convertir a formato [1, 3, H, W] para pytorch-msssim
    def to_tensor_format(img):
        if len(img.shape) == 3:
            return img.permute(2, 0, 1).unsqueeze(0)
        return img
    
    img1_tensor = to_tensor_format(img1)
    img2_tensor = to_tensor_format(img2)
    
    # Redimensionar si es necesario
    if img1_tensor.shape != img2_tensor.shape:
        img2_tensor = F.interpolate(img2_tensor, size=img1_tensor.shape[2:], mode='bicubic')
    
    try:
        # Calcular SSIM
        ssim_val = ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
        
        # Calcular MS-SSIM
        ms_ssim_val = ms_ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
        
        return {
            'ssim': float(ssim_val),
            'ms_ssim': float(ms_ssim_val)
        }
    except Exception as e:
        print(f"⚠️  Error calculando SSIM: {e}")
        return {'ssim': None, 'ms_ssim': None}


def calculate_mse(img1, img2):
    """Calcula MSE entre dos imágenes"""
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2.unsqueeze(0).permute(2, 0, 1).unsqueeze(0), 
                           size=img1.shape[:2], mode='bicubic').squeeze(0).permute(1, 2, 0)
    
    mse = torch.mean((img1 - img2) ** 2)
    return float(mse)


def resize_to_match(img1, img2):
    """Redimensiona img2 para que coincida con img1 si es necesario"""
    if img1.shape != img2.shape:
        print(f"⚠️  Redimensionando: {img2.shape} → {img1.shape}")
        img2 = F.interpolate(img2.unsqueeze(0).permute(2, 0, 1).unsqueeze(0), 
                           size=img1.shape[:2], mode='bicubic').squeeze(0).permute(1, 2, 0)
    return img2


def compare_images(generated_path, ground_truth_path, kimianet_weights_path=None, output_json=None):
    """
    Compara una imagen generada con su ground truth
    
    Args:
        generated_path: Ruta a la imagen generada por SwinIR
        ground_truth_path: Ruta a la imagen de referencia (ground truth)
        kimianet_weights_path: Ruta a los pesos KimiaNet (opcional)
        output_json: Ruta para guardar resultados en JSON (opcional)
    """
    
    print("🔍 COMPARACIÓN DE IMÁGENES SWINIR")
    print("=" * 50)
    
    # Cargar imágenes
    print("📁 Cargando imágenes...")
    generated_img = load_image(generated_path)
    ground_truth_img = load_image(ground_truth_path)
    
    if generated_img is None or ground_truth_img is None:
        print("❌ Error cargando las imágenes")
        return None
    
    print(f"   Generada: {generated_path}")
    print(f"   Dimensiones: {generated_img.shape}")
    print(f"   Ground Truth: {ground_truth_path}")
    print(f"   Dimensiones: {ground_truth_img.shape}")
    
    # Ajustar dimensiones si es necesario
    ground_truth_img = resize_to_match(generated_img, ground_truth_img)
    
    print("\n📊 Calculando métricas...")
    
    # Inicializar resultados
    results = {
        'generated_path': generated_path,
        'ground_truth_path': ground_truth_path,
        'generated_shape': list(generated_img.shape),
        'ground_truth_shape': list(ground_truth_img.shape)
    }
    
    # 1. PSNR
    print("📈 Calculando PSNR...")
    psnr_value = calculate_psnr(generated_img, ground_truth_img)
    results['psnr'] = psnr_value
    print(f"✅ PSNR: {psnr_value:.4f} dB")
    
    # 2. SSIM y MS-SSIM
    print("📊 Calculando SSIM y MS-SSIM...")
    ssim_metrics = calculate_ssim_metrics(generated_img, ground_truth_img)
    results.update(ssim_metrics)
    if ssim_metrics['ssim'] is not None:
        print(f"✅ SSIM: {ssim_metrics['ssim']:.6f}")
    if ssim_metrics['ms_ssim'] is not None:
        print(f"✅ MS-SSIM: {ssim_metrics['ms_ssim']:.6f}")
    
    # 3. MSE
    print("📉 Calculando MSE...")
    mse_value = calculate_mse(generated_img, ground_truth_img)
    results['mse'] = mse_value
    print(f"✅ MSE: {mse_value:.8f}")
    
    # 4. Índice Perceptual con KimiaNet
    if TF_AVAILABLE and kimianet_weights_path:
        print("🧠 Calculando índice perceptual con KimiaNet...")
        try:
            perceptual_loss = KimiaNetPerceptualLoss(kimianet_weights_path)
            
            # Convertir tensores PyTorch a numpy para TensorFlow
            img1_np = (generated_img.numpy() * 255).astype(np.uint8)
            img2_np = (ground_truth_img.numpy() * 255).astype(np.uint8)
            
            perceptual_distance = perceptual_loss.calculate_perceptual_distance(img1_np, img2_np)
            results['perceptual_index'] = perceptual_distance
            print(f"✅ Índice Perceptual: {perceptual_distance:.6f}")
        except Exception as e:
            print(f"❌ Error calculando índice perceptual: {e}")
            results['perceptual_index'] = None
    else:
        results['perceptual_index'] = None
        if not TF_AVAILABLE:
            print("⚠️  Índice perceptual no disponible (TensorFlow no instalado)")
        else:
            print("⚠️  Índice perceptual no calculado (pesos KimiaNet no proporcionados)")
    
    # Mostrar resultados finales
    print("=" * 50)
    print("🏆 RESULTADOS FINALES")
    print("=" * 50)
    print(f"📈 PSNR: {results['psnr']:.4f} dB")
    if results.get('ssim'):
        print(f"📊 SSIM: {results['ssim']:.6f}")
    if results.get('ms_ssim'):
        print(f"📊 MS-SSIM: {results['ms_ssim']:.6f}")
    print(f"📉 MSE: {results['mse']:.8f}")
    if results.get('perceptual_index'):
        print(f"🧠 Índice Perceptual: {results['perceptual_index']:.6f}")
    
    print("\n💡 Interpretación:")
    print("   • PSNR: Más alto = mejor (típico: 20-35 dB)")
    print("   • SSIM/MS-SSIM: Más alto = mejor (rango: 0-1)")
    print("   • MSE: Más bajo = mejor")
    print("   • Índice Perceptual: Más bajo = mejor (típico: 0.001-0.1)")
    print("=" * 50)
    
    # Guardar resultados en JSON si se especifica
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"💾 Resultados guardados en: {output_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comparar UNA imagen generada por SwinIR vs ground truth")
    
    parser.add_argument(
        "--generated",
        required=True,
        help="Ruta a la imagen generada por SwinIR"
    )
    
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Ruta a la imagen ground truth"
    )
    
    parser.add_argument(
        "--kimianet_weights",
        default=None,
        help="Ruta a los pesos KimiaNet (opcional)"
    )
    
    parser.add_argument(
        "--output_json",
        default=None,
        help="Ruta para guardar resultados en JSON (opcional)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.generated):
        print(f"❌ No existe la imagen generada: {args.generated}")
        return 1
    
    if not os.path.exists(args.ground_truth):
        print(f"❌ No existe la imagen ground truth: {args.ground_truth}")
        return 1
    
    # Comparar imágenes
    try:
        results = compare_images(
            generated_path=args.generated,
            ground_truth_path=args.ground_truth,
            kimianet_weights_path=args.kimianet_weights,
            output_json=args.output_json
        )
        
        if results:
            print("🎉 Comparación completada exitosamente")
            return 0
        else:
            print("❌ Error durante la comparación")
            return 1
            
    except Exception as e:
        print(f"💥 Error durante la comparación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())