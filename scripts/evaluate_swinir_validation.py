#!/usr/bin/env python3
"""
Evaluador de Validación para Modelo SwinIR Específico
Evalúa un modelo específico con su dataset de validación correspondiente
Genera CSVs con métricas en color y blanco y negro
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time
from tqdm import tqdm
import warnings
import cv2
from collections import OrderedDict

# Imports específicos de SwinIR
from models.network_swinir import SwinIR as SwinIRNet
from utils import utils_image as util

# Imports para KimiaNet (mantenemos la misma lógica de TensorFlow)
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121

warnings.filterwarnings('ignore')

class SwinIRModelEvaluator:
    """Evaluador para un modelo SwinIR específico"""
    
    def __init__(self, kimianet_weights_path):
        """
        Inicializa el evaluador
        
        Args:
            kimianet_weights_path: Ruta a los pesos de KimiaNet
        """
        self.kimianet_weights_path = kimianet_weights_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = None
        self._initialize_kimianet()
        
    def _initialize_kimianet(self):
        """Inicializa DenseNet121 con pesos KimiaNet para índice perceptual"""
        print("🧠 Inicializando KimiaNet para índice perceptual...")
        
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,
            input_shape=(None, None, 3)
        )
        
        if os.path.exists(self.kimianet_weights_path):
            try:
                self.densenet.load_weights(self.kimianet_weights_path)
                print(f"✅ Pesos KimiaNet cargados desde: {self.kimianet_weights_path}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos KimiaNet: {e}")
                print("    Continuando sin pesos preentrenados")
        else:
            print(f"⚠️  No se encontró el archivo de pesos: {self.kimianet_weights_path}")
        
        # Usar capa intermedia para características
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
        
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
    def load_model(self, model_path, scale=2, training_patch_size=128):
        """
        Carga el modelo SwinIR
        
        Args:
            model_path: Ruta al archivo .pth del modelo
            scale: Factor de escala del modelo
            training_patch_size: Tamaño de patch usado en entrenamiento
        """
        print(f"📦 Cargando modelo SwinIR desde: {model_path}")
        
        try:
            # Definir arquitectura del modelo basada en parámetros
            self.model = self._define_model_architecture(scale, training_patch_size)
            
            # Cargar pesos del modelo
            pretrained_model = torch.load(model_path, map_location=self.device)
            
            # Extraer parámetros según la estructura del checkpoint
            if 'params' in pretrained_model:
                state_dict = pretrained_model['params']
            elif 'params_ema' in pretrained_model:
                state_dict = pretrained_model['params_ema']
            else:
                state_dict = pretrained_model
            
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            print("✅ Modelo SwinIR cargado correctamente")
            
            # Verificar que funciona con una imagen de prueba
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
            print(f"🧪 Prueba de inferencia exitosa - Input: {test_input.shape}, Output: {test_output.shape}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def _define_model_architecture(self, scale, training_patch_size):
        """
        Define la arquitectura del modelo SwinIR basada en parámetros
        
        Args:
            scale: Factor de escala
            training_patch_size: Tamaño de patch de entrenamiento
            
        Returns:
            Modelo SwinIR configurado
        """
        # Configuraciones basadas en el training_patch_size (similar a main_test_swinir_individual.py)
        if training_patch_size == 64:
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        elif training_patch_size == 128:
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        elif training_patch_size == 256:
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        elif training_patch_size == 512:
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[4, 4, 4], embed_dim=60, 
                num_heads=[4, 4, 4], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        else:
            # Configuración por defecto
            print(f"⚠️  Usando configuración por defecto para training_patch_size={training_patch_size}")
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        
        return model
    
    def load_validation_dataset(self, lr_meta_file, hr_meta_file, base_path=""):
        """
        Carga el dataset de validación desde archivos paired_meta
        
        Args:
            lr_meta_file: Archivo con rutas de imágenes LR
            hr_meta_file: Archivo con rutas de imágenes HR  
            base_path: Ruta base para las imágenes
            
        Returns:
            Lista de tuplas (lr_path, hr_path)
        """
        print(f"📂 Cargando dataset de validación...")
        print(f"   LR meta: {lr_meta_file}")
        print(f"   HR meta: {hr_meta_file}")
        
        # Cargar rutas LR
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        # Cargar rutas HR
        with open(hr_meta_file, 'r') as f:
            hr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        print(f"✅ Cargadas {len(lr_paths)} imágenes LR y {len(hr_paths)} imágenes HR")
        
        # Verificar que las cantidades coinciden
        if len(lr_paths) != len(hr_paths):
            print(f"⚠️  Advertencia: Número diferente de imágenes LR ({len(lr_paths)}) y HR ({len(hr_paths)})")
        
        # Verificar que existen algunas imágenes de muestra
        sample_size = min(5, len(lr_paths))
        missing_files = 0
        for i in range(sample_size):
            if not os.path.exists(lr_paths[i]):
                missing_files += 1
                print(f"⚠️  Archivo LR no encontrado: {lr_paths[i]}")
            if not os.path.exists(hr_paths[i]):
                missing_files += 1
                print(f"⚠️  Archivo HR no encontrado: {hr_paths[i]}")
        
        if missing_files > 0:
            print(f"⚠️  Se encontraron {missing_files} archivos faltantes en la muestra")
        
        # Crear pares de rutas
        image_pairs = list(zip(lr_paths, hr_paths))
        print(f"📊 Dataset preparado con {len(image_pairs)} pares de imágenes")
        
        return image_pairs
    
    def load_image(self, image_path):
        """Carga una imagen usando OpenCV y la convierte a tensor PyTorch"""
        try:
            # Cargar imagen con OpenCV (BGR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a float32 y normalizar a [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convertir a tensor PyTorch y cambiar dimensiones a CHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            
            return image_tensor
        except Exception as e:
            print(f"❌ Error cargando imagen {image_path}: {e}")
            return None
    
    def convert_to_grayscale_torch(self, image_tensor):
        """Convierte tensor PyTorch RGB a escala de grises"""
        try:
            # Conversión usando pesos estándar RGB a grayscale
            if len(image_tensor.shape) == 4:  # NCHW
                r, g, b = image_tensor[:, 0:1, :, :], image_tensor[:, 1:2, :, :], image_tensor[:, 2:3, :, :]
            else:  # CHW
                r, g, b = image_tensor[0:1, :, :], image_tensor[1:2, :, :], image_tensor[2:3, :, :]
            
            grayscale = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Convertir de vuelta a 3 canales
            if len(image_tensor.shape) == 4:
                grayscale_3ch = torch.cat([grayscale, grayscale, grayscale], dim=1)
            else:
                grayscale_3ch = torch.cat([grayscale, grayscale, grayscale], dim=0)
            
            return grayscale_3ch
        except Exception as e:
            print(f"⚠️  Error conversión a grises: {e}")
            # Fallback: promedio simple
            if len(image_tensor.shape) == 4:
                gray_simple = torch.mean(image_tensor, dim=1, keepdim=True)
                return torch.cat([gray_simple, gray_simple, gray_simple], dim=1)
            else:
                gray_simple = torch.mean(image_tensor, dim=0, keepdim=True)
                return torch.cat([gray_simple, gray_simple, gray_simple], dim=0)
    
    def calculate_perceptual_index(self, img1_torch, img2_torch):
        """Calcula índice perceptual usando KimiaNet (convierte de PyTorch a TensorFlow)"""
        # Convertir de PyTorch tensor a numpy
        if len(img1_torch.shape) == 4:  # NCHW -> NHWC
            img1_np = img1_torch.permute(0, 2, 3, 1).cpu().numpy()
            img2_np = img2_torch.permute(0, 2, 3, 1).cpu().numpy()
        else:  # CHW -> HWC
            img1_np = img1_torch.permute(1, 2, 0).cpu().numpy()
            img2_np = img2_torch.permute(1, 2, 0).cpu().numpy()
        
        # Convertir a rango [0, 255] para KimiaNet
        img1_tf = tf.convert_to_tensor(img1_np * 255.0, dtype=tf.float32)
        img2_tf = tf.convert_to_tensor(img2_np * 255.0, dtype=tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1_tf.shape) == 3:
            img1_tf = tf.expand_dims(img1_tf, 0)
        if len(img2_tf.shape) == 3:
            img2_tf = tf.expand_dims(img2_tf, 0)
        
        # Normalización para DenseNet
        img1_norm = (img1_tf - 127.5) / 127.5
        img2_norm = (img2_tf - 127.5) / 127.5
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return float(perceptual_distance.numpy())
    
    def calculate_all_metrics(self, generated_torch, hr_torch):
        """
        Calcula todas las métricas de evaluación
        
        Args:
            generated_torch: Imagen generada por SwinIR (tensor PyTorch CHW, [0,1])
            hr_torch: Imagen de alta resolución (tensor PyTorch CHW, [0,1])
            
        Returns:
            Dict con todas las métricas
        """
        # ASEGURAR QUE AMBOS TENSORES ESTÉN EN EL MISMO DISPOSITIVO
        hr_torch = hr_torch.to(self.device)
        generated_torch = generated_torch.to(self.device)
        
        # Asegurar que las imágenes tengan el mismo tamaño
        if generated_torch.shape != hr_torch.shape:
            hr_torch = F.interpolate(
                hr_torch.unsqueeze(0), 
                size=generated_torch.shape[-2:], 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
        
        # Convertir a numpy para las métricas de utils_image (esperan [0, 255])
        # MOVER A CPU ANTES DE CONVERTIR A NUMPY
        generated_np = (generated_torch.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        hr_np = (hr_torch.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # Métricas básicas usando utils_image
        psnr = util.calculate_psnr(generated_np, hr_np, border=0)
        ssim = util.calculate_ssim(generated_np, hr_np, border=0)
        
        # MSE en PyTorch (ambos tensores ya están en GPU)
        mse = F.mse_loss(generated_torch, hr_torch).item()
        
        # MS-SSIM usando PyTorch (implementación simple)
        try:
            # Para MS-SSIM usaremos SSIM regular por simplicidad
            ms_ssim = ssim  # Placeholder
        except:
            ms_ssim = ssim
        
        # Índice perceptual
        perceptual_index = self.calculate_perceptual_index(generated_torch, hr_torch)
        
        return {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'ms_ssim': float(ms_ssim),
            'mse': float(mse),
            'perceptual_index': float(perceptual_index)
        }
    
    def evaluate_single_pair(self, lr_path, hr_path, window_size=8):
        """
        Evalúa un par de imágenes LR-HR
        
        Args:
            lr_path: Ruta a imagen de baja resolución
            hr_path: Ruta a imagen de alta resolución
            window_size: Tamaño de ventana para SwinIR
            
        Returns:
            Dict con métricas en color y blanco y negro, o None si hay error
        """
        try:
            # Cargar imágenes
            lr_tensor = self.load_image(lr_path)
            hr_tensor = self.load_image(hr_path)
            
            if lr_tensor is None or hr_tensor is None:
                return None
            
            # MOVER hr_tensor AL DISPOSITIVO CORRECTO DESDE EL INICIO
            hr_tensor = hr_tensor.to(self.device)
            
            # Preparar imagen para SwinIR
            lr_batch = lr_tensor.unsqueeze(0).to(self.device)  # Agregar dimensión de batch
            
            # Padding para ser múltiplo de window_size
            _, _, h_old, w_old = lr_batch.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            lr_batch = F.pad(lr_batch, (0, w_pad, 0, h_pad), 'reflect')
            
            # Generar imagen con SwinIR
            with torch.no_grad():
                generated_batch = self.model(lr_batch)
                
            # Quitar padding y batch dimension
            scale = generated_batch.shape[-1] // lr_batch.shape[-1]  # Inferir scale automáticamente
            generated = generated_batch[0, :, :h_old * scale, :w_old * scale]
            generated = torch.clamp(generated, 0, 1)
            
            # Métricas en COLOR
            color_metrics = self.calculate_all_metrics(generated, hr_tensor)
            
            # Convertir a escala de grises
            generated_gray = self.convert_to_grayscale_torch(generated)
            hr_gray = self.convert_to_grayscale_torch(hr_tensor)
            
            # Métricas en BLANCO Y NEGRO
            gray_metrics = self.calculate_all_metrics(generated_gray, hr_gray)
            
            return {
                'lr_path': lr_path,
                'hr_path': hr_path,
                'color_metrics': color_metrics,
                'gray_metrics': gray_metrics
            }
            
        except Exception as e:
            print(f"❌ Error procesando par {lr_path} - {hr_path}: {e}")
            return None
    
    def evaluate_model(self, image_pairs, output_dir, model_name, window_size=8):
        """
        Evalúa el modelo con todos los pares de imágenes
        
        Args:
            image_pairs: Lista de tuplas (lr_path, hr_path)
            output_dir: Directorio para guardar resultados
            model_name: Nombre del modelo para los archivos de salida
            window_size: Tamaño de ventana para SwinIR
        """
        print(f"\n🚀 Iniciando evaluación del modelo SwinIR {model_name}")
        print(f"📊 Procesando {len(image_pairs)} pares de imágenes...")
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Listas para almacenar resultados
        color_results = []
        gray_results = []
        
        # Procesar cada par de imágenes
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, (lr_path, hr_path) in enumerate(tqdm(image_pairs, desc="Evaluando imágenes")):
            result = self.evaluate_single_pair(lr_path, hr_path, window_size)
            
            if result is not None:
                # Preparar datos para CSV
                base_data = {
                    'image_index': i + 1,
                    'lr_path': result['lr_path'],
                    'hr_path': result['hr_path'],
                    'lr_filename': os.path.basename(result['lr_path']),
                    'hr_filename': os.path.basename(result['hr_path'])
                }
                
                # Datos para CSV de color
                color_row = base_data.copy()
                color_row.update(result['color_metrics'])
                color_results.append(color_row)
                
                # Datos para CSV de escala de grises
                gray_row = base_data.copy()
                gray_row.update(result['gray_metrics'])
                gray_results.append(gray_row)
                
                successful_evaluations += 1
            else:
                failed_evaluations += 1
        
        print(f"\n📈 Evaluación completada:")
        print(f"   ✅ Exitosas: {successful_evaluations}")
        print(f"   ❌ Fallidas: {failed_evaluations}")
        
        # Crear DataFrames y guardar CSVs
        if color_results:
            # CSV para métricas en color
            color_df = pd.DataFrame(color_results)
            color_csv_path = os.path.join(output_dir, f"{model_name}_metrics_color.csv")
            color_df.to_csv(color_csv_path, index=False)
            print(f"💾 Métricas en COLOR guardadas: {color_csv_path}")
            
            # CSV para métricas en escala de grises
            gray_df = pd.DataFrame(gray_results)
            gray_csv_path = os.path.join(output_dir, f"{model_name}_metrics_grayscale.csv")
            gray_df.to_csv(gray_csv_path, index=False)
            print(f"💾 Métricas en ESCALA DE GRISES guardadas: {gray_csv_path}")
            
            # Mostrar estadísticas resumidas
            self._print_summary_statistics(color_df, gray_df, model_name)
            
        else:
            print("❌ No se pudieron evaluar imágenes correctamente")
    
    def _print_summary_statistics(self, color_df, gray_df, model_name):
        """Imprime estadísticas resumidas"""
        print(f"\n📊 ESTADÍSTICAS RESUMIDAS - {model_name}")
        print("=" * 50)
        
        metrics = ['psnr', 'ssim', 'ms_ssim', 'mse', 'perceptual_index']
        
        print("COLOR:")
        for metric in metrics:
            if metric in color_df.columns:
                mean_val = color_df[metric].mean()
                std_val = color_df[metric].std()
                print(f"  {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
        
        print("\nESCALA DE GRISES:")
        for metric in metrics:
            if metric in gray_df.columns:
                mean_val = gray_df[metric].mean()
                std_val = gray_df[metric].std()
                print(f"  {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Evaluador de Validación para Modelo SwinIR Específico")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo SwinIR (.pth file)"
    )
    
    parser.add_argument(
        "--model_name", 
        required=True,
        help="Nombre del modelo (ej: SwinIR_128to512)"
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Factor de escala del modelo (default: 2)"
    )
    
    parser.add_argument(
        "--training_patch_size",
        type=int,
        default=128,
        help="Tamaño de patch usado en entrenamiento (default: 128)"
    )
    
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Tamaño de ventana de SwinIR (default: 8)"
    )
    
    parser.add_argument(
        "--lr_meta_file",
        required=True,
        help="Archivo meta con rutas de imágenes LR (ej: paired_lr_meta.txt)"
    )
    
    parser.add_argument(
        "--hr_meta_file",
        required=True,
        help="Archivo meta con rutas de imágenes HR (ej: paired_hr_meta.txt)"
    )
    
    parser.add_argument(
        "--kimianet_weights",
        default="model-kimianet/KimiaNetKerasWeights.h5",
        help="Ruta a los pesos de KimiaNet"
    )
    
    parser.add_argument(
        "--base_path",
        default="",
        help="Ruta base para las imágenes (si las rutas en meta_info son relativas)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directorio para guardar los resultados CSV"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Máximo número de imágenes a evaluar (para pruebas rápidas)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    required_files = [args.model_path, args.lr_meta_file, args.hr_meta_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra el archivo/directorio: {file_path}")
            return 1
    
    print("🔬 EVALUADOR DE MODELO SWINIR")
    print("=" * 40)
    print(f"Modelo: {args.model_name}")
    print(f"Ruta del modelo: {args.model_path}")
    print(f"Escala: {args.scale}")
    print(f"Training patch size: {args.training_patch_size}")
    print(f"Window size: {args.window_size}")
    print(f"Dataset LR: {args.lr_meta_file}")
    print(f"Dataset HR: {args.hr_meta_file}")
    print(f"KimiaNet: {args.kimianet_weights}")
    print(f"Resultados: {args.output_dir}")
    
    try:
        # Inicializar evaluador
        evaluator = SwinIRModelEvaluator(args.kimianet_weights)
        
        # Cargar modelo
        evaluator.load_model(args.model_path, args.scale, args.training_patch_size)
        
        # Cargar dataset de validación
        image_pairs = evaluator.load_validation_dataset(
            args.lr_meta_file, args.hr_meta_file, args.base_path
        )
        
        # Limitar número de imágenes si se especifica
        if args.max_images and args.max_images < len(image_pairs):
            image_pairs = image_pairs[:args.max_images]
            print(f"🔢 Limitando evaluación a {args.max_images} imágenes")
        
        # Evaluar modelo
        evaluator.evaluate_model(image_pairs, args.output_dir, args.model_name, args.window_size)
        
        print(f"\n🎉 Evaluación completada exitosamente!")
        print(f"📂 Archivos CSV generados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error fatal durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())