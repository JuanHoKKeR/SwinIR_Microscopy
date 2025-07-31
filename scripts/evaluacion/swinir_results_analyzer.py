#!/usr/bin/env python3
"""
Analizador de Resultados de Superresolución para SwinIR
Compara imagen LR, genera predicción con SwinIR, compara con HR original y genera análisis visual completo
Adaptado del sr_results_analyzer.py para funcionar con modelos SwinIR en formato .pt
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
import cv2
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar TensorFlow para KimiaNet
try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow no disponible. Índice perceptual no estará disponible.")
    TF_AVAILABLE = False

# Configurar fuente Computer Modern Roman para matplotlib
plt.rcParams.update({
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 11
})


class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet+KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no está disponible")
            
        print("🧠 Cargando DenseNet121 con pesos KimiaNet...")
        
        # Cargar DenseNet121 sin la capa final
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,
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
        """Calcula distancia perceptual entre dos imágenes usando KimiaNet"""
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
        
        return perceptual_distance


class SwinIRSuperResolutionAnalyzer:
    """Analizador completo de resultados de superresolución para SwinIR"""
    
    def __init__(self, model_path, kimianet_weights_path=None, device='auto'):
        """Inicializa el analizador con modelo SwinIR y evaluador perceptual"""
        
        # Configurar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Usando dispositivo: {self.device}")
        
        # Cargar modelo SwinIR
        self.model = self.load_swinir_model(model_path)
        
        # Inicializar evaluador perceptual
        self.perceptual_evaluator = None
        if kimianet_weights_path and TF_AVAILABLE:
            try:
                self.perceptual_evaluator = KimiaNetPerceptualLoss(kimianet_weights_path)
            except Exception as e:
                print(f"⚠️  Error inicializando evaluador perceptual: {e}")
    
    def load_swinir_model(self, model_path):
        """Carga un modelo SwinIR optimizado (.pt)"""
        print(f"📦 Cargando modelo SwinIR desde: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        try:
            # Cargar modelo optimizado (TorchScript)
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.eval()
            
            print("✅ Modelo SwinIR TorchScript cargado correctamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def load_image_torch(self, image_path, target_size=None):
        """Carga una imagen y la prepara como tensor PyTorch"""
        try:
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
            
            # Redimensionar si se especifica
            if target_size is not None:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Convertir a tensor PyTorch y cambiar dimensiones a CHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            
            # Agregar dimensión de batch
            image_tensor = image_tensor.unsqueeze(0)
            
            print(f"✅ Imagen cargada - Tamaño: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            print(f"❌ Error cargando {image_path}: {e}")
            return None
    
    def load_image_tf(self, image_path, target_size=None):
        """Carga una imagen y la devuelve como tensor TensorFlow para métricas"""
        try:
            # Usar OpenCV para cargar (consistente con PyTorch)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a float32
            image = image.astype(np.float32)
            
            # Redimensionar si se especifica
            if target_size is not None:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Convertir a tensor TensorFlow
            image = tf.constant(image)
            
            return image
            
        except Exception as e:
            print(f"❌ Error cargando {image_path}: {e}")
            return None
    
    def perform_super_resolution(self, lr_tensor):
        """Realiza superresolución con el modelo SwinIR"""
        print("🚀 Realizando superresolución con SwinIR...")
        
        # Mover tensor a dispositivo
        lr_tensor = lr_tensor.to(self.device)
        
        # Realizar inferencia
        with torch.no_grad():
            start_time = time.time()
            sr_tensor = self.model(lr_tensor)
            inference_time = time.time() - start_time
        
        print(f"✅ Superresolución completada en {inference_time:.3f} segundos")
        print(f"   Input: {lr_tensor.shape} -> Output: {sr_tensor.shape}")
        
        return sr_tensor
    
    def tensor_to_tf(self, pytorch_tensor):
        """Convierte tensor PyTorch a TensorFlow para métricas"""
        # Mover a CPU si está en GPU
        if pytorch_tensor.is_cuda:
            pytorch_tensor = pytorch_tensor.cpu()
        
        # Convertir a numpy y luego a TensorFlow
        numpy_array = pytorch_tensor.squeeze(0).permute(1, 2, 0).numpy()
        numpy_array = numpy_array * 255.0  # Desnormalizar a [0, 255]
        
        return tf.constant(numpy_array, dtype=tf.float32)
    
    def calculate_metrics(self, predicted_tensor, ground_truth_img):
        """Calcula todas las métricas de evaluación"""
        
        # Convertir tensor PyTorch a TensorFlow
        predicted_img = self.tensor_to_tf(predicted_tensor)
        
        # Asegurar mismo tamaño
        if predicted_img.shape != ground_truth_img.shape:
            print(f"⚠️  Redimensionando ground truth: {ground_truth_img.shape} → {predicted_img.shape}")
            ground_truth_img = tf.image.resize(ground_truth_img, predicted_img.shape[:2], method='bicubic')
        
        # PSNR y SSIM
        psnr = tf.image.psnr(predicted_img, ground_truth_img, max_val=255.0)
        ssim = tf.image.ssim(predicted_img, ground_truth_img, max_val=255.0)
        
        # MS-SSIM con manejo de errores
        try:
            ms_ssim = tf.image.ssim_multiscale(
                tf.expand_dims(predicted_img, 0),
                tf.expand_dims(ground_truth_img, 0),
                max_val=255.0
            )
            ms_ssim = tf.squeeze(ms_ssim)
        except:
            ms_ssim = ssim
        
        # MSE y MAE
        mse = tf.reduce_mean(tf.square(predicted_img - ground_truth_img))
        mae = tf.reduce_mean(tf.abs(predicted_img - ground_truth_img))
        
        metrics = {
            'psnr': float(psnr.numpy()),
            'ssim': float(ssim.numpy()),
            'ms_ssim': float(ms_ssim.numpy()),
            'mse': float(mse.numpy()),
            'mae': float(mae.numpy())
        }
        
        # Índice perceptual si está disponible
        if self.perceptual_evaluator:
            try:
                perceptual_dist = self.perceptual_evaluator.calculate_perceptual_distance(
                    predicted_img, ground_truth_img
                )
                metrics['perceptual_index'] = float(perceptual_dist.numpy())
            except Exception as e:
                print(f"⚠️  Error calculando índice perceptual: {e}")
                metrics['perceptual_index'] = None
        else:
            metrics['perceptual_index'] = None
        
        return metrics
    
    def calculate_absolute_difference(self, predicted_tensor, ground_truth_img):
        """Calcula diferencia absoluta normalizada"""
        # Convertir tensor PyTorch a TensorFlow y normalizar
        predicted_img = self.tensor_to_tf(predicted_tensor)
        pred_norm = predicted_img / 255.0
        gt_norm = ground_truth_img / 255.0
        
        # Asegurar mismo tamaño
        if pred_norm.shape != gt_norm.shape:
            gt_norm = tf.image.resize(gt_norm, pred_norm.shape[:2], method='bicubic')
        
        # Diferencia absoluta
        abs_diff = tf.abs(pred_norm - gt_norm)
        
        return abs_diff
    
    def generate_latex_table(self, metrics, output_path, image_name=""):
        """Genera tabla LaTeX con las métricas"""
        
        latex_table = f"""\\begin{{table}}[!htb]
\\centering
\\caption{{Métricas de evaluación para superresolución SwinIR: {image_name}}}
\\label{{tab:swinir_metrics_{image_name.lower().replace(' ', '_')}}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} \\\\
\\hline
PSNR (dB) & {metrics['psnr']:.4f} \\\\
SSIM & {metrics['ssim']:.4f} \\\\
MS-SSIM & {metrics['ms_ssim']:.4f} \\\\
MSE & {metrics['mse']:.2f} \\\\
MAE & {metrics['mae']:.2f} \\\\
"""
        
        if metrics['perceptual_index'] is not None:
            latex_table += f"Índice Perceptual & {metrics['perceptual_index']:.6f} \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\textbf{Notas:} PSNR más alto indica mejor fidelidad. SSIM evalúa similitud estructural (0-1, más alto mejor). MS-SSIM es extensión multi-escala de SSIM. MSE y MAE miden error medio cuadrático y absoluto respectivamente. Índice Perceptual basado en KimiaNet (más bajo mejor).
\\end{table}
"""
        
        # Guardar archivo LaTeX
        latex_file = output_path.replace('.png', '_metrics.tex').replace('.jpg', '_metrics.tex')
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"📄 Tabla LaTeX guardada en: {latex_file}")
        return latex_file
    
    def create_comprehensive_analysis(self, lr_path, hr_path, output_path, image_name="Análisis SwinIR"):
        """
        Crea análisis visual completo de resultados de superresolución con SwinIR
        """
        
        print("🎨 CREANDO ANÁLISIS VISUAL COMPLETO PARA SWINIR")
        print("=" * 60)
        
        # Cargar imagen LR para PyTorch (inferencia)
        print("📁 Cargando imagen LR para inferencia...")
        lr_tensor = self.load_image_torch(lr_path)
        
        if lr_tensor is None:
            print("❌ Error cargando imagen LR")
            return None
        
        # Realizar superresolución
        print("🚀 Generando predicción con SwinIR...")
        predicted_tensor = self.perform_super_resolution(lr_tensor)
        
        # Cargar imágenes para visualización/métricas (TensorFlow)
        print("📁 Cargando imágenes para análisis...")
        lr_img = self.load_image_tf(lr_path)
        hr_img = self.load_image_tf(hr_path)
        
        if any(img is None for img in [lr_img, hr_img]):
            print("❌ Error cargando alguna de las imágenes")
            return None
        
        # Obtener formas para display
        lr_shape = lr_tensor.shape[2:]  # [H, W]
        predicted_shape = predicted_tensor.shape[2:]  # [H, W]
        hr_shape = hr_img.shape[:2]  # [H, W]
        
        print(f"   LR: {lr_shape}")
        print(f"   Predicha: {predicted_shape}")
        print(f"   HR: {hr_shape}")
        
        # Calcular factor de escala
        scale_factor = predicted_shape[0] // lr_shape[0]
        
        # Calcular métricas
        print("📊 Calculando métricas...")
        metrics = self.calculate_metrics(predicted_tensor, hr_img)
        
        # Mostrar métricas
        print(f"\n📈 MÉTRICAS CALCULADAS:")
        print(f"   PSNR: {metrics['psnr']:.4f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"   MS-SSIM: {metrics['ms_ssim']:.4f}")
        print(f"   MSE: {metrics['mse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        if metrics['perceptual_index'] is not None:
            print(f"   Índice Perceptual: {metrics['perceptual_index']:.6f}")
        
        # Calcular diferencia absoluta
        abs_diff = self.calculate_absolute_difference(predicted_tensor, hr_img)
        
        # Generar tabla LaTeX
        self.generate_latex_table(metrics, output_path, image_name)
        
        # Crear visualización IDÉNTICA al script original
        print("🎨 Generando visualización...")
        
        fig = plt.figure(figsize=(18, 5))
        
        # Convertir tensores a numpy para matplotlib
        def tensor_to_numpy(tensor, is_pytorch=False):
            if is_pytorch:
                # Tensor PyTorch [1, 3, H, W] -> [H, W, 3]
                array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                return np.clip((array * 255).astype(np.uint8), 0, 255)
            else:
                # Tensor TensorFlow [H, W, 3]
                return np.clip(tensor.numpy().astype(np.uint8), 0, 255)
        
        # Título principal
        fig.suptitle(f'Análisis Completo de Superresolución: {image_name}', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Configurar subplot con LR MUCHO más pequeña para impacto visual
        gs = GridSpec(1, 5, figure=fig, width_ratios=[0.5, 0.3, 2, 2, 2], 
                     wspace=0.2, left=0.05, right=0.95, top=0.85, bottom=0.05)
        
        # 1. Imagen LR (MUCHO más pequeña visualmente)
        ax1 = fig.add_subplot(gs[0, 0])
        lr_display = tensor_to_numpy(lr_tensor, is_pytorch=True)
        ax1.imshow(lr_display)
        ax1.set_title(f'LR Input\n{lr_shape[0]}×{lr_shape[1]}', 
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Agregar borde rojo MÁS GRUESO para destacar
        rect = patches.Rectangle((0, 0), lr_shape[1]-1, lr_shape[0]-1, 
                               linewidth=4, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # Agregar texto "PEQUEÑA" para enfatizar
        ax1.text(0.5, -0.15, '¡Pequeña!', ha='center', va='top', 
                fontsize=9, fontweight='bold', color='red',
                transform=ax1.transAxes)
        
        # 2. Flecha y texto de modelo
        ax_arrow = fig.add_subplot(gs[0, 1])
        ax_arrow.axis('off')
        
        # Flecha más dramática
        arrow = patches.FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                                       connectionstyle="arc3", 
                                       arrowstyle='->', mutation_scale=30,
                                       transform=ax_arrow.transAxes, 
                                       color='blue', linewidth=4)
        ax_arrow.add_patch(arrow)
        
        # Texto del modelo
        ax_arrow.text(0.5, 0.7, f'SwinIR\n×{scale_factor}', ha='center', va='center', 
                     fontsize=9, fontweight='bold', color='blue',
                     transform=ax_arrow.transAxes)
        
        # 3. Imagen predicha (GRANDE)
        ax2 = fig.add_subplot(gs[0, 2])
        predicted_display = tensor_to_numpy(predicted_tensor, is_pytorch=True)
        ax2.imshow(predicted_display)
        ax2.set_title(f'Predicción SwinIR\n{predicted_shape[0]}×{predicted_shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Agregar texto "GRANDE" para contrastar
        ax2.text(0.5, -0.08, '¡Grande!', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='green',
                transform=ax2.transAxes)
        
        # 4. Imagen HR original (GRANDE)
        ax3 = fig.add_subplot(gs[0, 3])
        hr_display = tensor_to_numpy(hr_img)
        ax3.imshow(hr_display)
        ax3.set_title(f'HR Original\n{hr_shape[0]}×{hr_shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # 5. Diferencia absoluta con mapa de calor
        ax4 = fig.add_subplot(gs[0, 4])
        diff_display = np.mean(abs_diff.numpy(), axis=-1)  # Promedio de canales RGB
        
        im = ax4.imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
        ax4.set_title(f'Diferencia Absoluta\nError Promedio: {np.mean(diff_display):.4f}', 
                     fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # Colorbar para diferencia absoluta
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Error Absoluto', rotation=270, labelpad=12, fontsize=9)
        
        # Ajustar layout compacto
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Guardar imagen
        print(f"💾 Guardando visualización en: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        
        # Resumen en consola
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL ANÁLISIS SWINIR")
        print("=" * 60)
        
        print(f"🔍 Factor de escala detectado: ×{scale_factor}")
        print(f"📏 Resolución: {lr_shape[0]}×{lr_shape[1]} → {predicted_shape[0]}×{predicted_shape[1]}")
        
        # Evaluación de calidad
        if metrics['psnr'] > 25:
            quality = "Excelente"
        elif metrics['psnr'] > 22:
            quality = "Buena"
        elif metrics['psnr'] > 18:
            quality = "Moderada"
        else:
            quality = "Baja"
        
        print(f"⭐ Calidad estimada: {quality}")
        print(f"💡 Error promedio: {np.mean(diff_display):.4f}")
        
        if metrics['perceptual_index'] is not None:
            if metrics['perceptual_index'] < 0.01:
                perceptual_quality = "Excelente fidelidad biológica"
            elif metrics['perceptual_index'] < 0.05:
                perceptual_quality = "Buena fidelidad biológica"
            else:
                perceptual_quality = "Fidelidad biológica moderada"
            print(f"🧠 Evaluación perceptual: {perceptual_quality}")
        
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Analizador completo de resultados de superresolución SwinIR")
    
    parser.add_argument("--lr", required=True, help="Imagen LR de entrada")
    parser.add_argument("--hr", required=True, help="Imagen HR original (ground truth)")
    parser.add_argument("--model", required=True, help="Modelo SwinIR optimizado (.pt)")
    parser.add_argument("--output", required=True, help="Ruta de salida para el análisis")
    parser.add_argument("--kimianet_weights", 
                       default=None,
                       help="Pesos KimiaNet para índice perceptual")
    parser.add_argument("--name", default="Análisis SwinIR", help="Nombre para el análisis")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Dispositivo a usar (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Verificar que todos los archivos existen
    required_files = [args.lr, args.hr, args.model]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ No existe el archivo: {file_path}")
            return 1
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Crear analizador
        analyzer = SwinIRSuperResolutionAnalyzer(
            model_path=args.model,
            kimianet_weights_path=args.kimianet_weights,
            device=args.device
        )
        
        # Realizar análisis
        metrics = analyzer.create_comprehensive_analysis(
            lr_path=args.lr,
            hr_path=args.hr,
            output_path=args.output,
            image_name=args.name
        )
        
        if metrics:
            print(f"✅ Análisis completado: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())