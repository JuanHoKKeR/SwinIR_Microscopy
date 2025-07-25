#!/usr/bin/env python3
"""
Benchmark de Tiempo de Inferencia SwinIR - VERSIÓN REALISTA
Usa imágenes reales del dataset de validación para medición precisa
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import psutil
import platform
from pathlib import Path
from tqdm import tqdm
import json
import random
import cv2
import warnings

# Imports específicos de SwinIR
from models.network_swinir import SwinIR as SwinIRNet

warnings.filterwarnings('ignore')

class SwinIRRealisticTimingBenchmark:
    """Benchmark de timing usando dataset real de validación para SwinIR"""
    
    def __init__(self, device='auto'):
        """
        Inicializa el benchmark
        
        Args:
            device: 'cpu', 'gpu', o 'auto'
        """
        self.device_type = device
        self.device = None
        self.model = None
        self.device_info = self._get_device_info()
        self._configure_device()
        
    def _get_device_info(self):
        """Obtiene información del hardware"""
        device_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__
        }
        
        # Información de GPU
        if torch.cuda.is_available():
            device_info['gpu_available'] = True
            device_info['gpu_count'] = torch.cuda.device_count()
            device_info['cuda_version'] = torch.version.cuda
            try:
                gpu_details = []
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_details.append(f"GPU_{i}: {gpu_name} ({gpu_memory:.1f}GB)")
                device_info['gpu_details'] = gpu_details
            except:
                device_info['gpu_details'] = "GPU info not available"
        else:
            device_info['gpu_available'] = False
            device_info['gpu_count'] = 0
        
        return device_info
    
    def _configure_device(self):
        """Configura el dispositivo para el benchmark"""
        if self.device_type.lower() == 'cpu':
            # Forzar uso de CPU
            self.device = torch.device('cpu')
            # Opcional: También puedes configurar las variables de entorno
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("🖥️  Configurado para usar CPU (GPU deshabilitada)")
            
        elif self.device_type.lower() == 'gpu':
            # Verificar que GPU esté disponible
            if not torch.cuda.is_available():
                print("⚠️  GPU solicitada pero no disponible, usando CPU")
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
            else:
                self.device = torch.device('cuda')
                print(f"🚀 Configurado para usar GPU (CUDA: {torch.version.cuda})")
                print(f"   GPU disponible: {torch.cuda.get_device_name()}")
                
        else:  # auto
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.device_type = 'gpu'
                print(f"🚀 Auto-detectado: usando GPU ({torch.cuda.get_device_name()})")
            else:
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
                print("🖥️  Auto-detectado: usando CPU")
    
    def define_model_architecture(self, scale, training_patch_size):
        """Define la arquitectura del modelo SwinIR"""
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
            # Definir arquitectura del modelo
            self.model = self.define_model_architecture(scale, training_patch_size)
            
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
    
    def load_validation_images(self, lr_meta_file, base_path="", max_images=50):
        """
        Carga imágenes reales del dataset de validación
        
        Args:
            lr_meta_file: Archivo meta con rutas de imágenes LR
            base_path: Ruta base para las imágenes
            max_images: Máximo número de imágenes a cargar
            
        Returns:
            Lista de tensores de imágenes cargadas
        """
        print(f"📂 Cargando imágenes reales desde: {lr_meta_file}")
        
        # Cargar rutas LR
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        # Limitar número de imágenes
        if len(lr_paths) > max_images:
            lr_paths = random.sample(lr_paths, max_images)
            print(f"🎲 Muestreadas {max_images} imágenes de {len(lr_paths)} disponibles")
        
        print(f"📊 Cargando {len(lr_paths)} imágenes...")
        
        images = []
        loaded_count = 0
        
        for lr_path in tqdm(lr_paths, desc="Cargando imágenes"):
            try:
                # Cargar imagen con OpenCV
                image = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"⚠️  No se pudo cargar: {lr_path}")
                    continue
                
                # Convertir BGR a RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convertir a float32 y normalizar a [0, 1]
                image = image.astype(np.float32) / 255.0
                
                # Convertir a tensor PyTorch y cambiar dimensiones a CHW
                image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                
                # Verificar que la imagen se cargó correctamente
                if image_tensor.shape[1] > 0 and image_tensor.shape[2] > 0:
                    images.append(image_tensor)
                    loaded_count += 1
                else:
                    print(f"⚠️  Imagen vacía saltada: {lr_path}")
                    
            except Exception as e:
                print(f"❌ Error cargando {lr_path}: {e}")
                continue
        
        print(f"✅ Cargadas exitosamente {loaded_count} imágenes")
        
        if loaded_count == 0:
            raise ValueError("No se pudieron cargar imágenes del dataset")
        
        return images
    
    def prepare_batch_images(self, images, batch_size=1, target_size=None, window_size=8):
        """
        Prepara imágenes en batches del tamaño requerido con padding para SwinIR
        
        Args:
            images: Lista de imágenes cargadas (tensores CHW)
            batch_size: Tamaño del batch
            target_size: Tamaño objetivo (height, width) o None para usar tamaño original
            window_size: Tamaño de ventana para padding
            
        Returns:
            Lista de batches preparados
        """
        batches = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Si el batch no está completo, rellenar con imágenes repetidas
            while len(batch_images) < batch_size:
                batch_images.extend(batch_images[:batch_size-len(batch_images)])
            
            processed_images = []
            for img in batch_images[:batch_size]:
                # Redimensionar si es necesario
                if target_size:
                    img = F.interpolate(
                        img.unsqueeze(0), 
                        size=target_size, 
                        mode='bicubic', 
                        align_corners=False
                    ).squeeze(0)
                
                # Aplicar padding para SwinIR (múltiplo de window_size)
                _, h, w = img.shape
                h_pad = (h // window_size + 1) * window_size - h
                w_pad = (w // window_size + 1) * window_size - w
                img_padded = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')
                
                processed_images.append(img_padded)
            
            # Crear batch tensor
            batch_tensor = torch.stack(processed_images)
            batches.append(batch_tensor)
        
        return batches
    
    def measure_inference_time_realistic(self, images, batch_size=1, num_warmup=5, num_runs=20, 
                                      target_size=None, window_size=8):
        """
        Mide el tiempo de inferencia usando imágenes reales
        
        Args:
            images: Lista de imágenes del dataset de validación
            batch_size: Tamaño del batch
            num_warmup: Número de ejecuciones de calentamiento
            num_runs: Número de ejecuciones para medir tiempo
            target_size: Tamaño objetivo (height, width) o None
            window_size: Tamaño de ventana para SwinIR
            
        Returns:
            Dict con estadísticas de timing
        """
        print(f"⏱️  Midiendo tiempo de inferencia con imágenes reales...")
        print(f"   Imágenes disponibles: {len(images)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Warmup runs: {num_warmup}")
        print(f"   Timing runs: {num_runs}")
        print(f"   Dispositivo: {self.device}")
        
        # Preparar batches
        batches = self.prepare_batch_images(images, batch_size, target_size, window_size)
        
        if len(batches) == 0:
            raise ValueError("No se pudieron crear batches de imágenes")
        
        print(f"   Batches preparados: {len(batches)}")
        
        # Mover batches al dispositivo
        batches = [batch.to(self.device) for batch in batches]
        
        # Medir memoria antes de empezar
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Memoria GPU si está disponible
        gpu_memory_before = 0
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        # Warmup runs
        print("🔥 Ejecutando warmup...")
        with torch.no_grad():
            for i in range(num_warmup):
                batch = batches[i % len(batches)]
                _ = self.model(batch)
                
                # Sincronizar GPU si es necesario
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Medir tiempos
        print("⏲️  Midiendo tiempos...")
        times = []
        
        with torch.no_grad():
            for i in tqdm(range(num_runs), desc="Timing runs"):
                batch = batches[i % len(batches)]
                
                # Sincronizar antes de medir
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                output = self.model(batch)
                
                # Sincronizar después de medir
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                times.append(inference_time)
        
        # Medir memoria después
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        gpu_memory_after = 0
        if self.device.type == 'cuda':
            gpu_memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        # Calcular estadísticas
        times = np.array(times)
        
        # Obtener información de la imagen
        sample_batch = batches[0]
        input_shape = sample_batch.shape
        
        stats = {
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'median_time_ms': float(np.median(times) * 1000),
            'p95_time_ms': float(np.percentile(times, 95) * 1000),
            'p99_time_ms': float(np.percentile(times, 99) * 1000),
            'fps': float(batch_size / np.mean(times)),
            'memory_increase_mb': float(memory_after - memory_before),
            'gpu_memory_increase_mb': float(gpu_memory_after - gpu_memory_before) if self.device.type == 'cuda' else 0,
            'num_runs': num_runs,
            'batch_size': batch_size,
            'input_height': int(input_shape[2]),  # H en NCHW
            'input_width': int(input_shape[3]),   # W en NCHW
            'num_images_used': len(images),
            'num_batches_used': len(batches),
            'using_real_images': True,
            'device_type': self.device.type
        }
        
        return stats
    
    def benchmark_model_realistic(self, model_path, model_name, lr_meta_file, 
                                base_path="", configurations=None, scale=2, 
                                training_patch_size=128, window_size=8):
        """
        Ejecuta benchmark completo usando dataset real de validación
        
        Args:
            model_path: Ruta al modelo
            model_name: Nombre del modelo
            lr_meta_file: Archivo meta con rutas de imágenes LR
            base_path: Ruta base para imágenes
            configurations: Lista de configuraciones a probar
            scale: Factor de escala del modelo
            training_patch_size: Tamaño de patch usado en entrenamiento
            window_size: Tamaño de ventana de SwinIR
        
        Returns:
            Lista de resultados
        """
        print(f"\n🏁 BENCHMARKING REALISTA SWINIR - MODELO: {model_name}")
        print("=" * 60)
        
        # Cargar modelo
        self.load_model(model_path, scale, training_patch_size)
        
        # Cargar imágenes reales del dataset
        images = self.load_validation_images(lr_meta_file, base_path, max_images=100)
        
        # Configuraciones por defecto si no se proporcionan
        if configurations is None:
            configurations = [
                {'batch_size': 1, 'num_warmup': 5, 'num_runs': 20},
            ]
        
        results = []
        
        for i, config in enumerate(configurations):
            print(f"\n📊 Configuración {i+1}/{len(configurations)}")
            
            try:
                timing_stats = self.measure_inference_time_realistic(
                    images=images,
                    batch_size=config.get('batch_size', 1),
                    num_warmup=config.get('num_warmup', 5),
                    num_runs=config.get('num_runs', 20),
                    target_size=config.get('target_size', None),
                    window_size=window_size
                )
                
                # Agregar información del contexto
                result = {
                    'model_name': model_name,
                    'device': self.device_type,
                    'scale': scale,
                    'training_patch_size': training_patch_size,
                    'window_size': window_size,
                    'timestamp': time.time(),
                    'lr_meta_file': lr_meta_file,
                    **timing_stats,
                    **self.device_info
                }
                
                results.append(result)
                
                print(f"✅ Tiempo promedio: {timing_stats['mean_time_ms']:.2f} ms")
                print(f"   FPS: {timing_stats['fps']:.2f}")
                print(f"   Memoria RAM adicional: {timing_stats['memory_increase_mb']:.1f} MB")
                if timing_stats['gpu_memory_increase_mb'] > 0:
                    print(f"   Memoria GPU adicional: {timing_stats['gpu_memory_increase_mb']:.1f} MB")
                print(f"   Imágenes reales usadas: {timing_stats['num_images_used']}")
                
            except Exception as e:
                print(f"❌ Error en configuración {i+1}: {e}")
                
        return results
    
    def save_results(self, results, output_dir, filename_prefix):
        """
        Guarda los resultados en CSV y JSON
        
        Args:
            results: Lista de resultados del benchmark
            output_dir: Directorio de salida
            filename_prefix: Prefijo para los archivos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not results:
            print("⚠️  No hay resultados para guardar")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Guardar CSV
        csv_file = os.path.join(output_dir, f"{filename_prefix}_{self.device_type}_timing.csv")
        df.to_csv(csv_file, index=False)
        print(f"💾 Resultados CSV guardados: {csv_file}")
        
        # Guardar JSON con más detalles
        json_file = os.path.join(output_dir, f"{filename_prefix}_{self.device_type}_timing.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Resultados JSON guardados: {json_file}")
        
        # Crear resumen estadístico
        self._create_timing_summary(df, output_dir, filename_prefix)
    
    def _create_timing_summary(self, df, output_dir, filename_prefix):
        """Crea un resumen estadístico de los tiempos"""
        summary_file = os.path.join(output_dir, f"{filename_prefix}_{self.device_type}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"RESUMEN DE TIMING REALISTA SWINIR - {self.device_type.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("INFORMACIÓN DEL SISTEMA:\n")
            f.write(f"Dispositivo: {self.device_type.upper()}\n")
            f.write(f"Platform: {self.device_info['platform']}\n")
            f.write(f"CPU: {self.device_info['processor']}\n")
            f.write(f"Cores: {self.device_info['cpu_count']}\n")
            f.write(f"RAM: {self.device_info['ram_gb']} GB\n")
            if self.device_info['gpu_available']:
                f.write(f"GPU: Disponible ({self.device_info['gpu_count']} unidades)\n")
                for gpu_detail in self.device_info.get('gpu_details', []):
                    f.write(f"     {gpu_detail}\n")
            else:
                f.write("GPU: No disponible\n")
            f.write(f"PyTorch: {self.device_info['torch_version']}\n")
            if self.device_info['gpu_available']:
                f.write(f"CUDA: {self.device_info.get('cuda_version', 'N/A')}\n")
            f.write("\n")
            
            f.write("ESTADÍSTICAS DE TIMING CON IMÁGENES REALES:\n")
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                f.write(f"\n{model}:\n")
                f.write(f"  Tiempo promedio: {model_data['mean_time_ms'].mean():.2f} ms\n")
                f.write(f"  FPS promedio: {model_data['fps'].mean():.2f}\n")
                f.write(f"  Memoria RAM promedio: {model_data['memory_increase_mb'].mean():.1f} MB\n")
                if 'gpu_memory_increase_mb' in model_data.columns and model_data['gpu_memory_increase_mb'].mean() > 0:
                    f.write(f"  Memoria GPU promedio: {model_data['gpu_memory_increase_mb'].mean():.1f} MB\n")
                f.write(f"  Imágenes reales usadas: {model_data['num_images_used'].iloc[0]}\n")
                f.write(f"  Tamaño imagen: {model_data['input_height'].iloc[0]}x{model_data['input_width'].iloc[0]}\n")
                f.write(f"  Escala: {model_data['scale'].iloc[0]}x\n")
                f.write(f"  Training patch size: {model_data['training_patch_size'].iloc[0]}\n")
        
        print(f"📄 Resumen guardado: {summary_file}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Benchmark Realista de Tiempo de Inferencia SwinIR")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo SwinIR (.pth file)"
    )
    
    parser.add_argument(
        "--model_name",
        required=True,
        help="Nombre del modelo (ej: SwinIR_128to256)"
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
        help="Archivo meta con rutas de imágenes LR del dataset de validación"
    )
    
    parser.add_argument(
        "--base_path",
        default="",
        help="Ruta base para las imágenes"
    )
    
    parser.add_argument(
        "--device",
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help="Dispositivo a usar para benchmark"
    )
    
    parser.add_argument(
        "--batch_sizes",
        nargs='+',
        type=int,
        default=[1],
        help="Tamaños de batch a probar (ej: 1 2 4)"
    )
    
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Número de ejecuciones para timing"
    )
    
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Número de ejecuciones de warmup"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Máximo número de imágenes del dataset a usar"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./realistic_timing_results",
        help="Directorio para guardar resultados"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    required_files = [args.model_path, args.lr_meta_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra: {file_path}")
            return 1
    
    print("⚡ BENCHMARK REALISTA DE TIEMPO DE INFERENCIA SWINIR")
    print("=" * 60)
    print(f"Modelo: {args.model_name}")
    print(f"Ruta: {args.model_path}")
    print(f"Escala: {args.scale}")
    print(f"Training patch size: {args.training_patch_size}")
    print(f"Window size: {args.window_size}")
    print(f"Dataset LR: {args.lr_meta_file}")
    print(f"Dispositivo: {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs: {args.num_runs}")
    print(f"Imágenes máx: {args.max_images}")
    
    try:
        # Inicializar benchmark
        benchmark = SwinIRRealisticTimingBenchmark(device=args.device)
        
        # Crear configuraciones de prueba
        configurations = []
        for batch_size in args.batch_sizes:
            configurations.append({
                'batch_size': batch_size,
                'num_runs': args.num_runs,
                'num_warmup': args.num_warmup
            })
        
        # Ejecutar benchmark realista
        results = benchmark.benchmark_model_realistic(
            args.model_path,
            args.model_name,
            args.lr_meta_file,
            args.base_path,
            configurations,
            args.scale,
            args.training_patch_size,
            args.window_size
        )
        
        # Guardar resultados
        benchmark.save_results(results, args.output_dir, args.model_name)
        
        print(f"\n🎉 Benchmark realista completado exitosamente!")
        print(f"📂 Resultados guardados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())