#!/usr/bin/env python3
"""
Script para cambiar la resolución de imágenes
Permite diferentes métodos de interpolación: bicúbico, bilineal, lanczos, etc.
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'font.size': 12
})
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ImageResizer:
    """Clase para cambiar la resolución de imágenes con diferentes métodos"""
    
    def __init__(self):
        """Inicializa el redimensionador de imágenes"""
        self.methods = {
            'bicubic': cv2.INTER_CUBIC,
            'bilinear': cv2.INTER_LINEAR,
            'lanczos': cv2.INTER_LANCZOS4,
            'nearest': cv2.INTER_NEAREST,
            'area': cv2.INTER_AREA
        }
        
        self.pil_methods = {
            'bicubic': Image.BICUBIC,
            'bilinear': Image.BILINEAR,
            'lanczos': Image.LANCZOS,
            'nearest': Image.NEAREST,
            'box': Image.BOX,
            'hamming': Image.HAMMING
        }
    
    def load_image(self, image_path):
        """Carga una imagen desde archivo"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encuentra el archivo: {image_path}")
            
            # Cargar con OpenCV para mejor control
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print(f"✅ Imagen cargada: {image.shape}")
            return image
            
        except Exception as e:
            print(f"❌ Error cargando imagen {image_path}: {e}")
            return None
    
    def resize_opencv(self, image, target_size, method='bicubic'):
        """
        Redimensiona imagen usando OpenCV
        
        Args:
            image: Imagen numpy array [H, W, C]
            target_size: Tupla (width, height) o (height, width)
            method: Método de interpolación
            
        Returns:
            Imagen redimensionada
        """
        if method not in self.methods:
            raise ValueError(f"Método no válido: {method}. Opciones: {list(self.methods.keys())}")
        
        # OpenCV usa (width, height) para resize
        target_width, target_height = target_size
        
        # Redimensionar
        resized = cv2.resize(image, (target_width, target_height), 
                           interpolation=self.methods[method])
        
        return resized
    
    def resize_pil(self, image, target_size, method='bicubic'):
        """
        Redimensiona imagen usando PIL
        
        Args:
            image: Imagen numpy array [H, W, C]
            target_size: Tupla (width, height)
            method: Método de interpolación
            
        Returns:
            Imagen redimensionada
        """
        if method not in self.pil_methods:
            raise ValueError(f"Método no válido: {method}. Opciones: {list(self.pil_methods.keys())}")
        
        # Convertir a PIL Image
        pil_image = Image.fromarray(image)
        
        # Redimensionar
        resized_pil = pil_image.resize(target_size, self.pil_methods[method])
        
        # Convertir de vuelta a numpy
        resized = np.array(resized_pil)
        
        return resized
    
    def resize_image(self, image, target_size, method='bicubic', library='opencv'):
        """
        Redimensiona imagen usando el método y librería especificados
        
        Args:
            image: Imagen numpy array
            target_size: Tupla (width, height)
            method: Método de interpolación
            library: 'opencv' o 'pil'
            
        Returns:
            Imagen redimensionada
        """
        original_size = image.shape[:2]
        target_width, target_height = target_size
        
        print(f"🔄 Redimensionando: {original_size[1]}x{original_size[0]} → {target_width}x{target_height}")
        print(f"   Método: {method} | Librería: {library}")
        
        if library == 'opencv':
            resized = self.resize_opencv(image, target_size, method)
        elif library == 'pil':
            resized = self.resize_pil(image, target_size, method)
        else:
            raise ValueError(f"Librería no válida: {library}. Opciones: 'opencv', 'pil'")
        
        print(f"✅ Redimensionado completado: {resized.shape}")
        return resized
    
    def compare_methods(self, image, target_size, methods=['bicubic', 'bilinear', 'lanczos', 'nearest']):
        """
        Compara diferentes métodos de interpolación
        
        Args:
            image: Imagen original
            target_size: Tamaño objetivo
            methods: Lista de métodos a comparar
            
        Returns:
            Diccionario con resultados de cada método
        """
        results = {}
        
        print(f"\n🔬 COMPARANDO MÉTODOS DE INTERPOLACIÓN")
        print("=" * 50)
        
        for method in methods:
            print(f"\n📊 Probando método: {method}")
            
            try:
                # Probar con OpenCV
                resized_opencv = self.resize_opencv(image, target_size, method)
                
                # Probar con PIL
                resized_pil = self.resize_pil(image, target_size, method)
                
                # Calcular diferencias entre métodos
                diff = np.abs(resized_opencv.astype(float) - resized_pil.astype(float))
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                
                results[method] = {
                    'opencv': resized_opencv,
                    'pil': resized_pil,
                    'mean_difference': mean_diff,
                    'max_difference': max_diff
                }
                
                print(f"   OpenCV vs PIL - Diferencia media: {mean_diff:.4f}, Máxima: {max_diff:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error con método {method}: {e}")
                results[method] = None
        
        return results
    
    def save_comparison(self, original, results, output_path, target_size):
        """Guarda comparación visual de métodos"""
        n_methods = len([r for r in results.values() if r is not None])
        
        if n_methods == 0:
            print("❌ No hay métodos válidos para comparar")
            return
        
        # Calcular layout
        cols = min(3, n_methods + 1)  # +1 para imagen original
        rows = (n_methods + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Imagen original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(f'Original\n{original.shape[1]}x{original.shape[0]}', 
                           fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Métodos
        method_idx = 1
        for method, result in results.items():
            if result is None:
                continue
                
            row = method_idx // cols
            col = method_idx % cols
            
            # Mostrar resultado OpenCV
            axes[row, col].imshow(result['opencv'])
            axes[row, col].set_title(f'{method.upper()}\n{target_size[0]}x{target_size[1]}\n'
                                   f'Diff: {result["mean_difference"]:.4f}', 
                                   fontweight='bold', fontsize=10)
            axes[row, col].axis('off')
            
            method_idx += 1
        
        # Ocultar ejes vacíos
        for i in range(method_idx, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Comparación de Métodos de Interpolación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Comparación guardada: {output_path}")
    
    def save_image(self, image, output_path, quality=95):
        """Guarda imagen en formato PNG o JPG"""
        try:
            # Convertir a PIL para guardar
            pil_image = Image.fromarray(image)
            
            # Determinar formato por extensión
            ext = Path(output_path).suffix.lower()
            
            if ext == '.png':
                pil_image.save(output_path, 'PNG')
            elif ext in ['.jpg', '.jpeg']:
                pil_image.save(output_path, 'JPEG', quality=quality)
            else:
                # Por defecto PNG
                output_path = str(Path(output_path).with_suffix('.png'))
                pil_image.save(output_path, 'PNG')
            
            print(f"💾 Imagen guardada: {output_path}")
            
        except Exception as e:
            print(f"❌ Error guardando imagen: {e}")
    
    def process_image(self, input_path, output_path, target_size, method='bicubic', 
                     library='opencv', compare_methods=False, quality=95):
        """
        Procesa una imagen: carga, redimensiona y guarda
        
        Args:
            input_path: Ruta de imagen de entrada
            output_path: Ruta de imagen de salida
            target_size: Tupla (width, height)
            method: Método de interpolación
            library: Librería a usar ('opencv' o 'pil')
            compare_methods: Si comparar diferentes métodos
            quality: Calidad para JPG (1-100)
        """
        print(f"\n🎯 PROCESANDO IMAGEN")
        print("=" * 50)
        print(f"Entrada: {input_path}")
        print(f"Salida: {output_path}")
        print(f"Tamaño objetivo: {target_size[0]}x{target_size[1]}")
        
        # Cargar imagen
        image = self.load_image(input_path)
        if image is None:
            return None
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if compare_methods:
            # Comparar métodos
            methods_to_compare = ['bicubic', 'bilinear', 'lanczos', 'nearest']
            results = self.compare_methods(image, target_size, methods_to_compare)
            
            # Guardar comparación visual
            comparison_path = str(Path(output_path).with_suffix('')) + '_comparison.png'
            self.save_comparison(image, results, comparison_path, target_size)
            
            # Guardar imagen con método principal
            resized = self.resize_image(image, target_size, method, library)
            self.save_image(resized, output_path, quality)
            
            # Guardar métricas de comparación
            metrics_path = str(Path(output_path).with_suffix('')) + '_metrics.json'
            metrics = {}
            for method, result in results.items():
                if result is not None:
                    metrics[method] = {
                        'mean_difference': float(result['mean_difference']),
                        'max_difference': float(result['max_difference'])
                    }
            
            with open(metrics_path, 'w') as f:
                json.dump({
                    'original_size': image.shape[:2][::-1],  # (width, height)
                    'target_size': target_size,
                    'comparison_metrics': metrics
                }, f, indent=2)
            
            print(f"📄 Métricas guardadas: {metrics_path}")
            
        else:
            # Solo redimensionar con método especificado
            resized = self.resize_image(image, target_size, method, library)
            self.save_image(resized, output_path, quality)
        
        print(f"✅ Procesamiento completado!")
        return resized

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Redimensionador de Imágenes con Diferentes Métodos")
    
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta de la imagen de entrada"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de la imagen de salida"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Ancho objetivo"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Alto objetivo"
    )
    
    parser.add_argument(
        "--method",
        default="bicubic",
        choices=['bicubic', 'bilinear', 'lanczos', 'nearest', 'area'],
        help="Método de interpolación"
    )
    
    parser.add_argument(
        "--library",
        default="opencv",
        choices=['opencv', 'pil'],
        help="Librería a usar"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparar diferentes métodos de interpolación"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Calidad para JPG (1-100)"
    )
    
    args = parser.parse_args()
    
    # Verificar archivo de entrada
    if not os.path.exists(args.input):
        print(f"❌ Error: No se encuentra el archivo: {args.input}")
        return 1
    
    print("🖼️  REDIMENSIONADOR DE IMÁGENES")
    print("=" * 50)
    print(f"Entrada: {args.input}")
    print(f"Salida: {args.output}")
    print(f"Tamaño: {args.width}x{args.height}")
    print(f"Método: {args.method}")
    print(f"Librería: {args.library}")
    print(f"Comparar métodos: {args.compare}")
    
    try:
        # Crear redimensionador
        resizer = ImageResizer()
        
        # Procesar imagen
        result = resizer.process_image(
            args.input,
            args.output,
            (args.width, args.height),
            args.method,
            args.library,
            args.compare,
            args.quality
        )
        
        if result is not None:
            print(f"\n🎉 Redimensionado exitoso!")
            print(f"📂 Resultados guardados en: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())