#!/usr/bin/env python3
"""
Exportador de Modelos SwinIR a TorchScript
Convierte modelos .pth a formato optimizado .pt (TorchScript)
Similar a SavedModel de TensorFlow - incluye arquitectura + pesos optimizados
"""

import torch
import torch.nn.functional as F
import os
import argparse
from models.network_swinir import SwinIR as SwinIRNet

class SwinIRExporter:
    """Exportador de modelos SwinIR a TorchScript"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
            model = SwinIRNet(
                upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffle', resi_connection='1conv'
            )
        
        return model
    
    def export_to_torchscript(self, model_path, output_path, scale=2, training_patch_size=128, 
                             sample_size=(256, 256), optimize=True):
        """
        Exporta modelo SwinIR a TorchScript optimizado
        
        Args:
            model_path: Ruta al modelo .pth original
            output_path: Ruta de salida para el modelo .pt optimizado
            scale: Factor de escala del modelo
            training_patch_size: Tamaño de patch usado en entrenamiento
            sample_size: Tamaño de muestra para tracing (height, width)
            optimize: Si aplicar optimizaciones adicionales
        """
        print(f"🔄 Exportando modelo SwinIR a TorchScript...")
        print(f"   Modelo original: {model_path}")
        print(f"   Modelo optimizado: {output_path}")
        print(f"   Configuración: scale={scale}, patch_size={training_patch_size}")
        
        # Cargar modelo original
        print("📦 Cargando modelo original...")
        model = self.define_model_architecture(scale, training_patch_size)
        
        # Cargar pesos
        pretrained_model = torch.load(model_path, map_location=self.device)
        if 'params' in pretrained_model:
            state_dict = pretrained_model['params']
        elif 'params_ema' in pretrained_model:
            state_dict = pretrained_model['params_ema']
        else:
            state_dict = pretrained_model
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        
        print("✅ Modelo cargado correctamente")
        
        # Crear input de ejemplo para tracing
        print(f"🧪 Creando input de ejemplo {sample_size}...")
        example_input = torch.randn(1, 3, sample_size[0], sample_size[1]).to(self.device)
        
        # Verificar que funciona
        with torch.no_grad():
            test_output = model(example_input)
        print(f"   Test exitoso: {example_input.shape} -> {test_output.shape}")
        
        # Exportar a TorchScript usando tracing
        print("🚀 Exportando a TorchScript...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Optimizar si se solicita
        if optimize:
            print("⚡ Aplicando optimizaciones...")
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Guardar modelo optimizado
        print("💾 Guardando modelo optimizado...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traced_model.save(output_path)
        
        # Verificar que el modelo exportado funciona
        print("🔍 Verificando modelo exportado...")
        loaded_model = torch.jit.load(output_path, map_location=self.device)
        with torch.no_grad():
            verify_output = loaded_model(example_input)
        
        # Comparar salidas
        diff = torch.abs(test_output - verify_output).max().item()
        print(f"   Diferencia máxima: {diff:.2e}")
        
        if diff < 1e-5:
            print("✅ Modelo exportado verificado correctamente")
        else:
            print("⚠️  Diferencia detectada en salidas")
        
        # Información de tamaños
        original_size = os.path.getsize(model_path) / (1024**2)
        optimized_size = os.path.getsize(output_path) / (1024**2)
        
        print(f"\n📊 RESUMEN:")
        print(f"   Modelo original: {original_size:.1f} MB")
        print(f"   Modelo optimizado: {optimized_size:.1f} MB")
        print(f"   Factor: {optimized_size/original_size:.2f}x")
        print(f"   Incluye arquitectura: ✅")
        print(f"   Optimizado para inferencia: {'✅' if optimize else '❌'}")
        
        return output_path
    
    def export_all_models(self, models_config, output_dir="optimized_models"):
        """Exporta todos los modelos a TorchScript"""
        print("🚀 EXPORTANDO TODOS LOS MODELOS A TORCHSCRIPT")
        print("=" * 60)
        
        results = []
        
        for config in models_config:
            try:
                print(f"\n📦 Procesando modelo: {config['name']}")
                
                output_path = os.path.join(output_dir, f"swinir_{config['name']}_optimized.pt")
                
                self.export_to_torchscript(
                    model_path=config["model_path"],
                    output_path=output_path,
                    scale=config["scale"],
                    training_patch_size=config["training_patch_size"],
                    sample_size=(config["training_patch_size"], config["training_patch_size"]),
                    optimize=True
                )
                
                results.append({
                    'name': config['name'],
                    'original': config['model_path'],
                    'optimized': output_path,
                    'success': True
                })
                
            except Exception as e:
                print(f"❌ Error exportando {config['name']}: {e}")
                results.append({
                    'name': config['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # Resumen final
        print(f"\n🎉 EXPORTACIÓN COMPLETADA")
        print("=" * 40)
        successful = len([r for r in results if r['success']])
        total = len(results)
        print(f"Exitosos: {successful}/{total}")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['name']}")
        
        return results

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Exportar modelos SwinIR a TorchScript optimizado")
    
    parser.add_argument(
        "--model_path",
        help="Ruta al modelo específico a exportar"
    )
    
    parser.add_argument(
        "--output_path",
        help="Ruta de salida para modelo optimizado"
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Factor de escala del modelo"
    )
    
    parser.add_argument(
        "--training_patch_size",
        type=int,
        default=128,
        help="Tamaño de patch usado en entrenamiento"
    )
    
    parser.add_argument(
        "--export_all",
        action='store_true',
        help="Exportar todos los modelos automáticamente"
    )
    
    parser.add_argument(
        "--output_dir",
        default="optimized_models",
        help="Directorio para modelos optimizados (modo --export_all)"
    )
    
    args = parser.parse_args()
    
    exporter = SwinIRExporter()
    
    if args.export_all:
        # Configuraciones de todos los modelos
        models_config = [
            {
                "name": "64to128",
                "model_path": "superresolution/SwinIR_SR_64to128_v0/665000_G.pth",
                "scale": 2,
                "training_patch_size": 64
            },
            {
                "name": "128to256", 
                "model_path": "superresolution/SwinIR_SR_128to256/615000_G.pth",
                "scale": 2,
                "training_patch_size": 128
            },
            {
                "name": "256to512",
                "model_path": "superresolution/SwinIR_SR_256to512/700000_G.pth", 
                "scale": 2,
                "training_patch_size": 256
            },
            {
                "name": "512to1024",
                "model_path": "superresolution/SwinIR_SR_512to1024/500000_G.pth",
                "scale": 2, 
                "training_patch_size": 512
            }
        ]
        
        exporter.export_all_models(models_config, args.output_dir)
        
    else:
        if not args.model_path or not args.output_path:
            print("❌ Error: Especifica --model_path y --output_path, o usa --export_all")
            return 1
        
        exporter.export_to_torchscript(
            args.model_path,
            args.output_path,
            args.scale,
            args.training_patch_size
        )
    
    print("\n💡 Para usar los modelos optimizados:")
    print("   model = torch.jit.load('modelo_optimizado.pt')")
    print("   output = model(input_tensor)  # Sin especificar arquitectura!")

if __name__ == "__main__":
    exit(main())