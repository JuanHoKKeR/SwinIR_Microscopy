#!/usr/bin/env python3
"""
Script Simple para Ejecutar Evaluaciones de SwinIR
Interface fácil para el evaluador automático
"""

import os
import sys
import subprocess
from datetime import datetime

def show_banner():
    """Muestra banner inicial"""
    print("\n" + "="*60)
    print("🤖 EVALUADOR AUTOMÁTICO DE MODELOS SWINIR")
    print("="*60)
    print("Ejecuta evaluación completa para todos o algunos modelos")
    print("Para cada modelo: Métricas → Timing GPU → Timing CPU")
    print("="*60)

def show_available_models():
    """Muestra modelos disponibles"""
    models = [
        ("64to128", "64×64 → 128×128 (Scale 2x)"),
        ("128to256", "128×128 → 256×256 (Scale 2x)"), 
        ("256to512", "256×256 → 512×512 (Scale 2x)"),
        ("512to1024", "512×512 → 1024×1024 (Scale 2x)")
    ]
    
    print("\n📦 MODELOS DISPONIBLES:")
    for i, (name, desc) in enumerate(models, 1):
        print(f"   {i}. {name:<12} - {desc}")
    
    return [name for name, _ in models]

def get_user_selection():
    """Obtiene selección del usuario"""
    print("\n🎯 OPCIONES DE EVALUACIÓN:")
    print("   1. Evaluar TODOS los modelos (completo)")
    print("   2. Evaluar modelos ESPECÍFICOS")
    print("   3. Solo VERIFICAR configuración (dry run)")
    print("   4. Salir")
    
    while True:
        try:
            choice = int(input("\n👉 Selecciona una opción (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("⚠️  Opción inválida. Usa 1, 2, 3 o 4.")
        except ValueError:
            print("⚠️  Por favor ingresa un número.")

def select_specific_models():
    """Permite seleccionar modelos específicos"""
    available_models = show_available_models()
    
    print("\n📋 SELECCIÓN DE MODELOS:")
    print("Ingresa los números de los modelos a evaluar (ej: 1,3 o 1 2 4)")
    print("Presiona Enter para evaluar todos")
    
    selection = input("👉 Tu selección: ").strip()
    
    if not selection:
        return available_models
    
    try:
        # Procesar selección (comas o espacios)
        if ',' in selection:
            indices = [int(x.strip()) for x in selection.split(',')]
        else:
            indices = [int(x) for x in selection.split()]
        
        selected_models = []
        for idx in indices:
            if 1 <= idx <= len(available_models):
                selected_models.append(available_models[idx - 1])
            else:
                print(f"⚠️  Índice {idx} inválido. Ignorando.")
        
        if not selected_models:
            print("⚠️  No se seleccionaron modelos válidos. Usando todos.")
            return available_models
        
        return selected_models
        
    except ValueError:
        print("⚠️  Formato inválido. Usando todos los modelos.")
        return available_models

def get_evaluation_settings():
    """Obtiene configuraciones de evaluación"""
    print("\n⚙️  CONFIGURACIONES:")
    
    # Número de imágenes para métricas
    while True:
        try:
            max_images = input("📊 Máximo de imágenes para métricas (default: 1000): ").strip()
            max_images = 1000 if not max_images else int(max_images)
            if max_images > 0:
                break
            else:
                print("⚠️  Debe ser un número positivo.")
        except ValueError:
            print("⚠️  Ingresa un número válido.")
    
    # Número de runs para timing
    while True:
        try:
            num_runs = input("⏱️  Número de runs para timing (default: 50): ").strip()
            num_runs = 50 if not num_runs else int(num_runs)
            if num_runs > 0:
                break
            else:
                print("⚠️  Debe ser un número positivo.")
        except ValueError:
            print("⚠️  Ingresa un número válido.")
    
    # Directorio de salida
    output_dir = input("📂 Directorio de salida (default: evaluation_results): ").strip()
    output_dir = output_dir if output_dir else "evaluation_results"
    
    return max_images, num_runs, output_dir

def confirm_execution(models, max_images, num_runs, output_dir, dry_run=False):
    """Confirma la ejecución con el usuario"""
    print("\n" + "="*50)
    print("📋 RESUMEN DE EVALUACIÓN")
    print("="*50)
    
    if dry_run:
        print("🔍 MODO: Verificación solamente (dry run)")
    else:
        print("🚀 MODO: Evaluación completa")
    
    print(f"📦 Modelos: {', '.join(models)}")
    print(f"📊 Imágenes máx: {max_images}")
    print(f"⏱️  Timing runs: {num_runs}")
    print(f"📂 Directorio: {output_dir}")
    
    if not dry_run:
        # Estimar tiempo
        estimated_time = len(models) * 15  # ~15 min por modelo estimado
        print(f"⏰ Tiempo estimado: ~{estimated_time} minutos")
        
        print("\n📝 Para cada modelo se ejecutará:")
        print("   1. Evaluación de métricas (PSNR, SSIM, KimiaNet)")
        print("   2. Benchmark de tiempo en GPU")
        print("   3. Benchmark de tiempo en CPU")
    
    print("="*50)
    
    while True:
        confirm = input("¿Proceder con la evaluación? (s/n): ").strip().lower()
        if confirm in ['s', 'si', 'y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("⚠️  Responde 's' para sí o 'n' para no.")

def build_command(models, max_images, num_runs, output_dir, dry_run=False):
    """Construye el comando a ejecutar"""
    cmd = ["python", "-m", "scripts.auto_evaluate_swinir"]
    
    if models:
        cmd.extend(["--models"] + models)
    
    cmd.extend([
        "--max_images", str(max_images),
        "--num_runs", str(num_runs),
        "--output_dir", output_dir
    ])
    
    if dry_run:
        cmd.append("--dry_run")
    
    return cmd

def execute_evaluation(cmd):
    """Ejecuta la evaluación"""
    print("\n🚀 INICIANDO EVALUACIÓN...")
    print("="*60)
    print(f"📝 Comando: {' '.join(cmd)}")
    print("="*60)
    
    try:
        # Ejecutar el comando
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 EVALUACIÓN COMPLETADA EXITOSAMENTE!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error durante la evaluación (código {e.returncode})")
        return False
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluación interrumpida por el usuario")
        return False
        
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        return False

def show_quick_start():
    """Muestra opciones de inicio rápido"""
    print("\n⚡ INICIO RÁPIDO:")
    print("   1. python scripts/run_evaluation.py")
    print("   2. Evaluar solo modelo 64→128:")
    print("      python -m scripts.auto_evaluate_swinir --models 64to128")
    print("   3. Verificar configuración:")
    print("      python -m scripts.auto_evaluate_swinir --dry_run")
    print()

def main():
    """Función principal"""
    show_banner()
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("scripts"):
        print("❌ Error: Ejecuta este script desde el directorio SwinIR/KAIR/")
        print("   Estructura esperada: SwinIR/KAIR/scripts/")
        return 1
    
    # Verificar que existe el script automatizador
    if not os.path.exists("scripts/auto_evaluate_swinir.py"):
        print("❌ Error: No se encuentra scripts/auto_evaluate_swinir.py")
        print("   Asegúrate de haber copiado todos los scripts.")
        return 1
    
    try:
        while True:
            choice = get_user_selection()
            
            if choice == 4:  # Salir
                print("👋 ¡Hasta luego!")
                break
                
            elif choice == 1:  # Todos los modelos
                models = None  # None significa todos
                max_images, num_runs, output_dir = get_evaluation_settings()
                
                if confirm_execution(['TODOS'], max_images, num_runs, output_dir):
                    cmd = build_command(models, max_images, num_runs, output_dir)
                    execute_evaluation(cmd)
                else:
                    print("❌ Evaluación cancelada")
                
                break
                
            elif choice == 2:  # Modelos específicos
                models = select_specific_models()
                max_images, num_runs, output_dir = get_evaluation_settings()
                
                if confirm_execution(models, max_images, num_runs, output_dir):
                    cmd = build_command(models, max_images, num_runs, output_dir)
                    execute_evaluation(cmd)
                else:
                    print("❌ Evaluación cancelada")
                
                break
                
            elif choice == 3:  # Dry run
                models = None
                
                print("\n🔍 VERIFICANDO CONFIGURACIÓN...")
                cmd = build_command(models, 1000, 50, "evaluation_results", dry_run=True)
                
                if execute_evaluation(cmd):
                    print("\n✅ Configuración verificada. Todo listo para evaluar.")
                    show_quick_start()
                else:
                    print("\n❌ Problemas en la configuración. Revisa los archivos faltantes.")
                
                break
    
    except KeyboardInterrupt:
        print("\n👋 Evaluación interrumpida por el usuario. ¡Hasta luego!")
        return 0
    
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())