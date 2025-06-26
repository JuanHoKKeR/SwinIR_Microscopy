import os
import torch
import wandb
from data.select_dataset import define_Dataset
from models.select_model import define_Model
import json

def verify_setup(config_path):
    """Verifica que todo esté configurado correctamente"""
    
    print("🔍 Verificando configuración...")
    
    # 1. Verificar archivo de configuración
    if not os.path.exists(config_path):
        print(f"❌ Archivo de configuración no encontrado: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        opt = json.load(f)
    
    # 2. Verificar archivos de dataset
    train_hr = opt['datasets']['train']['dataroot_H']
    train_lr = opt['datasets']['train']['dataroot_L']
    test_hr = opt['datasets']['test']['dataroot_H']
    test_lr = opt['datasets']['test']['dataroot_L']
    
    for file_path in [train_hr, train_lr, test_hr, test_lr]:
        if not os.path.exists(file_path):
            print(f"❌ Archivo meta_info no encontrado: {file_path}")
            return False
        
        # Verificar que contenga rutas
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(f"❌ Archivo meta_info vacío: {file_path}")
                return False
            print(f"✅ {file_path}: {len(lines)} imágenes")
    
    # 3. Verificar dependencias
    try:
        import pytorch_msssim
        print("✅ pytorch-msssim disponible")
    except ImportError:
        print("⚠️  pytorch-msssim no disponible. Instalar con: pip install pytorch-msssim")
    
    try:
        import wandb
        print("✅ wandb disponible")
    except ImportError:
        print("❌ wandb no disponible. Instalar con: pip install wandb")
        return False
    
    # 4. Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  GPU no disponible, entrenamiento será muy lento")
    
    print("✅ Verificación completada. Listo para entrenar!")
    return True

if __name__ == "__main__":
    verify_setup("options/swinir/train_swinir_sr_custom.json")