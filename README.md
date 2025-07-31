# SwinIR for Histopathology Super-Resolution

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Una implementación especializada de **SwinIR** (Image Restoration using Swin Transformer) optimizada para imágenes de histopatología de cáncer de mama. Este proyecto forma parte de un **Trabajo de Grado** enfocado en super-resolución para aplicaciones médicas mediante arquitecturas Transformer.

## 🎯 **Objetivo del Proyecto**

Adaptar y evaluar SwinIR, una arquitectura state-of-the-art basada en Vision Transformers, para super-resolución de imágenes de microscopia histopatológica, aprovechando su capacidad superior de capturar dependencias a largo alcance y preservar detalles finos.

## ✨ **Características Principales**

- **🔬 Especializado en Histopatología**: Optimizado para imágenes de cáncer de mama
- **🏗️ Arquitectura Transformer**: Utiliza Swin Transformer blocks para capturar patrones complejos
- **📊 Excelente Calidad**: Mejores métricas PSNR y SSIM comparado con CNNs tradicionales
- **⚡ Eficiencia de Memory**: Uso optimizado de VRAM mediante configuraciones adaptables
- **📈 Sistema de Evaluación Comprehensive**: Métricas especializadas para imágenes médicas

## 🔄 **Diferencias con el Proyecto Original**

Este repositorio está basado en [KAIR](https://github.com/cszn/KAIR) de Kai Zhang pero incluye adaptaciones específicas:

| Aspecto | KAIR Original | Esta Implementación |
|---------|---------------|-------------------|
| **Dominio** | Imágenes naturales | Histopatología específica |
| **Configuración** | Configuraciones fijas | Adaptables por limitaciones de memoria |
| **Evaluación** | Métricas básicas | Sistema comprehensive con evaluación médica |
| **Dataset** | DIV2K, Flickr2K | Dataset histopatológico especializado |
| **Escalas** | Multi-escala completa | Optimizado para ×2 (limitaciones de hardware) |

## 🚀 **Inicio Rápido**

### Prerequisitos
- GPU NVIDIA con drivers compatibles (recomendado)
- NVIDIA Container Toolkit para soporte GPU
- ~8GB VRAM mínimo para entrenamiento

### 1. Clonar el Repositorio
```bash
git clone https://github.com/JuanHoKKeR/SwinIR_Microscopy.git
cd SwinIR_Microscopy
```

### 2. Preparar el Dataset
Organiza tu dataset con la siguiente estructura:
```
trainsets/
├── trainH/              # Imágenes de alta resolución (ground truth)
└── trainL/              # Imágenes de baja resolución (input)
testsets/
├── Set5/
│   ├── HR/             # Ground truth para evaluación
│   └── LR_bicubic/     # Imágenes de entrada para testing
└── histopatologia/     # Dataset personalizado
    ├── HR/
    └── LR_bicubic/
```

### 3. Configurar el Entrenamiento
Edita el archivo de configuración JSON en `options/`:

```json
{
  "task": "sr",
  "model": "plain", 
  "scale": 2,
  "gpu_ids": [0],
  
  "datasets": {
    "train": {
      "name": "histopatologia_train",
      "type": "DatasetSR",
      "dataroot_H": "trainsets/trainH",
      "dataroot_L": "trainsets/trainL",
      "H_size": 256,
      "dataloader_shuffle": true,
      "dataloader_num_workers": 8,
      "dataloader_batch_size": 4
    },
    "test": {
      "name": "histopatologia_test", 
      "type": "DatasetSR",
      "dataroot_H": "testsets/histopatologia/HR",
      "dataroot_L": "testsets/histopatologia/LR_bicubic"
    }
  },

  "netG": {
    "net_type": "swinir",
    "upscale": 2,
    "in_chans": 3,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv"
  }
}
```

### 4. Ejecutar

#### Entrenamiento
```bash
# Ejecutar entrenamiento
python main_train_psnr.py \
    --opt options\swinir\train_swinir_sr_custom.json
```

#### Inferencia/Testing
```bash
# Evaluar modelo entrenado
python main_test_swinir.py \
    --opt options\swinir\test_swinir_sr_custom.json
```

## 📁 **Estructura del Proyecto**

```
SwinIR_Microscopy/
├── options/                    # Archivos de configuración JSON
│   ├── train_swinir_sr_*.json  # Configuraciones de entrenamiento
│   └── test_swinir_sr_*.json   # Configuraciones de testing
├── models/                     # Arquitecturas y modelos
│   ├── network_swinir.py       # Implementación SwinIR
│   ├── model_plain.py          # Wrapper del modelo
│   └── select_network.py       # Selector de redes
├── data/                       # Manejo de datasets
│   ├── dataset_sr.py           # Dataset para super-resolución
│   └── select_dataset.py       # Selector de datasets
├── utils/                      # Utilidades
│   ├── utils_image.py          # Procesamiento de imágenes
│   ├── utils_logger.py         # Sistema de logging
│   └── utils_option.py         # Parseo de configuraciones
├── main_train_psnr.py          # Script principal de entrenamiento
├── main_test_swinir.py         # Script principal de testing
├── experiments/                # Resultados y modelos entrenados
│   └── [experiment_name]/
│       ├── models/             # Checkpoints del modelo
│       ├── images/             # Imágenes de validación
│       └── log/                # Logs de entrenamiento
└── testsets/                   # Datasets de evaluación
```

## 🧠 **Arquitectura SwinIR**

### Componentes Principales

#### **1. Swin Transformer Blocks**
```python
# En models/network_swinir.py
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4.):
        # Window-based Multi-head Self Attention (W-MSA)
        # Shifted Window Multi-head Self Attention (SW-MSA)
        # Feed-forward Network (FFN)
```

#### **2. Arquitectura Completa**
- **Shallow Feature Extraction**: Extracción inicial de características
- **Deep Feature Extraction**: Stack de Residual Swin Transformer Blocks (RSTB)
- **High-quality Image Reconstruction**: Upsampling y reconstrucción final

### Configuraciones Disponibles

#### **Configuración Estándar (Recomendada)**
```json
"netG": {
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "window_size": 8,
    "img_size": 64
}
```

#### **Configuración Optimizada para Memoria**
```json
"netG": {
    "depths": [6, 6, 6, 6],          # Reducir layers externos
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6],       # Correspondiente a depths
    "window_size": 8,
    "img_size": 48                   # Reducir tamaño de entrada
}
```

#### **Configuración Mínima (Hardware Limitado)**
```json
"netG": {
    "depths": [4, 4, 4],
    "embed_dim": 60,                 # Reducir dimensión embedding
    "num_heads": [4, 4, 4],
    "window_size": 8,
    "img_size": 32
}
```

## 🚀 **Scripts Principales**

### 1. **Entrenamiento de Modelos**

#### `main_train_psnr.py`
Script principal para entrenar modelos SwinIR optimizando PSNR.

```bash
python main_train_psnr.py --opt options/train_swinir_sr_histopatologia.json
```

**Parámetros de configuración importantes:**
```json
{
  "train": {
    "G_lossfn_type": "l1",           # l1, l2, ssim
    "G_lossfn_weight": 1.0,
    "G_optimizer_type": "adam",
    "G_optimizer_lr": 2e-4,
    "G_optimizer_wd": 0,
    "G_scheduler_type": "MultiStepLR",
    "G_scheduler_milestones": [250000, 400000, 450000, 475000],
    "G_scheduler_gamma": 0.5,
    "checkpoint_save": 5000,
    "checkpoint_print": 100
  }
}
```

### 2. **Testing y Evaluación**

#### `main_test_swinir.py`
Evaluación de modelos entrenados con métricas comprehensive.

```bash
python main_test_swinir.py --opt options/test_swinir_sr_histopatologia.json
```

**Configuración de testing:**
```json
{
  "path": {
    "pretrained_netG": "experiments/swinir_histopatologia/models/G.pth",
    "root": "testsets",
    "results_root": "results"
  },
  "datasets": {
    "test": {
      "name": "histopatologia_test",
      "type": "DatasetSR", 
      "dataroot_H": "testsets/histopatologia/HR",
      "dataroot_L": "testsets/histopatologia/LR_bicubic"
    }
  }
}
```

### 3. **Evaluación con Métricas Especializadas**

#### `evaluate_model_comprehensive.py`
Script personalizado para evaluación detallada con métricas médicas.

```bash
python evaluate_model_comprehensive.py \
    --model_path experiments/swinir_histopatologia/models/G.pth \
    --test_dir testsets/histopatologia \
    --results_dir evaluation_results \
    --max_images 1000
```

**Métricas calculadas:**
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MS-SSIM**: Multi-Scale SSIM
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Métricas perceptuales**: Basadas en características VGG

### 4. **Análisis Visual y Comparativo**

#### `visual_comparison_analyzer.py`
Genera análisis visual comprehensive entre diferentes resoluciones.

```bash
python visual_comparison_analyzer.py \
    --lr_image path/to/lr_image.png \
    --hr_image path/to/hr_image.png \
    --model_path experiments/swinir_histopatologia/models/G.pth \
    --output_dir visual_analysis
```

**Genera:**
- Comparación lado a lado (LR → Predicción → GT)
- Mapas de diferencia absoluta
- Análisis de preservación de estructuras histológicas
- Estadísticas detalladas por región

## 📊 **Resultados y Rendimiento**

### **Modelos Entrenados y Evaluados**

| Modelo | Input → Output | PSNR (dB) | SSIM | MS-SSIM | Tiempo GPU (ms) | VRAM (MB) |
|--------|----------------|-----------|------|---------|-----------------|-----------|
| 64→128 | 64×64 → 128×128 | 23.19±1.8 | 0.741±0.05 | 0.954±0.02 | 89.81 | 0.5 |
| 128→256 | 128×128 → 256×256 | 25.42±2.1 | 0.802±0.04 | 0.975±0.01 | 131.87 | 1.7 |
| 256→512 | 256×256 → 512×512 | 28.56±1.9 | 0.847±0.03 | 0.982±0.01 | 267.34 | 6.4 |
| **512→1024** | **512×512 → 1024×1024** | **32.89±1.6** | **0.912±0.02** | **0.960±0.02** | **501.25** | **24.8** |

*Evaluado en ~1000 muestras aleatorias del dataset de histopatología*

### **Comparación con Otras Arquitecturas**

| Métrica | SwinIR 512→1024 | ESRGAN 512→1024 | EDSR 512→1024 |
|---------|-----------------|-----------------|---------------|
| **PSNR (dB)** | **32.89** ↑ | 30.51 | 32.06 |
| **SSIM** | **0.912** ↑ | 0.886 | 0.888 |
| **MS-SSIM** | **0.960** ↑ | 0.972 | 0.968 |
| **Tiempo GPU (ms)** | 501.25 | **61.10** ↑ | 384.80 |
| **VRAM (MB)** | **24.8** ↑ | 171.5 | 24.0 |

**Ventajas de SwinIR:**
✅ **Mejor calidad**: PSNR y SSIM superiores  
✅ **Eficiencia de memoria**: Menor uso de VRAM que ESRGAN  
✅ **Preservación de estructuras**: Mejor captura de patrones histológicos  
✅ **Estabilidad**: Entrenamiento más estable sin modo collapse  



### **Evaluación Visual - Preservación de Estructuras Histológicas**

```
Análisis Histopatológico:
├── Preservación de bordes nucleares: ✅ Excelente
├── Definición de interfaces tisulares: ✅ Superior 
├── Textura de estroma: ✅ Bien preservada
├── Arquitectura glandular: ✅ Fiel al original
└── Ausencia de artefactos: ✅ Mínimos
```

## ⚙️ **Configuración Detallada**

### **Configuraciones por Resolución**

#### **64→128 (Básica)**
```json
{
  "netG": {
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "img_size": 64,
    "window_size": 8
  },
  "datasets": {
    "train": {
      "H_size": 128,
      "dataloader_batch_size": 8
    }
  }
}
```

#### **256→512 (Intermedia)**
```json
{
  "netG": {
    "depths": [6, 6, 6, 6],           # Reducir para optimizar memoria
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6],
    "img_size": 256,
    "window_size": 8
  },
  "datasets": {
    "train": {
      "H_size": 512,
      "dataloader_batch_size": 2       # Reducir batch size
    }
  }
}
```

#### **512→1024 (Alta Resolución)**
```json
{
  "netG": {
    "depths": [4, 4, 4],              # Configuración mínima
    "embed_dim": 60,                  # Reducir embedding dimension
    "num_heads": [4, 4, 4],
    "img_size": 512,
    "window_size": 8
  },
  "datasets": {
    "train": {
      "H_size": 1024,
      "dataloader_batch_size": 1       # Batch size mínimo
    }
  }
}
```

### **Optimizaciones de Entrenamiento**

#### **Learning Rate Scheduling**
```json
{
  "train": {
    "G_optimizer_lr": 2e-4,
    "G_scheduler_type": "CosineAnnealingRestartLR", 
    "G_scheduler_periods": [250000, 250000, 250000, 250000],
    "G_scheduler_restart_weights": [1, 0.5, 0.5, 0.5],
    "G_scheduler_eta_min": 1e-7
  }
}
```

#### **Data Augmentation**
```json
{
  "datasets": {
    "train": {
      "use_hflip": true,              # Horizontal flip
      "use_rot": true,                # 90° rotations
      "use_shuffle": true,
      "dataloader_shuffle": true,
      "H_size": 256,
      "use_crop": true
    }
  }
}
```

## 🐛 **Solución de Problemas Comunes**

### **1. Problemas de Memoria GPU**

**Error**: `CUDA out of memory`

**Soluciones:**
```json
// Reducir batch size
"dataloader_batch_size": 1

// Usar configuración mínima
"depths": [4, 4, 4],
"embed_dim": 60,

// Reducir tamaño de imagen
"img_size": 32,
"H_size": 128
```

### **2. Convergencia Lenta**

**Síntomas**: Loss no disminuye después de muchas iteraciones

**Soluciones:**
```json
// Ajustar learning rate
"G_optimizer_lr": 1e-4,  // Reducir si oscila
"G_optimizer_lr": 5e-4,  // Aumentar si muy lento

// Cambiar scheduler
"G_scheduler_type": "StepLR",
"G_scheduler_step_size": 100000,
"G_scheduler_gamma": 0.8
```

### **3. Artefactos en Resultados**

**Problema**: Imágenes con patrones extraños o bloque

**Causas y soluciones:**
```json
// Window size inadecuado
"window_size": 8,  // Probar 4 o 16

// Modelo undertrained
// Entrenar por más iteraciones: 500000+

// Dataset insuficiente  
// Usar data augmentation agresivo
```

## 📈 **Monitoreo**

### **Checkpointing Automático**
```json
{
  "train": {
    "checkpoint_save": 5000,         # Guardar cada 5k iteraciones
    "checkpoint_test": 5000,         # Validar cada 5k iteraciones
    "checkpoint_print": 100          # Log cada 100 iteraciones
  }
}
```


## 🤝 **Contribución**

Este proyecto es parte de un Trabajo de Grado enfocado en super-resolución médica. Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 **Reconocimientos**

- **Proyecto Original**: [KAIR](https://github.com/cszn/KAIR) por Kai Zhang
- **SwinIR Paper**: Liang, Jingyun, et al. "SwinIR: Image restoration using swin transformer." ICCVW 2021.
- **Swin Transformer**: Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." ICCV 2021.
- **Framework PyTorch**: Por el excelente framework de deep learning

## 📞 **Contacto**

- **Autor**: Juan David Cruz Useche
- **Proyecto**: Trabajo de Grado - Super-Resolución para Histopatología
- **GitHub**: [@JuanHoKKeR](https://github.com/JuanHoKKeR)
- **Repositorio**: [SwinIR_Microscopy](https://github.com/JuanHoKKeR/SwinIR_Microscopy)

## 📚 **Referencias**

```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  pages={1833--1844},
  year={2021}
}

@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}

@misc{zhang2021kair,
  title={KAIR: An image restoration toolbox},
  author={Zhang, Kai},
  howpublished={\url{https://github.com/cszn/KAIR}},
  year={2021}
}
```

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**