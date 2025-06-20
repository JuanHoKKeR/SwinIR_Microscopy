# Entrenamiento Personalizado de SwinIR con Weights & Biases

Este documento explica cómo entrenar SwinIR con tu propio dataset usando archivos de texto con rutas de imágenes y Weights & Biases para el seguimiento de métricas.

## Características Implementadas

- ✅ **Dataset personalizado**: Soporte para archivos de texto con rutas de imágenes
- ✅ **Weights & Biases**: Integración completa para seguimiento de experimentos
- ✅ **Métricas adicionales**: PSNR, SSIM, MS-SSIM, MSE
- ✅ **Configuración para GPU única**: Optimizado para RTX 4090
- ✅ **Logging detallado**: Métricas de entrenamiento y validación

## Estructura del Dataset

Tu dataset debe tener la siguiente estructura:

```
tu_proyecto/
├── trainset/
│   ├── train_hr.txt    # Rutas de imágenes HR de entrenamiento
│   └── train_lr.txt    # Rutas de imágenes LR de entrenamiento
├── testset/
│   ├── test_hr.txt     # Rutas de imágenes HR de prueba
│   └── test_lr.txt     # Rutas de imágenes LR de prueba
└── ...
```

### Formato de los archivos de texto

Cada archivo `.txt` debe contener una ruta absoluta por línea:

```
/path/to/image1_hr.png
/path/to/image2_hr.png
/path/to/image3_hr.png
...
```

**Importante**: Las líneas en `train_hr.txt` y `train_lr.txt` deben corresponder (misma imagen en diferentes resoluciones).

## Preparación del Dataset

### Opción 1: Usar el script proporcionado

1. Modifica el script `scripts/prepare_custom_dataset.py` con tus rutas:

```python
hr_train_dir = "path/to/your/hr/train/images"
lr_train_dir = "path/to/your/lr/train/images"
hr_test_dir = "path/to/your/hr/test/images"
lr_test_dir = "path/to/your/lr/test/images"
```

2. Ejecuta el script:

```bash
python scripts/prepare_custom_dataset.py
```

### Opción 2: Crear manualmente

Crea los archivos de texto manualmente siguiendo el formato descrito arriba.

## Instalación de Dependencias

```bash
pip install -r requirements.txt
```

## Configuración de Weights & Biases

1. Instala Weights & Biases:
```bash
pip install wandb
```

2. Inicia sesión:
```bash
wandb login
```

3. Sigue las instrucciones para obtener tu API key.

## Configuración del Entrenamiento

### Archivo de configuración

El archivo `options/swinir/train_swinir_sr_custom.json` está configurado para:

- **GPU única**: `"gpu_ids": [0]`
- **Sin distribución**: `"dist": false`
- **Batch size reducido**: `"dataloader_batch_size": 8`
- **Weights & Biases habilitado**: `"use_wandb": true`

### Personalización

Puedes modificar los siguientes parámetros:

```json
{
  "scale": 2,                    // Factor de escala (2, 3, 4, 8)
  "n_channels": 3,               // Canales de color (1 para escala de grises)
  "datasets": {
    "train": {
      "H_size": 96,              // Tamaño del patch HR
      "dataloader_batch_size": 8, // Batch size (ajustar según tu GPU)
      "dataloader_num_workers": 4 // Workers para data loading
    }
  },
  "train": {
    "G_optimizer_lr": 2e-4,      // Learning rate
    "checkpoint_test": 5000,     // Frecuencia de validación
    "checkpoint_save": 5000,     // Frecuencia de guardado
    "use_wandb": true,           // Habilitar W&B
    "wandb_project": "swinir_custom_sr", // Nombre del proyecto
    "wandb_run_name": "swinir_x2_custom" // Nombre del experimento
  }
}
```

## Iniciar el Entrenamiento

```bash
python main_train_psnr.py --opt options/swinir/train_swinir_sr_custom.json
```

## Seguimiento con Weights & Biases

Durante el entrenamiento, podrás ver en tiempo real:

### Métricas de Entrenamiento
- **Loss**: Pérdida del modelo
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MS-SSIM**: Multi-Scale SSIM
- **MSE**: Mean Squared Error
- **Learning Rate**: Tasa de aprendizaje actual

### Métricas de Validación
- **PSNR**: PSNR promedio en el conjunto de prueba
- **SSIM**: SSIM promedio en el conjunto de prueba
- **MS-SSIM**: MS-SSIM promedio en el conjunto de prueba
- **MSE**: MSE promedio en el conjunto de prueba

### Visualizaciones
- Gráficos de métricas a lo largo del tiempo
- Comparación de experimentos
- Configuración del modelo

## Estructura de Salida

El entrenamiento generará:

```
superresolution/swinir_sr_custom_x2/
├── models/           # Modelos guardados
├── images/           # Imágenes de prueba
├── options/          # Configuración guardada
└── train.log         # Log de entrenamiento
```

## Optimización para RTX 4090

### Configuración recomendada:

```json
{
  "gpu_ids": [0],
  "dist": false,
  "datasets": {
    "train": {
      "dataloader_batch_size": 8,    // Ajustar según memoria
      "dataloader_num_workers": 4,   // Workers para data loading
      "H_size": 96                   // Tamaño de patch
    }
  }
}
```

### Ajustes de memoria:

- **Batch size**: Comienza con 8, aumenta si hay memoria disponible
- **Patch size**: 96 es un buen balance entre calidad y memoria
- **Workers**: 4-8 para data loading eficiente

## Solución de Problemas

### Error de memoria GPU
- Reduce `dataloader_batch_size`
- Reduce `H_size`
- Reduce `dataloader_num_workers`

### Error de archivos no encontrados
- Verifica que las rutas en los archivos `.txt` sean correctas
- Asegúrate de que las imágenes existan
- Usa rutas absolutas

### Error de Weights & Biases
- Verifica que estés logueado: `wandb login`
- Comprueba tu conexión a internet
- Deshabilita W&B temporalmente: `"use_wandb": false`

## Comandos Útiles

### Verificar instalación
```bash
python -c "import wandb; print('W&B instalado correctamente')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Verificar GPU
```bash
python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Limpiar caché de W&B
```bash
wandb sync --clean
```

## Notas Importantes

1. **Orden de las imágenes**: Las líneas en los archivos HR y LR deben corresponder
2. **Formato de imágenes**: PNG, JPG, BMP son soportados
3. **Memoria**: Monitorea el uso de memoria GPU durante el entrenamiento
4. **Backup**: Guarda copias de tus archivos de configuración
5. **Experimentos**: Usa diferentes nombres en `wandb_run_name` para cada experimento

## Soporte

Si encuentras problemas:

1. Revisa los logs en `superresolution/swinir_sr_custom_x2/train.log`
2. Verifica la configuración en el archivo JSON
3. Comprueba que todas las dependencias estén instaladas
4. Revisa el dashboard de Weights & Biases para errores 