import torch

# Cargar el modelo preentrenado
model_path = "superresolution/SwinIR_SR_256to512/models/700000_G.pth"
pretrained_model = torch.load(model_path, map_location='cpu')

print("Claves disponibles en el modelo:")
if isinstance(pretrained_model, dict):
    for key in pretrained_model.keys():
        print(f"  {key}")
    
    # Si hay una clave 'params' o similar, examinar su contenido
    if 'params' in pretrained_model:
        state_dict = pretrained_model['params']
    elif 'state_dict' in pretrained_model:
        state_dict = pretrained_model['state_dict']
    else:
        state_dict = pretrained_model
else:
    state_dict = pretrained_model

print("\nPrimeras 10 claves del state_dict:")
keys = list(state_dict.keys())
for i, key in enumerate(keys[:10]):
    print(f"  {key}: {state_dict[key].shape}")

print(f"\nTotal de parámetros: {len(keys)}")

# Buscar patrones que indiquen la arquitectura
depth_layers = [key for key in keys if 'layers.' in key and '.residual_group.blocks.' in key]
if depth_layers:
    layer_nums = set()
    for key in depth_layers:
        layer_num = int(key.split('.')[1])
        layer_nums.add(layer_num)
    print(f"Capas detectadas: {sorted(layer_nums)}")
    print(f"Número total de capas: {len(layer_nums)}")