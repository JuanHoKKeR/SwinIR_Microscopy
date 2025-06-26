# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as torch_ssim, ms_ssim


class MetricsCalculator:
    """"Calculadora de métricas para super-resolución"""

    def __init__(self, device='cuda'):
        self.device = device

    def calculate_psnr(self, img1, img2, max_val=1.0):
        """Calcula PSNR entre dos imágenes."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * torch.log10(max_val / torch.sqrt(mse))  # Retorna el PSNR en dB
    
    def calculate_mse(self, img1, img2):
        """Calcula MSE entre dos imágenes."""
        return torch.mean((img1 - img2) ** 2)
    
    def calculate_ssim_torch(self, img1, img2):
        """Calcula SSIM usando PyTorch."""
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)

        return torch_ssim(img1, img2, data_range=1.0)
    
    def calculate_msssim_torch(self, img1, img2):
        """Calcula MS-SSIM usando PyTorch."""
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)

        h, w = img1.shape[2], img1.shape[3]
        if h < 160 or w < 160:
            return torch_ssim(img1, img2, data_range=1.0)

        return ms_ssim(img1, img2, data_range=1.0)
    
    def calculate_ssim_skimage(self, img1, img2):
        """Calcula SSIM usando skimage."""
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        if img1.ndim == 4:
            img1 = img1[0].transpose(1, 2, 0)
        elif img1.ndim == 3:
            img1 = img1.transpose(1, 2, 0)
        
        if img2.ndim == 4:
            img2 = img2[0].transpose(1, 2, 0)
        elif img2.ndim == 3:
            img2 = img2.transpose(1, 2, 0)

        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        if img1.ndim == 3:
            return ssim(img1, img2, multichannel=True, data_range=1.0)
        else:
            return ssim(img1, img2, data_range=1.0)
        
    def calculate_all_metrics(self, pred, target, use_torch_ssim=True):
        """Calcula todas las métricas de una vez."""
        metrics = {}

        # PSNR
        metrics['PSNR'] = self.calculate_psnr(pred, target).item()

        # MSE
        metrics['MSE'] = self.calculate_mse(pred, target).item()

        # SSIM
        if use_torch_ssim:
            metrics['SSIM'] = self.calculate_ssim_torch(pred, target).item()
            metrics['MS-SSIM'] = self.calculate_msssim_torch(pred, target).item()
        else:
            metrics['SSIM'] = self.calculate_ssim_skimage(pred, target)
            metrics['MS-SSIM'] = self.calculate_msssim_torch(pred, target).item()
            
        return metrics
    
    def calculate_metrics_batch(pred_batch, target_batch, device='cuda'):
        """Calcula métricas promedio para un batch de imágenes."""
        calculator = MetricsCalculator(device=device)
        batch_metrics = { 'PSNR': [], 'MSE': [], 'SSIM': [], 'MS-SSIM': [] }

        for i in range(pred_batch.shape[0]):
            pred = pred_batch[i:i+1]
            target = target_batch[i:i+1]
            metrics = calculator.calculate_all_metrics(pred, target)
            for key in batch_metrics:
                batch_metrics[key].append(metrics[key])

        # Calcular promedios
        avg_metrics = {}
        for key, values in batch_metrics.items():
            avg_metrics[key] = np.mean(values)

        return avg_metrics
        