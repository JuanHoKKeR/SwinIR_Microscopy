import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import cv2

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils.utils_metrics import MetricsCalculator

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import wandb


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        # ------------------------------------
        # Log modelo de forma limpia
        # ------------------------------------
        def log_model_summary(model):
            """Log resumen limpio del modelo"""
            # Obtener información básica
            model_name = model.netG.__class__.__name__
            total_params = sum(p.numel() for p in model.netG.parameters())
            trainable_params = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)
            
            # Calcular tamaño en MB
            param_size_mb = sum(p.numel() * p.element_size() for p in model.netG.parameters()) / 1024 / 1024
            
            # Log información esencial
            logger.info('🏗️  Model Summary:')
            logger.info(f'   📝 Architecture: {model_name}')
            logger.info(f'   🔢 Total Parameters: {total_params:,}')
            logger.info(f'   🎯 Trainable Parameters: {trainable_params:,}')
            logger.info(f'   💾 Model Size: {param_size_mb:.2f} MB')
            logger.info(f'   ⚙️  Scale Factor: {opt["scale"]}x')
            logger.info(f'   🖼️  Input Channels: {opt["n_channels"]}')
            
            # Información de configuración específica de SwinIR
            if hasattr(model.netG, 'embed_dim'):
                logger.info(f'   🧠 Embed Dimension: {model.netG.embed_dim}')
            if hasattr(model.netG, 'depths'):
                logger.info(f'   📚 Depths: {model.netG.depths}')
            if hasattr(model.netG, 'num_heads'):
                logger.info(f'   👁️  Attention Heads: {model.netG.num_heads}')
            
            logger.info('') # Línea en blanco
        
        # Usar el logging personalizado
        log_model_summary(model)

    # ----------------------------------------
    # initialize metrics calculator for validation
    # ----------------------------------------
    metrics_calculator = MetricsCalculator(device=model.device)

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(500000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_ms_ssim = 0.0
                avg_mse = 0.0
                idx = 0

                # Lista para almacenar imagenes para wandb
                wandb_images = []

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    # Solo guardar imágenes de las primeras 5
                    if idx <= 5:
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    L_img = util.tensor2uint(visuals['L'])

                    # Solo guardar imagen física de las primeras 5
                    if idx <= 5:
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate metrics
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    
                    # Convert to tensor for metrics calculation
                    E_tensor = util.uint2tensor4(E_img).to(model.device)
                    H_tensor = util.uint2tensor4(H_img).to(model.device)
                    
                    # Calculate additional metrics
                    test_metrics = metrics_calculator.calculate_all_metrics(E_tensor, H_tensor)
                    current_ssim = test_metrics['SSIM']
                    current_ms_ssim = test_metrics['MS-SSIM']
                    current_mse = test_metrics['MSE']

                    display_name = img_name[:15] + "..." if len(img_name) > 15 else img_name

                    if idx % 5000 == 0 or idx <= 5:
                        logger.info('{:->4d}--> {:>15s} | PSNR: {:<4.2f}dB | SSIM: {:<4.4f} | MS-SSIM: {:<4.4f} | MSE: {:<4.4f}'.format(
                            idx, display_name, current_psnr, current_ssim, current_ms_ssim, current_mse))

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_ms_ssim += current_ms_ssim
                    avg_mse += current_mse

                    # Agregar imágenes para wandb (Solo las primeras 3 para no sobrecargar)
                    if idx <= 3:  # Cambiado de 1 a 3 para tener más imágenes
                        # Convertir imágenes a formato RGB para wandb
                        L_img_rgb = cv2.cvtColor(L_img, cv2.COLOR_BGR2RGB)
                        E_img_rgb = cv2.cvtColor(E_img, cv2.COLOR_BGR2RGB)
                        H_img_rgb = cv2.cvtColor(H_img, cv2.COLOR_BGR2RGB)

                        # Crear visualización comparativa
                        wandb_images.extend([
                            wandb.Image(
                                L_img_rgb,
                                caption=f"Input_{idx}: (LR) - {L_img.shape[1]}x{L_img.shape[0]}"
                            ),
                            wandb.Image(
                                E_img_rgb,
                                caption=f"Predicted_{idx}: (SR) - PSNR: {current_psnr:.2f}dB - {E_img.shape[1]}x{E_img.shape[0]}"
                            ),
                            wandb.Image(
                                H_img_rgb,
                                caption=f"GT_{idx}: (HR) - {H_img.shape[1]}x{H_img.shape[0]}"
                            )
                        ])

                        # Log individual images metrics
                        if hasattr(model, 'use_wandb') and model.use_wandb:
                            wandb.log({
                                f'val/{display_name}_psnr': current_psnr,
                                f'val/{display_name}_ssim': current_ssim,
                                f'val/{display_name}_ms_ssim': current_ms_ssim,
                                f'val/{display_name}_mse': current_mse,
                                'val/step': current_step
                            })

                # Al final del loop de testing
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_ms_ssim = avg_ms_ssim / idx
                avg_mse = avg_mse / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR: {:<.2f}dB, SSIM: {:<.4f}, MS-SSIM: {:<.4f}, MSE: {:<.4f}\n'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_ms_ssim, avg_mse))

                # Log validation metrics to wandb
                if hasattr(model, 'use_wandb') and model.use_wandb:
                    wandb.log({
                        'val/images': wandb_images,
                        'val/psnr': avg_psnr,
                        'val/ssim': avg_ssim,
                        'val/ms_ssim': avg_ms_ssim,
                        'val/mse': avg_mse,
                        'val/step': current_step
                    })

if __name__ == '__main__':
    main()
