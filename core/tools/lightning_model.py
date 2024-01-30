# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import random
import torchvision
from copy import deepcopy
import pytorch_lightning as ptl

from core.libs.FLAME.FLAME import FLAME
from core.data.tools import unnorm_transform
from core.utils.utils import visulize, vis_depth
from core.utils.metric import calc_psnr, calc_ssim
# from core.libs.nerf_camera import NerfCamera, GridNerfRenderer

class LightningOSBase(ptl.LightningModule):
    def __init__(self, config_dict, data_meta_info):
        super().__init__()
        # lightning params
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.automatic_optimization = False
        # model config
        self.config_dict = config_dict
        self.data_meta_info = data_meta_info
        self._best_metric_is_max = True
        self._last_best_model_path = None
        self._best_metric = -1e10 if self._best_metric_is_max else 1e10

    def get_point_cloud(self, poses, expressions, shapes, refine=False):
        if not hasattr(self, 'points_scale'):
            self.points_scale = self.data_meta_info['flame_scale']
            self.flame_model = FLAME(100, 50).to(self._device)
        flame_verts, _, _ = self.flame_model(
            shape_params=shapes, expression_params=expressions, 
            pose_params=poses
        )
        points = flame_verts * self.points_scale
        return points

    def training_step(self, batch, batch_idx):
        # # ema
        # if not hasattr(self, 'avg_param_G'):
        #     self.avg_param_G = copy_params(self.generator)
        optim = self.optimizers()
        # train 
        results = self.forward_train(**batch)
        loss_metrics, psnr = self.calc_metrics(**results)
        losses = sum([v for k, v in loss_metrics.items()])
        if torch.isnan(losses):
            raise Exception('Loss is NAN.......:', loss_metrics)
        optim.zero_grad()
        self.manual_backward(losses)
        optim.step()
        sche = self.lr_schedulers()
        sche.step()
        # # ema
        # for p, avg_p in zip(self.generator.parameters(), self.avg_param_G):
        #     avg_p.mul_(0.999).add_(0.001 * p.data)
        # logger
        self.log_dict(loss_metrics)
        self.log('PSNR', psnr.item(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        results = self.forward_val(**batch)
        # processing data
        f_images = self.merge_multiple_images(batch['f_images'], batch['d_images'].shape[-2:])
        gen_fine = self.resize(results['gen_fine'], batch['d_images'])
        gen_depth = self.resize(vis_depth(results['depth']), batch['d_images'])
        psnr = calc_psnr(results['gen_sr'], batch['d_images']).item()
        ssim = calc_ssim(results['gen_sr'], batch['d_images']).item()
        visulize_images = torch.cat(
            [f_images, batch['d_images'], results['gen_sr'], gen_fine, gen_depth]
        )
        images = torchvision.utils.make_grid(visulize_images, nrow=visulize_images.shape[0], padding=0)
        images = unnorm_transform(images, self.config_dict['DATASET_AUGMENT'])
        self.validation_step_outputs.append({'PSNR': psnr, 'SSIM': ssim, 'Image': images})

    def on_validation_epoch_end(self):
        # # ema
        # backup_para = copy_params(self.generator)
        # load_params(self.generator, self.avg_param_G)
        # build path
        this_step = self.trainer.global_step
        if self.logger is not None and self.trainer.is_global_zero:
            os.makedirs(os.path.join(self.logger.log_dir, 'examples'), exist_ok=True)
            os.makedirs(os.path.join(self.logger.log_dir, 'checkpoints'), exist_ok=True)
            model_root_path = os.path.join(self.logger.log_dir, 'checkpoints')
            validation_path = os.path.join(self.logger.log_dir, 'examples', f'{this_step}.jpg')
            record_path = os.path.join(self.logger.log_dir, 'record.md')
        else:
            validation_path = os.path.join('outputs', 'debug.jpg')
            record_path = os.path.join('outputs', 'debug.md')
        # gather and auto logging
        all_val_out = self.all_gather(self.validation_step_outputs)
        if all_val_out[0]['Image'].dim() > 3:
            images = torch.cat([d['Image'] for d in all_val_out], dim=0)
        else:
            images = torch.stack([d['Image'] for d in all_val_out], dim=0)
        images = torchvision.utils.make_grid(images[:8], nrow=2, padding=0)
        all_psnrs = sum([p['PSNR'] for p in self.validation_step_outputs])/len(self.validation_step_outputs)
        all_ssims = sum([p['SSIM'] for p in self.validation_step_outputs])/len(self.validation_step_outputs)
        merged_psnrs = self.all_gather(all_psnrs).mean().item()
        merged_ssims = self.all_gather(all_ssims).mean().item()
        merged_val_metric = merged_ssims
        visulize(images, save_path=validation_path)
        self.log("VAL_PSNR", all_psnrs, sync_dist=True)
        self.log("VAL_SSIM", all_ssims, sync_dist=True)
        del all_val_out
        # manual logging
        if self.trainer.is_global_zero:
            log_str = 'Epoch: {},\tStep: {},\tPSNR: {:.2f},\tSSIM: {:.4f}.\n'.format(
                self.trainer.current_epoch, this_step, merged_psnrs, merged_ssims
            )
            with open(record_path, 'a') as f:
                f.write(log_str)
            print(log_str)
        if self.logger is not None and self.trainer.is_global_zero:
            model_ckpts_dict = {
                'config_dict': dict(self.config_dict),
                'data_meta_info': self.data_meta_info,
                'step': this_step, 'metric': merged_val_metric, 
                'state_dict': self.state_dict(), 
            }
            best_model_str = 'best_{}_{:.3f}.ckpt'.format(this_step, merged_val_metric)
            save_best = merged_val_metric >= self._best_metric if self._best_metric_is_max else merged_val_metric <= self._best_metric
            if save_best:
                self._best_metric = merged_val_metric
                if self._last_best_model_path is not None:
                    os.remove(self._last_best_model_path)
                self._last_best_model_path = os.path.join(model_root_path, best_model_str)
                torch.save(
                    model_ckpts_dict, os.path.join(model_root_path, best_model_str)
                )
            torch.save(model_ckpts_dict, os.path.join(model_root_path, 'last.ckpt'))
        # clear
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        learning_rate = self.config_dict['TRAIN']['OPTIMIZER']['LR'] # * len(self.config_dict['DEVICES'])
        if self.trainer.is_global_zero:
            print('Learning rate: {}'.format(learning_rate))
        # params
        normal_params, renderer_params, attn_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'style_mlp' in name or 'final_linear' in name:
            # if 'up_renderer' in name:
                # print(name)
                renderer_params.append(param)
            elif 'attn_module' in name:
                # print(name)
                attn_params.append(param)
            else:
                normal_params.append(param)
        # optimizer
        optimizer = torch.optim.Adam([
                {'params': normal_params, 'lr': learning_rate},
                {'params': attn_params, 'lr': learning_rate},
                {'params': renderer_params, 'lr': learning_rate*0.1},
            ], lr=learning_rate, betas=(0.0, 0.99)
        )
        if self.config_dict['TRAIN']['SCHEDULER']['TYPE'] == 'LinearDecay':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=self.config_dict['TRAIN']['SCHEDULER']['DECAY_RATE'], 
                total_iters=self.config_dict['TRAIN']['SCHEDULER']['DECAY_STEP'], verbose=True if self.logger is None else False
            )
        elif self.config_dict['TRAIN']['SCHEDULER']['TYPE'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=self.config_dict['TRAIN']['SCHEDULER']['DECAY_STEP'], 
                gamma=self.config_dict['TRAIN']['SCHEDULER']['DECAY_RATE'],
            )
        else:
            raise NotImplementedError
        return {"optimizer": optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def merge_multiple_images(batch_images, tgt_size, nrow=2):
        if batch_images.shape[1] == 1:
            input_feature_image = batch_images[:, 0]
        else:
            input_feature_image = []
            for images in batch_images:
                if images.shape[0] < 4:
                    images = torch.cat([
                            images, 
                            images.new_zeros(4-images.shape[0], images.shape[1], images.shape[2], images.shape[3])
                        ], dim=0
                    )
                input_feature_image.append(torchvision.utils.make_grid(images, nrow=2, padding=0))
            input_feature_image = torch.stack(input_feature_image, dim=0)
        input_feature_image = torchvision.transforms.functional.resize(input_feature_image, tgt_size, antialias=True)
        return input_feature_image

    @staticmethod
    def resize(frames, tgt_size=(256, 256)):
        if isinstance(tgt_size, torch.Tensor):
            tgt_size = (tgt_size.shape[-2], tgt_size.shape[-1])
        frames = torch.nn.functional.interpolate(
            frames, size=tgt_size,
            mode='bilinear', align_corners=False, antialias=True
        )
        # frames = torchvision.transforms.functional.resize(
        #     frames, tgt_size, antialias=True
        # )
        return frames

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)
