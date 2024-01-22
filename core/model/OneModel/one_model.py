# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import random
import inspect
import torchvision

from core.model.modules.style_unet import StyleUNet
from core.model.modules.tp_transformer import MTAttention
from core.model.modules.points_decoder import PointsDecoder
from core.libs.nerf_camera import CubicNeRFCamera
from core.libs.tri_planes.tp_enc import sample_from_planes
from core.utils.registry import MODEL_REGISTRY
from core.utils.perceptual import PerceptualLoss
from core.tools.lightning_model import LightningOSBase

@MODEL_REGISTRY.register()
class OneModel(LightningOSBase):
    def __init__(self, config_dict, data_meta_info):
        super().__init__(config_dict, data_meta_info)
        # model config
        self.raw_cam_size = 128
        self.config_dict = config_dict
        self.data_meta_info = data_meta_info
        # standard points
        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        _head_points_path = os.path.join(_abs_script_path, 'mean_head.pth')
        standard_points = torch.load(_head_points_path, map_location='cpu')[None]
        standard_points *= data_meta_info['flame_scale']
        # random.seed(42)
        # self.points_id = torch.tensor(random.choices(list(range(standard_points.shape[-2])), k=2000)).long()
        # self.standard_points = torch.nn.Parameter(standard_points[:, self.points_id], requires_grad=False)
        self.standard_points = torch.nn.Parameter(standard_points, requires_grad=False)
        print('Verts_scale: {}, verts_number: {}'.format(data_meta_info['flame_scale'], self.standard_points.shape[-2]))

        # trainable params
        self.points_tplane = torch.nn.Parameter(torch.randn(1, 3, 32, 256, 256), requires_grad=True)
        self.style_tplane = StyleUNet(in_size=512, out_size=256, in_dim=3, out_dim=96, activation=False)
        self.nerf_mlp = PointsDecoder(in_dim=32, out_dim=32, points_number=5023, points_k=8)
        self.up_renderer = StyleUNet(in_size=512, out_size=512, in_dim=32, out_dim=3)
        # attn module
        self.query_style_tplane = torch.nn.Parameter(torch.randn(1, 1, 3, 32, 256, 256), requires_grad=True)
        self.attn_module = MTAttention(dim=32, qkv_bias=True)
        # from core.libs.GFPGAN import GFPGANv1Clean
        # self.up_renderer = GFPGANv1Clean(out_size=512, pretrained=True)
        # loss
        self.percep_loss = PerceptualLoss()
        self.nerf_camera = CubicNeRFCamera(data_meta_info['focal_length'], [128, 128], 32)

    def get_points_feature(self, batch_size):
        point_tplanes = self.points_tplane.expand(batch_size, -1, -1, -1, -1)
        points_features = sample_from_planes(
            self.standard_points.expand(batch_size, -1, -1), point_tplanes, return_type='mean'
        )
        return points_features

    def forward_train(self, f_images, d_images, d_points, d_transforms, d_bbox, **kwargs):
        if hasattr(self, 'points_id'):
            d_points = d_points[:, self.points_id]
        # process input_images
        f_images = f_images.flip(dims=(-2,))
        batch_size, v, c, h, w = f_images.shape
        f_images = f_images.reshape(-1, c, h, w)
        # render
        # set camera pose
        self.nerf_camera.set_position(transform_matrix=d_transforms)
        # tri-planes
        tex_tplanes = self.style_tplane(f_images)
        tex_tplanes = tex_tplanes.reshape(
            batch_size, v, 3, -1, tex_tplanes.shape[-2], tex_tplanes.shape[-1]
        )
        # random merge
        if random.random() < 0.3:
            tex_tplanes = tex_tplanes[:, :1]
        if tex_tplanes.shape[1] > 1:
            tex_tplanes = self.attn_module(
                self.query_style_tplane.expand(batch_size, -1, -1, -1, -1, -1),
                tex_tplanes
            )
        else:
            tex_tplanes = tex_tplanes[:, 0]
        # render
        points_features = self.get_points_feature(batch_size)
        gen_coarse, gen_fine, params_dict = self.nerf_camera.render(
            nerf_query_fn=self.nerf_mlp, noise=True, background=0.0, 
            # nerf_fn params
            points_position=d_points, points_features=points_features, tex_tplanes=tex_tplanes
        )
        gen_sr = self.up_renderer(gen_fine)
        gen_coarse, gen_fine = gen_coarse[:, :3], gen_fine[:, :3]
        # import ipdb; ipdb.set_trace()
        # gather
        results = {
            'images':d_images, 'bbox':d_bbox, 
            'gen_coarse': gen_coarse, 'gen_fine': gen_fine, 'gen_sr': gen_sr,
            'densities': params_dict['coarse']['densities'],
        }
        return results

    @torch.no_grad()
    def forward_val(self, f_images, d_points, d_transforms, **kwargs):
        if hasattr(self, 'points_id'):
            points = points[:, self.points_id]
        # merge multiple input images
        f_images = f_images.flip(dims=(-2,))
        batch_size, v, c, h, w = f_images.shape
        f_images = f_images.reshape(-1, c, h, w)
        # render
        # set camera pose
        self.nerf_camera.set_position(transform_matrix=d_transforms)
        # tri-planes
        tex_tplanes = self.style_tplane(f_images)
        tex_tplanes = tex_tplanes.reshape(
            batch_size, v, 3, -1, tex_tplanes.shape[-2], tex_tplanes.shape[-1]
        )
        if tex_tplanes.shape[1] > 1:
            tex_tplanes = self.attn_module(
                self.query_style_tplane.expand(batch_size, -1, -1, -1, -1, -1),
                tex_tplanes
            )
        else:
            tex_tplanes = tex_tplanes[:, 0]
        # render
        points_features = self.get_points_feature(batch_size)
        _, gen_fine, params_dict = self.nerf_camera.render(
            nerf_query_fn=self.nerf_mlp, noise=False, background=0.0, 
            # nerf_fn params
            points_position=d_points, points_features=points_features, tex_tplanes=tex_tplanes
        )
        gen_sr = self.up_renderer(gen_fine)
        gen_fine = gen_fine[:, :3]
        # gather
        results = {
            'f_images':f_images, 'gen_fine': gen_fine, 'gen_sr': gen_sr, 'depth': params_dict['depth']
        }
        return results

    @torch.no_grad()
    def forward_attention(
            self, f_images, infos, **kwargs
        ):
        batch_size, v, c, h, w = f_images.shape
        # merge multiple input images
        f_images = f_images.flip(dims=(-2,))
        f_images = f_images.reshape(-1, c, h, w)
        # processing data
        tex_tplanes = self.style_tplane(f_images)
        tex_tplanes = tex_tplanes.reshape(
            batch_size, v, 3, -1, tex_tplanes.shape[-2], tex_tplanes.shape[-1]
        )
        attn = self.attn_module(
            self.query_style_tplane.expand(batch_size, -1, -1, -1, -1, -1),
            tex_tplanes, only_attn=True
        )
        return attn
        
    @torch.no_grad()
    def forward_inference(
            self, f_images, d_points, d_transforms, infos, **kwargs
        ):
        if hasattr(self, 'points_id'):
            points = points[:, self.points_id]
        batch_size, v, c, h, w = f_images.shape
        if not hasattr(self, 'texture_planes'):
            # merge multiple input images
            f_images = f_images.flip(dims=(-2,))
            f_images = f_images.reshape(-1, c, h, w)
            # processing data
            tex_tplanes = self.style_tplane(f_images)
            tex_tplanes = tex_tplanes.reshape(
                batch_size, v, 3, -1, tex_tplanes.shape[-2], tex_tplanes.shape[-1]
            )
            self.texture_planes = self.attn_module(
                self.query_style_tplane.expand(batch_size, -1, -1, -1, -1, -1),
                tex_tplanes
            )
            self.points_features = self.get_points_feature(batch_size)
            print('Tri-Planes built.')
        # render
        # set camera pose
        self.nerf_camera.set_position(transform_matrix=d_transforms)
        # render
        _, gen_fine, params_dict = self.nerf_camera.render(
            nerf_query_fn=self.nerf_mlp, noise=False, background=0.0, 
            # nerf_fn params
            points_position=d_points, points_features=self.points_features, tex_tplanes=self.texture_planes
        )
        gen_sr = self.up_renderer(gen_fine)
        gen_fine = gen_fine[:, :3]
        # gather
        results = {
            'f_images':f_images, 'gen_fine': gen_fine, 'gen_sr': gen_sr, 'depth': params_dict['depth']
        }
        return results

    def calc_metrics(self, images, gen_coarse, gen_fine, gen_sr, densities, bbox, **kwargs):
        loss_fn = torch.nn.functional.l1_loss
        gt_small, gt_large = self.resize(images, gen_coarse), images

        pec_loss_0 = self.percep_loss(gen_coarse, gt_small)
        pec_loss_1 = self.percep_loss(gen_fine, gt_small)
        pec_loss_2 = self.percep_loss(gen_sr, gt_large)
        img_loss_0 = loss_fn(gen_coarse, gt_small)
        img_loss_1 = loss_fn(gen_fine, gt_small)
        img_loss_2 = loss_fn(gen_sr, gt_large)
        box_loss_0 = self.calc_box_loss(gen_coarse, gt_small, bbox, loss_fn)
        box_loss_1 = self.calc_box_loss(gen_fine, gt_small, bbox, loss_fn)
        box_loss_2 = self.calc_box_loss(gen_sr, gt_large, bbox, loss_fn)
        pec_loss = (pec_loss_0 + pec_loss_1 + pec_loss_2) / 3 * 1e-2
        img_loss = (img_loss_0 + img_loss_1 + img_loss_2)
        box_loss = (box_loss_0 + box_loss_1 + box_loss_2)
        if densities is not None:
            densities_loss = torch.norm(densities, p=2) * 1e-5
        else:
            densities_loss = torch.tensor(0.0).cuda()
        metrics = {
            'percep_loss': pec_loss, 'img_loss': img_loss, 
            'box_loss': box_loss, 'density_loss': densities_loss, 
        }
        # print(metrics)
        psnr = -10.0 * torch.log10(torch.nn.functional.mse_loss(gen_sr, images).detach())
        return metrics, psnr

    def calc_box_loss(self, image, gt_image, bbox, loss_fn, resize_size=256):
        bbox = bbox.clamp(min=0, max=1)
        bbox = (bbox * image.shape[-1]).long()
        pred_croped, gt_croped = [], []
        for b_idx, box in enumerate(bbox):
            gt_croped.append(
                self.resize(
                    gt_image[b_idx:b_idx+1, :, box[1]:box[3], box[0]:box[2]], 
                    (resize_size, resize_size)
                )
            )
            pred_croped.append(
                self.resize(
                    image[b_idx:b_idx+1, :, box[1]:box[3], box[0]:box[2]], 
                    (resize_size, resize_size)
                )
            )
        pred_croped = torch.cat(pred_croped, dim=0)
        gt_croped = torch.cat(gt_croped, dim=0)
        box_1_loss = self.percep_loss(pred_croped, gt_croped) * 1e-2
        box_2_loss = loss_fn(pred_croped, gt_croped)
        box_loss = (box_1_loss + box_2_loss) / 2
        return box_loss
