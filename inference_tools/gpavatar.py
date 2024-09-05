# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import random
import inspect
import torchvision

from .libs import flame_lite, CubicNeRFCamera
from .modules import StyleUNet, MTAttention, PointsDecoder

class gpavatar_r2g(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        # model config
        self.raw_cam_size = 128
        # trainable params
        self.style_tplane = StyleUNet(in_size=512, out_size=256, in_dim=3, out_dim=96, activation=False)
        self.nerf_mlp = PointsDecoder(in_dim=32, out_dim=32, points_number=5023, points_k=8)
        self.up_renderer = StyleUNet(in_size=512, out_size=512, in_dim=32, out_dim=3)
        # attn module
        self.query_style_tplane = torch.nn.Parameter(torch.randn(1, 1, 3, 32, 256, 256), requires_grad=True)
        self.attn_module = MTAttention(dim=32, qkv_bias=True)
        # models
        self.flame = flame_lite(n_shape=100, n_exp=50, scale=5.0)
        self.nerf_camera = CubicNeRFCamera(12.0, [128, 128], 32)
        # load ckpt
        self._abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        _ckpt_path = os.path.join(self._abs_script_path, 'checkpoints', 'one_model.ckpt')
        ckpt = torch.load(_ckpt_path, map_location='cpu')['state_dict']
        for key in list(ckpt.keys()):
            if 'percep_loss' in key:
                del ckpt[key]
        self.load_state_dict(ckpt, strict=False)

    def build_avatar(self, inp_track=None):
        if inp_track is None:
            inp_track = torch.load(os.path.join(self._abs_script_path, 'checkpoints', 'elon_musk.pth'), map_location='cpu')
        inp_image = inp_track['feature_image'][None].cuda()
        inp_shape, inp_trans = inp_track['feature_shape'], inp_track['frame_trans']
        # merge multiple input images
        batch_size, v, c, h, w = inp_image.shape
        inp_image = inp_image.flip(dims=(-2,))
        inp_image = inp_image.reshape(-1, c, h, w)
        # processing data
        tex_tplanes = self.style_tplane(inp_image)
        tex_tplanes = tex_tplanes.reshape(
            batch_size, v, 3, -1, tex_tplanes.shape[-2], tex_tplanes.shape[-1]
        )
        self.texture_planes = self.attn_module(
            self.query_style_tplane.expand(batch_size, -1, -1, -1, -1, -1),
            tex_tplanes
        )
        self.inp_shape, self.inp_trans = inp_shape, inp_trans
        print('Avatar built.')

    def forward(self, expression, pose, transform_matrix=None):
        # get flame points
        points = self.flame(
            expression_params=expression, pose_params=pose,
            shape_params=self.inp_shape[None].to(expression.device), 
        )
        # set camera
        if transform_matrix is None:
            transform_matrix = self.inp_trans.to(expression.device)
        self.nerf_camera.set_position(transform_matrix=transform_matrix)
        # render results
        _, gen_fine, params_dict = self.nerf_camera.render(
            nerf_query_fn=self.nerf_mlp, noise=False, background=0.0, 
            # nerf_fn params
            points_position=points, tex_tplanes=self.texture_planes
        )
        gen_sr = self.up_renderer(gen_fine)
        return gen_sr
