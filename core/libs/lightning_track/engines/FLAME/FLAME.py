"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""
# Modified from smplx code for FLAME
import os

import torch
import pickle
import inspect
import numpy as np
import torch.nn as nn

from .lbs import lbs, batch_rodrigues, vertices2landmarks

class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, n_shape, n_exp):
        super(FLAME, self).__init__()
        # print("creating the FLAME Model")
        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.flame_path = os.path.join(_abs_script_path, 'assets')
        flame_ckpt = torch.load(os.path.join(self.flame_path, 'FLAME.pth'))
        flame_model = flame_ckpt['flame_model']
        flame_lmk = flame_ckpt['lmk_embeddings']
        flame_lmk_mediapipe = flame_ckpt['lmk_embeddings_mediapipe']
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', flame_model['f'])
        self.register_buffer('v_template', flame_model['v_template'])
        shapedirs = flame_model['shapedirs']
        self.register_buffer('shapedirs', torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], 2))
        num_pose_basis = flame_model['posedirs'].shape[-1]
        self.register_buffer('posedirs', flame_model['posedirs'].reshape(-1, num_pose_basis).T)
        self.register_buffer('J_regressor', flame_model['J_regressor'])
        parents = flame_model['kintree_table'][0]
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', flame_model['weights'])
        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=torch.float32, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=torch.float32, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        self.register_buffer('lmk_faces_idx', flame_lmk['static_lmk_faces_idx'])
        self.register_buffer('lmk_bary_coords', flame_lmk['static_lmk_bary_coords'].to(dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', flame_lmk['dynamic_lmk_faces_idx'].to(dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', flame_lmk['dynamic_lmk_bary_coords'].to(dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx', flame_lmk['full_lmk_faces_idx'].to(dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords', flame_lmk['full_lmk_bary_coords'].to(dtype=self.dtype))

        # static MEDIAPIPE landmark embeddings for FLAME
        self.register_buffer('lmk_faces_idx_mediapipe', flame_lmk_mediapipe['lmk_face_idx'])
        self.register_buffer('lmk_bary_coords_mediapipe', flame_lmk_mediapipe['lmk_b_coords'])
        self.register_buffer('mediapipe_idx', flame_lmk_mediapipe['landmark_indices'])

        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        # print("FLAME Model Done.")

    def get_faces(self, ):
        return self.faces_tensor.long()

    def _find_dynamic_lmk_idx_and_bcoords(
            self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
            neck_kin_chain, dtype=torch.float32
        ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    # @torch.no_grad()
    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1) # TODO: is this correct?
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([
                pose_params[:, :3], self.neck_pose.expand(batch_size, -1), 
                pose_params[:, 3:], eye_pose_params
            ], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        )
        # find lmk
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        return vertices, landmarks2d, landmarks3d


class FLAMEDense(FLAME): 
    def __init__(
            self, n_shape, n_exp
        ):
        super().__init__(n_shape, n_exp)
        
    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        vertices, landmarks2d_68, landmarks3d = super().forward(
            shape_params, expression_params, pose_params, eye_pose_params
        )
        batch_size = shape_params.shape[0]
        lmk_faces_idx_mediapipe = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords_mediapipe = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(batch_size, -1, -1).contiguous()
        landmarks2d_mediapipe = vertices2landmarks(
            vertices, self.faces_tensor,
            lmk_faces_idx_mediapipe,
            lmk_bary_coords_mediapipe
        )
        return vertices, landmarks2d_68, landmarks2d_mediapipe


class FLAMETex(nn.Module):
    def __init__(self, n_tex=50):
        super(FLAMETex, self).__init__()
        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.flame_path = os.path.join(_abs_script_path, 'assets')
        tex_space = torch.load(os.path.join(self.flame_path, 'FLAME_texture.pth'))
        # FLAME texture
        scale = 1
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = texture_mean.float()[None, ...] * scale
        texture_basis = texture_basis[:, :n_tex].float()[None, ...] * scale
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)
        self.register_buffer('verts_uvs', tex_space['verts_uvs'].float())
        self.register_buffer('verts_idx', tex_space['verts_idx'].long())
        self.register_buffer('textures_idx', tex_space['textures_idx'].long())
        # MASK
        self.masks = tex_space['FLAME_masks']
        # with open(os.path.join(self.flame_path, 'FLAME_masks.pkl'), 'rb') as f:
        #     ss = pickle.load(f, encoding='latin1')
        #     self.masks = Struct(**ss)
        # with open(os.path.join('FLAME_masks.pkl'), 'rb') as f:
        #     ss = pickle.load(f, encoding='latin1')

    def get_face_mask(self, ):
        # 'eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 
        # 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region'
        return self.masks['face']

    def get_tuv(self, ):
        return {'verts_uvs': self.verts_uvs, 'verts_idx': self.verts_idx, 'textures_idx': self.textures_idx}

    def forward(self, texcode, image_size=512):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = torch.nn.functional.interpolate(texture, image_size, mode='bilinear')
        texture = texture[:, [2, 1, 0], :, :]
        return texture / 255.


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

