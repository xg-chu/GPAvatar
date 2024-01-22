# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import torch
import random
import pickle
import numpy as np
from core.libs.FLAME import FLAME
from core.data.tools import norm_transform
from core.utils.lmdb_utils import LMDBEngine
from core.utils.registry import DATA_BUILDER_REGISTRY

@DATA_BUILDER_REGISTRY.register()
class MVLightningPoints(torch.utils.data.Dataset):
    def __init__(self, dataset_config, augment_config, stage):
        super().__init__()
        # build path
        self._stage = stage
        self._training = stage == 'train'
        assert stage in ['train', 'val', 'test']
        self._augment_config = augment_config
        self._records_path = dataset_config['RECORDS_PATH']
        self._meta_path = dataset_config['METADATA_PATH']
        self._img_lmdb_path = dataset_config['DATASET_PATH']
        self._background_value = dataset_config['BACKGROUND_VALUE']
        self._sample_number = augment_config['SAMPLE_NUMBER']
        # build records
        with open(self._records_path, 'rb') as f:
            _pickle_data = pickle.load(f)
            self._records_data = _pickle_data['records']
            self._records_frames = _pickle_data['meta_info'][stage]
        with open(self._meta_path, 'r') as f:
            self._meta_info = json.load(f)
        self.flame_model = FLAME(n_shape=100, n_exp=50)
        # build video_id info
        self._id_info = {}
        for key in self._records_frames:
            video_id = key.split('_')[1]
            if video_id not in self._id_info.keys():
                self._id_info[video_id] = []
            self._id_info[video_id].append(key)
        for video_id in self._id_info.keys():
            self._id_info[video_id] = sorted(
                self._id_info[video_id], key=lambda x:int(x.split('_')[-1])
            )
        if self._stage == 'test':
            self._records_frames = [key for key in self._records_frames if int(key.split('_')[-1]) != 0]
        # build cross id dataset
        self.cid = False
        if self.cid:
            self.cid_mapping = {}
            video_ids = list(self._id_info.keys())
            video_ids = sorted(video_ids)
            for idx, video_id in enumerate(video_ids):
                if idx < len(video_ids) - 1:
                    self.cid_mapping[video_id] = video_ids[idx+1]
                else:
                    self.cid_mapping[video_id] = video_ids[0]

    def slice(self, slice):
        self._records_frames = self._records_frames[:slice]

    def __getitem__(self, index):
        frame_key = self._records_frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._records_frames)

    @property
    def meta_info(self, ):
        return self._meta_info

    def _init_lmdb_database(self):
        # print('Init the LMDB Database!')
        self._lmdb_engine = LMDBEngine(self._img_lmdb_path, write=False)

    def _choose_feature_image(self, frame_key, number=1):
        video_id = frame_key.split('_')[1]
        if self.cid:
            video_id = self.cid_mapping[video_id]
        if not self._training:
            if number == 1:
                feature_image_key = [self._id_info[video_id][0]]
            elif number == 2:
                feature_image_key = [self._id_info[video_id][0], self._id_info[video_id][-1]]
            elif number == 3:
                feature_image_key = [
                    self._id_info[video_id][0], 
                    self._id_info[video_id][len(self._id_info[video_id])//2], 
                    self._id_info[video_id][-1]
                ]
            elif number == 4:
                feature_image_key = [
                    self._id_info[video_id][0], 
                    self._id_info[video_id][int(len(self._id_info[video_id])*0.3)], 
                    self._id_info[video_id][int(len(self._id_info[video_id])*0.6)], 
                    self._id_info[video_id][-1]
                ]
            else:
                feature_image_key = random.sample(self._id_info[video_id], k=number)
            # feature_image_key = [frame_key] * number
        else:
            candidate_key = [key for key in self._id_info[video_id] if key != frame_key]
            # candidate_key = self._id_info[video_id]
            feature_image_key = random.sample(candidate_key, k=number)
        of_images, f_images, f_shapes = [], [], []
        for key in feature_image_key:
            this_record = self._records_data[key]
            feature_image_tensor = self._lmdb_engine[key].float()
            feature_image_tensor = norm_transform(feature_image_tensor, self._augment_config)
            of_images.append(feature_image_tensor.clone())
            feature_image_transform = torch.tensor(this_record['transform_matrix']).float()
            feature_image_tensor = perspective_transform(
                feature_image_tensor[None], feature_image_transform[None], self.meta_info,
                fill=self._background_value
            )[0]
            f_images.append(feature_image_tensor.float())
            f_shapes.append(torch.tensor(this_record['mica_shape']).float())
        of_images = torch.stack(of_images)
        f_images = torch.stack(f_images)
        f_shapes = torch.stack(f_shapes).mean(dim=0)
        return of_images, f_images, f_shapes

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        # feature image
        of_images, f_images, f_shape = self._choose_feature_image(
            frame_key, number=self._sample_number
        )
        # driven image
        this_record = self._records_data[frame_key]
        image_tensor = self._lmdb_engine[frame_key].float()
        image_tensor = norm_transform(image_tensor, self._augment_config)
        record_bbox = torch.tensor(this_record['bbox']).float()
        record_expression = torch.tensor(this_record['emoca_expression']).float()
        record_shape = torch.tensor(this_record['mica_shape']).float()
        record_pose = torch.tensor(this_record['emoca_pose']).float()
        record_transform = torch.tensor(this_record['transform_matrix']).float()
        point_tensor, _, _ = self.flame_model(
            shape_params=record_shape[None], 
            expression_params=record_expression[None],
            pose_params=record_pose[None],
        )
        point_tensor = point_tensor[0].float() * self._meta_info['flame_scale']
        one_data = {
            'f_images': f_images, 'f_shape': f_shape, 'of_images': of_images,
            'd_images': image_tensor, 'd_points': point_tensor, 
            'd_bbox': record_bbox, 'd_transforms': record_transform, 
            'd_expressions': record_expression, 'd_poses': record_pose,
            'background': self._background_value, 'infos': {'frame_key':frame_key},
        }
        return one_data

    def _build_mask_from_bbox(self, bbox_tensor, image_shape):
        bbox_tensor = bbox_tensor.float().clamp(min=0, max=max(image_shape))
        x1, y1, x2, y2 = bbox_tensor.to(torch.long)
        bbox_mask = torch.zeros((1, )+image_shape[1:], dtype=torch.bool)
        bbox_mask[:, y1:y2, x1:x2] = True
        return bbox_mask


def perspective_transform(feature, transforms, meta_info, fill=1.0):
    def _build_cameras_kwargs(batch_size, image_size, focal_length, principal_point, device):
        screen_size = torch.tensor(
            [image_size, image_size], device=device, dtype=torch.float32
        )[None].repeat(batch_size, 1)
        principal_point = torch.tensor(
            principal_point, device=device, dtype=torch.float32
        )[None].repeat(batch_size, 1)
        focal_length = torch.tensor(focal_length, device=device, dtype=torch.float32)
        cameras_kwargs = {
            'principal_point': principal_point, 'focal_length': focal_length, 
            'image_size': screen_size, 'device': device
        }
        return cameras_kwargs

    assert feature.shape[-2] == feature.shape[-1]
    from pytorch3d.renderer import PerspectiveCameras
    from torchvision.transforms.functional import perspective
    batch_size, image_size = feature.shape[0], feature.shape[-1]
    corner_points = torch.tensor(
        [[-1, 1, 0], [1, 1, 0], 
         [1, -1, 0], [-1, -1, 0]]
    ).type_as(feature)[None].repeat(feature.shape[0], 1, 1)
    cameras = PerspectiveCameras(
        **_build_cameras_kwargs(
            batch_size, image_size, meta_info['focal_length'], meta_info['principal_point'], transforms.device
        )
    )
    ori_corners = cameras.transform_points_screen(
        corner_points, R=transforms[:, :3, :3], T=transforms[:, :3, 3]
    )[..., :2]
    ori_corners = ori_corners.int().cpu().numpy().tolist()
    corners = [[0, 0], [image_size-1, 0], [image_size-1, image_size-1], [0, image_size-1]]
    feature_perspective = torch.stack(
        [perspective(f, ori_corners[idx], corners, fill=fill) for idx, f in enumerate(feature)]
    )
    return feature_perspective
