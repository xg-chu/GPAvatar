# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import time
import torch
import pickle
import random
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from core.utils.lmdb_utils import LMDBEngine
from core.data.tools import norm_transform

class InferenceLightning(torch.utils.data.Dataset):
    def __init__(self, lightning_path, augment_config, meta_info, if_video=False):
        super().__init__()
        # build path
        self.if_video = if_video
        self._background_value = 0.0
        self._augment_config = augment_config
        if os.path.isdir(lightning_path):
            self._img_lmdb_path = os.path.join(lightning_path, 'img_lmdb')
            if not os.path.exists(self._img_lmdb_path):
                print('Only driving data, no image data.')
                self._img_lmdb_path = None
            self._records_path = os.path.join(lightning_path, 'lightning.pkl')
        else:
            self._img_lmdb_path = None
            self._records_path = lightning_path
        self._meta_info = meta_info
        with open(self._records_path, 'rb') as f:
            _records = pickle.load(f)
            for fkey in _records.keys():
                for pkey in _records[fkey].keys():
                    if isinstance(_records[fkey][pkey], np.ndarray):
                        _records[fkey][pkey] = torch.tensor(_records[fkey][pkey])
            self._records = _records
        _records_keys = [key for key in self._records.keys()]
        print('Total frames: {}'.format(len(_records_keys)))
        self._records_keys = sorted(_records_keys, key=lambda x: int(x.split('_')[-1]))
        if if_video:
            self._records = smooth_records(self._records, self._records_keys)
        else:
            random.shuffle(self._records_keys)

    def slice(self, slice):
        self._records_keys = self._records_keys[:slice]

    def __getitem__(self, index):
        frame_key = self._records_keys[index]
        return self._load_one_record(self._records[frame_key], frame_key)

    def __len__(self, ):
        return len(self._records_keys)

    def _init_lmdb_database(self):
        # print('Init the LMDB Database!')
        self._img_lmdb_engine = LMDBEngine(self._img_lmdb_path, write=False)

    def _load_one_record(self, record, frame_key):
        if not hasattr(self, '_img_lmdb_engine') and self._img_lmdb_path is not None:
            self._init_lmdb_database()
        one_data = {
            'd_transforms': record['transform_matrix'].float(), 
            'd_shapes': record['mica_shape'].float(),
            'd_expressions': record['emoca_expression'].float(),
            'd_poses': record['emoca_pose'].float(),
            'infos': {'frame_key':frame_key},
        }
        if hasattr(self, '_img_lmdb_engine'):
            image_tensor = self._img_lmdb_engine[frame_key].float()
            image_tensor = norm_transform(image_tensor, self._augment_config)
            one_data['d_images'] = image_tensor.float().detach()
        return one_data


def smooth_records(records, records_keys, alpha=0.9):
    def smooth_sequence(batched_tensor, alpha=0.9):
        smoothed_data = [batched_tensor[0]]  # Initialize the smoothed data with the first value of the input data
        for i in range(1, batched_tensor.shape[0]):
            smoothed_value = alpha * batched_tensor[i] + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(smoothed_value)
        smoothed_data = torch.stack(smoothed_data)
        return smoothed_data
    smoothed_exp, smoothed_pose, smoothed_transform = [], [], []
    for key in records_keys:
        smoothed_exp.append(records[key]['emoca_expression'])
        smoothed_pose.append(records[key]['emoca_pose'])
        smoothed_transform.append(records[key]['transform_matrix'])
    smoothed_exp = smooth_sequence(torch.stack(smoothed_exp), alpha=alpha)
    smoothed_pose = smooth_sequence(torch.stack(smoothed_pose), alpha=alpha)
    batched_transform = torch.stack(smoothed_transform)
    smoothed_rotate = smooth_sequence(matrix_to_rotation_6d(batched_transform[:, :3, :3]), alpha=alpha)
    smoothed_translate = smooth_sequence(batched_transform[:, :3, 3], alpha=alpha)
    smoothed_transform = torch.cat([rotation_6d_to_matrix(smoothed_rotate), smoothed_translate[..., None]], dim=-1)
    for idx, key in enumerate(records_keys):
        records[key]['emoca_expression'] = smoothed_exp[idx]
        records[key]['emoca_pose'] = smoothed_pose[idx]
        records[key]['transform_matrix'] = smoothed_transform[idx]
    return records
