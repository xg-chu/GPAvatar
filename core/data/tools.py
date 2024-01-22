# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import io
import json
import lmdb
import torch
import torchvision
import numpy as np

def load_img(file_name, lmdb_txn=None, channel=3):
    # load image as [channel(RGB), image_height, image_width]
    if channel == 3:
        _mode = torchvision.io.ImageReadMode.RGB 
    elif channel == 4:
        _mode = torchvision.io.ImageReadMode.RGB_ALPHA
    else:
        _mode = torchvision.io.ImageReadMode.GRAY
    if lmdb_txn is not None:
        image_buf = lmdb_txn.get(file_name.encode())
        image_buf = torch.tensor(np.frombuffer(image_buf, dtype=np.uint8))
        image = torchvision.io.decode_image(image_buf, mode=_mode)
    else:
        image = torchvision.io.read_image(file_name, mode=_mode)
    assert image is not None, file_name
    return image

def load_torch(file_name, lmdb_txn=None):
    torch_buf = io.BytesIO(lmdb_txn.get(file_name.encode()))
    torch_obj = torch.load(torch_buf)
    return torch_obj

def load_records_from_json(json_path, stage):
    with open(json_path, 'r') as f:
        records = json.load(f)
    return records[stage]

def load_records_from_lmdb(lmdb_path, stage):
    _lmdb_env = lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
    )
    records = []
    with _lmdb_env.begin() as txn:
        all_keys = list(txn.cursor().iternext(values=False))
        print('Load data, length:{}.'.format(len(all_keys)))
        for idx, key in enumerate(all_keys):
            if 'jpg' in key.decode() or 'png' in key.decode() or 'jpeg' in key.decode():
                records.append(key.decode())
            else:
                print(f'{key.decode()} is not images.')
    return records


def norm_transform(data, aug_config, norm=True):
    # resize
    target_size = aug_config['RESIZE']
    h0, w0 = data.shape[1], data.shape[2]  # orig hw
    scale_r = target_size / min(h0, w0)
    h1, w1 = max(int(h0 * scale_r), target_size), max(int(w0 * scale_r), target_size)
    if scale_r != 1:
        data = torchvision.transforms.functional.resize(data, (h1, w1), antialias=True)
    if not norm:
        return data
    # normalization
    assert aug_config['RANGE'] in [[0, 1], [-1, 1]], "Only support (0, 1) and (-1, 1)"
    if aug_config['RANGE'] == [0, 1]:
        norm_mean = (0, 0, 0); norm_std = (256, 256, 256)
    else:
        norm_mean = (128, 128, 128); norm_std = (128, 128, 128)
    if data.shape[0] == 3:
        data = torchvision.transforms.functional.normalize(data, norm_mean, norm_std)
    elif data.shape[0] == 1 and data.dtype==torch.bool:
        data = data
    else:
        raise NotImplementedError
    return data


def unnorm_transform(data, aug_config):
    assert aug_config['RANGE'] in [[0, 1], [-1, 1]], "Only support (0, 1) and (-1, 1)"
    if aug_config['RANGE'] == [0, 1]:
        data = torchvision.transforms.functional.normalize(data, (0, 0, 0), (1.0/255, 1.0/255, 1.0/255))
    else:
        data = torchvision.transforms.functional.normalize(data, (-1, -1, -1), (1.0/128, 1.0/128, 1.0/128))
    return data


def perspective_input(feature, transforms, meta_info, fill=1.0):
    def _build_cameras_kwargs(batch_size, image_size, focal_length, principal_point, device):
        screen_size = torch.tensor(
            [image_size, image_size], device=device, dtype=torch.float32
        )[None].repeat(batch_size, 1)
        if isinstance(principal_point, torch.Tensor):
            principal_point = principal_point[None].repeat(batch_size, 1)
        else:
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
