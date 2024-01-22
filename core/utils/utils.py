# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import re
import json
import torch
import colored
import logging
import matplotlib
from colored import stylize

from core.utils.rtqdm import tqdm
import core.utils.distributed as dist

def mprint(input_str, write=False, coloring=True, set_color=None):
    if dist.get_rank() == 0:
        if coloring:
            input_str = stylize(
                str(input_str), 
                colored.fg(7 if set_color is None else set_color) # + colored.attr("bold")
            )
            print(input_str)
        else:
            print(str(input_str))
        if write:
            assert isinstance(input_str, str), 'Must be str: {}.'.format(type(input_str))
            input_str = re.sub(r'\[38;5;[1-9]*m', '', input_str)
            input_str = re.sub(r'\[0m', '', input_str)
            logging.getLogger('TRAIN_LOG').info(input_str)
    else:
        pass


def vis_depth(depth_map, color_map='magma'):
    # [inferno', 'magma', 'bone', 'gray', 'Greys']
    assert depth_map.dim() == 4, depth_map.dim()
    depth_map = depth_map.permute(0, 2, 3, 1)
    assert depth_map.shape[-1] == 1, depth_map.shape
    depth_map -= depth_map.min()
    depth_map /= depth_map.max()
    device = depth_map.device
    depth_color = matplotlib.colormaps[color_map]
    depth_map = torch.tensor(
        depth_color(1-depth_map[..., 0].cpu().numpy())[..., :3], device=device
    ).permute(0, 3, 1, 2)
    return depth_map


def visulize(images, save_path, bbox=None, mask=None, all_write=False, nrow=4):
    # bbox: [xmin, ymin, xmax, ymax] in range(0, 1).
    import torch
    import torchvision
    assert isinstance(images, torch.Tensor), "Must be tensor: {}.".format(type(images))
    if all_write or dist.get_rank() == 0:
        if bbox is not None:
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * images.shape[-1] # width
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * images.shape[-2] # height
            images = torch.stack([
                    torchvision.utils.draw_bounding_boxes(
                        f.to(torch.uint8), bbox[idx:idx+1],
                        width=3, colors=(255, 255, 255)
                    ) for idx, f in enumerate(images)
                ]
            )
        if mask is not None:
            images[~mask.expand(images.shape)] *= 0.5
        images = torchvision.utils.make_grid(images, nrow=nrow, padding=0) / 255.0
        torchvision.utils.save_image(images, save_path)
    else:
        pass


def to_gpu(batch_data_dict, device):
    import torch
    for key in batch_data_dict.keys():
        if isinstance(batch_data_dict[key], torch.Tensor):
            batch_data_dict[key] = batch_data_dict[key].to(device)
    return batch_data_dict


def set_random(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_json(path):
    with open(path, 'r') as f:
        result = json.load(f)
    return result


def save_json(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f)


def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ':\n'
            out_line += pretty_dict(value, indent+1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(1)) + '\n'
            else:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(2)) + '\n'
    if indent == 0:
        max_length = 0
        for line in out_line.split('\n'):
            max_length = max(max_length, len(line.split('\t')[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split('\n'):
            if '\t' in line:
                aligned_number = max_length - len(line.split('\t')[0])
                line = line.replace('\t',  aligned_number * ' ')
            aligned_line += line+'\n'
        return aligned_line[:-2]
    return out_line


def list_all_files(dir_path):
    pair = os.walk(dir_path)
    result = []
    for path, dirs, files in pair:
        if len(files):
            for file_name in files:
                result.append(os.path.join(path, file_name))
    return result


def device_parser(str_device):
    def parser_dash(str_device):
        if not len(str_device):
            return []
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1])+1)]
        return device_id

    if 'cpu' in str_device:
        device_id = ['cpu']
    else:
        split_str = str_device.split(',')
        device_id = []
        for s in split_str:
            device_id += parser_dash(s)
    return device_id


def build_time_string(start_time, this_time, this_iter, max_iter):
    frame_time = (this_time - start_time) / max(this_iter, 1)
    its = 1 / max(frame_time, 1e-5)
    remain_time = frame_time * (max_iter - this_iter)
    results = stylize('{}/{} '.format(this_iter+1, max_iter), colored.fg(2)) + ' [' + \
              stylize('{:.1f}s'.format(frame_time*this_iter), colored.fg(3)) + ' < ' + \
              stylize('{:.1f}s'.format(remain_time), colored.fg(4)) + '] ' + \
              stylize(', {:.1f} it/s.'.format(its), colored.fg(1))
    return results


tqdm = tqdm
