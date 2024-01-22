import os
import json
import colored
from colored import stylize

def device_parser(str_device):
    def parser_dash(str_device):
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1])+1)]
        return device_id
    if 'cpu' in str_device:
        device_id = ['cpu']
    else:
        device_id = str_device.split(',')
        device_id = [parser_dash(i) for i in device_id]
    res = []
    for i in device_id:
        res += i
    return res


def save_json(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f)


def set_devices(target_devices: str):
    if target_devices == 'cpu':
        return ['cpu']
    devices = device_parser(target_devices)
    devices_str = [str(d) for d in devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices_str)
    import torch
    if torch.cuda.device_count() != len(devices):
        raise Exception(
            'Pytorch is imported before setting the devices: {}!'.format(
                torch.cuda.device_count()
            )
        )
    return devices


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
